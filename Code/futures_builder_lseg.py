"""
futures_builder_lseg.py — Per-Contract-Month OHLCV+OI Database (LSEG)
=========================================================================
LSEG-API replacement for ICEBREAKER/Futures/Code/futures_builder.py
(icepython-based). Same output schema, same 7-commodity universe, same
FND/LTD contract-calendar logic (carried over unchanged — it's the same
vendor-agnostic pandas/numpy math already used in the Rollex migration).
Only the price fetch and symbol construction changed.

LSEG outright-contract RIC convention: <root><month_code><last digit of
year>, with a "^N" suffix disambiguating repeat occurrences of the same
month-code+year-digit pair across decades (e.g. KCH1^1 = Mar 2011,
KCH1^2 = Mar 2021). Not currently-listed contracts resolve bare (no
suffix); everything else needs the right ^N, discovered here by trying
candidates in order against the real fetch rather than guessing a formula.

Output: ../Database/{comm}_futures.parquet (one file per commodity)
Schema: Date, commodity, ice_symbol, month, year, FND, LTD,
        Open, High, Low, settlement, volume, open_interest

Usage:
    python futures_builder_lseg.py              # all commodities, incremental
    python futures_builder_lseg.py --full        # all commodities, full rebuild
    python futures_builder_lseg.py KC RC         # specific commodities
    python futures_builder_lseg.py KC --full     # single commodity, full rebuild
"""

import argparse
import calendar as cal_module
import datetime
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
pd.set_option("future.no_silent_downcasting", True)  # silences a harmless lseg.data internal FutureWarning
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, EasterMonday,
    USMemorialDay, USLaborDay, USThanksgivingDay,
)
from pandas.tseries.offsets import CustomBusinessDay

# ── PATHS / LOGGING ──────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).resolve().parent
DB_DIR   = CODE_DIR.parent / "Database"
DB_DIR.mkdir(exist_ok=True)
LOG_DIR  = CODE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_DIR / "futures_builder_lseg.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)

START_YEAR = 2011
FIELDS     = ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS", "OPINT_1"]
COL_MAP    = {"OPEN_PRC": "Open", "HIGH_1": "High", "LOW_1": "Low",
              "TRDPRC_1": "settlement", "ACVOL_UNS": "volume", "OPINT_1": "open_interest"}

# ── HOLIDAY CALENDARS (identical to Rollex/COT_ALL migrations) ─────────────

class USExchangeHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay",     month=1,  day=1,  observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth",      month=6,  day=19, observance=nearest_workday),
        Holiday("IndependenceDay", month=7,  day=4,  observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas",       month=12, day=25, observance=nearest_workday),
    ]


def _build_uk_holidays(start_year: int = 2004, end_year: int = 2040) -> pd.DatetimeIndex:
    base = AbstractHolidayCalendar(rules=[
        Holiday("NewYearsDay", month=1,  day=1,  observance=nearest_workday),
        GoodFriday,
        EasterMonday,
        Holiday("Christmas",   month=12, day=25, observance=nearest_workday),
        Holiday("BoxingDay",   month=12, day=26, observance=nearest_workday),
    ])
    hols = list(base.holidays(start=pd.Timestamp(f"{start_year}-01-01"), end=pd.Timestamp(f"{end_year}-12-31")))
    for year in range(start_year, end_year + 1):
        d = pd.Timestamp(year, 5, 1)
        while d.dayofweek != 0:
            d += pd.Timedelta(days=1)
        hols.append(d)
        d = pd.Timestamp(year, 5, cal_module.monthrange(year, 5)[1])
        while d.dayofweek != 0:
            d -= pd.Timedelta(days=1)
        hols.append(d)
        d = pd.Timestamp(year, 8, cal_module.monthrange(year, 8)[1])
        while d.dayofweek != 0:
            d -= pd.Timedelta(days=1)
        hols.append(d)
    return pd.DatetimeIndex(sorted(set(hols)))


US_BDAY  = CustomBusinessDay(calendar=USExchangeHolidayCalendar())
UK_BDAY  = CustomBusinessDay(holidays=_build_uk_holidays())
BDAY_CAL = {"US": US_BDAY, "UK": UK_BDAY}

# ── DATE HELPERS / FND-LTD RULE FACTORIES (identical to Rollex migration) ──

def first_bd(year, month, bday):
    d = pd.Timestamp(year=year, month=month, day=1).normalize()
    while d != (d + 0 * bday):
        d += pd.Timedelta(days=1)
    return d

def last_bd(year, month, bday):
    last_day = cal_module.monthrange(year, month)[1]
    d = pd.Timestamp(year=year, month=month, day=last_day).normalize()
    while d != (d + 0 * bday):
        d -= pd.Timedelta(days=1)
    return d

def nth_bd(year, month, n, bday):
    d = first_bd(year, month, bday)
    for _ in range(n - 1):
        d = (d + 1 * bday).normalize()
    return d

def preceding_month(year, month):
    return (year - 1, 12) if month == 1 else (year, month - 1)

def fnd_first_bd_minus(n):
    return lambda year, month, bday: (first_bd(year, month, bday) - n * bday).normalize()

def fnd_nth_bd_minus(nth, n):
    return lambda year, month, bday: (nth_bd(year, month, nth, bday) - n * bday).normalize()

def ltd_last_bd_minus(n):
    return lambda year, month, bday: (last_bd(year, month, bday) - n * bday).normalize()

def ltd_last_bd_preceding_month():
    def calc(year, month, bday):
        py, pm = preceding_month(year, month)
        return last_bd(py, pm, bday)
    return calc

def ltd_calendar_days_before_month_start(n, roll="preceding"):
    def calc(year, month, bday):
        d = (pd.Timestamp(year, month, 1) - pd.Timedelta(days=n)).normalize()
        step = 1 if roll == "following" else -1
        while d != (d + 0 * bday):
            d += pd.Timedelta(days=step)
        return d
    return calc

# ── COMMODITY CONFIG — LSEG roots + identical FND/LTD rules ────────────────

@dataclass
class CommodityConfig:
    lseg_root: str
    calendar:  str
    months:    List[str]
    month_num: Dict[str, int]
    fnd_rule:  object
    ltd_rule:  Callable

CONTRACT_CONFIG = {
    "KC": CommodityConfig("KC",  "US", ["H","K","N","U","Z"], {"H":3,"K":5,"N":7,"U":9,"Z":12},
                           fnd_first_bd_minus(7), ltd_last_bd_minus(8)),
    "CC": CommodityConfig("CC",  "US", ["H","K","N","U","Z"], {"H":3,"K":5,"N":7,"U":9,"Z":12},
                           fnd_nth_bd_minus(nth=6, n=10), ltd_last_bd_minus(11)),
    "CT": CommodityConfig("CT",  "US", ["H","K","N","V","Z"], {"H":3,"K":5,"N":7,"V":10,"Z":12},
                           fnd_first_bd_minus(5), ltd_last_bd_minus(17)),
    "SB": CommodityConfig("SB",  "US", ["H","K","N","V"], {"H":3,"K":5,"N":7,"V":10},
                           "after_ltd", ltd_last_bd_preceding_month()),
    "RC": CommodityConfig("LRC", "UK", ["F","H","K","N","U","X"], {"F":1,"H":3,"K":5,"N":7,"U":9,"X":11},
                           fnd_first_bd_minus(4), ltd_last_bd_minus(4)),
    "LCC":CommodityConfig("LCC", "UK", ["H","K","N","U","Z"], {"H":3,"K":5,"N":7,"U":9,"Z":12},
                           "after_ltd", ltd_last_bd_minus(11)),
    "LSU":CommodityConfig("LSU", "UK", ["H","K","Q","V","Z"], {"H":3,"K":5,"Q":8,"V":10,"Z":12},
                           ltd_calendar_days_before_month_start(15, roll="following"),
                           ltd_calendar_days_before_month_start(16, roll="preceding")),
}
COMMODITIES = list(CONTRACT_CONFIG.keys())


def fnd_ltd(cfg: CommodityConfig, year: int, month: int):
    bday = BDAY_CAL[cfg.calendar]
    ltd = cfg.ltd_rule(year, month, bday)
    fnd = (ltd + 1 * bday).normalize() if cfg.fnd_rule == "after_ltd" else cfg.fnd_rule(year, month, bday)
    return fnd, ltd

# ── LSEG SYMBOL RESOLUTION + FETCH ──────────────────────────────────────────

def resolve_and_fetch(ld, root: str, month_code: str, year: int, start: str, end: str, bday):
    """Try the bare RIC then ^1/^2/^3 in order; return (symbol, dataframe) for
    whichever candidate actually has data, or (None, None) if none do.

    LSEG's raw per-contract series have real gaps on days the exchange was
    open (same issue found and fixed in the Rollex/Roll Yield migrations —
    checked here too: ~80% density on a representative contract). Reindexes
    onto the contract's own exchange business-day calendar between its own
    first and last real print, and linearly interpolates strictly-internal
    holes only — never extrapolating past the real data's own edges."""
    base = f"{root}{month_code}{year % 10}"
    for cand in (base, f"{base}^1", f"{base}^2", f"{base}^3"):
        try:
            df = ld.get_history(universe=[cand], fields=FIELDS, start=start, end=end,
                                 interval="daily", count=10000)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df.rename(columns=COL_MAP)
        if "settlement" not in df.columns:
            continue
        df = df[df["settlement"].notna()]
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index).normalize()
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=bday)
        df = df.reindex(full_idx)
        for col in ["Open", "High", "Low", "settlement", "volume", "open_interest"]:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit_area="inside")
        df = df[df["settlement"].notna()]
        return cand, df
    return None, None


def build_commodity(ld, comm: str, start_year: int, end_year: int, incremental_from: dict | None) -> pd.DataFrame:
    cfg = CONTRACT_CONFIG[comm]
    rows = []
    today = pd.Timestamp.today().normalize()

    for year in range(start_year, end_year + 1):
        for month_code in cfg.months:
            month_num = cfg.month_num[month_code]
            fnd, ltd = fnd_ltd(cfg, year, month_num)

            if incremental_from is not None:
                key = (month_code, year)
                if key not in incremental_from:
                    continue
                fetch_start = incremental_from[key]
                fetch_end = min(ltd, today).strftime("%Y-%m-%d") if ltd <= today else today.strftime("%Y-%m-%d")
            else:
                fetch_start = (ltd - pd.Timedelta(days=1400)).strftime("%Y-%m-%d")
                fetch_end   = min(ltd, today).strftime("%Y-%m-%d") if ltd <= today else today.strftime("%Y-%m-%d")

            if pd.Timestamp(fetch_start) > pd.Timestamp(fetch_end):
                continue

            bday = BDAY_CAL[cfg.calendar]
            sym, df = resolve_and_fetch(ld, cfg.lseg_root, month_code, year, fetch_start, fetch_end, bday)
            if df is None or df.empty:
                continue

            df = df.copy()
            df.index = pd.to_datetime(df.index).normalize()
            df.index.name = "Date"
            df = df.reset_index()
            df["commodity"]  = comm
            df["ice_symbol"] = sym
            df["month"]      = month_code
            df["year"]       = year
            df["FND"]        = fnd
            df["LTD"]        = ltd
            keep = ["Date", "commodity", "ice_symbol", "month", "year", "FND", "LTD",
                    "Open", "High", "Low", "settlement", "volume", "open_interest"]
            rows.append(df[[c for c in keep if c in df.columns]])
            log.info(f"  {comm} {sym}: {len(df)} rows ({df['Date'].min().date()} -> {df['Date'].max().date()})")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def save(comm: str, new: pd.DataFrame, full: bool):
    out_path = DB_DIR / f"{comm.lower()}_futures.parquet"
    if not full and out_path.exists():
        old = pd.read_parquet(out_path)
        old["Date"] = pd.to_datetime(old["Date"])
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["Date", "ice_symbol"], keep="last")
        merged = merged.sort_values(["ice_symbol", "Date"]).reset_index(drop=True)
    else:
        merged = new.sort_values(["ice_symbol", "Date"]).reset_index(drop=True)
    merged.to_parquet(out_path, index=False)
    log.info(f"Saved {comm} -> {out_path.name} | {len(merged):,} rows | "
              f"{merged['Date'].min().date()} -> {merged['Date'].max().date()} | "
              f"{merged['ice_symbol'].nunique()} contracts")


def incremental_targets(comm: str) -> dict | None:
    """Contracts whose LTD is within the last 14 days or still to come get
    refetched from (last known date - 3 days), same window logic as the ICE
    source. Returns {(month, year): fetch_start_str} or None if full mode."""
    out_path = DB_DIR / f"{comm.lower()}_futures.parquet"
    if not out_path.exists():
        return {}
    df = pd.read_parquet(out_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"]  = pd.to_datetime(df["LTD"])
    today = pd.Timestamp.today().normalize()
    cfg = CONTRACT_CONFIG[comm]

    targets = {}
    for (month, year), grp in df.groupby(["month", "year"]):
        ltd = grp["LTD"].iloc[0]
        if ltd < today - pd.Timedelta(days=14):
            continue
        last_known = grp["Date"].max()
        targets[(month, year)] = (last_known - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

    # also pick up any not-yet-first-fetched contract within the forward window
    end_year = today.year + 2
    for year in range(today.year - 1, end_year + 1):
        for month_code in cfg.months:
            key = (month_code, year)
            if key in targets:
                continue
            fnd, ltd = fnd_ltd(cfg, year, cfg.month_num[month_code])
            if ltd < today - pd.Timedelta(days=14):
                continue
            exists = ((df["month"] == month_code) & (df["year"] == year)).any()
            if not exists:
                targets[key] = (ltd - pd.Timedelta(days=1400)).strftime("%Y-%m-%d")
    return targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("commodities", nargs="*", default=None)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    target_commodities = [c.upper() for c in args.commodities] if args.commodities else COMMODITIES
    end_year = pd.Timestamp.today().year + 2

    import lseg.data as ld
    ld.open_session()
    log.info("=" * 60)
    log.info(f"Futures Builder (LSEG) | mode={'FULL' if args.full else 'INCREMENTAL'} | {target_commodities}")
    log.info("=" * 60)

    try:
        for comm in target_commodities:
            if comm not in CONTRACT_CONFIG:
                log.warning(f"Unknown commodity {comm}, skipping")
                continue
            log.info(f"--- {comm} ---")
            if args.full:
                new = build_commodity(ld, comm, START_YEAR, end_year, incremental_from=None)
            else:
                targets = incremental_targets(comm)
                if not targets:
                    log.info(f"  {comm}: nothing to update")
                    continue
                new = build_commodity(ld, comm, START_YEAR, end_year, incremental_from=targets)
            if new.empty:
                log.warning(f"  {comm}: no data fetched")
                continue
            save(comm, new, args.full)
    finally:
        ld.close_session()

    log.info("=" * 60)
