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
# `settlement` is LSEG's SETTLE field, not TRDPRC_1. TRDPRC_1 is the LAST TRADE,
# which is null on any day a contract did not trade and otherwise sits a few
# points off the official settlement: measured on KC it equalled SETTLE on only
# 1-4% of days (mean gap 0.7-1.9 points), and for a deferred month roughly 30%
# of days had no trade at all. Anchoring the series on TRDPRC_1 therefore
# (a) stored last-trade prices under the name "settlement", and (b) dropped
# every no-trade session outright — including an expiring month's final days,
# whose real OI and settlement were then missing or interpolated.
FIELDS     = ["OPEN_PRC", "HIGH_1", "LOW_1", "SETTLE", "ACVOL_UNS", "OPINT_1"]
COL_MAP    = {"OPEN_PRC": "Open", "HIGH_1": "High", "LOW_1": "Low",
              "SETTLE": "settlement", "ACVOL_UNS": "volume", "OPINT_1": "open_interest"}
TOTAL_OI_FIELD = "TOTCNTROI"      # LSEG's own whole-market open interest

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
    horizon_years: int = 2   # how far forward to fetch listed contracts

CONTRACT_CONFIG = {
    "KC": CommodityConfig("KC",  "US", ["H","K","N","U","Z"], {"H":3,"K":5,"N":7,"U":9,"Z":12},
                           fnd_first_bd_minus(7), ltd_last_bd_minus(8)),
    "CC": CommodityConfig("CC",  "US", ["H","K","N","U","Z"], {"H":3,"K":5,"N":7,"U":9,"Z":12},
                           fnd_nth_bd_minus(nth=6, n=10), ltd_last_bd_minus(11)),
    "CT": CommodityConfig("CT",  "US", ["H","K","N","V","Z"], {"H":3,"K":5,"N":7,"V":10,"Z":12},
                           fnd_first_bd_minus(5), ltd_last_bd_minus(17)),
    # Sugar #11 trades much further out than the other softs — extend the
    # forward horizon so the board doesn't cut off 3 live contracts early.
    "SB": CommodityConfig("SB",  "US", ["H","K","N","V"], {"H":3,"K":5,"N":7,"V":10},
                           "after_ltd", ltd_last_bd_preceding_month(), horizon_years=3),
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

# Circuit breaker: if LSEG itself is unresponsive (Workspace closed,
# connection dropped, etc.), every single candidate for every contract for
# every commodity times out — resolve_and_fetch tries 4 candidates per
# contract, so a fully-dead session silently burns 10s of minutes retrying
# calls that were never going to succeed, before anyone notices. Track
# consecutive request EXCEPTIONS (not "empty response", which is a normal,
# expected outcome for a candidate ticker that just doesn't exist) across
# the whole run and abort fast once it's clearly not LSEG's data, it's the
# connection.
_CONSEC_FAILS = {"n": 0}
_FAIL_THRESHOLD = 8


class LSEGUnresponsive(RuntimeError):
    pass


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
            _CONSEC_FAILS["n"] = 0  # the request itself went through — LSEG is alive
        except Exception as e:
            _CONSEC_FAILS["n"] += 1
            if _CONSEC_FAILS["n"] >= _FAIL_THRESHOLD:
                raise LSEGUnresponsive(
                    f"{_FAIL_THRESHOLD} consecutive LSEG request failures — the session looks "
                    f"unresponsive (check LSEG Workspace is open and connected). "
                    f"Last error on {cand}: {e}"
                ) from e
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
        real = df["settlement"].notna()      # sessions LSEG actually published
        # Only settlement and open interest are interpolated, and only across
        # sessions LSEG is missing outright. Open/High/Low stay null on a day
        # nothing traded (there is no such price), and volume is a flow, not a
        # level: a session with a settlement but no volume print traded zero
        # lots, and a hole has no known volume — neither is a straight line
        # between its neighbours (the old code stored 2.33 and 3.67 lots of
        # KCU6 "volume" on two days that had none).
        for col in ["settlement", "open_interest"]:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear", limit_area="inside")
        if "volume" in df.columns:
            df.loc[real, "volume"] = df.loc[real, "volume"].fillna(0)
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


# Rollex is a sibling project with its own builder and schedule; this run only
# copies what the Futures dashboard needs (price index and its daily return)
# into Database/ so the deployed app, which cannot see ../Rollex, can read it.
# Slim on purpose: the full Rollex files are ~3 MB and would add a binary diff
# to every daily commit.
ROLLEX_DIR = CODE_DIR.parent.parent / "Rollex" / "Database"
ROLLEX_PATH = DB_DIR / "rollex.parquet"


def sync_rollex() -> int:
    """Refresh Database/rollex.parquet (Date, commodity, rollex_px, rollex_ret)
    from the Rollex project's own parquets. Never raises: the price matrix is
    optional, so a missing file or a bad read is logged and the run goes on.
    Rewrites the file only when its content changed, to keep commits quiet."""
    try:
        frames = []
        for comm in COMMODITIES:
            p = ROLLEX_DIR / f"rollex_{comm}.parquet"
            if not p.exists():
                log.warning(f"  rollex: {p.name} not found, skipped")
                continue
            d = pd.read_parquet(p, columns=["rollex_px", "rollex_ret"])
            d.index = pd.to_datetime(d.index).normalize()
            d.index.name = "Date"
            d = d.astype("float64").reset_index()
            d.insert(1, "commodity", comm)
            frames.append(d)
        if not frames:
            return 0
        out = (pd.concat(frames, ignore_index=True)
                 .drop_duplicates(subset=["commodity", "Date"], keep="last")
                 .sort_values(["commodity", "Date"]).reset_index(drop=True))
        if ROLLEX_PATH.exists():
            try:
                if pd.read_parquet(ROLLEX_PATH).equals(out):
                    log.info(f"  rollex: unchanged ({len(out):,} rows, latest {out['Date'].max().date()})")
                    return 0
            except Exception:
                pass                     # unreadable old copy: just replace it
        out.to_parquet(ROLLEX_PATH, index=False)
        log.info(f"  rollex: copied {len(out):,} rows for {out['commodity'].nunique()} markets, "
                 f"latest {out['Date'].max().date()}")
        return len(out)
    except Exception as e:
        log.warning(f"  rollex: sync failed ({type(e).__name__}: {str(e)[:100]})")
        return 0


TOTAL_OI_PATH = DB_DIR / "total_oi.parquet"
TOTAL_OI_START = "2000-01-01"     # LSEG carries it back to 2000 for the US markets


def update_total_oi(ld, comm: str, full: bool) -> int:
    """Store LSEG's whole-market open interest (TOTCNTROI) for one commodity.

    This is the exchange-wide futures total across every listed month, so the
    dashboard can plot it directly instead of summing the per-contract table —
    which under-reads on any session where an expiring or illiquid month is
    missing a row, and only starts once the database holds the full board
    (2011). The value is identical whichever contract's RIC it is queried
    through (verified on KCc1/c2/c3/KCZ6/KCH7), and it matched the CFTC's
    "Futures Only" total open interest exactly on every Tuesday checked.

    Incremental runs re-fetch a 10-day overlap so a session stored before its
    figure was final is corrected, not left as first written.
    """
    ric = f"{CONTRACT_CONFIG[comm].lseg_root}c2"
    old = None
    if TOTAL_OI_PATH.exists():
        old = pd.read_parquet(TOTAL_OI_PATH)
        old["Date"] = pd.to_datetime(old["Date"])
    mine = old[old["commodity"] == comm] if old is not None else None
    if full or mine is None or mine.empty:
        start = TOTAL_OI_START
    else:
        start = (mine["Date"].max() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        h = ld.get_history(universe=[ric], fields=[TOTAL_OI_FIELD], start=start,
                           end=pd.Timestamp.today().strftime("%Y-%m-%d"),
                           interval="daily", count=10000)
        _CONSEC_FAILS["n"] = 0
    except Exception as e:
        log.warning(f"  {comm}: total OI fetch failed ({ric}): {str(e)[:120]}")
        return 0
    if h is None or h.empty:
        log.warning(f"  {comm}: total OI returned nothing ({ric})")
        return 0
    s = h.iloc[:, 0].dropna()
    s = s[s > 0]
    new = pd.DataFrame({"Date": pd.to_datetime(s.index).normalize(), "commodity": comm,
                        "total_oi": s.astype("float64").to_numpy()})
    keep = old[old["commodity"] != comm] if (old is not None and full) else old
    merged = pd.concat([keep, new], ignore_index=True) if keep is not None else new
    merged = (merged.drop_duplicates(subset=["commodity", "Date"], keep="last")
                    .sort_values(["commodity", "Date"]).reset_index(drop=True))
    merged.to_parquet(TOTAL_OI_PATH, index=False)
    log.info(f"  {comm}: total OI {len(new):,} rows ({new['Date'].min().date()} -> "
             f"{new['Date'].max().date()}), latest {int(new['total_oi'].iloc[-1]):,}")
    return len(new)


def topup_open_interest(ld, comm: str) -> int:
    """Fill the last completed session's open interest from the real-time quote.

    `get_history` reads Refinitiv's daily timeseries file, which is rebuilt once
    a day and does not carry the newest session's open interest until after
    midnight — so a morning or afternoon run stores the session's settlement and
    volume with OPINT_1 still null. The real-time quote for the same RIC already
    carries it. Verified across LCC/RC/LSU on 2026-09-09: the snapshot OPINT_1
    matched the stored previous-session OI exactly on 6 of 6 contracts, and
    never the session before it — so the quote's open interest belongs to the
    last COMPLETED session, not to today and not to an intraday tally.

    Only that one session is ever written. A snapshot says nothing about older
    gaps, so those are left for the historical fetch to repair on its own.
    """
    out_path = DB_DIR / f"{comm.lower()}_futures.parquet"
    if not out_path.exists():
        return 0
    df = pd.read_parquet(out_path)
    df["Date"] = pd.to_datetime(df["Date"])

    today = pd.Timestamp.today().normalize()
    prior = df[df["Date"] < today]
    if prior.empty:
        return 0
    target = prior["Date"].max()          # the last completed session

    need = df[(df["Date"] == target) & df["settlement"].notna()
              & df["open_interest"].isna()]
    if need.empty:
        log.info(f"  {comm}: OI already complete for {target.date()}")
        return 0

    rics = sorted(need["ice_symbol"].unique())
    try:
        snap = ld.get_data(universe=list(rics), fields=["OPINT_1"])
    except Exception as e:
        log.warning(f"  {comm}: OI top-up quote failed: {e}")
        return 0
    snap = snap.set_index("Instrument")["OPINT_1"]

    # Freshness is decided per MARKET, not per contract. If the exchange has not
    # published the new session's OI yet, every quote still shows the PREVIOUS
    # session's number and writing those onto `target` would shift a whole
    # session's OI back by one day.
    #
    # Judging contract by contract does not work in either direction: a quiet
    # back month can genuinely carry an unchanged OI for days (so "unchanged"
    # alone does not prove staleness), and a contract with no prior OI at all
    # has nothing to compare against — that hole wrote two stale LSU values onto
    # 2026-09-09 before this was tightened. So compare every contract that CAN
    # be compared, and accept the batch only if the market as a whole has moved.
    prev_oi = (df[(df["Date"] < target) & df["open_interest"].notna()]
               .sort_values("Date").groupby("ice_symbol")["open_interest"].last())

    comparable = [r for r in rics if pd.notna(prev_oi.get(r)) and pd.notna(snap.get(r))]
    if not comparable:
        log.info(f"  {comm}: no contract with a prior OI to check the quote against — "
                 f"skipped, left for the historical fetch")
        return 0
    moved = [r for r in comparable
             if abs(float(snap[r]) - float(prev_oi[r])) >= 1]
    if len(moved) * 2 < len(comparable):
        log.info(f"  {comm}: only {len(moved)}/{len(comparable)} contracts show a "
                 f"changed OI — {target.date()} not published yet, left for the "
                 f"next run")
        return 0

    filled = 0
    for ric in rics:
        v = snap.get(ric)
        if pd.isna(v):
            continue
        df.loc[(df["Date"] == target) & (df["ice_symbol"] == ric),
               "open_interest"] = float(v)
        filled += 1
    if filled:
        df = df.sort_values(["ice_symbol", "Date"]).reset_index(drop=True)
        df.to_parquet(out_path, index=False)
        log.info(f"  {comm}: filled OI on {target.date()} for {filled} contract(s)")
    return filled


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
    end_year = today.year + cfg.horizon_years
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
    parser.add_argument("--oi-topup-only", action="store_true",
                        help="Skip the history fetch; only fill the last completed "
                             "session's open interest from the real-time quote. "
                             "Idempotent — safe to run repeatedly during the day.")
    args = parser.parse_args()

    target_commodities = [c.upper() for c in args.commodities] if args.commodities else COMMODITIES
    today_year = pd.Timestamp.today().year

    # A local file copy: needs no LSEG session, so it runs even if Workspace is down.
    sync_rollex()

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
            update_total_oi(ld, comm, args.full)
            if args.oi_topup_only:
                topup_open_interest(ld, comm)
                continue
            end_year = today_year + CONTRACT_CONFIG[comm].horizon_years
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
            # Refinitiv's daily timeseries lags the newest session's open
            # interest; the quote already has it. Runs after every build so the
            # daily job self-heals rather than leaving the dashboard a session
            # behind on the US markets.
            topup_open_interest(ld, comm)
    except LSEGUnresponsive as e:
        log.error(f"ABORTING: {e}")
        print(f"ABORTING: {e}", file=sys.stderr)  # run_updater.py's email body quotes stderr
        sys.exit(1)
    finally:
        ld.close_session()

    log.info("=" * 60)
