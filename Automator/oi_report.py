"""oi_report.py — per-contract OI/volume/price detail for the daily email.

Kept out of run_updater.py so it can be exercised on its own: run_updater runs
the whole pipeline at import time, so anything defined inside it can only be
tested by triggering a real update and a real email.

The report deliberately reports the open-interest date separately from the
settlement date. Refinitiv's daily timeseries carries a session's settlement
before its open interest, so a single "last date" reads as if the OI were
current when it can be a session behind.
"""

import pandas as pd

NAMES = {"KC": "Coffee", "CC": "Cocoa", "CT": "Cotton", "SB": "Sugar #11",
         "RC": "Robusta", "LCC": "Liffe Cocoa", "LSU": "Liffe Sugar"}


def _load(db_dir, comm: str):
    p = db_dir / f"{comm.lower()}_futures.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"] = pd.to_datetime(df["LTD"])
    return df


def market_dates(db_dir, comm: str):
    """(latest settlement date, latest open-interest date) for one market."""
    df = _load(db_dir, comm)
    if df is None or df.empty:
        return None, None
    px = df.loc[df["settlement"].notna(), "Date"].max()
    oi = df.loc[df["open_interest"].notna() & (df["open_interest"] > 0), "Date"].max()
    return (px if pd.notna(px) else None), (oi if pd.notna(oi) else None)


def _contract_rows(df: pd.DataFrame, oi_date: pd.Timestamp):
    """Every contract carrying OI on `oi_date`, near expiry first, with the
    change against each contract's own previous OI print (not a fixed
    yesterday — a contract can miss a session the rest of the board traded)."""
    cur = df[(df["Date"] == oi_date) & df["open_interest"].notna()
             & (df["open_interest"] > 0)].copy()
    if cur.empty:
        return cur

    hist = df[df["Date"] < oi_date].sort_values("Date")
    prev_oi = (hist[hist["open_interest"].notna()]
               .groupby("ice_symbol")["open_interest"].last())
    prev_px = (hist[hist["settlement"].notna()]
               .groupby("ice_symbol")["settlement"].last())

    cur["oi_chg"] = cur["open_interest"] - cur["ice_symbol"].map(prev_oi)
    pv = cur["ice_symbol"].map(prev_px)
    cur["px_chg_pct"] = (cur["settlement"] / pv - 1) * 100
    cur["dte"] = (cur["LTD"] - cur["Date"]).dt.days
    return cur.sort_values("LTD")


def build_oi_detail(db_dir, commodities) -> str:
    """Plain-text per-contract detail, one block per market."""
    out = []
    for comm in commodities:
        df = _load(db_dir, comm)
        if df is None or df.empty:
            out += [f"=== {comm} - no data ===", ""]
            continue
        px_date, oi_date = market_dates(db_dir, comm)
        if oi_date is None:
            out += [f"=== {comm} - no open interest on record ===", ""]
            continue

        lag = "" if (px_date is None or px_date <= oi_date) else \
              f"   [settlement to {px_date:%Y-%m-%d} - OI is a session behind]"
        out.append(f"=== {comm} - {NAMES.get(comm, comm)}  |  OI as of "
                   f"{oi_date:%a %d %b %Y}{lag} ===")

        rows = _contract_rows(df, oi_date)
        if rows.empty:
            out += ["  (no contracts)", ""]
            continue

        out.append(f"{'CONTRACT':<10} {'DTE':>5} {'SETTLE':>10} {'CHG%':>7} "
                   f"{'VOLUME':>10} {'OI':>11} {'OI CHG':>10}")
        out.append("-" * 68)
        for _, r in rows.iterrows():
            px = f"{r['settlement']:,.2f}" if pd.notna(r["settlement"]) else "-"
            pc = f"{r['px_chg_pct']:+.2f}" if pd.notna(r["px_chg_pct"]) else "-"
            vol = f"{r['volume']:,.0f}" if pd.notna(r["volume"]) else "-"
            oc = f"{r['oi_chg']:+,.0f}" if pd.notna(r["oi_chg"]) else "-"
            out.append(f"{r['ice_symbol']:<10} {r['dte']:>5.0f} {px:>10} {pc:>7} "
                       f"{vol:>10} {r['open_interest']:>11,.0f} {oc:>10}")

        tot_oi = rows["open_interest"].sum()
        tot_chg = rows["oi_chg"].sum(min_count=1)
        tot_vol = rows["volume"].sum(min_count=1)
        out.append("-" * 68)
        out.append(f"{'TOTAL':<10} {'':>5} {'':>10} {'':>7} "
                   f"{(f'{tot_vol:,.0f}' if pd.notna(tot_vol) else '-'):>10} "
                   f"{tot_oi:>11,.0f} "
                   f"{(f'{tot_chg:+,.0f}' if pd.notna(tot_chg) else '-'):>10}")
        out.append("")
    return "\n".join(out)
