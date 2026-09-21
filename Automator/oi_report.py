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
        out.append(f"=== {comm}  |  OI as of "
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


# -- HTML report -----------------------------------------------------------
# The plain-text version repeats a header, a rule and a total line per market,
# which is ~110 lines for seven markets. One table with the market as a column
# carries the same numbers in roughly half the height, and Outlook renders it
# denser still. run_log.txt keeps the plain-text version.
_TH = ("padding:3px 7px;text-align:right;font-weight:600;font-size:10px;"
       "color:#6b7280;border-bottom:1px solid #d1d5db;white-space:nowrap")
_TD = "padding:2px 7px;text-align:right;border-bottom:1px solid #f0f0f0;white-space:nowrap"


def _sign(v, fmt="{:+,.0f}"):
    if pd.isna(v):
        return '<td style="%s">-</td>' % _TD
    colour = "#16a34a" if v >= 0 else "#dc2626"
    return '<td style="%s;color:%s">%s</td>' % (_TD, colour, fmt.format(v))


def build_html_report(db_dir, commodities, run_dt: str, summary: list) -> str:
    """summary: list of dicts with comm, upserted, total, last_date, oi_date, status."""
    h = ['<div style="font-family:Segoe UI,Arial,sans-serif;font-size:12px;color:#1a1a1a">',
         '<div style="font-size:14px;font-weight:600">Futures Database (LSEG) &mdash; Daily Update</div>',
         '<div style="color:#6b7280;font-size:11px;margin:2px 0 10px">Run %s</div>' % run_dt]

    # --- summary ---
    h.append('<table style="border-collapse:collapse;font-size:11px">')
    h.append("<tr>" + "".join(
        '<th style="%s">%s</th>' % (_TH, c) for c in
        ("MKT", "UPSERTED", "TOTAL ROWS", "LAST DATE", "OI DATE", "STATUS")) + "</tr>")
    lagging = []
    for r in summary:
        bad = r["status"] != "OK"
        if r["status"] == "OI LAGS":
            lagging.append(r["comm"])
        colour = "#b45309" if r["status"] == "OI LAGS" else ("#dc2626" if bad else "#16a34a")
        h.append("<tr>"
                 + '<td style="%s;text-align:left;font-weight:600">%s</td>' % (_TD, r["comm"])
                 + '<td style="%s">%s</td>' % (_TD, f"{r['upserted']:,}")
                 + '<td style="%s">%s</td>' % (_TD, f"{r['total']:,}")
                 + '<td style="%s">%s</td>' % (_TD, r["last_date"])
                 + '<td style="%s">%s</td>' % (_TD, r["oi_date"])
                 + '<td style="%s;color:%s;font-weight:600">%s</td>' % (_TD, colour, r["status"])
                 + "</tr>")
    h.append("</table>")
    if lagging:
        h.append('<div style="font-size:10.5px;color:#b45309;margin:6px 0 0">'
                 'OI LAGS = settlement is in for a session whose open interest the vendor '
                 'has not published yet; the quote top-up fills it on the next run.</div>')

    # --- one detail table for every market ---
    h.append('<div style="font-size:12px;font-weight:600;margin:14px 0 4px">'
             'Per-contract detail</div>')
    h.append('<table style="border-collapse:collapse;font-size:11px">')
    h.append("<tr>" + "".join(
        '<th style="%s">%s</th>' % (_TH, c) for c in
        ("MKT", "CONTRACT", "DTE", "SETTLE", "CHG%", "VOLUME", "OI", "OI CHG")) + "</tr>")

    for comm in commodities:
        df = _load(db_dir, comm)
        if df is None or df.empty:
            continue
        _, oi_date = market_dates(db_dir, comm)
        if oi_date is None:
            continue
        rows = _contract_rows(df, oi_date)
        if rows.empty:
            continue
        first = True
        for _, r in rows.iterrows():
            mkt = comm if first else ""
            first = False
            h.append("<tr>"
                     + '<td style="%s;text-align:left;font-weight:600">%s</td>' % (_TD, mkt)
                     + '<td style="%s;text-align:left">%s</td>' % (_TD, r["ice_symbol"])
                     + '<td style="%s;color:#6b7280">%.0f</td>' % (_TD, r["dte"])
                     + '<td style="%s">%s</td>' % (_TD, f"{r['settlement']:,.2f}"
                                                   if pd.notna(r["settlement"]) else "-")
                     + _sign(r["px_chg_pct"], "{:+.2f}")
                     + '<td style="%s;color:#374151">%s</td>' % (_TD, f"{r['volume']:,.0f}"
                                                                 if pd.notna(r["volume"]) else "-")
                     + '<td style="%s">%s</td>' % (_TD, f"{r['open_interest']:,.0f}")
                     + _sign(r["oi_chg"])
                     + "</tr>")
        tot_oi = rows["open_interest"].sum()
        tot_chg = rows["oi_chg"].sum(min_count=1)
        tot_vol = rows["volume"].sum(min_count=1)
        tot_td = _TD + ";background:#f9fafb;font-weight:600;border-bottom:1px solid #d1d5db"
        h.append("<tr>"
                 + '<td style="%s"></td>' % tot_td
                 + '<td style="%s;text-align:left">%s total</td>' % (tot_td, comm)
                 + '<td style="%s"></td><td style="%s"></td><td style="%s"></td>' % (tot_td, tot_td, tot_td)
                 + '<td style="%s">%s</td>' % (tot_td, f"{tot_vol:,.0f}" if pd.notna(tot_vol) else "-")
                 + '<td style="%s">%s</td>' % (tot_td, f"{tot_oi:,.0f}")
                 + (('<td style="%s;color:%s">%s</td>'
                     % (tot_td, "#16a34a" if tot_chg >= 0 else "#dc2626",
                        "{:+,.0f}".format(tot_chg)))
                    if pd.notna(tot_chg) else '<td style="%s">-</td>' % tot_td)
                 + "</tr>")

    h.append("</table></div>")
    return "\n".join(h)

