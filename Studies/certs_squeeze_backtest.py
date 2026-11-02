# -*- coding: utf-8 -*-
"""KC certs-squeeze backtest: does a short-of-certs expiring contract pay a
bull spread (long expiring, short next) into FND?

Trade: buy the expiring/next spread at the settlement of entry day T_e (trading
days to FND), sell it at T-1 settlement. P&L in cents/lb; 1c = $375 per lot.

Bias controls:
- Signal uses the session BEFORE entry (OI and certs publish after the close,
  and US OI often a session late), so nothing unknown at entry is used.
- "Few certs" is decided only against contracts whose FND already passed
  (expanding median/tercile), never against the full sample.
- A cost per round trip is deducted from every trade.
- Benchmark: the same trade on every contract, no filter.

Run:  python Studies/certs_squeeze_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Dashboard"))
from common import DB_PATH  # noqa: E402

CERTS = (HERE.parents[2] / "Fundamental" / "Daily Miner" / "Database" / "Main" / "KC" / "kc_certs.parquet")
BAGS_PER_LOT = 37_500 * 0.45359237 / 60
USD_PER_CENT = 375.0
COST_C = 0.5          # cents per spread round trip (2 legs in, 2 legs out, thin expiring leg)
MIN_HISTORY = 8       # contracts of history needed before the filter can fire
ENTRIES = [-15, -10, -5]


def load():
    df = pd.read_parquet(DB_PATH / "kc_futures.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["settlement", "open_interest"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df[df["open_interest"] > 0]
    certs = pd.read_parquet(CERTS, columns=["Date", "KC-TOT-TOT"])
    certs = certs.assign(Date=pd.to_datetime(certs["Date"])).set_index("Date")["KC-TOT-TOT"]
    return df, certs.astype("float64").dropna().sort_index()


def trades(df, certs, t_entry):
    days = pd.DatetimeIndex(np.sort(df["Date"].unique()))
    ltd = df.groupby("ice_symbol")["LTD"].first().sort_values()
    nxt = dict(zip(ltd.index[:-1], ltd.index[1:]))
    px = df.pivot_table(index="Date", columns="ice_symbol", values="settlement")
    rows = []
    for sym, g in df.groupby("ice_symbol"):
        fnd = pd.Timestamp(g["FND"].iloc[0])
        if sym not in nxt or g["LTD"].iloc[0] > df["Date"].max() or fnd < certs.index[0] + pd.Timedelta(days=90):
            continue
        g = g.assign(t=days.get_indexer(g["Date"]) - days.searchsorted(fnd)).set_index("t")
        g = g[~g.index.duplicated()]
        need = [t_entry - 1, t_entry, -1]
        if not all(t in g.index for t in need):
            continue
        d_sig, d_in, d_out = (g.at[t, "Date"] for t in need)
        try:
            sp_in = px.at[d_in, sym] - px.at[d_in, nxt[sym]]
            sp_out = px.at[d_out, sym] - px.at[d_out, nxt[sym]]
        except KeyError:
            continue
        mult = g.at[t_entry - 1, "open_interest"] * BAGS_PER_LOT / certs.asof(d_sig)
        if np.isnan([sp_in, sp_out, mult]).any():
            continue
        rows.append(dict(contract=f"{g['month'].iloc[0]}{g['year'].iloc[0] % 100:02d}", fnd=fnd,
                         mult=mult, gross=sp_out - sp_in))
    t = pd.DataFrame(rows).sort_values("fnd").reset_index(drop=True)
    t["net"] = t["gross"] - COST_C
    # Past-only thresholds: median / top tercile of multiples of contracts already through FND
    past = [t["mult"].iloc[:i] for i in range(len(t))]
    t["past_med"] = [p.median() if len(p) >= MIN_HISTORY else np.nan for p in past]
    t["past_p67"] = [p.quantile(2 / 3) if len(p) >= MIN_HISTORY else np.nan for p in past]
    return t


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return dict(trades=0)
    tstat = x.mean() / (x.std(ddof=1) / np.sqrt(n)) if n > 2 and x.std() > 0 else np.nan
    return {"trades": n, "win %": round((x > 0).mean() * 100), "avg c": round(x.mean(), 2),
            "median c": round(x.median(), 2), "total c": round(x.sum(), 1),
            "total $/lot": round(x.sum() * USD_PER_CENT), "worst c": round(x.min(), 2),
            "best c": round(x.max(), 2), "t-stat": round(tstat, 2)}


def main():
    df, certs = load()
    pd.set_option("display.width", 200)
    for te in ENTRIES:
        t = trades(df, certs, te)
        live = t["past_med"].notna()
        rules = {
            "All contracts (no filter)": live,
            "Few certs: above past median": live & (t["mult"] > t["past_med"]),
            "Very few certs: top past third": live & (t["mult"] > t["past_p67"]),
            "Plenty certs: below past median": live & (t["mult"] <= t["past_med"]),
        }
        out = []
        for name, m in rules.items():
            for era, e in [("all", t["fnd"].notna()), ("2011-23", t["fnd"] < "2024-01-01"),
                           ("2024+", t["fnd"] >= "2024-01-01")]:
                out.append({"rule": name, "era": era, **stats(t.loc[m & e, "net"])})
        print(f"\n=== Entry T{te}, exit T-1, net of {COST_C}c cost (cents/lb; 1c = ${USD_PER_CENT:.0f}/lot) ===")
        print(pd.DataFrame(out).to_string(index=False))
        sel = t[live & (t["mult"] > t["past_p67"])]
        print("Top-third trades:", ", ".join(f"{r.contract} {r.net:+.1f}" for r in sel.itertuples()))


if __name__ == "__main__":
    main()
