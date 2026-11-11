# -*- coding: utf-8 -*-
"""Shared data layer for the Futures dashboard pages.

Both pages read the same per-contract-month parquets, so the loaders, the
commodity/month maps and the colour tokens live here instead of being
duplicated per page — one `@st.cache_data` entry per commodity is then shared
across pages rather than one cache per page holding its own copy of the same
frame. Deliberately thin: data access and constants only, no Streamlit UI.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# No vertical gridlines on any chart in either app (horizontal ones stay)
pio.templates["no_x_grid"] = go.layout.Template(layout=dict(xaxis=dict(showgrid=False)))
pio.templates.default = "plotly+no_x_grid"
from pathlib import Path

# Delivery-month codes in calendar order. Needed by the seasonals page to work
# out which leg opens a basket's season (Z+H runs Dec->Mar, not Mar->Dec).
MONTH_ORDER = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5,  "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

DB_PATH = Path(__file__).parent.parent / "Database"

# Display order is the desk's: coffee, cocoa, sugar, cotton, each NY then London.
# The second field is the label used everywhere on screen (selector, titles) --
# the bare acronym, by request, not the commodity's full name.
COMMODITIES = {
    "KC":  ("kc_futures.parquet",  "KC"),
    "RC":  ("rc_futures.parquet",  "RC"),
    "CC":  ("cc_futures.parquet",  "CC"),
    "LCC": ("lcc_futures.parquet", "LCC"),
    "SB":  ("sb_futures.parquet",  "SB"),
    "LSU": ("lsu_futures.parquet", "LSU"),
    "CT":  ("ct_futures.parquet",  "CT"),
}

# Contract size in metric tonnes, for turning a lot count into tonnage. From the
# exchange contract specs: Coffee "C" 37,500 lb, Sugar No.11 112,000 lb, Cotton
# No.2 50,000 lb (1 lb = 0.45359237 kg); ICE US cocoa, London cocoa and Robusta
# 10 tonnes; London No.5 white sugar 50 tonnes. A KC lot and an RC lot are
# different sizes, so lots only add up sensibly across markets once converted.
LOT_TONNES = {
    "KC":  37_500 * 0.45359237 / 1000,
    "RC":  10.0,
    "CC":  10.0,
    "LCC": 10.0,
    "SB":  112_000 * 0.45359237 / 1000,
    "LSU": 50.0,
    "CT":  50_000 * 0.45359237 / 1000,
    "CC+LCC": 10.0,
}

MONTH_NAMES = {
    "F": "January", "G": "February", "H": "March",  "J": "April",
    "K": "May",     "M": "June",     "N": "July",   "Q": "August",
    "U": "September","V": "October", "X": "November","Z": "December",
}

C = {
    "oi_outer":  "rgba(31, 138, 156, 0.08)",
    "oi_mid":    "rgba(31, 138, 156, 0.16)",
    "oi_inner":  "rgba(31, 138, 156, 0.28)",
    "oi_avg":    "#4a5578",
    "sh_outer":  "rgba(31, 138, 156, 0.08)",
    "sh_mid":    "rgba(31, 138, 156, 0.16)",
    "sh_inner":  "rgba(31, 138, 156, 0.28)",
    "sh_avg":    "#4a5578",
    "vr_outer":  "rgba(31, 138, 156, 0.08)",
    "vr_mid":    "rgba(31, 138, 156, 0.16)",
    "vr_inner":  "rgba(31, 138, 156, 0.28)",
    "vr_avg":    "#4a5578",
    "vs_outer":  "rgba(31, 138, 156, 0.08)",
    "vs_mid":    "rgba(31, 138, 156, 0.16)",
    "vs_inner":  "rgba(31, 138, 156, 0.28)",
    "vs_avg":    "#4a5578",
    "rv_outer":  "rgba(31, 138, 156, 0.08)",
    "rv_mid":    "rgba(31, 138, 156, 0.16)",
    "rv_inner":  "rgba(31, 138, 156, 0.28)",
    "rv_avg":    "#4a5578",
    # Common
    "current":   "#0a2463",
    "prev":      "#c94a4a",
    "individual":"rgba(138,148,168,0.4)",
    "grid":      "rgba(10,36,99,0.08)",
    "bg":        "#ffffff",
    "font":      "#1a1a2e",
    "vline":     "rgba(10,36,99,0.18)",
}


# Synthetic combined cocoa market, offered only by the OI progression page (it
# adds the key to its own copy of COMMODITIES, so other pages are unaffected).
# Both legs are 10 t lots on the same H/K/N/U/Z cycle and LCC symbols are the CC
# symbol with an 'L' prefix, so each CC contract absorbs its LCC twin: OI and
# volume are summed, while prices/LTD/FND stay CC's (USD) -- adding a USD and a
# GBP price makes no sense. Only sessions both exchanges printed are kept, so a
# one-exchange holiday does not show as a half-size market.
COMBINED = {"CC+LCC": ("CC", "LCC")}


# ── Data loaders ──────────────────────────────────────────────────────────────
def _mtime(commodity: str) -> float:
    if commodity in COMBINED:
        return max(_mtime(c) for c in COMBINED[commodity])
    filename, _ = COMMODITIES[commodity]
    p = DB_PATH / filename
    return p.stat().st_mtime if p.exists() else 0.0


_NUMERIC_COLS = ["Open", "High", "Low", "settlement", "volume", "open_interest"]


@st.cache_data
def load_data(commodity: str, mtime: float = 0.0) -> pd.DataFrame:
    if commodity in COMBINED:
        return _load_combined(commodity, mtime)
    filename, _ = COMMODITIES[commodity]
    df = pd.read_parquet(DB_PATH / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"]  = pd.to_datetime(df["LTD"])
    # The parquet stores prices/OI as pandas' nullable Float64, whose missing
    # value is pd.NA, not np.nan. pd.NA does not behave like nan once a scalar
    # escapes the Series: float(pd.NA) and bool(pd.NA) both RAISE, and so does
    # f-string formatting of it — so one missing settlement pulled out with
    # .at[] can take down a whole table builder. Downcast once, here, so every
    # consumer on both pages gets plain float64 + np.nan semantics.
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["days_to_expiry"] = (df["LTD"] - df["Date"]).dt.days
    return df[df["open_interest"] > 0].copy()


def _load_combined(commodity: str, mtime: float = 0.0) -> pd.DataFrame:
    base, add = COMBINED[commodity]
    b, a = load_data(base, mtime), load_data(add, mtime)
    common_dates = np.intersect1d(b["Date"].unique(), a["Date"].unique())
    b = b[b["Date"].isin(common_dates)].copy()
    a = a[a["Date"].isin(common_dates)]
    a = a.assign(ice_symbol=a["ice_symbol"].str[len(add) - len(base):])  # 'LCCZ6' -> 'CCZ6'
    # Outer join: a session where the CC leg has no row (OI filtered as 0/NA,
    # mostly back months) must still carry its LCC OI, not drop it. Such rows
    # take the CC contract's own LTD/FND when known and have no price.
    keys = ["Date", "ice_symbol"]
    m = b.merge(a[keys + ["open_interest", "volume", "month", "year", "LTD", "FND"]],
                on=keys, how="outer", suffixes=("", "_a"))
    for col in ("month", "year", "LTD", "FND"):
        m[col] = m[col].fillna(m[col + "_a"])
    sym_ltd = b.groupby("ice_symbol")["LTD"].first()
    sym_fnd = b.groupby("ice_symbol")["FND"].first()
    only_a = m["open_interest"].isna()
    m.loc[only_a, "LTD"] = m.loc[only_a, "ice_symbol"].map(sym_ltd).fillna(m.loc[only_a, "LTD"])
    m.loc[only_a, "FND"] = m.loc[only_a, "ice_symbol"].map(sym_fnd).fillna(m.loc[only_a, "FND"])
    m["open_interest"] = m["open_interest"].fillna(0) + m["open_interest_a"].fillna(0)
    m["volume"] = m["volume"].fillna(0) + m["volume_a"].fillna(0)
    m["year"] = m["year"].astype("int64")
    m["commodity"] = commodity
    m["days_to_expiry"] = (m["LTD"] - m["Date"]).dt.days
    m = m.drop(columns=[c for c in m.columns if c.endswith("_a")])
    return m.sort_values(keys).reset_index(drop=True)


def _rollex_mtime() -> float:
    p = DB_PATH / "rollex.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_rollex(commodity: str, mtime: float = 0.0):
    """Daily roll-adjusted return of the desk's Rollex index for one commodity,
    as a Date-indexed float Series, or None if the builder has not copied it.
    The builder copies it from the Rollex project on every run."""
    if commodity in COMBINED:                 # prices are the base (CC) leg's
        return load_rollex(COMBINED[commodity][0], mtime)
    p = DB_PATH / "rollex.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[df["commodity"] == commodity]
    if df.empty:
        return None
    s = df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date")["rollex_ret"]
    return s.astype("float64").sort_index().dropna()


def _total_oi_mtime() -> float:
    p = DB_PATH / "total_oi.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_total_oi(commodity: str, mtime: float = 0.0):
    """LSEG's own whole-market futures open interest (TOTCNTROI) for one
    commodity, as a Date-indexed float Series — or None if the builder has not
    stored it yet, so callers can fall back to summing the per-contract table.

    Unlike that sum it needs no complete board of contracts behind it, so it
    is not short on a session where an expiring month has no row, and it runs
    back to 2000 for the US markets rather than starting where the database's
    earliest stored contract expires (2011)."""
    if commodity in COMBINED:                 # sum, on sessions both legs have
        parts = [load_total_oi(c, mtime) for c in COMBINED[commodity]]
        if any(s is None for s in parts):
            return None
        return (parts[0] + parts[1]).dropna()
    p = DB_PATH / "total_oi.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[df["commodity"] == commodity]
    if df.empty:
        return None
    s = df.assign(Date=pd.to_datetime(df["Date"])).set_index("Date")["total_oi"]
    return _clean_total_oi(s.astype("float64").sort_index())


TOTAL_OI_RELIABLE_FROM = "2009-01-01"


def _clean_total_oi(s: pd.Series) -> pd.Series:
    """LSEG's whole-market series, restricted to where it can be trusted.

    Two problems in the raw series, both checked rather than assumed:
    - Before 2009 it is not reliable. KC's Tuesday values sit a median 25-30%
      off the CFTC futures-only total in 2006-07 (400k against a true ~120k in
      2007) and 85% of 2008 matches, then it agrees exactly from 2009 on
      (100% of Tuesdays 2009-2011, and 2011+ also equals the per-contract sum).
      Other markets show the same regime of wild swings before then, and
      Robusta prints single digits in 2008. So the series starts in 2009.
    - Isolated one-day glitches: a single session printing far off two
      neighbours that agree with each other (CT 24 Dec 2007 224k -> 90k ->
      224k, CC 13 May 2010 130k -> 183k -> 130k), typically on a holiday when
      only some contracts print. Such a day is dropped, not smoothed, so it
      shows as a gap rather than an invented number. A real move never looks
      like this: it persists into the next session.
    """
    s = s[s.index >= TOTAL_OI_RELIABLE_FROM]
    prev, nxt = s.shift(1), s.shift(-1)
    neighbours_agree = (prev / nxt - 1).abs() < 0.10
    off_both = ((s / prev - 1).abs() > 0.15) & ((s / nxt - 1).abs() > 0.15)
    return s[~(neighbours_agree & off_both)]


@st.cache_data
def load_enriched(commodity: str, mtime: float = 0.0) -> pd.DataFrame:
    """Adds oi_share_pct, vol_share_pct, vol_oi_ratio to every row."""
    df = load_data(commodity, mtime)
    tot_oi  = df.groupby("Date")["open_interest"].sum().rename("total_oi")
    # The per-contract sum is only the whole board from the earliest stored
    # contract's LTD on; before that the months that had already expired are
    # missing and the sum runs as low as 0-40% of the market, inflating every
    # share. Prefer LSEG's own market total (equal to the sum from 2011 on);
    # where it is absent too, a pre-board date has no trustworthy total at all.
    lseg = load_total_oi(commodity, _total_oi_mtime())
    if lseg is not None:
        tot_oi = lseg.reindex(tot_oi.index).fillna(tot_oi).rename("total_oi")
        tot_oi[(tot_oi.index < df.groupby("ice_symbol")["LTD"].first().min())
               & lseg.reindex(tot_oi.index).isna()] = np.nan
    else:
        tot_oi[tot_oi.index < df.groupby("ice_symbol")["LTD"].first().min()] = np.nan
    # No LSEG market-volume total is stored, so pre-board dates have no full total
    tot_vol = df.groupby("Date")["volume"].sum().rename("total_vol")
    tot_vol[tot_vol.index < df.groupby("ice_symbol")["LTD"].first().min()] = np.nan
    df = df.merge(tot_oi, on="Date").merge(tot_vol, on="Date")
    df["oi_share_pct"] = df["open_interest"] / df["total_oi"]  * 100
    df["vol_share_pct"]= df["volume"]        / df["total_vol"] * 100
    df["vol_oi_ratio"] = df["volume"]        / df["open_interest"]
    return df

# -- Conditional-formatting helpers ---------------------------------------
# Shared so the seasonals tables carry the same bars and tints as the
# comprehensive grid rather than a second, near-miss implementation.

def _safe(v, default=1.0):
    v = float(v) if pd.notna(v) else default
    return v if v > 0 else default

def _oi_heatmap_style(v, vmin, vmax):
    if pd.isna(v):
        return ""
    vmin = float(vmin) if pd.notna(vmin) else 0.0
    vmax = float(vmax) if pd.notna(vmax) else vmin + 1.0
    span = vmax - vmin
    t = min(max((float(v) - vmin) / span, 0.0), 1.0) if span > 0 else 0.0
    # White -> a medium, still-readable green (not near-black at the top end).
    r = round(255 + t * (150 - 255))
    g = round(255 + t * (200 - 255))
    b = round(255 + t * (165 - 255))
    return f"background-color:rgb({r},{g},{b});color:#1a1a1a"

def _bar_style(v, vmax, color):
    if pd.isna(v) or v == 0:
        return ""
    pct = min(abs(float(v)) / _safe(vmax), 1.0) * 100
    return f"background:linear-gradient(to right, {color} {pct:.1f}%, transparent {pct:.1f}%)"

def _diverging_bar_style(v, vmax, pos_color, neg_color):
    """Bar grows outward from the cell's center: green to the right for
    positive values, red to the left for negative — instead of both signs
    growing from the left edge, which made a small negative and a small
    positive look like they were on different scales."""
    if pd.isna(v) or v == 0:
        return ""
    half_pct = min(abs(float(v)) / _safe(vmax), 1.0) * 50
    if v >= 0:
        lo, hi, color = 50.0, 50.0 + half_pct, pos_color
    else:
        lo, hi, color = 50.0 - half_pct, 50.0, neg_color
    return (f"background:linear-gradient(to right, transparent {lo:.1f}%, "
            f"{color} {lo:.1f}%, {color} {hi:.1f}%, transparent {hi:.1f}%)")

def _oi_chg_style(v, vmax):
    if pd.isna(v):
        return ""
    return _diverging_bar_style(v, vmax, "rgba(22,163,74,0.55)", "rgba(220,38,38,0.55)")

def _vol_style(v, vmax):
    return _bar_style(v, vmax, "rgba(56,189,248,0.55)")


# -- Data freshness panel --------------------------------------------------
# load_data() filters open_interest > 0, so it cannot show the settlement-vs-OI
# lag: a session whose OI has not been published yet is simply absent from it.
# This reads the two date columns raw so the gap stays visible.
@st.cache_data(show_spinner=False)
def _freshness_row(commodity: str, mtime: float = 0.0):
    filename, _ = COMMODITIES[commodity]
    df = pd.read_parquet(DB_PATH / filename, columns=["Date", "settlement", "open_interest"])
    df["Date"] = pd.to_datetime(df["Date"])
    px = df.loc[df["settlement"].notna(), "Date"].max()
    oi = df.loc[df["open_interest"].notna() & (df["open_interest"] > 0), "Date"].max()
    return px, oi


def render_data_freshness(st_target=None):
    """Latest settlement and open-interest date per market.

    Refinitiv's daily timeseries carries a session's settlement before its open
    interest — the US markets routinely sit a session behind until the builder's
    quote top-up runs — so the two dates are shown separately rather than as one
    "last updated", which would have implied the OI was current when it was not.
    """
    tgt = st_target or st
    rows, newest_oi, any_lag = [], None, False
    for c in COMMODITIES:
        try:
            px, oi = _freshness_row(c, _mtime(c))
        except Exception:
            continue
        if pd.isna(oi):
            continue
        newest_oi = oi if newest_oi is None else max(newest_oi, oi)
        lag = pd.notna(px) and px > oi
        any_lag = any_lag or lag
        note = f" <span class='fr-px'>px {px:%d %b}</span>" if lag else ""
        rows.append(
            f"<tr><td class='fr-c'>{c}</td>"
            f"<td class='fr-d'>{oi:%d %b}{note}</td>"
            f"<td class='fr-s'><span class='fr-dot{' lag' if lag else ''}'></span></td></tr>")

    if not rows:
        return
    tip = ("<span class='fr-tip lag'>OI not out yet</span>" if any_lag
           else "<span class='fr-tip'>All up to date</span>")
    css = ("<style>"
           ".fr-card{background:rgba(255,255,255,.55);border:1px solid rgba(255,255,255,.95);border-radius:14px;"
           "padding:10px 12px 6px;margin-top:6px;font-family:'Plus Jakarta Sans','Inter',sans-serif;"
           "box-shadow:0 1px 2px rgba(10,36,99,.05),0 8px 24px rgba(10,36,99,.07),inset 0 1px 0 #fff}"
           ".fr-h{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-size:9.5px;"
           "font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:#7a86a8}"
           ".fr-tip{font-size:10px;font-weight:600;letter-spacing:.02em;text-transform:none;color:#16a34a}"
           ".fr-tip.lag{color:#b45309}"
           ".fr-t{width:100%;border-collapse:collapse;font-size:11.5px;font-variant-numeric:tabular-nums}"
           ".fr-t td{padding:3px 2px;border-bottom:1px solid rgba(43,74,138,.07)}"
           ".fr-t tr:last-child td{border-bottom:none}"
           ".fr-c{font-weight:700;color:#1e3a6e;letter-spacing:.04em}"
           ".fr-d{text-align:right;color:#334155}.fr-px{color:#9ca3af;font-size:10px;margin-left:4px}"
           ".fr-s{text-align:right;width:16px}"
           ".fr-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#16a34a;"
           "box-shadow:0 0 0 3px rgba(22,163,74,.15)}"
           ".fr-dot.lag{background:#f59e0b;box-shadow:0 0 0 3px rgba(245,158,11,.18)}"
           "</style>")
    tgt.markdown(
        css + "<div class='fr-card'><div class='fr-h'><span>OI as of</span>" + tip + "</div>"
        "<table class='fr-t'>" + "".join(rows) + "</table></div>",
        unsafe_allow_html=True)

