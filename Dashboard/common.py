# -*- coding: utf-8 -*-
"""Shared data layer for the Futures dashboard pages.

Both pages read the same per-contract-month parquets, so the loaders, the
commodity/month maps and the colour tokens live here instead of being
duplicated per page — one `@st.cache_data` entry per commodity is then shared
across pages rather than one cache per page holding its own copy of the same
frame. Deliberately thin: data access and constants only, no Streamlit UI.
"""
import streamlit as st
import pandas as pd
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
}

MONTH_NAMES = {
    "F": "January", "G": "February", "H": "March",  "J": "April",
    "K": "May",     "M": "June",     "N": "July",   "Q": "August",
    "U": "September","V": "October", "X": "November","Z": "December",
}

C = {
    # OI charts
    "oi_outer":  "rgba(99, 149, 237, 0.10)",
    "oi_inner":  "rgba(99, 149, 237, 0.28)",
    "oi_avg":    "#4A7FD4",
    # OI share
    "sh_outer":  "rgba(52, 168, 83, 0.10)",
    "sh_inner":  "rgba(52, 168, 83, 0.25)",
    "sh_avg":    "#34A853",
    # Vol/OI ratio
    "vr_outer":  "rgba(20, 184, 166, 0.10)",
    "vr_inner":  "rgba(20, 184, 166, 0.28)",
    "vr_avg":    "#0D9488",
    # Vol market share
    "vs_outer":  "rgba(139, 92, 246, 0.10)",
    "vs_inner":  "rgba(139, 92, 246, 0.28)",
    "vs_avg":    "#7C3AED",
    # Rolling volume
    "rv_outer":  "rgba(245, 158, 11, 0.10)",
    "rv_inner":  "rgba(245, 158, 11, 0.28)",
    "rv_avg":    "#D97706",
    # Common
    "current":   "#E8470A",
    "individual":"rgba(160,160,160,0.4)",
    "grid":      "rgba(0,0,0,0.07)",
    "bg":        "#ffffff",
    "font":      "#1a1a1a",
    "vline":     "rgba(0,0,0,0.18)",
}


# ── Data loaders ──────────────────────────────────────────────────────────────
def _mtime(commodity: str) -> float:
    filename, _ = COMMODITIES[commodity]
    p = DB_PATH / filename
    return p.stat().st_mtime if p.exists() else 0.0


_NUMERIC_COLS = ["Open", "High", "Low", "settlement", "volume", "open_interest"]


@st.cache_data
def load_data(commodity: str, mtime: float = 0.0) -> pd.DataFrame:
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


def _rollex_mtime() -> float:
    p = DB_PATH / "rollex.parquet"
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data
def load_rollex(commodity: str, mtime: float = 0.0):
    """Daily roll-adjusted return of the desk's Rollex index for one commodity,
    as a Date-indexed float Series, or None if the builder has not copied it.
    The builder copies it from the Rollex project on every run."""
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
    tot_vol = df.groupby("Date")["volume"].sum().rename("total_vol")
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
        dot = "#f59e0b" if lag else "#16a34a"
        note = f" <span style='color:#9ca3af'>(px {px:%d %b})</span>" if lag else ""
        rows.append(
            f"<tr><td style='color:#6b7280'>{c}</td>"
            f"<td style='text-align:right'>{oi:%d %b}{note}</td>"
            f"<td style='text-align:right'><span style='color:{dot}'>&#9679;</span></td></tr>")

    if not rows:
        return
    tip = "Amber: OI not out yet." if any_lag else "All up to date."
    tgt.markdown(
        "<div style='font-size:.70rem;font-weight:600;color:#6b7280;"
        "letter-spacing:.02em;margin:.2rem 0 .25rem'>OI AS OF</div>"
        "<table style='width:100%;font-size:.68rem;border-collapse:collapse'>"
        + "".join(rows) + "</table>",
        unsafe_allow_html=True)
    tgt.caption(tip)

