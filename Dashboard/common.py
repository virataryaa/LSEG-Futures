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

COMMODITIES = {
    "KC":  ("kc_futures.parquet",  "Coffee (KC)"),
    "CC":  ("cc_futures.parquet",  "Cocoa (CC)"),
    "CT":  ("ct_futures.parquet",  "Cotton (CT)"),
    "SB":  ("sb_futures.parquet",  "Sugar #11 (SB)"),
    "RC":  ("rc_futures.parquet",  "Robusta (RC)"),
    "LCC": ("lcc_futures.parquet", "Liffe Cocoa (LCC)"),
    "LSU": ("lsu_futures.parquet", "Liffe Sugar (LSU)"),
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


@st.cache_data
def load_data(commodity: str, mtime: float = 0.0) -> pd.DataFrame:
    filename, _ = COMMODITIES[commodity]
    df = pd.read_parquet(DB_PATH / filename)
    df["Date"] = pd.to_datetime(df["Date"])
    df["LTD"]  = pd.to_datetime(df["LTD"])
    df["days_to_expiry"] = (df["LTD"] - df["Date"]).dt.days
    return df[df["open_interest"] > 0].copy()


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
