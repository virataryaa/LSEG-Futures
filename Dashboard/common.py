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
