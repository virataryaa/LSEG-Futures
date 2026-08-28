# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Futures Dashboard", page_icon="📈", layout="wide")

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


def _most_active_month(df: pd.DataFrame) -> str:
    """Pick whichever currently-listed (not-yet-expired) contract has the
    highest Open Interest as of the latest available date, and return its
    delivery month — i.e. what's most active right now, not historically."""
    today  = pd.Timestamp(date.today())
    active = df[df["LTD"] >= today]
    if active.empty:
        active = df
    latest_date = active["Date"].max()
    latest      = active[active["Date"] == latest_date]
    return latest.loc[latest["open_interest"].idxmax(), "month"]


# ── Band computation ──────────────────────────────────────────────────────────
def _split_contracts(dm):
    today = pd.Timestamp(date.today())
    ltd   = dm.groupby("ice_symbol")["LTD"].first()
    return (ltd[ltd >= today].sort_values().index.tolist(),
            ltd[ltd <  today].sort_values().index.tolist())


def _densify_by_dte(hist_df, metric_col):
    """Reindex each contract onto a continuous integer DTE grid and
    linearly interpolate gaps (weekends/holidays). Removes the sparsity
    teeth artifacts that show up when only some contracts have a data
    point at a given integer DTE."""
    pieces = []
    for sym, grp in hist_df.groupby("ice_symbol"):
        g = (grp[["days_to_expiry", metric_col]]
             .drop_duplicates("days_to_expiry")
             .set_index("days_to_expiry")
             .sort_index())
        if len(g) < 2:
            continue
        full_idx = pd.RangeIndex(int(g.index.min()), int(g.index.max()) + 1)
        g = g.reindex(full_idx).interpolate(method="linear")
        g["ice_symbol"] = sym
        pieces.append(g.rename_axis("days_to_expiry").reset_index())
    return pd.concat(pieces, ignore_index=True) if pieces else hist_df


@st.cache_data(max_entries=200, show_spinner=False)
def _compute_dte_max(commodity, month, hist_year_range, mtime=0.0):
    """DTE_max: for each historical year, find the days-to-expiry where that
    year's OI peaked, then average those DTEs across years. This is a single
    reference day-to-expiry ('OI typically peaks here'), not a divisor."""
    df = load_data(commodity, mtime)
    dm = df[df["month"] == month].copy()
    _, hist_syms = _split_contracts(dm)
    hist = dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]
    if hist.empty:
        return None
    peak_dtes = hist.loc[hist.groupby("ice_symbol")["open_interest"].idxmax(), "days_to_expiry"]
    return int(round(peak_dtes.mean()))


def _hist_band(dense_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """min/max/mean/q25/q75 of metric_col per days_to_expiry, then a 7-point
    rolling smooth. Uses groupby(...).quantile() (pandas' own vectorized
    path) rather than .agg(hist_q25=lambda x: x.quantile(...)) — the lambda
    form calls quantile once per group via slow Python dispatch instead of
    pandas' optimized C-level implementation; on a wide days-to-expiry range
    this was the single biggest cost in a cold commodity switch (~5s of a
    ~10s total in profiling, from ~23k quantile calls)."""
    g = dense_df.groupby("days_to_expiry")[metric_col]
    band = g.agg(hist_min="min", hist_max="max", hist_mean="mean").reset_index()
    q25 = g.quantile(0.25).rename("hist_q25").reset_index()
    q75 = g.quantile(0.75).rename("hist_q75").reset_index()
    band = band.merge(q25, on="days_to_expiry").merge(q75, on="days_to_expiry")
    band = band.sort_values("days_to_expiry")
    for c in ["hist_mean", "hist_q25", "hist_q75"]:
        band[c] = band[c].rolling(7, center=True, min_periods=1).mean()
    return band


def _band_from_norm(dm, hist_syms, hist_year_range):
    hist_norm = dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]
    hist_norm_dense = _densify_by_dte(hist_norm, "open_interest")
    return _hist_band(hist_norm_dense, "open_interest")


@st.cache_data(max_entries=200, show_spinner=False)
def _normalize_oi_at_dte_max(commodity, month, hist_year_range, current_sym, mtime=0.0):
    """Method 2: every contract is divided by ITS OWN OI reading at DTE_max
    (the common reference day-to-expiry from _compute_dte_max) — not its own
    peak, so a year's line can go above/below 100% elsewhere. The current
    contract has no divisor (and isn't plotted) until it actually counts
    down past DTE_max, since there's no real reading yet."""
    df = load_data(commodity, mtime)
    dm = df[df["month"] == month].copy()
    active_syms, hist_syms = _split_contracts(dm)
    if not active_syms or not hist_syms:
        return None

    dte_max = _compute_dte_max(commodity, month, hist_year_range, mtime)
    if dte_max is None:
        return None

    hist_syms_range = sorted(dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]["ice_symbol"].unique().tolist())

    divisors = {}
    for sym in hist_syms_range + [current_sym]:
        g = (dm.loc[dm["ice_symbol"] == sym, ["days_to_expiry", "open_interest"]]
             .drop_duplicates("days_to_expiry")
             .set_index("days_to_expiry")
             .sort_index())
        if len(g) < 2:
            continue
        lo, hi = int(g.index.min()), int(g.index.max())
        if dte_max < lo or dte_max > hi:
            continue  # this contract's life never covered DTE_max (e.g. current, still to come)
        dense = g.reindex(range(lo, hi + 1)).interpolate(method="linear")
        divisors[sym] = float(dense.loc[dte_max, "open_interest"])

    hist_syms_ok = [s for s in hist_syms_range if divisors.get(s, 0) > 0]
    if not hist_syms_ok:
        return None

    dm = dm.copy()
    for s in hist_syms_ok:
        mask = dm["ice_symbol"] == s
        dm.loc[mask, "open_interest"] = dm.loc[mask, "open_interest"] / divisors[s] * 100

    current_reached = divisors.get(current_sym, 0) > 0
    curr_df = None
    if current_reached:
        mask = dm["ice_symbol"] == current_sym
        dm.loc[mask, "open_interest"] = dm.loc[mask, "open_interest"] / divisors[current_sym] * 100
        curr_df = dm[dm["ice_symbol"] == current_sym].sort_values("Date").copy()

    band = _band_from_norm(dm, hist_syms_ok, hist_year_range)
    hist_norm = dm[
        dm["ice_symbol"].isin(hist_syms_ok) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ].copy()
    return band, curr_df, dte_max, current_reached, hist_norm


@st.cache_data(max_entries=200, show_spinner=False)
def compute_band(commodity, month, hist_year_range, metric_col,
                 roll_n=None, use_enriched=False, mtime=0.0):
    """
    Generic band + current-contract computation for any metric.
    roll_n: if set, compute rolling(roll_n).mean() on 'volume' first (by Date order).
    Returns (band, curr_df, active_syms, hist_syms) or None.
    """
    df = load_enriched(commodity, mtime) if use_enriched else load_data(commodity, mtime)
    dm = df[df["month"] == month].copy()

    active_syms, hist_syms = _split_contracts(dm)
    if not active_syms:
        return None

    if roll_n:
        pieces = []
        for sym, grp in dm.groupby("ice_symbol"):
            g = grp.sort_values("Date").copy()
            g["_metric"] = g["volume"].rolling(roll_n, min_periods=1).mean()
            pieces.append(g)
        dm = pd.concat(pieces)
        metric_col = "_metric"

    hist_df = dm[
        dm["ice_symbol"].isin(hist_syms) &
        dm["year"].between(hist_year_range[0], hist_year_range[1])
    ]

    hist_dense = _densify_by_dte(hist_df, metric_col)
    band = _hist_band(hist_dense, metric_col)

    curr_df = dm[dm["ice_symbol"] == active_syms[0]].sort_values("Date").copy()
    return band, curr_df, active_syms, hist_syms


# ── Generic chart builder ─────────────────────────────────────────────────────
def build_chart(band, curr_df, metric_col, current_sym,
                title, y_title, y_fmt, y_suffix,
                outer_color, inner_color, avg_color,
                dte_range, dte_now,
                show_individual=False, hist_df=None, ind_metric=None,
                height=500):
    fig = go.Figure()

    # Outer band
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_max"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_min"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=outer_color,
        name="Min-Max Range", hoverinfo="skip"))

    # Inner band q25-q75
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q75"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q25"],
        mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=inner_color,
        name="25th-75th Pct", hoverinfo="skip"))

    # Mean line
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_mean"],
        mode="lines", line=dict(color=avg_color, width=2, dash="dash"),
        name="Historical Mean",
        hovertemplate=f"DTE: %{{x}}<br>Mean: %{{y:{y_fmt}}}{y_suffix}<extra>Mean</extra>"))

    # Individual years
    if show_individual and hist_df is not None and ind_metric:
        for sym, grp in hist_df.groupby("ice_symbol"):
            grp = grp.sort_values("days_to_expiry")
            fig.add_trace(go.Scatter(x=grp["days_to_expiry"], y=grp[ind_metric],
                mode="lines", line=dict(width=0.9, color=C["individual"]),
                name=sym, showlegend=True,
                hovertemplate=f"{sym} DTE:%{{x}} %{{y:{y_fmt}}}<extra></extra>"))

    # Current line (skipped when there's no current-year data yet, e.g. the
    # contract hasn't reached the DTE_max reference point under Method 2)
    if curr_df is not None and not curr_df.empty:
        fig.add_trace(go.Scatter(x=curr_df["days_to_expiry"], y=curr_df[metric_col],
            mode="lines", line=dict(color=C["current"], width=2.5),
            name=current_sym,
            hovertemplate=f"<b>{current_sym}</b><br>DTE: %{{x}}<br>%{{y:{y_fmt}}}{y_suffix}<extra></extra>"))

        # Latest dot
        latest = curr_df.iloc[-1]
        lat_val = latest[metric_col]
        lat_dte = int(latest["days_to_expiry"])
        lat_dt  = latest["Date"].strftime("%b %d, %Y")
        fig.add_trace(go.Scatter(x=[lat_dte], y=[lat_val],
            mode="markers",
            marker=dict(color=C["current"], size=8, line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate=f"<b>{lat_dt}</b><br>DTE: {lat_dte}<br>{lat_val:{y_fmt}}{y_suffix}<extra></extra>"))

    fig.add_vline(x=dte_now, line=dict(color=C["vline"], width=1, dash="dot"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=C["font"]), x=0.01),
        xaxis=dict(title="Days to Expiry", range=[dte_range[0], dte_range[1]],
                   showgrid=True, gridcolor=C["grid"], zeroline=False,
                   tickfont=dict(size=11, color=C["font"])),
        yaxis=dict(title=y_title, showgrid=True, gridcolor=C["grid"],
                   zeroline=False, tickformat=y_fmt.replace(",",",.0f").replace(".1f",".1f"),
                   ticksuffix=y_suffix, tickfont=dict(size=11, color=C["font"])),
        plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font=dict(color=C["font"], family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified", height=height,
        margin=dict(l=70, r=30, t=60, b=120),
    )
    return fig


def kpi_row(vals: list):
    """vals = list of (label, value, delta, delta_color) — delta/delta_color optional."""
    cols = st.columns(len(vals))
    for col, item in zip(cols, vals):
        label, value = item[0], item[1]
        delta        = item[2] if len(item) > 2 else None
        dc           = item[3] if len(item) > 3 else "normal"
        if delta is not None:
            col.metric(label, value, delta=delta, delta_color=dc)
        else:
            col.metric(label, value)


# ── Subplot trace helper (for 2x2 grid) ──────────────────────────────────────
def add_oi_traces(fig, band, curr_df, current_sym, oi_fmt,
                  show_individual=False, hist_df=None, row=None, col=None, show_legend=True):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_max"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_min"],
        mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C["oi_outer"],
        name="Min-Max", showlegend=show_legend, hoverinfo="skip", legendgroup="outer"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q75"],
        mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_q25"],
        mode="lines", line=dict(width=0), fill="tonexty", fillcolor=C["oi_inner"],
        name="25-75 Pct", showlegend=show_legend, hoverinfo="skip", legendgroup="inner"), **kw)
    fig.add_trace(go.Scatter(x=band["days_to_expiry"], y=band["hist_mean"],
        mode="lines", line=dict(color=C["oi_avg"], width=1.5, dash="dash"),
        name="Mean", showlegend=show_legend, legendgroup="mean",
        hovertemplate=f"DTE: %{{x}}<br>Mean: %{{y:{oi_fmt}}}<extra>Mean</extra>"), **kw)
    fig.add_trace(go.Scatter(x=curr_df["days_to_expiry"], y=curr_df["open_interest"],
        mode="lines", line=dict(color=C["current"], width=2),
        name=current_sym, showlegend=show_legend, legendgroup="curr",
        hovertemplate=f"<b>{current_sym}</b><br>DTE:%{{x}}<br>OI:%{{y:{oi_fmt}}}<extra></extra>"), **kw)
    latest = curr_df.iloc[-1]
    fig.add_trace(go.Scatter(x=[int(latest["days_to_expiry"])], y=[latest["open_interest"]],
        mode="markers", marker=dict(color=C["current"], size=7, line=dict(color="white", width=1.5)),
        showlegend=False,
        hovertemplate=f"<b>{latest['Date'].strftime('%b %d, %Y')}</b><br>OI:{latest['open_interest']:{oi_fmt}}<extra></extra>"), **kw)


# ── Daily OI/Volume-by-contract-month table (cached — building this HTML by
# hand for every row/column was the single most expensive uncached thing on
# the page, and it re-ran on every widget change anywhere in the app since
# Streamlit executes every tab's body on every rerun, not just the visible
# one) ──────────────────────────────────────────────────────────────────────
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


@st.cache_data(max_entries=50, show_spinner=False)
def build_oi_vol_table_html(commodity: str, table_lookback: int, mtime: float = 0.0):
    """Returns the full HTML string for the OI/Volume-by-contract-month table,
    or None if there's no data in the window."""
    df_all_tbl = load_data(commodity, mtime).sort_values(["ice_symbol", "Date"])
    df_all_tbl["oi_change"] = df_all_tbl.groupby("ice_symbol")["open_interest"].diff()
    df_all_tbl["px_change"] = df_all_tbl.groupby("ice_symbol")["settlement"].pct_change() * 100

    max_date_tbl = df_all_tbl["Date"].max()
    cutoff_tbl   = max_date_tbl - pd.Timedelta(days=table_lookback)
    win_tbl      = df_all_tbl[df_all_tbl["Date"] >= cutoff_tbl].copy()

    if win_tbl.empty:
        return None

    # Every contract month with any OI in the window — sorted near-to-far expiry.
    ltd_map  = win_tbl.groupby("ice_symbol")["LTD"].first()
    syms_tbl = ltd_map.sort_values().index.tolist()
    dates_tbl = sorted(win_tbl["Date"].unique(), reverse=True)

    oi_piv  = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="open_interest", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))
    chg_piv = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="oi_change", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))
    vol_piv = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="volume", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))
    px_piv  = (win_tbl.pivot_table(index="Date", columns="ice_symbol", values="px_change", aggfunc="last")
               .reindex(index=dates_tbl, columns=syms_tbl))

    # min_count=1: a date where every contract's OI change is null sums to
    # NaN, not a misleading 0 (same lesson as the Options project's OI bug).
    total_chg = chg_piv.sum(axis=1, min_count=1)
    total_vol = vol_piv.sum(axis=1, min_count=1)

    # Per-column scaling — a front-month contract's OI/volume dwarfs a
    # far-month one, so a single table-wide scale would make every far-month
    # cell look flat. Each contract's heatmap/bars are scaled to its own range.
    oi_col_min  = oi_piv.min(axis=0)
    oi_col_max  = oi_piv.max(axis=0)
    chg_col_max = chg_piv.abs().max(axis=0)
    vol_col_max = vol_piv.max(axis=0)
    px_col_max  = px_piv.abs().max(axis=0)
    total_chg_absmax = float(total_chg.abs().max()) if total_chg.notna().any() else 1.0
    total_vol_max    = float(total_vol.max())        if total_vol.notna().any() else 1.0
    total_chg_absmax = total_chg_absmax if total_chg_absmax > 0 else 1.0
    total_vol_max    = total_vol_max    if total_vol_max    > 0 else 1.0

    css = """
    <style>
    .oivol-wrap { overflow:auto; max-height:640px; border:1px solid #e5e7eb; border-radius:6px; }
    .oivol-tbl { border-collapse:collapse; font-size:9px; font-family:'Inter',sans-serif; white-space:nowrap; }
    .oivol-tbl th, .oivol-tbl td { padding:2px 5px; text-align:center; border-bottom:1px solid #f0f0f0; }
    .oivol-tbl th { position:sticky; top:0; background:#fafafa; font-weight:600; z-index:2; }
    .oivol-tbl .grp-h { background:#eef2f7; }
    /* box-shadow instead of border-left: border-collapse silently drops
       adjacent-cell borders depending on which side "wins" the merge, but
       box-shadow isn't part of the border-collapse model so it always shows. */
    .oivol-tbl .grp-start { box-shadow: inset 2px 0 0 0 #374151; }
    .oivol-tbl .date-cell { position:sticky; left:0; background:#fff; text-align:center;
                             font-weight:600; z-index:1; box-shadow: inset -2px 0 0 0 #374151; }
    .oivol-tbl .tot-cell { background:#fffbea; font-weight:600; }
    .oivol-tbl .sub-h { color:#888; font-weight:400; font-size:8px; }
    </style>
    """

    h1 = '<tr><th class="date-cell" rowspan="2">Date</th>'
    for s in syms_tbl:
        h1 += f'<th class="grp-h grp-start" colspan="4">{s}</th>'
    h1 += '<th class="tot-cell grp-start" colspan="2">Total</th></tr>'

    h2 = "<tr>"
    for s in syms_tbl:
        h2 += ('<th class="sub-h grp-h grp-start">OI</th>'
               '<th class="sub-h grp-h">ΔOI</th>'
               '<th class="sub-h grp-h">PxΔ%</th>'
               '<th class="sub-h grp-h">Vol</th>')
    h2 += '<th class="sub-h tot-cell grp-start">ΔOI</th><th class="sub-h tot-cell">Vol</th></tr>'

    rows = []
    for d in dates_tbl:
        d_str = pd.Timestamp(d).strftime("%d %b %Y")
        row = f'<tr><td class="date-cell">{d_str}</td>'
        for s in syms_tbl:
            oi_v  = oi_piv.at[d, s]
            chg_v = chg_piv.at[d, s]
            vol_v = vol_piv.at[d, s]
            px_v  = px_piv.at[d, s]
            oi_txt  = f"{oi_v:,.0f}"  if pd.notna(oi_v)  else ""
            chg_txt = f"{chg_v:+,.0f}" if pd.notna(chg_v) else ""
            vol_txt = f"{vol_v:,.0f}" if pd.notna(vol_v) else ""
            px_txt  = f"{px_v:+.2f}%" if pd.notna(px_v)  else ""
            row += f'<td class="grp-start" style="{_oi_heatmap_style(oi_v, oi_col_min[s], oi_col_max[s])}">{oi_txt}</td>'
            row += f'<td style="{_oi_chg_style(chg_v, chg_col_max[s])}">{chg_txt}</td>'
            row += f'<td style="{_oi_chg_style(px_v, px_col_max[s])}">{px_txt}</td>'
            row += f'<td style="{_vol_style(vol_v, vol_col_max[s])}">{vol_txt}</td>'
        tc, tv = total_chg.loc[d], total_vol.loc[d]
        tc_txt = f"{tc:+,.0f}" if pd.notna(tc) else ""
        tv_txt = f"{tv:,.0f}"  if pd.notna(tv) else ""
        row += f'<td class="tot-cell grp-start" style="{_oi_chg_style(tc, total_chg_absmax)}">{tc_txt}</td>'
        row += f'<td class="tot-cell" style="{_vol_style(tv, total_vol_max)}">{tv_txt}</td>'
        row += "</tr>"
        rows.append(row)

    return (css + f'<div class="oivol-wrap"><table class="oivol-tbl"><thead>{h1}{h2}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# ── Spot OI report — spot (most-active month) vs non-spot OI, with COT- ────────
#    Tuesday-aligned weekly snapshots, matching the desk's reference sheet.
def _fmt_num(v, signed: bool = False) -> str:
    if pd.isna(v):
        return ""
    return f"{v:+,.0f}" if signed else f"{v:,.0f}"


def _fmt_pct(v) -> str:
    if pd.isna(v):
        return ""
    return f"{v:+.1f}%"


def _flat_tint(v) -> str:
    """A flat, sign-only background wash (no magnitude bar) — reads clean
    across many narrow columns, where a magnitude-scaled bar tends to
    render as a thin, glitchy-looking sliver for small values."""
    if pd.isna(v) or v == 0:
        return ""
    return "background:rgba(22,163,74,.12)" if v > 0 else "background:rgba(220,38,38,.12)"


# Shared fixed widths so the contract-month columns line up pixel-for-pixel
# between the daily grid and the separate "per expiry" table below it — two
# independent tables won't auto-align on their own since each has a
# different total column count.
_CCOL_W = 72
_DATECOL_W = 96


def _sign_color(v) -> str:
    if pd.isna(v) or v == 0:
        return "#1d1d1f"
    return "#16a34a" if v > 0 else "#dc2626"


def _spot_series(oi_piv: pd.DataFrame, px_piv: pd.DataFrame):
    """For every date, find whichever contract has the highest OI (the
    'spot'/most-active month) and pull its OI and settlement price — the
    active month shifts over time as contracts roll (e.g. Z6 now, H7 once
    Z6 expires), unlike 'nearest expiry' which would stay on a thin,
    already-rolled-out-of front month."""
    syms_out, oi_out, px_out = [], [], []
    for d in oi_piv.index:
        row = oi_piv.loc[d]
        if row.notna().any():
            sym = row.idxmax()
            syms_out.append(sym)
            oi_out.append(row[sym])
            px_out.append(px_piv.at[d, sym] if sym in px_piv.columns else np.nan)
        else:
            syms_out.append(None); oi_out.append(np.nan); px_out.append(np.nan)
    idx = oi_piv.index
    return (pd.Series(syms_out, index=idx), pd.Series(oi_out, index=idx, dtype=float),
            pd.Series(px_out, index=idx, dtype=float))


@st.cache_data(max_entries=50, show_spinner=False)
def build_spot_oi_data(commodity: str, mtime: float = 0.0) -> dict:
    """Full-history spot/non-spot OI series for a commodity. Cached once
    per commodity; the report tab below slices this to its own lookback
    window for display, and diffs are computed here on the full series
    first so a display-window edge never truncates a change calculation."""
    df_all = load_data(commodity, mtime).sort_values(["ice_symbol", "Date"])
    today = pd.Timestamp(date.today())
    ltd_map = df_all.groupby("ice_symbol")["LTD"].first()
    # Only currently-unexpired contract months — matches _split_contracts'
    # convention elsewhere in this file. Restricting up front (rather than
    # showing every contract ever traded) keeps the column set to the
    # handful of months actually relevant right now.
    syms = ltd_map[ltd_map >= today].sort_values().index.tolist()

    oi_piv = df_all.pivot_table(index="Date", columns="ice_symbol", values="open_interest", aggfunc="last").reindex(columns=syms)
    px_piv = df_all.pivot_table(index="Date", columns="ice_symbol", values="settlement", aggfunc="last").reindex(columns=syms)

    total_oi = oi_piv.sum(axis=1, min_count=1)
    spot_sym, spot_oi, spot_price = _spot_series(oi_piv, px_piv)
    non_spot_oi = total_oi - spot_oi

    return dict(
        syms=syms, oi_piv=oi_piv, px_piv=px_piv, total_oi=total_oi,
        spot_sym=spot_sym, spot_oi=spot_oi, spot_price=spot_price, non_spot_oi=non_spot_oi,
        oi_chg=total_oi.diff(), spot_oi_chg=spot_oi.diff(), spot_oi_5d=spot_oi.rolling(5).mean(),
        price_chg_pct=spot_price.pct_change() * 100,
        per_contract_chg=oi_piv.diff(),
    )


def build_spot_summary_html(data: dict) -> str:
    """LAST (today's raw OI) + the day-over-day and since-last-COT deltas,
    then the last two confirmed COT Tuesdays with their own week-over-week
    deltas — 'confirmed' meaning strictly before the latest date, since
    the current week's Tuesday COT report isn't published until Friday
    even if today happens to be that Tuesday."""
    syms = data["syms"]; oi_piv = data["oi_piv"]; total_oi = data["total_oi"]; spot_price = data["spot_price"]
    dates = list(oi_piv.index)
    if not dates:
        return "<p>No data.</p>"
    max_date = dates[-1]
    prev_day = dates[-2] if len(dates) >= 2 else None
    tuesdays = [d for d in dates if pd.Timestamp(d).weekday() == 1 and d < max_date]
    last_cot  = tuesdays[-1] if len(tuesdays) >= 1 else None
    prev_cot  = tuesdays[-2] if len(tuesdays) >= 2 else None
    prev_cot2 = tuesdays[-3] if len(tuesdays) >= 3 else None

    def oi_row(d):
        return oi_piv.loc[d] if d is not None else pd.Series(index=syms, dtype=float)

    def px(d):
        return spot_price.loc[d] if d is not None else np.nan

    def px_chg_pct(d1, d0):
        p1, p0 = px(d1), px(d0)
        return (p1 - p0) / p0 * 100 if (d1 is not None and d0 is not None and p0) else np.nan

    css = """<style>
      .spotsum-wrap{overflow-x:auto;border:1px solid #e5e7eb;border-radius:6px;margin-bottom:8px}
      table.spotsum{border-collapse:collapse;width:100%;font-size:.66rem;font-family:'Inter',sans-serif;white-space:nowrap}
      table.spotsum th,table.spotsum td{padding:2px 6px;text-align:center;border-bottom:1px solid #f0f0f0}
      table.spotsum th{position:sticky;top:0;background:#fafafa;color:#1a1a1a;font-weight:600;
        font-size:.58rem;text-transform:uppercase;letter-spacing:.02em;border-bottom:2px solid #d1d5db}
      table.spotsum td.lbl{text-align:left;font-weight:600;color:#1d1d1f}
      table.spotsum tr.delta td.lbl{font-weight:400;color:#9ca3af;font-size:.62rem}
      table.spotsum tr.spacer td{padding:3px 0;border:none}
      table.spotsum td.tot{font-weight:700;background:#fafafa}
      table.spotsum tr.tue-row{background:#eceef1}
    </style>"""

    def value_row(label, d):
        if d is None:
            return ""
        tr_cls = " class='tue-row'" if pd.Timestamp(d).weekday() == 1 else ""
        cells = "".join(f"<td>{_fmt_num(oi_row(d).get(s))}</td>" for s in syms)
        px_txt = f"{px(d):.2f}" if pd.notna(px(d)) else ""
        return (f"<tr{tr_cls}><td class='lbl'>{label}</td>{cells}"
                f"<td class='tot'>{_fmt_num(total_oi.get(d))}</td><td>{px_txt}</td></tr>")

    def delta_row(label, d1, d0):
        if d1 is None or d0 is None:
            return ""
        delta = oi_row(d1) - oi_row(d0)
        cells = "".join(
            f"<td style='{_flat_tint(delta.get(s))};color:{_sign_color(delta.get(s))};font-weight:600'>"
            f"{_fmt_num(delta.get(s), True)}</td>" for s in syms
        )
        tot_delta = total_oi.get(d1) - total_oi.get(d0)
        px_pct = px_chg_pct(d1, d0)
        return (
            f"<tr class='delta'><td class='lbl'>{label}</td>{cells}"
            f"<td class='tot' style='color:{_sign_color(tot_delta)}'>{_fmt_num(tot_delta, True)}</td>"
            f"<td style='color:{_sign_color(px_pct)}'>{_fmt_pct(px_pct)}</td></tr>"
        )

    spacer = f"<tr class='spacer'><td colspan='{len(syms) + 3}'></td></tr>"
    header = "<tr><th class='lbl'>Date</th>" + "".join(f"<th>{s}</th>" for s in syms) + "<th>Total</th><th>Price</th></tr>"
    body = (
        value_row(pd.Timestamp(max_date).strftime("%d/%m/%Y"), max_date)
        + delta_row("+/- day", max_date, prev_day)
        + delta_row("+/- Last COT", max_date, last_cot)
        + spacer
        + value_row(pd.Timestamp(last_cot).strftime("%d/%m/%Y") if last_cot else "", last_cot)
        + delta_row("+/- week", last_cot, prev_cot)
        + spacer
        + value_row(pd.Timestamp(prev_cot).strftime("%d/%m/%Y") if prev_cot else "", prev_cot)
        + delta_row("+/- week", prev_cot, prev_cot2)
    )
    return f"{css}<div class='spotsum-wrap'><table class='spotsum'>{header}<tbody>{body}</tbody></table></div>"


@st.cache_data(max_entries=50, show_spinner=False)
def build_spot_daily_table_html(commodity: str, table_lookback: int, leg1: str, leg2: str,
                                price_source: str = "Spot (Most OI)", mtime: float = 0.0):
    """Daily grid: per-contract OI, Total, Price (spot by default — see
    price_source), day OI/Spot-OI changes, Non-Spot OI, a user-picked
    calendar-spread, and the spot month's 5-trading-day OI change."""
    data = build_spot_oi_data(commodity, mtime)
    syms = data["syms"]; oi_piv = data["oi_piv"]; px_piv = data["px_piv"]
    total_oi = data["total_oi"]
    oi_chg = data["oi_chg"]; spot_oi_chg = data["spot_oi_chg"]
    spot_oi_5d = data["spot_oi_5d"]

    if price_source != "Spot (Most OI)" and price_source in px_piv.columns:
        # A fixed, single contract's price throughout — unlike Spot, this
        # doesn't switch contracts as OI leadership rolls from one month
        # to the next, useful for tracking one specific expiry's own price.
        spot_price = px_piv[price_source]
        price_chg_pct = spot_price.pct_change() * 100
    else:
        spot_price = data["spot_price"]; price_chg_pct = data["price_chg_pct"]

    if oi_piv.empty:
        return None
    max_date = oi_piv.index.max()
    cutoff = max_date - pd.Timedelta(days=table_lookback)
    dates = [d for d in oi_piv.index if d >= cutoff]
    if not dates:
        return None
    dates_desc = sorted(dates, reverse=True)

    have_spread = leg1 in px_piv.columns and leg2 in px_piv.columns
    spread = (px_piv[leg1] - px_piv[leg2]) if have_spread else pd.Series(dtype=float)
    root_len = len(commodity)
    spread_label = f"{leg1}-{leg2[root_len:]}" if have_spread else "Spread"

    oi_chg_vmax = _safe(oi_chg.loc[dates].abs().max())

    css = f"""<style>
      .spotgrid-wrap{{overflow:auto;max-height:600px;border:1px solid #e5e7eb;border-radius:6px}}
      table.spotgrid{{border-collapse:collapse;width:100%;font-size:.66rem;font-family:'Inter',sans-serif;white-space:nowrap}}
      table.spotgrid th,table.spotgrid td{{padding:1px 6px;text-align:center;border-bottom:1px solid #f4f4f5}}
      table.spotgrid th{{position:sticky;top:0;background:#0a2463;color:#fff;font-weight:600;z-index:2;
        font-size:.6rem;text-transform:uppercase;letter-spacing:.02em;border-bottom:2px solid #0a2463}}
      table.spotgrid .date-cell{{position:sticky;left:0;background:#fff;font-weight:600;z-index:1;
        box-shadow:inset -1px 0 0 0 #e5e7eb;min-width:{_DATECOL_W}px}}
      table.spotgrid th.date-cell{{background:#0a2463;color:#fff;z-index:3}}
      table.spotgrid .ccol{{min-width:{_CCOL_W}px}}
      table.spotgrid .tot-cell{{background:#fafafa;font-weight:700}}
      table.spotgrid th.tot-cell{{background:#0a2463;color:#fff}}
      table.spotgrid tr.tue-row{{background:#eceef1}}
      table.spotgrid tbody tr:hover td{{background-color:rgba(10,36,99,.04)}}
    </style>"""

    if price_source == "Spot (Most OI)":
        latest_spot = data["spot_sym"].iloc[-1] if len(data["spot_sym"]) else None
        price_label = f"Price (Spot: {latest_spot})" if latest_spot else "Price (Spot)"
    else:
        price_label = f"Price ({price_source})"
    header = ("<tr><th class='date-cell'>Date</th>" + "".join(f"<th class='ccol'>{s}</th>" for s in syms) +
              f"<th class='tot-cell'>Total</th><th>{price_label}</th><th>+/-</th>"
              "<th>OI Chg</th><th>Spot OI +/-</th><th>Non Spot Chg</th><th>Date</th>"
              f"<th>{spread_label}</th><th>Spot OI 5d Avg</th></tr>")

    rows = []
    for d in dates_desc:
        d_str = pd.Timestamp(d).strftime("%d/%m/%Y")
        tr_cls = " class='tue-row'" if pd.Timestamp(d).weekday() == 1 else ""
        cells = f"<td class='date-cell'>{d_str}</td>"
        for s in syms:
            v = oi_piv.at[d, s] if s in oi_piv.columns else np.nan
            cells += f"<td class='ccol'>{_fmt_num(v)}</td>"
        px_v = spot_price.get(d)
        px_pct_v, oi_chg_v, spot_chg_v, spread_v, spot5d_v = (
            price_chg_pct.get(d), oi_chg.get(d), spot_oi_chg.get(d),
            spread.get(d) if have_spread else np.nan, spot_oi_5d.get(d),
        )
        # Non Spot Chg = Total OI change minus Spot OI change — algebraically
        # the same as diffing the Non-Spot level directly, since
        # Non Spot = Total - Spot.
        non_spot_chg_v = oi_chg_v - spot_chg_v if pd.notna(oi_chg_v) and pd.notna(spot_chg_v) else np.nan
        cells += f"<td class='tot-cell'>{_fmt_num(total_oi.get(d))}</td>"
        cells += f"<td>{px_v:.2f}</td>" if pd.notna(px_v) else "<td></td>"
        cells += f"<td style='{_flat_tint(px_pct_v)};color:{_sign_color(px_pct_v)}'>{_fmt_pct(px_pct_v)}</td>"
        cells += f"<td style='{_oi_chg_style(oi_chg_v, oi_chg_vmax)}'>{_fmt_num(oi_chg_v, True)}</td>"
        cells += f"<td style='{_flat_tint(spot_chg_v)};color:{_sign_color(spot_chg_v)};font-weight:600'>{_fmt_num(spot_chg_v, True)}</td>"
        cells += f"<td style='{_flat_tint(non_spot_chg_v)};color:{_sign_color(non_spot_chg_v)}'>{_fmt_num(non_spot_chg_v, True)}</td>"
        cells += f"<td style='color:#9ca3af'>{d_str}</td>"
        cells += (f"<td style='{_flat_tint(spread_v)};color:{_sign_color(spread_v)}'>{spread_v:+.2f}</td>"
                  if pd.notna(spread_v) else "<td></td>")
        cells += f"<td>{_fmt_num(spot5d_v)}</td>"
        rows.append(f"<tr{tr_cls}>{cells}</tr>")

    return f"{css}<div class='spotgrid-wrap'><table class='spotgrid'>{header}<tbody>{''.join(rows)}</tbody></table></div>"


@st.cache_data(max_entries=50, show_spinner=False)
def build_expiry_chg_table_html(commodity: str, table_lookback: int, mtime: float = 0.0):
    """Day-over-day OI change for every individual contract month (not the
    aggregate) — each column scaled to its own range, since a front month's
    change dwarfs a far month's."""
    data = build_spot_oi_data(commodity, mtime)
    syms = data["syms"]; per_chg = data["per_contract_chg"]

    if per_chg.empty:
        return None
    max_date = per_chg.index.max()
    cutoff = max_date - pd.Timedelta(days=table_lookback)
    dates = [d for d in per_chg.index if d >= cutoff]
    if not dates:
        return None
    dates_desc = sorted(dates, reverse=True)
    col_vmax = per_chg.loc[dates].abs().max()

    css = f"""<style>
      .expchg-wrap{{overflow:auto;max-height:480px;border:1px solid #e5e7eb;border-radius:6px}}
      table.expchg{{border-collapse:collapse;font-size:.66rem;font-family:'Inter',sans-serif;white-space:nowrap}}
      table.expchg th,table.expchg td{{padding:1px 6px;text-align:center;border-bottom:1px solid #f4f4f5}}
      table.expchg th{{position:sticky;top:0;background:#fafafa;color:#1a1a1a;font-weight:600;z-index:2;
        font-size:.6rem;text-transform:uppercase;letter-spacing:.02em;border-bottom:2px solid #d1d5db}}
      table.expchg .date-cell{{position:sticky;left:0;background:#fff;font-weight:600;z-index:1;
        box-shadow:inset -1px 0 0 0 #e5e7eb;min-width:{_DATECOL_W}px}}
      table.expchg .ccol{{min-width:{_CCOL_W}px}}
      table.expchg tr.tue-row{{background:#eceef1}}
      table.expchg tbody tr:hover td{{background-color:rgba(10,36,99,.04)}}
    </style>"""

    header = "<tr><th class='date-cell'>Date</th>" + "".join(f"<th class='ccol'>{s}</th>" for s in syms) + "</tr>"
    rows = []
    for d in dates_desc:
        d_str = pd.Timestamp(d).strftime("%d/%m/%Y")
        tr_cls = " class='tue-row'" if pd.Timestamp(d).weekday() == 1 else ""
        cells = f"<td class='date-cell'>{d_str}</td>"
        for s in syms:
            v = per_chg.at[d, s] if s in per_chg.columns else np.nan
            cells += f"<td class='ccol' style='{_oi_chg_style(v, col_vmax.get(s))}'>{_fmt_num(v, True)}</td>"
        rows.append(f"<tr{tr_cls}>{cells}</tr>")

    return f"{css}<div class='expchg-wrap'><table class='expchg'>{header}<tbody>{''.join(rows)}</tbody></table></div>"


@st.cache_data(max_entries=50, show_spinner=False)
def build_oi_spread_chart(commodity: str, table_lookback: int, oi_choice: str, leg1: str, leg2: str,
                          mtime: float = 0.0):
    """Dual-axis chart: an OI series (Spot by default, or any single
    contract) on the left axis, the chosen calendar spread on the right —
    same idea as the reference workbook's own OI-vs-spread chart."""
    data = build_spot_oi_data(commodity, mtime)
    oi_piv = data["oi_piv"]; px_piv = data["px_piv"]
    if oi_piv.empty:
        return None
    max_date = oi_piv.index.max()
    cutoff = max_date - pd.Timedelta(days=table_lookback)
    dates = sorted(d for d in oi_piv.index if d >= cutoff)
    if not dates:
        return None

    if oi_choice == "Spot (Most OI)":
        oi_series = data["spot_oi"].reindex(dates)
        latest_spot = data["spot_sym"].iloc[-1] if len(data["spot_sym"]) else None
        oi_name = f"Spot OI ({latest_spot})" if latest_spot else "Spot OI"
    else:
        oi_series = oi_piv[oi_choice].reindex(dates) if oi_choice in oi_piv.columns else pd.Series(dtype=float)
        oi_name = f"{oi_choice} OI"

    have_spread = leg1 in px_piv.columns and leg2 in px_piv.columns
    root_len = len(commodity)
    spread_name = f"{leg1}-{leg2[root_len:]} Spread" if have_spread else "Spread"
    spread_series = (px_piv[leg1] - px_piv[leg2]).reindex(dates) if have_spread else pd.Series(dtype=float)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=oi_series.values, name=oi_name,
                             line=dict(color="#1a56db", width=2)), secondary_y=False)
    if have_spread:
        fig.add_trace(go.Scatter(x=dates, y=spread_series.values, name=spread_name,
                                 line=dict(color="#f59e0b", width=2)), secondary_y=True)
    fig.update_layout(height=380, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(l=55, r=55, t=30, b=40))
    fig.update_yaxes(title_text=oi_name, secondary_y=False, gridcolor="rgba(0,0,0,.07)")
    fig.update_yaxes(title_text=spread_name, secondary_y=True, showgrid=False)
    return fig


@st.cache_data(max_entries=50, show_spinner=False)
def build_term_structure_chart(commodity: str, snapshot_date, mtime: float = 0.0):
    """Term structure snapshot for one date: OI per contract as grey bars
    (left axis), price per contract as a line (right axis) — the curve
    shape (contango/backwardation) and where OI sits along it, at a glance."""
    data = build_spot_oi_data(commodity, mtime)
    syms = data["syms"]; oi_piv = data["oi_piv"]; px_piv = data["px_piv"]
    if not syms or snapshot_date not in oi_piv.index:
        return None
    oi_row = oi_piv.loc[snapshot_date]
    px_row = px_piv.loc[snapshot_date]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=syms, y=oi_row.reindex(syms).values, name="OI",
                         marker_color="rgba(120,120,120,.55)",
                         hovertemplate="%{x}<br>OI: %{y:,.0f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=syms, y=px_row.reindex(syms).values, name="Price", mode="lines+markers",
                             line=dict(color="#1a56db", width=2), marker=dict(size=7),
                             hovertemplate="%{x}<br>Price: %{y:.2f}<extra></extra>"), secondary_y=True)
    fig.update_layout(height=380, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(l=55, r=55, t=30, b=40),
                      title=dict(text=f"Term Structure — {pd.Timestamp(snapshot_date).strftime('%d %b %Y')}",
                                 font=dict(size=12), x=0.01))
    fig.update_yaxes(title_text="Open Interest", secondary_y=False, gridcolor="rgba(0,0,0,.07)")
    fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False)
    return fig


@st.cache_data(max_entries=50, show_spinner=False)
def build_curve_spread_chart(commodity: str, snapshot_date, mtime: float = 0.0):
    """Every adjacent-month spread across the whole active curve, for one
    date — where the curve is steepest/most inverted, at a glance (vs. the
    OI-vs-Spread chart's single user-picked pair)."""
    data = build_spot_oi_data(commodity, mtime)
    syms = data["syms"]; px_piv = data["px_piv"]
    if len(syms) < 2 or snapshot_date not in px_piv.index:
        return None
    px_row = px_piv.loc[snapshot_date]
    root_len = len(commodity)
    pairs, spreads = [], []
    for i in range(len(syms) - 1):
        s1, s2 = syms[i], syms[i + 1]
        p1, p2 = px_row.get(s1), px_row.get(s2)
        if pd.isna(p1) or pd.isna(p2):
            continue
        pairs.append(f"{s1}-{s2[root_len:]}")
        spreads.append(p1 - p2)
    if not pairs:
        return None
    colors = ["#16a34a" if s >= 0 else "#dc2626" for s in spreads]

    fig = go.Figure(go.Bar(x=pairs, y=spreads, marker_color=colors,
                           hovertemplate="%{x}<br>Spread: %{y:+.2f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#cccccc", line_width=1)
    fig.update_layout(height=340, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      showlegend=False, margin=dict(l=55, r=25, t=30, b=40),
                      yaxis=dict(title="Spread", gridcolor="rgba(0,0,0,.07)"),
                      title=dict(text=f"Curve Spreads — {pd.Timestamp(snapshot_date).strftime('%d %b %Y')}",
                                 font=dict(size=12), x=0.01))
    return fig


@st.cache_data(max_entries=50, show_spinner=False)
def build_oi_price_scatter(commodity: str, table_lookback: int, contract_choice: str, mtime: float = 0.0):
    """OI change vs price change, one point per day — do the days with the
    biggest OI flow line up with the biggest price moves, or not?"""
    data = build_spot_oi_data(commodity, mtime)
    oi_piv = data["oi_piv"]; px_piv = data["px_piv"]
    if oi_piv.empty:
        return None
    max_date = oi_piv.index.max()
    cutoff = max_date - pd.Timedelta(days=table_lookback)
    dates = sorted(d for d in oi_piv.index if d >= cutoff)
    if not dates:
        return None

    if contract_choice == "Spot (Most OI)":
        oi_chg_s = data["spot_oi_chg"].reindex(dates)
        px_chg_s = data["price_chg_pct"].reindex(dates)
        name = "Spot"
    elif contract_choice in oi_piv.columns:
        oi_chg_s = oi_piv[contract_choice].diff().reindex(dates)
        px_chg_s = (px_piv[contract_choice].pct_change() * 100).reindex(dates)
        name = contract_choice
    else:
        return None

    valid = oi_chg_s.notna() & px_chg_s.notna()
    if not valid.any():
        return None
    dates_v = [d for d, ok in zip(dates, valid) if ok]

    fig = go.Figure(go.Scatter(
        x=px_chg_s[valid].values, y=oi_chg_s[valid].values, mode="markers",
        marker=dict(size=7, color="#1a56db", opacity=0.65, line=dict(width=0.5, color="#fff")),
        text=[d.strftime("%d %b %Y") for d in dates_v],
        hovertemplate="%{text}<br>Price Chg: %{x:+.2f}%<br>OI Chg: %{y:+,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.add_vline(x=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(height=380, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      showlegend=False, margin=dict(l=55, r=25, t=30, b=40),
                      xaxis=dict(title="Price Change (%)", gridcolor="rgba(0,0,0,.07)"),
                      yaxis=dict(title=f"{name} OI Change (lots)", gridcolor="rgba(0,0,0,.07)"))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    st.markdown("---")

    commodity = st.selectbox("Commodity", list(COMMODITIES.keys()),
                             format_func=lambda x: COMMODITIES[x][1])

    mt              = _mtime(commodity)
    df_sidebar      = load_data(commodity, mt)
    avail_months    = sorted(df_sidebar["month"].unique())
    most_active     = _most_active_month(df_sidebar)
    default_idx     = avail_months.index(most_active) if most_active in avail_months else 0
    selected_month  = st.selectbox("Contract Month", avail_months, index=default_idx,
                                   format_func=lambda x: f"{MONTH_NAMES.get(x,x)} ({x})")

    df_month    = df_sidebar[df_sidebar["month"] == selected_month].copy()
    today       = pd.Timestamp(date.today())
    active_syms, hist_syms = _split_contracts(df_month)

    if not active_syms:
        st.error("No active contract found.")
        st.stop()

    current_contract = st.selectbox("Current Contract", active_syms)
    st.markdown("---")

    years_all  = sorted(df_month[df_month["ice_symbol"].isin(hist_syms)]["year"].unique())
    hist_range = st.slider("Historical Years",
                           int(years_all[0]), int(years_all[-1]),
                           (int(years_all[0]), int(years_all[-1]))) if years_all else (0,0)
    st.markdown("---")

    max_dte      = int(df_month["days_to_expiry"].max())
    max_dte_r    = (max_dte // 10) * 10
    dte_opts_rev = list(range(max_dte_r, -1, -10))
    default_upper = 300 if 300 in dte_opts_rev else dte_opts_rev[0]
    dte_sel      = st.select_slider("Days to Expiry Range (Raw)",
                                    options=dte_opts_rev,
                                    value=(default_upper, dte_opts_rev[-1]))
    dte_range    = [dte_sel[0], dte_sel[1]]   # [high DTE, low DTE] — chart is reversed

    default_norm_upper = 150 if 150 in dte_opts_rev else dte_opts_rev[0]
    norm_dte_sel = st.select_slider("Days to Expiry Range (Normalized)",
                                    options=dte_opts_rev,
                                    value=(default_norm_upper, dte_opts_rev[-1]))
    norm_dte_range = [norm_dte_sel[0], norm_dte_sel[1]]

    st.markdown("---")
    show_individual = st.toggle("Show individual years", value=False)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size:0.70rem !important; color:#888; }
[data-testid="stMetricValue"] { font-size:1.10rem !important; font-weight:600; }
[data-testid="stMetricDelta"] { font-size:0.70rem !important; }
</style>""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_oi, tab_spot, tab_spot_charts, tab_vol, tab_flow, tab_grid = st.tabs(
    ["OI Progression", "All Futures OI Recap", "All Futures OI Charts",
     "Volume", "OI & Volume Flow", "Comprehensive Grid"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OI PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_oi:
    res = compute_band(commodity, selected_month, hist_range, "open_interest", mtime=mt)
    if res is None:
        st.error("No data available.")
        st.stop()

    band, curr_df, _, _ = res
    curr_df = df_month[df_month["ice_symbol"] == current_contract].sort_values("Date").copy()

    latest     = curr_df.iloc[-1]
    dte_now    = int(latest["days_to_expiry"])
    latest_oi  = latest["open_interest"]
    lat_date   = latest["Date"].strftime("%b %d, %Y")

    closest    = (band["days_to_expiry"] - dte_now).abs().idxmin()
    avg_oi     = band.loc[closest, "hist_mean"]
    pct_vs_avg = (latest_oi - avg_oi) / avg_oi * 100 if avg_oi > 0 else 0.0

    month_name = MONTH_NAMES.get(selected_month, selected_month)

    kpi_row([
        ("Contract",        current_contract),
        ("Current OI",      f"{latest_oi:,.0f}"),
        ("As of",           lat_date),
        ("Days to Expiry",  str(dte_now)),
        ("vs Hist Mean",    f"{avg_oi:,.0f}", f"{pct_vs_avg:+.1f}%"),
    ])

    hist_df_ind = df_month[df_month["ice_symbol"].isin(hist_syms)].copy() if show_individual else None
    col_raw, col_norm = st.columns(2)
    with col_raw:
        fig_oi = build_chart(
            band, curr_df, "open_interest", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Raw OI",
            y_title="Open Interest (contracts)",
            y_fmt=",.0f", y_suffix="",
            outer_color=C["oi_outer"], inner_color=C["oi_inner"], avg_color=C["oi_avg"],
            dte_range=dte_range, dte_now=dte_now,
            show_individual=show_individual, hist_df=hist_df_ind, ind_metric="open_interest",
            height=520,
        )
        st.plotly_chart(fig_oi, use_container_width=True)

    with col_norm:
        res_n2 = _normalize_oi_at_dte_max(commodity, selected_month, hist_range, current_contract, mt)
        if res_n2 is not None:
            band_n2, curr_n2, dte_max, current_reached, hist_norm_n2 = res_n2
            st.caption(
                f"DTE_max = {dte_max} days to expiry — average, across the selected historical years, "
                f"of the day each year's OI peaked.",
                help="DTE_max is the average, across the selected historical years, of the day-to-expiry "
                     "each year's OI hit its own high; every year's OI curve is then divided by that same "
                     "year's own OI reading on that DTE_max day (not its own peak) and shown as a %.",
            )
            if not current_reached:
                st.info(f"{current_contract} is still at {dte_now} days to expiry and hasn't "
                         f"counted down to DTE_max={dte_max} yet, so it has no line here until it does.")
            fig_n2 = build_chart(
                band_n2, curr_n2, "open_interest", current_contract,
                title=f"<b>{commodity} {month_name}</b>  |  Normalized (% of OI at DTE_max={dte_max})",
                y_title="Open Interest (% of OI at DTE_max)",
                y_fmt=".1f", y_suffix="%",
                outer_color=C["oi_outer"], inner_color=C["oi_inner"], avg_color=C["oi_avg"],
                dte_range=norm_dte_range, dte_now=dte_now,
                show_individual=show_individual, hist_df=hist_norm_n2, ind_metric="open_interest",
                height=520,
            )
            st.plotly_chart(fig_n2, use_container_width=True)

    # ── 2x2 Active contracts ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {COMMODITIES[commodity][1]} — 4 Active Contracts")

    df_all     = load_data(commodity, mt)
    ltd_all    = df_all.groupby("ice_symbol")[["LTD","month"]].first().reset_index()
    active_all = ltd_all[ltd_all["LTD"] >= today].sort_values("LTD").head(4)
    quad_syms  = list(active_all["ice_symbol"])
    quad_months= list(active_all["month"])

    quad_results = []  # (sym, month, band, curr_df)
    for sym_q, m_q in zip(quad_syms, quad_months):
        res_q = compute_band(commodity, m_q, hist_range, "open_interest", mtime=mt)
        if res_q is None:
            quad_results.append((sym_q, m_q, None, None))
            continue
        b_q = res_q[0]
        c_q = df_all[df_all["ice_symbol"] == sym_q].sort_values("Date").copy()
        quad_results.append((sym_q, m_q, b_q, c_q))

    fig4 = make_subplots(rows=2, cols=2,
        subplot_titles=[f"{s}  ({MONTH_NAMES.get(m, m)})" for s, m, _, _ in quad_results],
        horizontal_spacing=0.08, vertical_spacing=0.14)

    for idx, (sym_q, m_q, b_q, c_q) in enumerate(quad_results):
        if b_q is None:
            continue
        r, cl = idx//2+1, idx%2+1
        add_oi_traces(fig4, b_q, c_q, sym_q, ",.0f", row=r, col=cl, show_legend=False)

    fig4.update_layout(height=720, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                       font=dict(color=C["font"], family="Inter, sans-serif"),
                       showlegend=False, margin=dict(l=50,r=30,t=60,b=50))
    for i in range(1,5):
        fig4.update_xaxes(range=[dte_range[0], dte_range[1]], showgrid=True, gridcolor=C["grid"],
                          tickfont=dict(size=10), zeroline=False,
                          row=(i-1)//2+1, col=(i-1)%2+1)
        fig4.update_yaxes(showgrid=True, gridcolor=C["grid"], tickformat=",",
                          tickfont=dict(size=10), zeroline=False,
                          row=(i-1)//2+1, col=(i-1)%2+1)
    st.plotly_chart(fig4, use_container_width=True)

    # ── OI Market Share ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {current_contract} — Share of Total {commodity} Market OI (%)")
    st.caption("Contract OI / sum of all active contracts OI on that date.")

    res_sh = compute_band(commodity, selected_month, hist_range, "oi_share_pct", use_enriched=True, mtime=mt)
    if res_sh:
        b_sh, _, _, _ = res_sh
        df_enr  = load_enriched(commodity, mt)
        c_sh    = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        lat_sh  = c_sh.iloc[-1]
        dte_sh  = int(lat_sh["days_to_expiry"])
        val_sh  = lat_sh["oi_share_pct"]
        idx_sh  = (b_sh["days_to_expiry"] - dte_sh).abs().idxmin()
        avg_sh  = b_sh.loc[idx_sh, "hist_mean"]

        kpi_row([
            ("Contract",      current_contract),
            ("Current Share", f"{val_sh:.1f}%"),
            ("As of",         lat_sh["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",  f"{avg_sh:.1f}%", f"{val_sh-avg_sh:+.1f}pp"),
        ])

        fig_sh = build_chart(b_sh, c_sh, "oi_share_pct", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  OI Market Share",
            y_title="Share of Total Market OI", y_fmt=".1f", y_suffix="%",
            outer_color=C["sh_outer"], inner_color=C["sh_inner"], avg_color=C["sh_avg"],
            dte_range=dte_range, dte_now=dte_sh, height=480)
        st.plotly_chart(fig_sh, use_container_width=True)

    with st.expander("Current Contract Data", expanded=False):
        tbl = curr_df[["Date","days_to_expiry","open_interest","volume","settlement"]].copy()
        tbl = tbl.sort_values("Date", ascending=False)
        tbl.columns = ["Date","DTE","Open Interest","Volume","Settlement"]
        tbl["Date"] = tbl["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOLUME
# ═══════════════════════════════════════════════════════════════════════════════
with tab_vol:
    st.markdown(f"### {current_contract}  |  Volume Analysis")
    st.markdown("---")

    # Rolling window selector
    roll_n = st.slider("Rolling Window (days)", min_value=1, max_value=30,
                       value=10, step=1,
                       help="Applied to daily volume before plotting progression")

    month_name = MONTH_NAMES.get(selected_month, selected_month)
    df_enr     = load_enriched(commodity, mt)
    df_enr_m   = df_enr[df_enr["month"] == selected_month].copy()

    # ── Chart 1: Volume Market Share ──────────────────────────────────────────
    st.markdown("#### Volume Market Share (%)")
    st.caption("Contract daily volume as % of total commodity volume on that date.")

    res_vs = compute_band(commodity, selected_month, hist_range, "vol_share_pct", use_enriched=True, mtime=mt)
    if res_vs:
        b_vs, _, _, _ = res_vs
        c_vs  = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        lat   = c_vs.iloc[-1]
        v_now = lat["vol_share_pct"]
        d_now = int(lat["days_to_expiry"])
        idx_  = (b_vs["days_to_expiry"] - d_now).abs().idxmin()
        avg_  = b_vs.loc[idx_, "hist_mean"]

        kpi_row([
            ("Contract",       current_contract),
            ("Vol Share",      f"{v_now:.1f}%"),
            ("As of",          lat["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",   f"{avg_:.1f}%", f"{v_now-avg_:+.1f}pp"),
        ])

        fig_vs = build_chart(b_vs, c_vs, "vol_share_pct", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Volume Market Share",
            y_title="Share of Total Volume", y_fmt=".1f", y_suffix="%",
            outer_color=C["vs_outer"], inner_color=C["vs_inner"], avg_color=C["vs_avg"],
            dte_range=dte_range, dte_now=d_now, height=460)
        st.plotly_chart(fig_vs, use_container_width=True)

    st.markdown("---")

    # ── Chart 3: Rolling N-day Volume ─────────────────────────────────────────
    st.markdown(f"#### Rolling {roll_n}-Day Average Volume")
    st.caption(f"{roll_n}-day rolling mean of daily volume, aligned by days to expiry.")

    res_rv = compute_band(commodity, selected_month, hist_range,
                          metric_col="volume", roll_n=roll_n, mtime=mt)
    if res_rv:
        b_rv, c_rv_base, _, _ = res_rv

        # Compute rolling vol for current contract
        c_rv = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        c_rv["_metric"] = c_rv["volume"].rolling(roll_n, min_periods=1).mean()

        lat   = c_rv.iloc[-1]
        v_now = lat["_metric"]
        d_now = int(lat["days_to_expiry"])
        idx_  = (b_rv["days_to_expiry"] - d_now).abs().idxmin()
        avg_  = b_rv.loc[idx_, "hist_mean"]

        kpi_row([
            ("Contract",          current_contract),
            (f"{roll_n}d Avg Vol", f"{v_now:,.0f}"),
            ("As of",             lat["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",      f"{avg_:,.0f}", f"{(v_now-avg_)/avg_*100:+.1f}%"),
        ])

        fig_rv = build_chart(b_rv, c_rv, "_metric", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Rolling {roll_n}-Day Volume",
            y_title=f"{roll_n}-Day Avg Daily Volume (contracts)",
            y_fmt=",.0f", y_suffix="",
            outer_color=C["rv_outer"], inner_color=C["rv_inner"], avg_color=C["rv_avg"],
            dte_range=dte_range, dte_now=d_now, height=460)
        st.plotly_chart(fig_rv, use_container_width=True)

    st.markdown("---")

    # ── Chart: All Active Contracts — Rolling Volume (overlaid lines) ────────
    st.markdown(f"#### All Contracts — Rolling {roll_n}-Day Volume")
    st.caption("Rolling volume for every contract that traded within the lookback window "
               "(includes contracts that have since expired, so historical totals stay accurate).")

    df_vol_all = load_data(commodity, mt)

    lookback = st.slider("Lookback (calendar days)", 30, 365, 120, step=10,
                         key="vol_all_lookback")
    cutoff = df_vol_all["Date"].max() - pd.Timedelta(days=lookback)

    # Any symbol that traded in the window — not just ones still unexpired today,
    # otherwise a contract that rolled off mid-window vanishes from past totals/mix.
    ltd_full      = df_vol_all.groupby("ice_symbol")["LTD"].first()
    syms_in_window= df_vol_all.loc[df_vol_all["Date"] >= cutoff, "ice_symbol"].unique()
    relevant_syms = ltd_full[ltd_full.index.isin(syms_in_window)].sort_values().index.tolist()

    pieces = []
    for sym in relevant_syms:
        g = df_vol_all[df_vol_all["ice_symbol"] == sym].sort_values("Date").copy()
        g["_rv"] = g["volume"].rolling(roll_n, min_periods=1).mean()
        pieces.append(g[["Date", "ice_symbol", "_rv"]])
    vol_all = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    if not vol_all.empty:
        vol_win = vol_all[vol_all["Date"] >= cutoff].copy()
        colors  = px.colors.qualitative.Bold

        # Drop symbols with no real trading in this window (keeps legend clean)
        traded_totals = vol_win.groupby("ice_symbol")["_rv"].sum()
        active_win = [s for s in relevant_syms if traded_totals.get(s, 0) > 0]

        # Close weekend + holiday gaps on the date axis
        all_bdays   = pd.bdate_range(vol_win["Date"].min(), vol_win["Date"].max())
        missing_days= all_bdays.difference(pd.DatetimeIndex(vol_win["Date"].unique()))
        rangebreaks = [dict(bounds=["sat", "mon"])]
        if len(missing_days):
            rangebreaks.append(dict(values=missing_days))

        fig_allvol = go.Figure()
        for i, sym in enumerate(active_win):
            g = vol_win[vol_win["ice_symbol"] == sym].sort_values("Date")
            if g.empty:
                continue
            fig_allvol.add_trace(go.Scatter(
                x=g["Date"], y=g["_rv"], mode="lines", name=sym,
                line=dict(width=2, color=colors[i % len(colors)]),
                hovertemplate=f"<b>{sym}</b><br>%{{x|%b %d, %Y}}<br>%{{y:,.0f}}<extra></extra>"))
        fig_allvol.update_layout(
            title=dict(text=f"<b>{commodity}</b>  |  Rolling {roll_n}-Day Volume — All Contracts",
                       font=dict(size=16, color=C["font"]), x=0.01),
            xaxis=dict(title="Date", showgrid=True, gridcolor=C["grid"],
                      tickfont=dict(size=11, color=C["font"]),
                      rangebreaks=rangebreaks),
            yaxis=dict(title=f"{roll_n}-Day Avg Volume (contracts)", showgrid=True,
                      gridcolor=C["grid"], tickfont=dict(size=11, color=C["font"])),
            plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            font=dict(color=C["font"], family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                       bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", height=480, margin=dict(l=70, r=30, t=60, b=90),
        )
        st.plotly_chart(fig_allvol, use_container_width=True)

        # ── Stacked views: proportion (%) and absolute total ──────────────────
        pivot = (vol_win.pivot_table(index="Date", columns="ice_symbol", values="_rv", aggfunc="mean")
                        .reindex(columns=active_win)
                        .fillna(0.0))
        totals = pivot.sum(axis=1)
        pct    = pivot.div(totals.replace(0, pd.NA), axis=0) * 100

        col_pct, col_abs = st.columns(2)

        with col_pct:
            fig_pct = go.Figure()
            for i, sym in enumerate(active_win):
                fig_pct.add_trace(go.Bar(
                    x=pct.index, y=pct[sym], name=sym,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"<b>{sym}</b><br>%{{x|%b %d, %Y}}<br>%{{y:.1f}}%<extra></extra>"))
            fig_pct.update_layout(
                barmode="stack",
                title=dict(text=f"<b>{commodity}</b>  |  Rolling Volume Mix (%)",
                           font=dict(size=16, color=C["font"]), x=0.01),
                xaxis=dict(title="Date", showgrid=True, gridcolor=C["grid"],
                          tickfont=dict(size=11, color=C["font"]),
                          rangebreaks=rangebreaks),
                yaxis=dict(title="Share of Rolling Volume", range=[0, 100], ticksuffix="%",
                          showgrid=True, gridcolor=C["grid"], tickfont=dict(size=11, color=C["font"])),
                plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                font=dict(color=C["font"], family="Inter, sans-serif"),
                bargap=0.02,
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                hovermode="x unified", height=480, margin=dict(l=60, r=20, t=60, b=100),
            )
            st.plotly_chart(fig_pct, use_container_width=True)

        with col_abs:
            fig_abs = go.Figure()
            for i, sym in enumerate(active_win):
                fig_abs.add_trace(go.Bar(
                    x=pivot.index, y=pivot[sym], name=sym,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f"<b>{sym}</b><br>%{{x|%b %d, %Y}}<br>%{{y:,.0f}}<extra></extra>"))
            fig_abs.update_layout(
                barmode="stack",
                title=dict(text=f"<b>{commodity}</b>  |  Total Rolling Volume (stacked)",
                           font=dict(size=16, color=C["font"]), x=0.01),
                xaxis=dict(title="Date", showgrid=True, gridcolor=C["grid"],
                          tickfont=dict(size=11, color=C["font"]),
                          rangebreaks=rangebreaks),
                yaxis=dict(title=f"{roll_n}-Day Avg Volume (contracts)", showgrid=True,
                          gridcolor=C["grid"], tickfont=dict(size=11, color=C["font"])),
                plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                font=dict(color=C["font"], family="Inter, sans-serif"),
                bargap=0.02,
                legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
                hovermode="x unified", height=480, margin=dict(l=70, r=20, t=60, b=100),
            )
            st.plotly_chart(fig_abs, use_container_width=True)

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — OI FLOW (daily OI change vs Volume, same contract as the other tabs)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_flow:
    month_name = MONTH_NAMES.get(selected_month, selected_month)
    st.markdown(f"### {current_contract}  |  Daily OI Change vs Volume")

    flow_lookback = st.slider("Lookback (calendar days)", 30, 365, 120, step=10,
                              key="flow_lookback")

    curr_flow = df_month[df_month["ice_symbol"] == current_contract].sort_values("Date").copy()
    curr_flow["oi_change"] = curr_flow["open_interest"].diff()
    curr_flow = curr_flow.dropna(subset=["oi_change"])

    cutoff_flow = curr_flow["Date"].max() - pd.Timedelta(days=flow_lookback) if not curr_flow.empty else None
    flow_win = curr_flow[curr_flow["Date"] >= cutoff_flow] if cutoff_flow is not None else curr_flow

    if flow_win.empty:
        st.info("Not enough history for this contract to show daily OI change.")
    else:
        latest_flow   = flow_win.iloc[-1]
        latest_change = latest_flow["oi_change"]
        latest_vol    = latest_flow["volume"]
        turnover      = (latest_vol / abs(latest_change)) if latest_change else float("nan")

        kpi_row([
            ("Contract",        current_contract),
            ("Latest OI Chg",   f"{latest_change:+,.0f}"),
            ("Latest Volume",   f"{latest_vol:,.0f}"),
            ("As of",           latest_flow["Date"].strftime("%b %d, %Y")),
        ])

        bar_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in flow_win["oi_change"]]

        all_bdays_f    = pd.bdate_range(flow_win["Date"].min(), flow_win["Date"].max())
        missing_days_f = all_bdays_f.difference(pd.DatetimeIndex(flow_win["Date"].unique()))
        rangebreaks_f  = [dict(bounds=["sat", "mon"])]
        if len(missing_days_f):
            rangebreaks_f.append(dict(values=missing_days_f))

        fig_flow = go.Figure()
        fig_flow.add_trace(go.Bar(
            x=flow_win["Date"], y=flow_win["volume"], name="Volume",
            marker_color="rgba(120,120,120,0.30)", yaxis="y2",
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>",
        ))
        fig_flow.add_trace(go.Bar(
            x=flow_win["Date"], y=flow_win["oi_change"], name="OI Change",
            marker_color=bar_colors,
            hovertemplate="<b>%{x|%b %d, %Y}</b><br>OI Change: %{y:+,.0f}<extra></extra>",
        ))
        fig_flow.update_layout(
            barmode="overlay",
            title=dict(text=f"<b>{commodity} {month_name}</b>  |  Daily OI Change vs Volume",
                       font=dict(size=16, color=C["font"]), x=0.01),
            xaxis=dict(title="Date", showgrid=True, gridcolor=C["grid"],
                      tickfont=dict(size=11, color=C["font"]), rangebreaks=rangebreaks_f),
            yaxis=dict(title="OI Change (contracts)", showgrid=True, gridcolor=C["grid"],
                      zeroline=True, zerolinecolor="rgba(0,0,0,0.25)", zerolinewidth=1,
                      tickfont=dict(size=11, color=C["font"])),
            yaxis2=dict(title="Volume (contracts)", overlaying="y", side="right",
                       showgrid=False, tickfont=dict(size=11, color=C["font"])),
            plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            font=dict(color=C["font"], family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0,
                       bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", height=500, margin=dict(l=70, r=60, t=60, b=90),
        )
        st.plotly_chart(fig_flow, use_container_width=True)

        # ── Scatter: Volume vs OI Change ─────────────────────────────────────
        st.markdown("#### Volume vs OI Change — Scatter")
        oi_chg_mode = st.radio(
            "OI Δ", ["Signed", "Absolute"], horizontal=True,
            key="flow_scatter_mode",
        )
        x_scatter = flow_win["oi_change"] if oi_chg_mode == "Signed" else flow_win["oi_change"].abs()
        y_scatter = flow_win["volume"]

        if len(flow_win) < 5:
            st.info("Not enough days in this window for a scatter.")
        else:
            xs, ys = x_scatter.values, y_scatter.values
            slope, intercept = np.polyfit(xs, ys, 1)
            r2 = np.corrcoef(xs, ys)[0, 1] ** 2
            x_line = np.array([xs.min(), xs.max()])

            if oi_chg_mode == "Signed":
                pt_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in xs]
            else:
                pt_colors = "#4A7FD4"

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color=pt_colors, size=8, opacity=0.7,
                           line=dict(color="white", width=0.8)),
                name="Daily obs", showlegend=False,
                customdata=flow_win["Date"].dt.strftime("%b %d, %Y"),
                hovertemplate="<b>%{customdata}</b><br>OI Δ: %{x:+,.0f}<br>Volume: %{y:,.0f}<extra></extra>",
            ))
            fig_sc.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept, mode="lines",
                line=dict(color="#1a1a2e", width=1.5, dash="dash"),
                name=f"Fit (R²={r2:.2f})",
            ))
            fig_sc.add_trace(go.Scatter(
                x=[xs[-1]], y=[ys[-1]], mode="markers",
                marker=dict(color="#f59e0b", size=13, symbol="star",
                           line=dict(color="white", width=1)),
                name=f"Latest ({flow_win['Date'].iloc[-1].strftime('%b %d, %Y')})",
            ))
            fig_sc.update_layout(
                height=440, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                font=dict(color=C["font"], family="Inter, sans-serif"),
                margin=dict(l=60, r=30, t=20, b=60),
                xaxis=dict(title=("OI Change" if oi_chg_mode == "Signed" else "|OI Change|") + " (contracts)",
                          showgrid=True, gridcolor=C["grid"], zeroline=(oi_chg_mode == "Signed"),
                          zerolinecolor="rgba(0,0,0,0.25)", tickfont=dict(size=11, color=C["font"])),
                yaxis=dict(title="Volume (contracts)", showgrid=True, gridcolor=C["grid"],
                          tickfont=dict(size=11, color=C["font"])),
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DAILY GRID (OI level / OI change / Volume / Px change, per contract month)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_grid:
    st.markdown(f"### {COMMODITIES[commodity][1]}  |  Daily OI & Volume by Contract Month")
    table_lookback = st.slider("Lookback (calendar days)", 30, 365, 90, step=10,
                               key="oi_table_lookback")
    html = build_oi_vol_table_html(commodity, table_lookback, mt)
    if html is None:
        st.info("No data in this window.")
    else:
        st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SPOT OI REPORT (spot vs non-spot OI, COT-Tuesday-aligned snapshots)
# ═══════════════════════════════════════════════════════════════════════════════
def _spot_controls_css(container_key: str):
    st.markdown(f"""<style>
      .st-key-{container_key} div[data-testid="stSelectbox"],
      .st-key-{container_key} div[data-testid="stNumberInput"] {{ margin-bottom:-16px; }}
      .st-key-{container_key} label p {{ font-size:.68rem !important; margin-bottom:0 !important; }}
      .st-key-{container_key} div[data-baseweb="select"] {{ min-height:30px; }}
    </style>""", unsafe_allow_html=True)


@st.fragment
def _render_all_futures_oi_recap(commodity: str, mt: float):
    """Fragment-scoped: without this, changing the lookback/price/spread
    widgets below re-runs the ENTIRE script — including the other tabs'
    own expensive band/quadrant/volatility computations, even though
    they're not visible. Wrapping this tab in @st.fragment confines a
    rerun triggered by one of its own widgets to just this tab."""
    st.caption("Price = settlement of the **spot** contract — whichever unexpired month "
              "currently has the highest OI (shifts as contracts roll).")

    spot_data = build_spot_oi_data(commodity, mt)
    syms_spot = spot_data["syms"]
    if not syms_spot:
        st.info("No contract-month data available.")
        return

    latest_spot_sym = spot_data["spot_sym"].iloc[-1] if len(spot_data["spot_sym"]) else syms_spot[0]
    leg1_idx = syms_spot.index(latest_spot_sym) if latest_spot_sym in syms_spot else 0
    leg2_idx = min(leg1_idx + 1, len(syms_spot) - 1)

    # ── Controls (lookback, price source, spread legs) ──────────────────────
    _spot_controls_css("recap_controls")
    with st.expander("Controls", expanded=False):
        with st.container(key="recap_controls"):
            c0, c1, c2, c3 = st.columns(4)
            with c0:
                spot_lookback = st.number_input("Lookback (calendar days)", min_value=30, max_value=730,
                                                value=90, step=10, key="recap_table_lookback")
            with c1:
                spot_label = f"Spot (Most OI: {latest_spot_sym})" if latest_spot_sym else "Spot (Most OI)"
                price_opts = [spot_label] + syms_spot
                price_source_sel = st.selectbox(
                    "Price", price_opts, index=0, key="recap_price_source",
                    help="Spot (Most OI) is the default: whichever unexpired month "
                         "currently has the highest OI. Pick a specific contract instead "
                         "to track that one month's own price throughout, without it "
                         "switching as OI leadership rolls to the next month.",
                )
                price_source = "Spot (Most OI)" if price_source_sel == spot_label else price_source_sel
            with c2:
                leg1 = st.selectbox("Spread Leg 1", syms_spot, index=leg1_idx, key="recap_spread_leg1")
            with c3:
                leg2 = st.selectbox("Spread Leg 2", syms_spot, index=leg2_idx, key="recap_spread_leg2")

    # ── Main daily grid — always visible, no expander ───────────────────────
    html_spot = build_spot_daily_table_html(commodity, spot_lookback, leg1, leg2, price_source, mt)
    if html_spot is None:
        st.info("No data in this window.")
    else:
        st.markdown(html_spot, unsafe_allow_html=True)

    # ── Change wrt COT date — always visible, no expander ───────────────────
    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:14px 0 4px'>"
               "Change wrt COT date</div>", unsafe_allow_html=True)
    st.markdown(build_spot_summary_html(spot_data), unsafe_allow_html=True)

    # ── Daily OI Change per Expiry — always visible, no expander ────────────
    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:14px 0 4px'>"
               "Daily OI Change per Expiry</div>", unsafe_allow_html=True)
    html_expchg = build_expiry_chg_table_html(commodity, spot_lookback, mt)
    if html_expchg is None:
        st.info("No data in this window.")
    else:
        st.markdown(html_expchg, unsafe_allow_html=True)


@st.fragment
def _render_all_futures_oi_charts(commodity: str, mt: float):
    """Fragment-scoped for the same reason as the Recap tab above."""
    spot_data = build_spot_oi_data(commodity, mt)
    syms_spot = spot_data["syms"]
    if not syms_spot:
        st.info("No contract-month data available.")
        return

    latest_spot_sym = spot_data["spot_sym"].iloc[-1] if len(spot_data["spot_sym"]) else syms_spot[0]
    leg1_idx = syms_spot.index(latest_spot_sym) if latest_spot_sym in syms_spot else 0
    leg2_idx = min(leg1_idx + 1, len(syms_spot) - 1)

    all_dates = list(spot_data["oi_piv"].index)

    _spot_controls_css("charts_controls")
    with st.expander("Controls", expanded=False):
        with st.container(key="charts_controls"):
            c0, c1, c2, c3, c4 = st.columns(5)
            with c0:
                chart_lookback = st.number_input("Lookback (calendar days)", min_value=30, max_value=730,
                                                 value=90, step=10, key="charts_table_lookback")
            with c1:
                spot_label = f"Spot (Most OI: {latest_spot_sym})" if latest_spot_sym else "Spot (Most OI)"
                oi_opts = [spot_label] + syms_spot
                oi_choice_sel = st.selectbox("OI Series", oi_opts, index=0, key="charts_oi_choice")
                oi_choice = "Spot (Most OI)" if oi_choice_sel == spot_label else oi_choice_sel
            with c2:
                leg1 = st.selectbox("Spread Leg 1", syms_spot, index=leg1_idx, key="charts_spread_leg1")
            with c3:
                leg2 = st.selectbox("Spread Leg 2", syms_spot, index=leg2_idx, key="charts_spread_leg2")
            with c4:
                snapshot_date = st.select_slider(
                    "Snapshot Date", options=all_dates, value=all_dates[-1] if all_dates else None,
                    format_func=lambda d: pd.Timestamp(d).strftime("%d %b %Y"), key="charts_snapshot_date",
                    help="Which date the Term Structure and Curve Spread charts below show.",
                )

    fig_oi_spread = build_oi_spread_chart(commodity, chart_lookback, oi_choice, leg1, leg2, mt)
    if fig_oi_spread is not None:
        st.plotly_chart(fig_oi_spread, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        fig_term = build_term_structure_chart(commodity, snapshot_date, mt)
        if fig_term is not None:
            st.plotly_chart(fig_term, use_container_width=True)
    with cc2:
        fig_curve_spread = build_curve_spread_chart(commodity, snapshot_date, mt)
        if fig_curve_spread is not None:
            st.plotly_chart(fig_curve_spread, use_container_width=True)

    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:10px 0 4px'>"
               "OI Change vs Price Change</div>", unsafe_allow_html=True)
    fig_scatter = build_oi_price_scatter(commodity, chart_lookback, oi_choice, mt)
    if fig_scatter is not None:
        st.plotly_chart(fig_scatter, use_container_width=True)


with tab_spot:
    _render_all_futures_oi_recap(commodity, mt)

with tab_spot_charts:
    _render_all_futures_oi_charts(commodity, mt)
