# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import re
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import date

st.set_page_config(page_title="Futures Dashboard", page_icon="📈", layout="wide")

from common import (DB_PATH, COMMODITIES, MONTH_NAMES, MONTH_ORDER, C,
                    _mtime, _total_oi_mtime, load_data, load_enriched, load_total_oi,
                    _safe, _oi_heatmap_style, _bar_style,
                    _diverging_bar_style, _oi_chg_style, _vol_style,
                    render_data_freshness)


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


def _hist_band(dense_df: pd.DataFrame, metric_col: str, x: str = "days_to_expiry") -> pd.DataFrame:
    """min/max/mean/q25/q75 of metric_col per `x` (days_to_expiry unless the
    caller aligns on something else, e.g. day of year), then a 7-point
    rolling smooth. Uses groupby(...).quantile() (pandas' own vectorized
    path) rather than .agg(hist_q25=lambda x: x.quantile(...)) — the lambda
    form calls quantile once per group via slow Python dispatch instead of
    pandas' optimized C-level implementation; on a wide days-to-expiry range
    this was the single biggest cost in a cold commodity switch (~5s of a
    ~10s total in profiling, from ~23k quantile calls)."""
    g = dense_df.groupby(x)[metric_col]
    band = g.agg(hist_min="min", hist_max="max", hist_mean="mean").reset_index()
    q25 = g.quantile(0.25).rename("hist_q25").reset_index()
    q75 = g.quantile(0.75).rename("hist_q75").reset_index()
    band = band.merge(q25, on=x).merge(q75, on=x)
    band = band.sort_values(x)
    for c in ["hist_mean", "hist_q25", "hist_q75"]:
        band[c] = band[c].rolling(7, center=True, min_periods=1).mean()
    # min/max stay the true extremes (unsmoothed), so at a local spike the
    # smoothed quartiles could sit outside them and the inner fill would
    # render outside the outer one. Widen the envelope rather than smoothing
    # min/max, which would stop them being extremes at all.
    band["hist_max"] = band[["hist_max", "hist_q75"]].max(axis=1)
    band["hist_min"] = band[["hist_min", "hist_q25"]].min(axis=1)
    return band


_SEAS_REF_YEAR = 2000   # a leap year, so Feb 29 has a slot on the shared axis


@st.cache_data(max_entries=50, show_spinner=False)
def build_total_oi_seasonal(commodity, hist_year_range, mtime=0.0, oi_mtime=0.0):
    """Whole-market OI by calendar day of year, one series per year, plus a
    historical band.

    The series is LSEG's own market total (TOTCNTROI, stored by the builder in
    total_oi.parquet). Only if that has not been stored does it fall back to
    summing every contract in the per-contract table — the fallback path
    below, with its own start-of-board and half-published-session handling.

    Aligned on month/day mapped into a leap reference year rather than raw
    dayofyear: raw dayofyear shifts every date after Feb 28 by one in a leap
    year, so Mar 1 would sit on day 60 in 2023 but day 61 in 2024. Each year is
    then interpolated across weekends/holidays onto a continuous daily grid,
    the same densify-then-band treatment the DTE charts use.

    Trailing dates where noticeably fewer contracts printed than usual are
    dropped: OI lands contract by contract, and summing a half-published
    session reads as a sudden board-wide liquidation that never happened.
    """
    stored = load_total_oi(commodity, oi_mtime)
    if stored is not None and len(stored) > 30:
        # LSEG's early history is not always a real total (Robusta opens 2008 at
        # a value of 1); start from the first date it reaches a plausible level.
        typical = float(stored.iloc[-250:].median())
        ok = stored[stored >= 0.2 * typical]
        daily = pd.DataFrame({"total_oi": ok})
        source, board_from, trimmed = "LSEG", ok.index.min(), 0
    else:
        daily, source, board_from, trimmed = None, "sum", None, 0
    if daily is None:
        df = load_data(commodity, mtime)
    # The database carries contracts from a given expiry onward, not the whole
    # board as it stood on its first date. Until the earliest stored contract's
    # last trading day, every date is missing the months that had already
    # expired before it, so the "total" is a fraction of the real board — KC
    # opens on 2008-04-17 with a single contract (18 lots), and averages ~4x
    # below its post-2011 level until 2011-03-21. Summing those dates would drag
    # the seasonal band's early-year lower edge down and start the history
    # chart near zero. From that LTD on, every missing month has expired.
    if daily is None:
        board_from = df.groupby("ice_symbol")["LTD"].first().min()
        df = df[df["Date"] >= board_from]
        daily = (df.groupby("Date")
                   .agg(total_oi=("open_interest", "sum"), n=("ice_symbol", "nunique"))
                   .sort_index())
        n = daily["n"].to_numpy()
        keep = len(daily)
        while keep > 11 and n[keep - 1] < 0.8 * np.median(n[keep - 11:keep - 1]):
            keep -= 1
        trimmed = len(daily) - keep
        daily = daily.iloc[:keep]
    if daily.empty:
        return None

    idx = daily.index
    ref = pd.to_datetime(pd.DataFrame({"year": _SEAS_REF_YEAR, "month": idx.month, "day": idx.day}))
    daily = daily.assign(year=idx.year, doy=ref.dt.dayofyear.to_numpy())

    pieces = []
    for yr, g in daily.groupby("year"):
        s = g.set_index("doy")["total_oi"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if len(s) < 2:
            continue
        # A year that spans the calendar is padded to Jan 1 / Dec 31: the first
        # print is usually Jan 2-3 and the last Dec 29-31, and without padding
        # the band would be one year short on those few days and step there.
        lo = 1 if s.index.min() <= 15 else int(s.index.min())
        hi = 366 if s.index.max() >= 350 and yr != daily["year"].max() else int(s.index.max())
        s = s.reindex(pd.RangeIndex(lo, hi + 1)).interpolate(limit_direction="both")
        pieces.append(pd.DataFrame({"year": yr, "doy": s.index, "total_oi": s.to_numpy()}))
    if not pieces:
        return None
    dense = pd.concat(pieces, ignore_index=True)

    last_date = daily.index.max()
    cur_year = int(last_date.year)
    # A year only enters the band if it spans (almost) the whole calendar. A
    # part-year such as the first one after board_from (RC opens 25 Jan, KC
    # 21 Mar) would be absent from the early days and present later, so the
    # min/max/quartile edges would step at the day it starts instead of
    # reflecting the market.
    cover = dense.groupby("year")["doy"].agg(["min", "max"])
    full_years = {int(y) for y, r in cover.iterrows() if r["min"] <= 15 and r["max"] >= 350}
    band_years = [int(y) for y in sorted(dense["year"].unique())
                  if hist_year_range[0] <= y <= hist_year_range[1]
                  and y != cur_year and y in full_years]
    band = (_hist_band(dense[dense["year"].isin(band_years)], "total_oi", x="doy")
            if band_years else None)
    return dict(dense=dense, band=band, band_years=band_years, cur_year=cur_year,
                series=daily["total_oi"], board_from=board_from, source=source,
                last_date=last_date, last_oi=float(daily["total_oi"].iloc[-1]),
                last_doy=int(daily["doy"].iloc[-1]), trimmed=trimmed)


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
    Generic historical-band computation for any metric.
    roll_n: if set, compute rolling(roll_n).mean() on 'volume' first (by Date order).
    Returns (band, active_syms, hist_syms) or None.

    Deliberately does NOT return a current-contract frame: it used to hand
    back active_syms[0] (nearest expiry), but every call site immediately
    discarded it and rebuilt from the sidebar's selected contract, so the
    returned frame was both unused and wrong whenever the two differed.
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

    return band, active_syms, hist_syms


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
                   zeroline=False, tickformat=y_fmt,
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
    """vals = list of (label, value, delta) — delta optional. One compact,
    single-line strip of chips — st.metric's default size/padding was the
    biggest thing on the page for what's essentially a caption."""
    chips = []
    for item in vals:
        label, value = item[0], item[1]
        delta = item[2] if len(item) > 2 else None
        delta_html = ""
        if delta:
            dc = "#dc2626" if str(delta).strip().startswith("-") else "#16a34a"
            delta_html = f"<b style='color:{dc};font-weight:700;margin-left:4px'>{delta}</b>"
        chips.append(
            f"<span class='kpichip'><span class='kpil'>{label}</span> "
            f"<span class='kpiv'>{value}</span>{delta_html}</span>"
        )
    st.markdown(
        "<style>.kpirow{display:flex;flex-wrap:wrap;gap:0;border:1px solid #e5e7eb;"
        "border-radius:6px;background:#fafbfc;margin:4px 0 10px;overflow:hidden;width:fit-content}"
        ".kpichip{padding:4px 12px;border-right:1px solid #e5e7eb;white-space:nowrap;font-size:.72rem}"
        ".kpichip:last-child{border-right:none}"
        ".kpil{color:#9ca3af;font-size:.62rem;text-transform:uppercase;letter-spacing:.03em;margin-right:4px}"
        ".kpiv{font-weight:700;color:#1a1a1a}</style>"
        f"<div class='kpirow'>{''.join(chips)}</div>",
        unsafe_allow_html=True,
    )


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
@st.cache_data(max_entries=50, show_spinner=False)
def build_oi_vol_table_html(commodity: str, table_lookback: int, mtime: float = 0.0):
    """Returns the full HTML string for the OI/Volume-by-contract-month table,
    or None if there's no data in the window."""
    df_all_tbl = load_data(commodity, mtime).sort_values(["ice_symbol", "Date"])
    df_all_tbl["oi_change"] = df_all_tbl.groupby("ice_symbol")["open_interest"].diff()
    # fill_method=None: the default ('pad') carries the last settlement across a
    # gap and then prints a fake 0.00% move against it. A missing price should
    # leave the cell blank, not invent an unchanged day.
    df_all_tbl["px_change"] = df_all_tbl.groupby("ice_symbol")["settlement"].pct_change(fill_method=None) * 100

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
_CCOL_W = 54
_DATECOL_W = 76


_CONTRACT_SUFFIX = re.compile(r"^(.*)([FGHJKMNQUVXZ]\d+(?:\^\d+)?)$")


def _leg_suffix(sym: str) -> str:
    """Month+year tail of a contract symbol, e.g. 'LRCX6' -> 'X6'.

    Spread labels used to slice with len(commodity), which assumes the symbol
    root equals the commodity key. It does not for Robusta: the key is 'RC'
    but every symbol is rooted 'LRC', so leg2[2:] turned LRCU6-LRCX6 into
    "LRCU6-CX6". Matching the month code from the right works for all seven
    roots (KC/CC/CT/SB/LRC/LCC/LSU) and keeps the ^1/^2 decade suffix.
    """
    m = _CONTRACT_SUFFIX.match(sym)
    return m.group(2) if m else sym


def _sign_color(v) -> str:
    if pd.isna(v) or v == 0:
        return "#1d1d1f"
    return "#16a34a" if v > 0 else "#dc2626"


def _spot_series(oi_piv: pd.DataFrame, px_piv: pd.DataFrame):
    """For every date, find whichever contract has the highest OI (the
    'spot'/most-active month) and pull its OI and settlement price — the
    active month shifts over time as contracts roll (e.g. Z6 now, H7 once
    Z6 expires), unlike 'nearest expiry' which would stay on a thin,
    already-rolled-out-of front month.

    Levels are the stitched series, but every CHANGE is taken within the
    day's own spot contract, against that same contract's previous session:
    diffing the stitched level across a roll would book the calendar spread
    as a price/OI move. On KC's 2026-07-28 roll (KCU6 -> KCZ6) the stitched
    pct_change read -1.42% on a day KCZ6 actually settled +4.24%.

    Returns (sym, oi, price, oi_chg, px_chg_pct, oi_5d) — the last three
    all contract-consistent; oi_5d is the spot contract's own trailing
    5-session mean, not a mean of the stitched (roll-discontinuous) level.
    """
    prev_oi = oi_piv.shift(1)
    prev_px = px_piv.shift(1)
    roll5   = oi_piv.rolling(5).mean()

    syms_out, oi_out, px_out = [], [], []
    oichg_out, pxchg_out, oi5_out = [], [], []
    for d in oi_piv.index:
        row = oi_piv.loc[d]
        if not row.notna().any():
            syms_out.append(None)
            for lst in (oi_out, px_out, oichg_out, pxchg_out, oi5_out):
                lst.append(np.nan)
            continue
        sym = row.idxmax()
        has_px = sym in px_piv.columns
        p1, p0 = (px_piv.at[d, sym], prev_px.at[d, sym]) if has_px else (np.nan, np.nan)
        syms_out.append(sym)
        oi_out.append(row[sym])
        px_out.append(p1)
        oichg_out.append(row[sym] - prev_oi.at[d, sym])
        pxchg_out.append((p1 - p0) / p0 * 100 if pd.notna(p0) and p0 else np.nan)
        oi5_out.append(roll5.at[d, sym])

    idx = oi_piv.index
    S = lambda v: pd.Series(v, index=idx, dtype=float)
    return (pd.Series(syms_out, index=idx), S(oi_out), S(px_out),
            S(oichg_out), S(pxchg_out), S(oi5_out))


@st.cache_data(max_entries=50, show_spinner=False)
def build_spot_oi_data(commodity: str, mtime: float = 0.0) -> dict:
    """Full-history spot/non-spot OI series for a commodity. Cached once
    per commodity; the report tab below slices this to its own lookback
    window for display, and diffs are computed here on the full series
    first so a display-window edge never truncates a change calculation."""
    df_all = load_data(commodity, mtime).sort_values(["ice_symbol", "Date"])
    today = pd.Timestamp(date.today())
    ltd_map = df_all.groupby("ice_symbol")["LTD"].first().sort_values()
    # The pivots span EVERY contract ever traded, not just the ones still
    # unexpired today. Restricting the column set up front made "Total" mean
    # "OI of the months that happen to still be alive today" at every past
    # date: on KC at 2026-06-11 that understated true board OI by 15.2%,
    # because KCN6 had since expired and was dropped from the whole history.
    # It also let a past date's real spot month vanish, silently promoting
    # the 2nd or 3rd month to "spot". Columns actually SHOWN are narrowed
    # per view (_syms_in_window / _syms_on) — the aggregates stay whole.
    all_syms  = ltd_map.index.tolist()                       # LTD-ascending
    live_syms = ltd_map[ltd_map >= today].index.tolist()     # tradeable today

    oi_piv = df_all.pivot_table(index="Date", columns="ice_symbol", values="open_interest", aggfunc="last").reindex(columns=all_syms)
    px_piv = df_all.pivot_table(index="Date", columns="ice_symbol", values="settlement", aggfunc="last").reindex(columns=all_syms)

    total_oi = oi_piv.sum(axis=1, min_count=1)
    spot_sym, spot_oi, spot_price, spot_oi_chg, price_chg_pct, spot_oi_5d = _spot_series(oi_piv, px_piv)
    non_spot_oi = total_oi - spot_oi

    return dict(
        syms=live_syms, all_syms=all_syms, ltd_map=ltd_map,
        oi_piv=oi_piv, px_piv=px_piv, total_oi=total_oi,
        spot_sym=spot_sym, spot_oi=spot_oi, spot_price=spot_price, non_spot_oi=non_spot_oi,
        oi_chg=total_oi.diff(), spot_oi_chg=spot_oi_chg, spot_oi_5d=spot_oi_5d,
        price_chg_pct=price_chg_pct,
        per_contract_chg=oi_piv.diff(),
    )


def _syms_in_window(data: dict, dates) -> list:
    """Contract months carrying OI anywhere in the displayed window, expiry-
    ascending — the same rule the Comprehensive Grid uses, so the two tabs
    agree. Includes months that expired mid-window, which is exactly what
    keeps the earlier rows' Total honest."""
    live = data["oi_piv"].loc[list(dates)].notna().any(axis=0)
    return [s for s in data["all_syms"] if live.get(s, False)]


def _syms_on(data: dict, snapshot_date) -> list:
    """Contract months with OI on one specific date, expiry-ascending — used
    by the snapshot (term-structure / matrix) views so a historical snapshot
    shows the curve as it actually stood then, not today's listed months."""
    row = data["oi_piv"].loc[snapshot_date].notna()
    return [s for s in data["all_syms"] if row.get(s, False)]


def build_spot_summary_html(data: dict) -> str:
    """LAST (today's raw OI) + the day-over-day and since-last-COT deltas,
    then the last two COT Tuesdays (the session itself, not the Friday
    report) with their own week-over-week deltas. This table only tracks
    OI on Tuesday sessions — it doesn't read the published COT report — so
    the most recent Tuesday with OI data is used as soon as it's in, with
    no wait for Friday's release."""
    oi_piv = data["oi_piv"]; total_oi = data["total_oi"]; spot_price = data["spot_price"]
    dates = list(oi_piv.index[oi_piv.notna().any(axis=1)])
    if not dates:
        return "<p>No data.</p>"
    max_date = dates[-1]
    prev_day = dates[-2] if len(dates) >= 2 else None
    # Columns: months carrying OI over the span this table actually quotes.
    syms = _syms_in_window(data, dates[-30:])
    tuesdays = [d for d in dates if pd.Timestamp(d).weekday() == 1]
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
      table.spotsum{border-collapse:collapse;width:auto;font-size:.66rem;font-family:'Inter',sans-serif;white-space:nowrap}
      table.spotsum th,table.spotsum td{padding:1px 8px;text-align:center;border-bottom:1px solid #f4f4f5}
      table.spotsum th{position:sticky;top:0;background:#0a2463;color:#fff;font-weight:600;
        font-size:.6rem;text-transform:uppercase;letter-spacing:.02em;border-bottom:2px solid #0a2463}
      table.spotsum td.lbl{text-align:left;font-weight:600;color:#1d1d1f;padding-left:10px}
      table.spotsum tr.delta td.lbl{font-weight:400;color:#9ca3af;font-size:.62rem}
      table.spotsum tr.spacer td{padding:2px 0;border:none}
      table.spotsum td.tot{font-weight:700;background:#fafafa}
      table.spotsum tr.tue-row{background:#eceef1}
      table.spotsum tbody tr:hover td{background-color:rgba(10,36,99,.04)}
    </style>"""

    # Shared scale across every delta row/column so the bars stay comparable
    # to each other — a per-row max would make a quiet day's bar look as
    # "full" as the heaviest week's.
    all_deltas = []
    for d1, d0 in [(max_date, prev_day), (max_date, last_cot), (last_cot, prev_cot), (prev_cot, prev_cot2)]:
        if d1 is not None and d0 is not None:
            all_deltas.append((oi_row(d1) - oi_row(d0)).abs())
    delta_vmax = _safe(pd.concat(all_deltas).max()) if all_deltas else 1.0

    def value_row(label, d):
        if d is None:
            return ""
        tr_cls = " class='tue-row'" if pd.Timestamp(d).weekday() == 1 else ""
        cells = "".join(f"<td class='ccol'>{_fmt_num(oi_row(d).get(s))}</td>" for s in syms)
        px_txt = f"{px(d):.2f}" if pd.notna(px(d)) else ""
        return (f"<tr{tr_cls}><td class='lbl'>{label}</td>{cells}"
                f"<td class='tot'>{_fmt_num(total_oi.get(d))}</td><td>{px_txt}</td></tr>")

    def delta_row(label, d1, d0):
        if d1 is None or d0 is None:
            return ""
        delta = oi_row(d1) - oi_row(d0)
        cells = "".join(
            f"<td class='ccol' style='{_oi_chg_style(delta.get(s), delta_vmax)};color:{_sign_color(delta.get(s))};font-weight:600'>"
            f"{_fmt_num(delta.get(s), True)}</td>" for s in syms
        )
        tot_delta = total_oi.get(d1) - total_oi.get(d0)
        px_pct = px_chg_pct(d1, d0)
        return (
            f"<tr class='delta'><td class='lbl'>{label}</td>{cells}"
            f"<td class='tot' style='{_oi_chg_style(tot_delta, delta_vmax)};color:{_sign_color(tot_delta)}'>{_fmt_num(tot_delta, True)}</td>"
            f"<td style='{_flat_tint(px_pct)};color:{_sign_color(px_pct)}'>{_fmt_pct(px_pct)}</td></tr>"
        )

    spacer = f"<tr class='spacer'><td colspan='{len(syms) + 3}'></td></tr>"
    header = "<tr><th class='lbl'>Date</th>" + "".join(f"<th class='ccol'>{s}</th>" for s in syms) + "<th>Total</th><th>Price</th></tr>"
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
    calendar-spread, and the trailing 5-session mean of the daily OI Chg (Total).

    Contract columns cover every month with OI in the window, including any
    that expired mid-window — so Total is the real board figure on every
    row and matches the Comprehensive Grid tab."""
    data = build_spot_oi_data(commodity, mtime)
    oi_piv = data["oi_piv"]; px_piv = data["px_piv"]
    total_oi = data["total_oi"]
    oi_chg = data["oi_chg"]; spot_oi_chg = data["spot_oi_chg"]
    oi_chg_5d = oi_chg.rolling(5).mean()

    if price_source != "Spot (Most OI)" and price_source in px_piv.columns:
        # A fixed, single contract's price throughout — unlike Spot, this
        # doesn't switch contracts as OI leadership rolls from one month
        # to the next, useful for tracking one specific expiry's own price.
        spot_price = px_piv[price_source]
        price_chg_pct = spot_price.pct_change(fill_method=None) * 100
    else:
        spot_price = data["spot_price"]; price_chg_pct = data["price_chg_pct"]

    if oi_piv.empty:
        return None
    max_date = oi_piv.index.max()
    cutoff = max_date - pd.Timedelta(days=table_lookback)
    has_oi = oi_piv.notna().any(axis=1)
    dates = [d for d in oi_piv.index if d >= cutoff and has_oi.at[d]]
    if not dates:
        return None
    dates_desc = sorted(dates, reverse=True)
    # Only currently-live (unexpired) contract months as columns — a month
    # that has since expired mid-window still counts in Total/OI Chg (those
    # come from the full oi_piv regardless of which columns are shown), it
    # just no longer gets its own column here.
    syms = [s for s in _syms_in_window(data, dates) if s in data["syms"]]

    have_spread = leg1 != leg2 and leg1 in px_piv.columns and leg2 in px_piv.columns
    spread = (px_piv[leg1] - px_piv[leg2]) if have_spread else pd.Series(dtype=float)
    spread_label = f"{leg1}-{_leg_suffix(leg2)}" if have_spread else "Spread"

    oi_chg_vmax = _safe(oi_chg.loc[dates].abs().max())

    css = f"""<style>
      .spotgrid-wrap{{overflow:auto;max-height:600px;border:1px solid #e5e7eb;border-radius:6px}}
      table.spotgrid{{border-collapse:collapse;width:auto;font-size:.6rem;font-family:'Inter',sans-serif;white-space:nowrap}}
      table.spotgrid th,table.spotgrid td{{padding:1px 4px;text-align:center;border-bottom:1px solid #f4f4f5}}
      table.spotgrid th{{position:sticky;top:0;background:#0a2463;color:#fff;font-weight:600;z-index:2;
        font-size:.54rem;text-transform:uppercase;letter-spacing:.02em;border-bottom:2px solid #0a2463}}
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
              f"<th>{spread_label}</th><th>OI Chg 5d Avg</th></tr>")

    rows = []
    for d in dates_desc:
        d_str = pd.Timestamp(d).strftime("%d/%m/%Y")
        tr_cls = " class='tue-row'" if pd.Timestamp(d).weekday() == 1 else ""
        cells = f"<td class='date-cell'>{d_str}</td>"
        for s in syms:
            v = oi_piv.at[d, s] if s in oi_piv.columns else np.nan
            cells += f"<td class='ccol'>{_fmt_num(v)}</td>"
        px_v = spot_price.get(d)
        px_pct_v, oi_chg_v, spot_chg_v, spread_v, oichg5d_v = (
            price_chg_pct.get(d), oi_chg.get(d), spot_oi_chg.get(d),
            spread.get(d) if have_spread else np.nan, oi_chg_5d.get(d),
        )
        # Non Spot Chg = Total OI change minus the spot month's OWN OI change.
        # Since spot_oi_chg now holds the day's spot contract fixed across
        # both sessions, this is exactly d(Total - that contract) — "the flow
        # in everything except the front month". Diffing a stitched Non-Spot
        # level instead would post a spurious jump on every roll day.
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
        cells += f"<td style='{_oi_chg_style(oichg5d_v, oi_chg_vmax)};color:{_sign_color(oichg5d_v)}'>{_fmt_num(oichg5d_v, True)}</td>"
        rows.append(f"<tr{tr_cls}>{cells}</tr>")

    return f"{css}<div class='spotgrid-wrap'><table class='spotgrid'>{header}<tbody>{''.join(rows)}</tbody></table></div>"


@st.cache_data(max_entries=50, show_spinner=False)
def build_expiry_chg_table_html(commodity: str, table_lookback: int, mtime: float = 0.0):
    """Day-over-day OI change for every individual contract month (not the
    aggregate) — each column scaled to its own range, since a front month's
    change dwarfs a far month's."""
    data = build_spot_oi_data(commodity, mtime)
    per_chg = data["per_contract_chg"]

    if per_chg.empty:
        return None
    max_date = per_chg.index.max()
    cutoff = max_date - pd.Timedelta(days=table_lookback)
    has_oi = data["oi_piv"].notna().any(axis=1)
    dates = [d for d in per_chg.index if d >= cutoff and has_oi.at[d]]
    if not dates:
        return None
    dates_desc = sorted(dates, reverse=True)
    syms = _syms_in_window(data, dates)
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

    have_spread = leg1 != leg2 and leg1 in px_piv.columns and leg2 in px_piv.columns
    spread_name = f"{leg1}-{_leg_suffix(leg2)} Spread" if have_spread else "Spread"
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
def build_term_structure_chart(commodity: str, snapshot_date, older_date=None, mtime: float = 0.0):
    """Term structure snapshot for one date: OI per contract as grey bars
    (left axis), price per contract as a line (right axis) — the curve
    shape (contango/backwardation) and where OI sits along it, at a glance.
    An optional second, older date overlays as a lighter bar + dashed line,
    so the curve's shape today can be compared against how it looked then."""
    data = build_spot_oi_data(commodity, mtime)
    oi_piv = data["oi_piv"]; px_piv = data["px_piv"]
    if snapshot_date not in oi_piv.index:
        return None
    # Months listed on the snapshot date itself — a date from before a roll
    # should draw the curve that existed then, front month included, rather
    # than only the months still unexpired today.
    syms = _syms_on(data, snapshot_date)
    if not syms:
        return None
    oi_row = oi_piv.loc[snapshot_date]
    px_row = px_piv.loc[snapshot_date]
    d_label = pd.Timestamp(snapshot_date).strftime("%d %b")

    have_older = older_date is not None and older_date in oi_piv.index and older_date != snapshot_date

    # One hue per DATE rather than per metric: the new date is blue (bar AND
    # price line), the older date amber, so a bar and the price line belonging
    # to the same snapshot are matched at a glance without reading the legend.
    NEW_LINE, NEW_BAR = "#1e3a8a", "rgba(30,58,138,.42)"
    OLD_LINE, OLD_BAR = "#f59e0b", "rgba(245,158,11,.38)"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=syms, y=oi_row.reindex(syms).values, name=f"OI ({d_label})",
                         marker_color=NEW_BAR, marker_line=dict(color=NEW_LINE, width=1),
                         hovertemplate="%{x}<br>OI: %{y:,.0f}<extra></extra>"), secondary_y=False)
    if have_older:
        oi_row_old = oi_piv.loc[older_date]
        old_label = pd.Timestamp(older_date).strftime("%d %b")
        fig.add_trace(go.Bar(x=syms, y=oi_row_old.reindex(syms).values, name=f"OI ({old_label})",
                             marker_color=OLD_BAR, marker_line=dict(color=OLD_LINE, width=1),
                             hovertemplate="%{x}<br>OI: %{y:,.0f}<extra></extra>"), secondary_y=False)
        fig.update_layout(barmode="group")

    fig.add_trace(go.Scatter(x=syms, y=px_row.reindex(syms).values, name=f"Price ({d_label})", mode="lines+markers",
                             line=dict(color=NEW_LINE, width=2), marker=dict(size=7),
                             hovertemplate="%{x}<br>Price: %{y:.2f}<extra></extra>"), secondary_y=True)
    if have_older:
        px_row_old = px_piv.loc[older_date]
        fig.add_trace(go.Scatter(x=syms, y=px_row_old.reindex(syms).values, name=f"Price ({old_label})",
                                 mode="lines+markers", line=dict(color=OLD_LINE, width=2, dash="dash"),
                                 marker=dict(size=6),
                                 hovertemplate="%{x}<br>Price: %{y:.2f}<extra></extra>"), secondary_y=True)

    fig.update_layout(height=380, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      legend=dict(orientation="h", y=1.1, x=0),
                      margin=dict(l=55, r=55, t=30, b=40))
    fig.update_yaxes(title_text="Open Interest", secondary_y=False, gridcolor="rgba(0,0,0,.07)")
    fig.update_yaxes(title_text="Price", secondary_y=True, showgrid=False)
    return fig


@st.cache_data(max_entries=50, show_spinner=False)
def build_curve_spread_chart(commodity: str, snapshot_date, mtime: float = 0.0):
    """Every adjacent-month spread across the whole active curve, for one
    date, plus each pair's min OI (liquidity) as light-grey bars — where the
    curve is steepest/most inverted AND how liquid that leg is, at a glance
    (vs. the OI-vs-Spread chart's single user-picked pair)."""
    data = build_spot_oi_data(commodity, mtime)
    px_piv = data["px_piv"]; oi_piv = data["oi_piv"]
    if snapshot_date not in px_piv.index:
        return None
    syms = _syms_on(data, snapshot_date)
    if len(syms) < 2:
        return None
    px_row = px_piv.loc[snapshot_date]
    oi_row = oi_piv.loc[snapshot_date]
    pairs, spreads, min_ois = [], [], []
    for i in range(len(syms) - 1):
        s1, s2 = syms[i], syms[i + 1]
        p1, p2 = px_row.get(s1), px_row.get(s2)
        if pd.isna(p1) or pd.isna(p2):
            continue
        o1, o2 = oi_row.get(s1, 0), oi_row.get(s2, 0)
        pairs.append(f"{s1}-{_leg_suffix(s2)}")
        spreads.append(p1 - p2)
        min_ois.append(min(o1, o2))
    if not pairs:
        return None
    marker_colors = ["#16a34a" if s >= 0 else "#dc2626" for s in spreads]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=pairs, y=min_ois, name="Min OI", marker_color="rgba(156,163,175,.55)",
                         hovertemplate="%{x}<br>Min OI: %{y:,.0f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=pairs, y=spreads, name="Spread", mode="lines+markers",
                             line=dict(color="#374151", width=2),
                             marker=dict(size=8, color=marker_colors),
                             hovertemplate="%{x}<br>Spread: %{y:+.2f}<extra></extra>"), secondary_y=True)
    fig.add_hline(y=0, line_color="#cccccc", line_width=1, secondary_y=True)
    fig.update_layout(height=340, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      legend=dict(orientation="h", y=1.12, x=0),
                      margin=dict(l=55, r=55, t=30, b=40))
    fig.update_yaxes(title_text="Min OI", secondary_y=False, gridcolor="rgba(0,0,0,.07)")
    fig.update_yaxes(title_text="Spread", secondary_y=True, showgrid=False)
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
        px_chg_s = (px_piv[contract_choice].pct_change(fill_method=None) * 100).reindex(dates)
        name = contract_choice
    else:
        return None

    valid = oi_chg_s.notna() & px_chg_s.notna()
    if not valid.any():
        return None
    dates_v = [d for d, ok in zip(dates, valid) if ok]

    fig = go.Figure(go.Scatter(
        x=oi_chg_s[valid].values, y=px_chg_s[valid].values, mode="markers",
        marker=dict(size=7, color="#1a56db", opacity=0.65, line=dict(width=0.5, color="#fff")),
        text=[d.strftime("%d %b %Y") for d in dates_v],
        hovertemplate="%{text}<br>OI Chg: %{x:+,.0f}<br>Price Chg: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#e5e7eb", line_width=1)
    fig.add_vline(x=0, line_color="#e5e7eb", line_width=1)
    fig.update_layout(height=380, plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Inter, sans-serif", color="#1a1a1a", size=11),
                      showlegend=False, margin=dict(l=55, r=25, t=30, b=40),
                      xaxis=dict(title=f"{name} OI Change (lots)", gridcolor="rgba(0,0,0,.07)"),
                      yaxis=dict(title="Price Change (%)", gridcolor="rgba(0,0,0,.07)"))
    return fig


def _spread_pairs(syms: list, px_row, oi_row) -> dict:
    """Upper-triangle-only (row = earlier month, col = later month, since
    syms is LTD-ascending) pairwise price spread and min-OI, keyed by
    (row_idx, col_idx)."""
    n = len(syms)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            r, c = syms[i], syms[j]
            p1, p2 = px_row.get(r), px_row.get(c)
            if pd.isna(p1) or pd.isna(p2):
                continue
            o1, o2 = oi_row.get(r), oi_row.get(c)
            min_oi = min(o1, o2) if pd.notna(o1) and pd.notna(o2) else None
            out[(i, j)] = (p1 - p2, min_oi)
    return out


def _matrix_table_open(css_extra: str = "") -> str:
    return f"""<style>
      .sprmat-wrap{{overflow-x:auto;border:1px solid #e5e7eb;border-radius:6px}}
      table.sprmat{{border-collapse:collapse;width:100%;font-size:.68rem;font-family:'Inter',sans-serif;white-space:nowrap}}
      table.sprmat th,table.sprmat td{{padding:3px 8px;text-align:center;border-bottom:1px solid #f4f4f5}}
      table.sprmat th{{background:#fafafa;color:#1a1a1a;font-weight:600;font-size:.62rem;
        text-transform:uppercase;letter-spacing:.02em;border-bottom:2px solid #d1d5db}}
      table.sprmat td.row-hdr{{background:#fafafa;font-weight:600;text-align:left}}
      table.sprmat td.blank{{background:#fbfbfc}}
      {css_extra}
    </style>"""


@st.cache_data(max_entries=50, show_spinner=False)
def build_min_oi_matrix_html(commodity: str, snapshot_date, mtime: float = 0.0):
    """Min(OI of the two legs) for every calendar-spread combination — a
    spread is only as tradeable as its thinner leg. White-to-green
    heatmap: deeper green = more size actually tradeable."""
    data = build_spot_oi_data(commodity, mtime)
    px_piv = data["px_piv"]; oi_piv = data["oi_piv"]
    if snapshot_date not in px_piv.index:
        return None
    syms = _syms_on(data, snapshot_date)
    if len(syms) < 2:
        return None
    pairs = _spread_pairs(syms, px_piv.loc[snapshot_date], oi_piv.loc[snapshot_date])
    oi_vals = [v[1] for v in pairs.values() if v[1] is not None]
    oi_min, oi_max = (min(oi_vals), max(oi_vals)) if oi_vals else (0, 1)

    css = _matrix_table_open()
    header = "<tr><th></th>" + "".join(f"<th>{s}</th>" for s in syms) + "</tr>"
    rows = []
    for i, r in enumerate(syms):
        cells = f"<td class='row-hdr'>{r}</td>"
        for j, c in enumerate(syms):
            if j <= i:
                cells += "<td class='blank'></td>"
                continue
            cell = pairs.get((i, j))
            if cell is None or cell[1] is None:
                cells += "<td></td>"
                continue
            min_oi = cell[1]
            cells += f"<td style='{_oi_heatmap_style(min_oi, oi_min, oi_max)};font-weight:600'>{_fmt_num(min_oi)}</td>"
        rows.append(f"<tr>{cells}</tr>")
    return f"{css}<div class='sprmat-wrap'><table class='sprmat'>{header}<tbody>{''.join(rows)}</tbody></table></div>"


@st.cache_data(max_entries=50, show_spinner=False)
def build_price_spread_matrix_html(commodity: str, snapshot_date, mtime: float = 0.0):
    """The price spread itself for every calendar-spread combination, as an
    Excel-style diverging data bar."""
    data = build_spot_oi_data(commodity, mtime)
    px_piv = data["px_piv"]; oi_piv = data["oi_piv"]
    if snapshot_date not in px_piv.index:
        return None
    syms = _syms_on(data, snapshot_date)
    if len(syms) < 2:
        return None
    pairs = _spread_pairs(syms, px_piv.loc[snapshot_date], oi_piv.loc[snapshot_date])
    vmax = max((abs(v[0]) for v in pairs.values()), default=1.0) or 1.0

    css = _matrix_table_open()
    header = "<tr><th></th>" + "".join(f"<th>{s}</th>" for s in syms) + "</tr>"
    rows = []
    for i, r in enumerate(syms):
        cells = f"<td class='row-hdr'>{r}</td>"
        for j, c in enumerate(syms):
            if j <= i:
                cells += "<td class='blank'></td>"
                continue
            cell = pairs.get((i, j))
            if cell is None:
                cells += "<td></td>"
                continue
            spread = cell[0]
            cells += (f"<td style='{_oi_chg_style(spread, vmax)};color:{_sign_color(spread)};font-weight:600'>"
                      f"{spread:+.2f}</td>")
        rows.append(f"<tr>{cells}</tr>")
    return f"{css}<div class='sprmat-wrap'><table class='sprmat'>{header}<tbody>{''.join(rows)}</tbody></table></div>"


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

    # Drives volume charts that now sit in two different Volume subtabs, so it
    # cannot live inside either one of them.
    roll_n = st.slider("Rolling Volume Window (days)", min_value=1, max_value=30,
                       value=10, step=1,
                       help="Applied to daily volume before plotting")

    st.markdown("---")
    render_data_freshness(st.sidebar)


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size:0.70rem !important; color:#888; }
[data-testid="stMetricValue"] { font-size:1.10rem !important; font-weight:600; }
[data-testid="stMetricDelta"] { font-size:0.70rem !important; }
</style>""", unsafe_allow_html=True)


# ── Section nav — same button-pill / underline-tab treatment as the COT
# Comprehensive dashboard: a top pill row picks the metric group (Open
# Interest / Volume / OI & Volume), a second underline row picks the scope
# within it. Only the selected (group, view) body actually runs each rerun —
# st.tabs used to build all 8 panes on every sidebar change and just hide
# the rest. Grouped by metric at the top level, then by scope inside:
# Progression is one contract against its own history on a days-to-expiry
# axis, Board is every contract on a calendar axis.
_NAV_ACCENT = C["oi_avg"]
st.markdown(f"""<style>
  .st-key-nav_section [data-testid="stButtonGroup"] > div {{
    display:inline-flex; gap:4px; padding:4px; background:#f1f3f7;
    border:1px solid #e3e7ee; border-radius:999px;
  }}
  .st-key-nav_section button[kind^="segmented_control"] {{
    border:none !important; border-radius:999px !important; margin:0 !important;
    padding:.35rem 1.25rem !important; min-height:0 !important;
    background:transparent !important; box-shadow:none !important;
    transition:background .15s ease, color .15s ease;
  }}
  .st-key-nav_section button[kind^="segmented_control"] p {{
    font-size:.84rem !important; font-weight:600 !important; letter-spacing:.02em;
    color:#5b6472 !important;
  }}
  .st-key-nav_section button[kind="segmented_control"]:hover {{ background:#e6e9f0 !important; }}
  .st-key-nav_section button[kind="segmented_controlActive"] {{
    background:{_NAV_ACCENT} !important; box-shadow:0 1px 3px rgba(0,0,0,.18) !important;
  }}
  .st-key-nav_section button[kind="segmented_controlActive"] p {{ color:#ffffff !important; }}

  .st-key-nav_view {{ margin-top:-.35rem; border-bottom:1px solid #e3e7ee; gap:0; }}
  .st-key-nav_view [data-testid="stButtonGroup"] > div {{ gap:2px; flex-wrap:wrap; }}
  .st-key-nav_view button[kind^="segmented_control"] {{
    border:none !important; border-radius:6px 6px 0 0 !important; margin:0 0 -1px 0 !important;
    padding:.45rem .9rem !important; min-height:0 !important;
    background:transparent !important; box-shadow:none !important;
    border-bottom:2px solid transparent !important;
    transition:color .15s ease, border-color .15s ease, background .15s ease;
  }}
  .st-key-nav_view button[kind^="segmented_control"] p {{
    font-size:.81rem !important; font-weight:500 !important; color:#6b7280 !important;
  }}
  .st-key-nav_view button[kind="segmented_control"]:hover {{ background:#f5f6f9 !important; }}
  .st-key-nav_view button[kind="segmented_control"]:hover p {{ color:#1f2937 !important; }}
  .st-key-nav_view button[kind="segmented_controlActive"] {{ border-bottom:2px solid {_NAV_ACCENT} !important; }}
  .st-key-nav_view button[kind="segmented_controlActive"] p {{ color:{_NAV_ACCENT} !important; font-weight:600 !important; }}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OI PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _view_oi():
    res = compute_band(commodity, selected_month, hist_range, "open_interest", mtime=mt)
    if res is None:
        st.error("No data available.")
        st.stop()

    band = res[0]
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

    # Normalized panel is computed BEFORE the columns open so that nothing but
    # the chart itself is emitted inside each column: any caption/info block
    # rendered above a chart pushes that column's plot down and breaks the
    # side-by-side vertical alignment. All prose now sits *below* both charts,
    # one line each, so the two plots start and end at the same height.
    res_n2 = _normalize_oi_at_dte_max(commodity, selected_month, hist_range, current_contract, mt)

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
        st.caption("Contracts outstanding vs history, by days to expiry.")

    with col_norm:
        if res_n2 is not None:
            band_n2, curr_n2, dte_max, current_reached, hist_norm_n2 = res_n2
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
            note = f"DTE_max={dte_max}: avg DTE of each year's OI peak."
            if not current_reached:
                note += f" {current_contract} at {dte_now} DTE — no line yet."
            st.caption(
                note,
                help="DTE_max is the average, across the selected historical years, of the day-to-expiry "
                     "each year's OI hit its own high; every year's OI curve is then divided by that same "
                     "year's own OI reading on that DTE_max day (not its own peak) and shown as a %. "
                     "The current contract only appears once it has counted down to DTE_max.",
            )

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
        b_sh = res_sh[0]
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

    # ── Total OI seasonality ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### {COMMODITIES[commodity][1]} — Total OI Seasonality")
    st.caption("Whole-market futures open interest (LSEG TOTCNTROI), by calendar day. Band and "
               "mean use the Historical Years in the sidebar; the current year is never in its "
               "own band.")

    seas = build_total_oi_seasonal(commodity, tuple(hist_range), mt, _total_oi_mtime())
    if seas is None:
        st.info("Not enough history for a seasonal view.")
    else:
        dense_s, band_s = seas["dense"], seas["band"]
        cy, doy_now, oi_now = seas["cur_year"], seas["last_doy"], seas["last_oi"]

        def _at(frame, col, doy):
            if frame is None:
                return np.nan
            r = frame.loc[frame["doy"] == doy, col]
            return float(r.iloc[0]) if len(r) else np.nan

        mean_now = _at(band_s, "hist_mean", doy_now)
        ly_now = _at(dense_s[dense_s["year"] == cy - 1], "total_oi", doy_now)
        n_yrs = len(seas["band_years"])
        kpi_row([
            ("Total OI",              f"{oi_now:,.0f}"),
            ("As of",                 seas["last_date"].strftime("%b %d, %Y")),
            (f"vs {n_yrs}Y Mean",     f"{mean_now:,.0f}" if pd.notna(mean_now) else "—",
             f"{(oi_now / mean_now - 1) * 100:+.1f}%" if pd.notna(mean_now) and mean_now > 0 else None),
            (f"vs {cy - 1} same day", f"{ly_now:,.0f}" if pd.notna(ly_now) else "—",
             f"{(oi_now / ly_now - 1) * 100:+.1f}%" if pd.notna(ly_now) and ly_now > 0 else None),
        ])

        _x0 = pd.Timestamp(f"{_SEAS_REF_YEAR}-01-01")

        def _dx(d):
            return _x0 + pd.to_timedelta(np.asarray(d) - 1, unit="D")

        fig_ts = go.Figure()
        if band_s is not None and not band_s.empty:
            bx = _dx(band_s["doy"])
            fig_ts.add_trace(go.Scatter(x=bx, y=band_s["hist_max"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_ts.add_trace(go.Scatter(x=bx, y=band_s["hist_min"], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=C["oi_outer"],
                name="Min-Max Range", hoverinfo="skip"))
            fig_ts.add_trace(go.Scatter(x=bx, y=band_s["hist_q75"], mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_ts.add_trace(go.Scatter(x=bx, y=band_s["hist_q25"], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=C["oi_inner"],
                name="25th-75th Pct", hoverinfo="skip"))
            fig_ts.add_trace(go.Scatter(x=bx, y=band_s["hist_mean"], mode="lines",
                line=dict(color=C["oi_avg"], width=2, dash="dash"), name=f"{n_yrs}Y Mean",
                hovertemplate="Mean: %{y:,.0f}<extra></extra>"))

        if show_individual:
            for yr in seas["band_years"]:
                g = dense_s[dense_s["year"] == yr]
                fig_ts.add_trace(go.Scatter(x=_dx(g["doy"]), y=g["total_oi"], mode="lines",
                    line=dict(width=0.9, color=C["individual"]), name=str(yr),
                    hovertemplate=f"{yr}: %{{y:,.0f}}<extra></extra>"))

        prev = dense_s[dense_s["year"] == cy - 1]
        if not prev.empty:
            fig_ts.add_trace(go.Scatter(x=_dx(prev["doy"]), y=prev["total_oi"], mode="lines",
                line=dict(color="#6b7280", width=1.6), name=str(cy - 1),
                hovertemplate=f"{cy - 1}: %{{y:,.0f}}<extra></extra>"))

        cur = dense_s[dense_s["year"] == cy]
        fig_ts.add_trace(go.Scatter(x=_dx(cur["doy"]), y=cur["total_oi"], mode="lines",
            line=dict(color=C["current"], width=2.5), name=str(cy),
            hovertemplate=f"<b>{cy}</b>: %{{y:,.0f}}<extra></extra>"))
        fig_ts.add_trace(go.Scatter(x=_dx([doy_now]), y=[oi_now], mode="markers",
            marker=dict(color=C["current"], size=8, line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip"))

        fig_ts.update_layout(
            height=500, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            font=dict(color=C["font"], family="Inter, sans-serif"),
            xaxis=dict(tickformat="%b", dtick="M1", hoverformat="%d %b", showgrid=True,
                       gridcolor=C["grid"], range=[_x0, _x0 + pd.Timedelta(days=365)],
                       zeroline=False, tickfont=dict(size=11, color=C["font"])),
            yaxis=dict(title="Total Open Interest (contracts)", tickformat=",.0f",
                       showgrid=True, gridcolor=C["grid"], zeroline=False,
                       tickfont=dict(size=11, color=C["font"])),
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", margin=dict(l=70, r=30, t=30, b=80),
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        if seas["trimmed"]:
            st.caption(f"{seas['trimmed']} latest session(s) left out: fewer contracts had "
                       f"published OI than usual, so the board total would read short.")

        # ── Total OI time series ──────────────────────────────────────────────
        st.markdown(f"#### {COMMODITIES[commodity][1]} — Total OI History")
        if seas["source"] == "LSEG":
            st.caption(f"LSEG whole-market total (TOTCNTROI), from {seas['board_from']:%d %b %Y}.")
        else:
            st.caption(f"Summed from the per-contract table (the stored LSEG total was not found); "
                       f"starts {seas['board_from']:%d %b %Y}, the first date the database holds "
                       f"every listed contract.")
        ts = seas["series"]
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines", name="Total OI",
            line=dict(color=C["oi_avg"], width=1.6),
            hovertemplate="%{x|%d %b %Y}<br>Total OI: %{y:,.0f}<extra></extra>"))
        fig_hist.add_trace(go.Scatter(x=[ts.index[-1]], y=[ts.iloc[-1]], mode="markers",
            marker=dict(color=C["current"], size=8, line=dict(color="white", width=1.5)),
            showlegend=False, hoverinfo="skip"))
        # Opens on the last 5 years: the full history back to 2010 squeezes
        # the recent moves flat. The range buttons widen it on demand.
        _end = ts.index[-1]
        _vis = ts[ts.index >= max(ts.index[0], _end - pd.DateOffset(years=5))]
        fig_hist.update_layout(
            height=420, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            font=dict(color=C["font"], family="Inter, sans-serif"),
            xaxis=dict(range=[max(ts.index[0], _end - pd.DateOffset(years=5)), _end],
                       showgrid=True, gridcolor=C["grid"], zeroline=False,
                       tickfont=dict(size=11, color=C["font"]),
                       rangeselector=dict(
                           buttons=[dict(count=6, label="6M", step="month", stepmode="backward"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                                    dict(step="all", label="All")],
                           bgcolor="#f3f4f6", activecolor="#dbe4f5",
                           font=dict(size=10, color=C["font"]), x=0, y=1.08)),
            # Plotly autoranges y over ALL the data, not the opening x-window, so
            # the 26-year history dragged the axis down to the 2000 level while
            # only the last 5 years show. Fit the opening window; the range
            # buttons still re-autorange when the user widens it.
            yaxis=dict(title="Total Open Interest (contracts)", tickformat=",.0f",
                       range=[float(_vis.min()) * 0.94, float(_vis.max()) * 1.04],
                       showgrid=True, gridcolor=C["grid"], zeroline=False,
                       tickfont=dict(size=11, color=C["font"])),
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", margin=dict(l=70, r=30, t=50, b=60),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOLUME
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _view_vol():
    st.markdown(f"### {current_contract}  |  Volume Analysis")
    st.markdown("---")

    month_name = MONTH_NAMES.get(selected_month, selected_month)
    df_enr     = load_enriched(commodity, mt)
    df_enr_m   = df_enr[df_enr["month"] == selected_month].copy()

    # ── Chart 1: Volume Market Share ──────────────────────────────────────────
    st.markdown("#### Volume Market Share (%)")
    st.caption("Contract daily volume as % of total commodity volume on that date.")

    res_vs = compute_band(commodity, selected_month, hist_range, "vol_share_pct", use_enriched=True, mtime=mt)
    if res_vs:
        b_vs = res_vs[0]
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
        b_rv = res_rv[0]

        # Compute rolling vol for current contract
        c_rv = df_enr[df_enr["ice_symbol"] == current_contract].sort_values("Date").copy()
        c_rv["_metric"] = c_rv["volume"].rolling(roll_n, min_periods=1).mean()

        lat   = c_rv.iloc[-1]
        v_now = lat["_metric"]
        d_now = int(lat["days_to_expiry"])
        idx_  = (b_rv["days_to_expiry"] - d_now).abs().idxmin()
        avg_  = b_rv.loc[idx_, "hist_mean"]

        # Guarded like the OI KPI in Tab 1: a historical mean of 0 (a contract
        # that simply did not trade around this DTE) otherwise renders "inf%".
        rv_delta = f"{(v_now-avg_)/avg_*100:+.1f}%" if pd.notna(avg_) and avg_ > 0 else None
        kpi_row([
            ("Contract",          current_contract),
            (f"{roll_n}d Avg Vol", f"{v_now:,.0f}"),
            ("As of",             lat["Date"].strftime("%b %d, %Y")),
            ("vs Hist Mean",      f"{avg_:,.0f}", rv_delta),
        ])

        fig_rv = build_chart(b_rv, c_rv, "_metric", current_contract,
            title=f"<b>{commodity} {month_name}</b>  |  Rolling {roll_n}-Day Volume",
            y_title=f"{roll_n}-Day Avg Daily Volume (contracts)",
            y_fmt=",.0f", y_suffix="",
            outer_color=C["rv_outer"], inner_color=C["rv_inner"], avg_color=C["rv_avg"],
            dte_range=dte_range, dte_now=d_now, height=460)
        st.plotly_chart(fig_rv, use_container_width=True)


# ==============================================================================
# VOLUME - BOARD (every contract, calendar axis)
# ==============================================================================
@st.fragment
def _view_vol_board():
    st.markdown(f"### {COMMODITIES[commodity][1]}  |  Volume by Contract")
    st.markdown(f"#### All Contracts — Rolling {roll_n}-Day Volume")
    st.caption("Rolling volume for every contract that traded within the lookback window "
               "(includes contracts that have since expired, so historical totals stay accurate).")

    df_vol_all = load_data(commodity, mt)

    _lb_col, _ = st.columns([1, 5])
    with _lb_col:
        lookback = st.number_input("Lookback (calendar days)", min_value=30, max_value=365,
                                   value=120, step=10, key="vol_all_lookback")
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
@st.fragment
def _view_flow():
    month_name = MONTH_NAMES.get(selected_month, selected_month)
    st.markdown(f"### {current_contract}  |  Daily OI Change vs Volume")

    _lb_col, _ = st.columns([1, 5])
    with _lb_col:
        flow_lookback = st.number_input("Lookback (calendar days)", min_value=30, max_value=365,
                                        value=120, step=10, key="flow_lookback")

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

        # Stacked panels rather than one overlaid axis pair: volume is an order
        # of magnitude larger than the daily OI change, so overlaying them left
        # the two bar sets sitting on top of each other and unreadable. Sharing
        # one x-axis keeps every date column locked between the two panels.
        fig_flow = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.05, row_heights=[0.5, 0.5])
        fig_flow.add_trace(go.Bar(
            x=flow_win["Date"], y=flow_win["oi_change"], name="OI Change",
            marker_color=bar_colors,
            hovertemplate="OI Change: %{y:+,.0f}<extra></extra>",
        ), row=1, col=1)
        fig_flow.add_trace(go.Bar(
            x=flow_win["Date"], y=flow_win["volume"], name="Volume",
            marker_color="rgba(120,120,120,0.55)",
            hovertemplate="Volume: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig_flow.update_layout(
            title=dict(text=f"<b>{commodity} {month_name}</b>  |  Daily OI Change (top) vs Volume (bottom)",
                       font=dict(size=16, color=C["font"]), x=0.01),
            plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
            font=dict(color=C["font"], family="Inter, sans-serif"),
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                       bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            hovermode="x unified", height=640, margin=dict(l=70, r=30, t=60, b=80),
        )
        fig_flow.update_xaxes(showgrid=True, gridcolor=C["grid"], rangebreaks=rangebreaks_f,
                              tickfont=dict(size=11, color=C["font"]))
        fig_flow.update_xaxes(title_text="Date", row=2, col=1)
        fig_flow.update_yaxes(title_text="OI Change (contracts)", showgrid=True, gridcolor=C["grid"],
                              zeroline=True, zerolinecolor="rgba(0,0,0,0.25)", zerolinewidth=1,
                              tickfont=dict(size=11, color=C["font"]), row=1, col=1)
        fig_flow.update_yaxes(title_text="Volume (contracts)", showgrid=True, gridcolor=C["grid"],
                              tickfont=dict(size=11, color=C["font"]), row=2, col=1)
        st.plotly_chart(fig_flow, use_container_width=True)

        # ── Scatter: Volume (x) vs OI Change (y) ─────────────────────────────
        st.markdown("#### Volume vs OI Change — Scatter")
        sc_c1, sc_c2, sc_c3 = st.columns([2, 1, 1])
        with sc_c1:
            scatter_scope = st.radio(
                "Scope", [f"This contract ({current_contract})", "All futures combined"],
                horizontal=True, key="flow_scatter_scope",
                help="All futures combined sums volume across every contract month on each "
                     "date and takes the day-over-day change in total board OI — the whole "
                     "curve's flow, so a roll that just moves OI between months nets out.",
            )
        with sc_c2:
            oi_chg_mode = st.radio("OI Δ", ["Signed", "Absolute"], horizontal=True,
                                   key="flow_scatter_mode")
        with sc_c3:
            smooth_mode = st.radio(
                "Smoothing", ["Daily", "5-day MA"], horizontal=True, key="flow_scatter_smooth",
                help="5-day MA averages BOTH volume and OI change over a trailing 5 sessions "
                     "before plotting, so each dot is a week of flow rather than one day — "
                     "it strips the day-to-day noise that flattens the daily fit. Works on "
                     "either scope.",
            )

        if scatter_scope == "All futures combined":
            df_all_flow = load_data(commodity, mt)
            sc_full = (df_all_flow.groupby("Date")
                       .agg(volume=("volume", "sum"), open_interest=("open_interest", "sum"))
                       .sort_index())
            sc_full["oi_change"] = sc_full["open_interest"].diff()
            sc_full = sc_full.dropna(subset=["oi_change"]).reset_index()
            sc_label = f"All {commodity} futures combined"
        else:
            sc_full = curr_flow
            sc_label = current_contract

        # |OI change| is taken BEFORE the rolling mean, not after: averaging the
        # signed series first and then taking the absolute value would let a
        # +2k day and a -2k day cancel to ~0 churn, which is the opposite of
        # what "Absolute" is asking for (average daily turnover of OI).
        sc_full = sc_full[["Date", "volume", "oi_change"]].copy()
        if oi_chg_mode == "Absolute":
            sc_full["oi_change"] = sc_full["oi_change"].abs()

        # Smoothing runs on the FULL history, then the lookback window is
        # sliced off it — smoothing the already-sliced window would burn the
        # first 4 days of the window on half-formed (NaN) averages.
        if smooth_mode == "5-day MA":
            sc_full["volume"]    = sc_full["volume"].rolling(5).mean()
            sc_full["oi_change"] = sc_full["oi_change"].rolling(5).mean()
            sc_full = sc_full.dropna(subset=["volume", "oi_change"])
            sc_label += "  —  5-day MA"

        sc_win = sc_full[sc_full["Date"] >= cutoff_flow] if cutoff_flow is not None else sc_full

        ma_sfx    = " , 5d MA" if smooth_mode == "5-day MA" else ""
        x_scatter = sc_win["volume"]
        y_scatter = sc_win["oi_change"]

        if len(sc_win) < 5:
            st.info("Not enough days in this window for a scatter.")
        else:
            xs, ys = x_scatter.values, y_scatter.values
            slope, intercept = np.polyfit(xs, ys, 1)
            r2 = np.corrcoef(xs, ys)[0, 1] ** 2
            x_line = np.array([xs.min(), xs.max()])

            if oi_chg_mode == "Signed":
                pt_colors = ["#16a34a" if v >= 0 else "#dc2626" for v in ys]
            else:
                pt_colors = "#4A7FD4"

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color=pt_colors, size=8, opacity=0.7,
                           line=dict(color="white", width=0.8)),
                name="Daily obs", showlegend=False,
                customdata=sc_win["Date"].dt.strftime("%b %d, %Y"),
                hovertemplate="<b>%{customdata}</b><br>Volume" + ma_sfx.replace(" , ", " ") +
                              ": %{x:,.0f}<br>OI Δ" + ma_sfx.replace(" , ", " ") +
                              ": %{y:+,.0f}<extra></extra>",
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
                name=f"Latest ({sc_win['Date'].iloc[-1].strftime('%b %d, %Y')})",
            ))
            fig_sc.update_layout(
                title=dict(text=f"<b>{sc_label}</b>", font=dict(size=13, color=C["font"]), x=0.01),
                height=440, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
                font=dict(color=C["font"], family="Inter, sans-serif"),
                margin=dict(l=70, r=30, t=45, b=60),
                xaxis=dict(title=f"Volume (contracts{ma_sfx})", showgrid=True, gridcolor=C["grid"],
                          tickfont=dict(size=11, color=C["font"])),
                yaxis=dict(title=("OI Change" if oi_chg_mode == "Signed" else "|OI Change|")
                                 + f" (contracts{ma_sfx})",
                          showgrid=True, gridcolor=C["grid"], zeroline=(oi_chg_mode == "Signed"),
                          zerolinecolor="rgba(0,0,0,0.25)", tickfont=dict(size=11, color=C["font"])),
                legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0,
                           bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
            )
            st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DAILY GRID (OI level / OI change / Volume / Px change, per contract month)
# ═══════════════════════════════════════════════════════════════════════════════
@st.fragment
def _view_grid():
    st.markdown(f"### {COMMODITIES[commodity][1]}  |  Daily OI & Volume by Contract Month")
    _lb_col, _ = st.columns([1, 5])
    with _lb_col:
        table_lookback = st.number_input("Lookback (calendar days)", min_value=30, max_value=365,
                                         value=90, step=10, key="oi_table_lookback")
    html = build_oi_vol_table_html(commodity, table_lookback, mt)
    if html is None:
        st.info("No data in this window.")
    else:
        st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SPOT OI REPORT (spot vs non-spot OI, COT-Tuesday-aligned snapshots)
# ═══════════════════════════════════════════════════════════════════════════════
def _calendar_date(label: str, all_dates: list, min_d, max_d, default_d, key_prefix: str):
    """Plain calendar widget, snapped to the nearest REAL available date.
    Uses only the value st.date_input returns THIS run — no session_state
    indirection, no secondary widget — so there is nothing in this
    function's own logic that can lag behind what the calendar displays."""
    picked = st.date_input(label, value=default_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_cal")
    return min(all_dates, key=lambda d: abs((d - pd.Timestamp(picked)).days))


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

    # Change wrt COT date sits ABOVE the daily grid: it's the summary read
    # first, so it shouldn't be buried under a long scrolling table.
    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:4px 0 4px'>"
               "Change wrt COT date</div>", unsafe_allow_html=True)
    st.markdown(build_spot_summary_html(spot_data), unsafe_allow_html=True)

    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:14px 0 4px'>"
               "Daily Grid</div>", unsafe_allow_html=True)
    html_spot = build_spot_daily_table_html(commodity, spot_lookback, leg1, leg2, price_source, mt)
    if html_spot is None:
        st.info("No data in this window.")
    else:
        st.markdown(html_spot, unsafe_allow_html=True)

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

    _spot_controls_css("charts_controls")
    with st.expander("Controls", expanded=False):
        with st.container(key="charts_controls"):
            c0, c1, c2, c3 = st.columns(4)
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

    fig_oi_spread = build_oi_spread_chart(commodity, chart_lookback, oi_choice, leg1, leg2, mt)
    if fig_oi_spread is not None:
        st.plotly_chart(fig_oi_spread, use_container_width=True)

    # ── OI Change vs Price Change — own "which future" selector right above ──
    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:10px 0 4px'>"
               "OI Change vs Price Change</div>", unsafe_allow_html=True)
    scatter_opts = [spot_label] + syms_spot
    scatter_sel = st.selectbox("Which future to study?", scatter_opts, index=0, key="charts_scatter_future")
    scatter_choice = "Spot (Most OI)" if scatter_sel == spot_label else scatter_sel
    fig_scatter = build_oi_price_scatter(commodity, chart_lookback, scatter_choice, mt)
    if fig_scatter is not None:
        st.plotly_chart(fig_scatter, use_container_width=True)


@st.fragment
def _render_spreads_tab(commodity: str, mt: float):
    """Fragment-scoped for the same reason as the other tabs above. All
    snapshot-date, cross-sectional views (term structure, curve spreads,
    the full spread matrix) live here — distinct from the time-series
    charts in 'All Futures OI Charts'."""
    spot_data = build_spot_oi_data(commodity, mt)
    syms_spot = spot_data["syms"]
    if not syms_spot:
        st.info("No contract-month data available.")
        return

    all_dates = list(spot_data["oi_piv"].index)
    min_d, max_d = pd.Timestamp(all_dates[0]).date(), pd.Timestamp(all_dates[-1]).date()
    default_older_d = max(min_d, (pd.Timestamp(max_d) - pd.Timedelta(days=7)).date())

    dc1, dc2 = st.columns(2)
    with dc1:
        snapshot_date = _calendar_date("New Date", all_dates, min_d, max_d, max_d, "spreads_snapshot")
    with dc2:
        older_date = _calendar_date("Older Date", all_dates, min_d, max_d, default_older_d, "spreads_older")

    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:10px 0 4px'>"
               f"Term Structure — {pd.Timestamp(snapshot_date).strftime('%d %b %Y')} vs "
               f"{pd.Timestamp(older_date).strftime('%d %b %Y')}</div>",
               unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        fig_term = build_term_structure_chart(commodity, snapshot_date, older_date, mt)
        if fig_term is not None:
            st.plotly_chart(fig_term, use_container_width=True)
    with cc2:
        st.markdown("<div style='font-size:.78rem;font-weight:600;color:#1a1a1a;margin-bottom:4px'>"
                   f"Curve Spreads (adjacent months) — New Date: "
                   f"{pd.Timestamp(snapshot_date).strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
        fig_curve_spread = build_curve_spread_chart(commodity, snapshot_date, mt)
        if fig_curve_spread is not None:
            st.plotly_chart(fig_curve_spread, use_container_width=True)

    # ── Spread Matrix — every pair, not just adjacent, two tables side by side ─
    st.markdown("<div style='font-size:.85rem;font-weight:600;color:#1a1a1a;margin:14px 0 4px'>"
               f"Spread Matrix (row minus column) — New Date: "
               f"{pd.Timestamp(snapshot_date).strftime('%d %b %Y')}</div>", unsafe_allow_html=True)
    st.caption("Each pair shown once — row's contract is the earlier month, so a cell is "
              "\"row price minus column price\" for that combination, e.g. the CCZ6 row / "
              "CCH7 column cell is CCZ6 - CCH7.")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("<div style='font-size:.72rem;font-weight:600;color:#6b7280;margin-bottom:4px'>"
                   "Min OI (Liquidity)</div>", unsafe_allow_html=True)
        html_oi = build_min_oi_matrix_html(commodity, snapshot_date, mt)
        if html_oi is None:
            st.info("No data for this date.")
        else:
            st.markdown(html_oi, unsafe_allow_html=True)
    with mc2:
        st.markdown("<div style='font-size:.72rem;font-weight:600;color:#6b7280;margin-bottom:4px'>"
                   "Price Spread</div>", unsafe_allow_html=True)
        html_price = build_price_spread_matrix_html(commodity, snapshot_date, mt)
        if html_price is None:
            st.info("No data for this date.")
        else:
            st.markdown(html_price, unsafe_allow_html=True)


def _view_spot():
    _render_all_futures_oi_recap(commodity, mt)

def _view_spot_charts():
    _render_all_futures_oi_charts(commodity, mt)

def _view_spreads():
    _render_spreads_tab(commodity, mt)


# ── Section nav — dispatch ───────────────────────────────────────────────────
NAV_GROUPS = {
    "Open Interest": {"Progression": _view_oi, "All Futures OI": _view_spot,
                      "Spread OI": _view_spreads, "Spot OI vs Spread": _view_spot_charts},
    "Volume":        {"Progression": _view_vol, "Board": _view_vol_board},
    "OI & Volume":   {"Flow": _view_flow, "Grid": _view_grid},
}

# A relabelled tab (e.g. "Board" -> "All Futures OI") leaves a stale value in
# a returning browser session's state — both the widget's own key and the
# plain "_last_view" fallback key persist across reruns/redeploys, and
# segmented_control errors if `default`/its stored value isn't in `options`
# any more. Drop anything that no longer matches before the widgets render.
if st.session_state.get("main_group") not in (None, *NAV_GROUPS):
    del st.session_state["main_group"]

with st.container(key="nav_section"):
    group = st.segmented_control("Section", list(NAV_GROUPS), default="Open Interest",
                                 key="main_group", label_visibility="collapsed") or "Open Interest"
group_views = NAV_GROUPS[group]

_view_widget_key = f"main_view_{group}"
if st.session_state.get(_view_widget_key) not in (None, *group_views):
    del st.session_state[_view_widget_key]

# Streamlit drops a widget's state while it isn't rendered, so the other
# group's last view is remembered in a plain session key and fed back as default.
_last_key = f"_last_view_{group}"
_last = st.session_state.get(_last_key)
if _last not in group_views:
    _last = next(iter(group_views))

with st.container(key="nav_view"):
    view = st.segmented_control("View", list(group_views), default=_last,
                                key=_view_widget_key, label_visibility="collapsed") or _last
st.session_state[_last_key] = view
group_views[view]()
