# -*- coding: utf-8 -*-
"""Deferred OI seasonals — total open interest across a basket of contracts
(any combination of markets and delivery months), aligned on days to the
basket's back-leg expiry, one line per crop year.

Replaces the desk's Excel sheet, which hardcodes a vendor RIC per leg and so
loses a leg silently every time a contract rolls off (an expired contract
needs a `^N` suffix the sheet does not carry), and which under-counts every
date one exchange is shut while another is open. Both are handled here
structurally rather than sheet by sheet.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Deferred OI Seasonals", page_icon="📈",
                   layout="wide")

from common import COMMODITIES, MONTH_NAMES, MONTH_ORDER, C, _mtime, load_data

# Categorical line colours, fixed order, never cycled. The dashboard's
# "current" orange is reserved for the live crop year, so comparison years take
# the remaining seven. Validated on the adjacent pairlist: worst CVD deltaE 9.1
# (target >= 8), worst normal-vision deltaE 19.6 (floor >= 15). Three of them sit
# under 3:1 contrast on white, which is why the data table below is not
# optional — it is the relief view for those.
CURRENT_COLOR = "#E8470A"
YEAR_COLORS = ["#2a78d6", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948"]
MAX_COMPARE = len(YEAR_COLORS)

MEAN_COLOR = "#1a1a2e"          # neutral reference line, not a series hue
BAND_INNER = "rgba(99,149,237,0.28)"
BAND_OUTER = "rgba(99,149,237,0.10)"

PRESETS = {
    "Cocoa — CC+LCC, Z+H":     {"CC": ["Z", "H"], "LCC": ["Z", "H"]},
    "Sugar — SB+LSU, K+N+V":   {"SB": ["K", "N", "V"], "LSU": ["K", "N", "V"]},
    "Coffee — KC Z+H, RC X+F": {"KC": ["Z", "H"], "RC": ["X", "F"]},
    "Cocoa NY only — CC, Z+H": {"CC": ["Z", "H"]},
}


def _cycle_start(months) -> int:
    """Which delivery month opens the basket's season.

    Z+H is Dec-then-Mar (a 3-month basket), not Mar-then-Dec (9 months), so the
    opening leg is whichever start makes the set span the fewest months. The
    same rule recovers U+Z (Sep opens), K+N+V (May) and X+F (Nov) without
    asking the user to say which leg is the front."""
    nums = sorted({MONTH_ORDER[m] for m in months})
    return min(nums, key=lambda s: max((n - s) % 12 for n in nums))


def basket_legs(basket: dict, crop_year: int):
    """(commodity, month, delivery_year) for one crop year of the basket.

    A leg whose month falls before the opening month has wrapped past December
    into the next calendar year — that is what makes Z25+H26 a single crop
    year, and what lets KC Z+H and RC X+F sit in one basket together."""
    start = _cycle_start([m for ms in basket.values() for m in ms])
    return [(c, m, crop_year + (0 if MONTH_ORDER[m] >= start else 1))
            for c, ms in basket.items() for m in ms], start


@st.cache_data(max_entries=400, show_spinner=False)
def build_crop_year(basket_key, crop_year: int, mtimes, stop_at_front: bool = True):
    """One crop year of basket OI, densified onto an integer days-to-expiry grid.

    The x-axis is days to the *back* leg's expiry — for Z+H that is "time to H
    exp", which is how the desk reads it — and the series stops when the front
    leg expires, because past that point the basket is no longer the basket
    that was selected. Returns (series, meta) or None."""
    basket = {c: list(ms) for c, ms in basket_key}
    legs, _ = basket_legs(basket, crop_year)

    frames = []
    for comm, m, y in legs:
        df = load_data(comm, mtimes.get(comm, 0.0))
        sub = df[(df["month"] == m) & (df["year"] == y)]
        if not sub.empty:
            frames.append(sub[["Date", "ice_symbol", "open_interest", "LTD"]])
    if not frames:
        return None

    allf = pd.concat(frames)
    ltd = allf.groupby("ice_symbol")["LTD"].first()
    back_ltd, front_ltd = ltd.max(), ltd.min()

    piv = allf.pivot_table(index="Date", columns="ice_symbol",
                           values="open_interest", aggfunc="last").sort_index()

    # NY and London keep different holiday calendars. On a UK bank holiday the
    # London legs simply have no row, so a raw row-wise sum silently halves a
    # NY+LD basket for that date, and the reverse happens on US holidays — 26%
    # of dates in a CC+LCC Z+H basket are affected. Carry each leg's last real
    # OI across the other exchange's closures, but only between that leg's own
    # first and last print, so nothing is invented before a contract lists or
    # after it expires.
    filled = 0
    for col in piv.columns:
        s = piv[col]
        lo, hi = s.first_valid_index(), s.last_valid_index()
        if lo is None:
            continue
        seg = s.loc[lo:hi]
        filled += int(seg.isna().sum())
        piv.loc[lo:hi, col] = seg.ffill()

    # The two exchanges' feeds land at different times, so on the newest date
    # one market can be written and the other not yet. Summing there yields a
    # half-basket that reads as a genuine collapse — the very artefact this
    # page exists to remove — so the series ends at the last date on which
    # every leg actually printed. For a completed crop year this is a no-op:
    # the plotted window already closes at the front leg's expiry, by which
    # point every leg still has data.
    last_common = min(piv[c].last_valid_index() for c in piv.columns)
    trimmed = int((piv.index > last_common).sum())
    piv = piv.loc[:last_common]

    total = piv.sum(axis=1, min_count=1).dropna()
    if total.empty:
        return None

    dte = (back_ltd - total.index).days
    s = pd.Series(total.values, index=dte).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if stop_at_front:
        s = s[s.index >= (back_ltd - front_ltd).days]
    s = s[s.index >= 0]
    if len(s) < 2:
        return None

    grid = np.arange(int(s.index.min()), int(s.index.max()) + 1)
    dense = pd.Series(np.interp(grid, s.index.values, s.values), index=grid)

    return dense, dict(back_ltd=back_ltd, front_ltd=front_ltd,
                       legs=list(piv.columns), filled=filled, trimmed=trimmed,
                       complete=front_ltd < pd.Timestamp(date.today()),
                       last_date=total.index.max())


def crop_label(basket: dict, crop_year: int) -> str:
    """"24/25" when the basket wraps a calendar year, plain "25" when it does not."""
    legs, _ = basket_legs(basket, crop_year)
    wraps = any(y != crop_year for _, _, y in legs)
    return (f"{crop_year % 100:02d}/{(crop_year + 1) % 100:02d}" if wraps
            else f"{crop_year % 100:02d}")


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### Deferred OI Seasonals")
st.caption("Total open interest across a basket of contracts, aligned on days to the "
           "basket's back-leg expiry — one line per crop year.")

MTIMES = {c: _mtime(c) for c in COMMODITIES}


@st.cache_data(show_spinner=False)
def _months_traded(commodity: str, mtime: float) -> list:
    df = load_data(commodity, mtime)
    return sorted(df["month"].unique(), key=lambda m: MONTH_ORDER[m])


with st.expander("Basket", expanded=True):
    preset_name = st.selectbox("Preset", list(PRESETS) + ["Custom"], index=0,
                               key="seas_preset")
    if preset_name == "Custom":
        markets = st.multiselect(
            "Markets", list(COMMODITIES), default=["KC", "RC"], key="seas_markets",
            help="Months are picked per market below, so the legs do not have to match "
                 "across markets — KC Z+H can be combined with RC X+F in one basket.")
        basket = {}
        if markets:
            cols = st.columns(len(markets))
            for col, mk in zip(cols, markets):
                with col:
                    opts = _months_traded(mk, MTIMES[mk])
                    picked = st.multiselect(
                        f"{mk} months", opts, default=[], key=f"seas_months_{mk}",
                        format_func=lambda m: f"{m} ({MONTH_NAMES[m][:3]})")
                    if picked:
                        basket[mk] = picked
    else:
        basket = {k: list(v) for k, v in PRESETS[preset_name].items()}
        st.caption("  ·  ".join(f"**{k}** {'+'.join(v)}" for k, v in basket.items()))

if not basket:
    st.info("Pick at least one market and one delivery month.")
    st.stop()

basket_key = tuple((c, tuple(ms)) for c, ms in sorted(basket.items()))

# ── Build every crop year the data supports ───────────────────────────────────
built, meta = {}, {}
for cy in range(2009, date.today().year + 2):
    r = build_crop_year(basket_key, cy, MTIMES, stop_at_front=True)
    if r is None:
        continue
    lbl = crop_label(basket, cy)
    built[lbl], meta[lbl] = r[0], r[1]

if not built:
    st.error("No contracts found for that basket.")
    st.stop()

labels_all = list(built)
complete   = [l for l in labels_all if meta[l]["complete"]]

# "Current" is the season being traded now — the one whose front leg expires
# next — not simply the newest one built. A crop year two seasons out is
# listed and technically incomplete, but carries a few thousand lots and is
# not what the desk means by the live season.
incomplete = [l for l in labels_all if not meta[l]["complete"]]
current    = (min(incomplete, key=lambda l: meta[l]["front_ltd"]) if incomplete
              else labels_all[-1])

# The year pickers are keyed to the basket. A keyed widget keeps its value
# across reruns and only drops entries missing from the new option list — so
# switching from cocoa (labels like "21/22") to sugar ("27") silently emptied
# both lists and left the average with nothing in it, rather than falling back
# to the defaults. A per-basket key makes each basket get its own widget.
bsig = "_".join(f"{c}{''.join(ms)}" for c, ms in basket_key)

with st.expander("Years & display", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        default_cmp = [l for l in complete[-MAX_COMPARE:] if l != current]
        cmp_years = st.multiselect(
            "Crop years to plot", labels_all, default=default_cmp, key=f"seas_cmp_{bsig}",
            help=f"The current crop year ({current}) is always drawn. Capped at "
                 f"{MAX_COMPARE} comparison lines — past that the colours stop being "
                 f"reliably distinguishable, so read the rest off the band instead.")
        if len(cmp_years) > MAX_COMPARE:
            st.warning(f"Showing the first {MAX_COMPARE} of {len(cmp_years)} — deselect "
                       f"some, or read the rest off the band.")
            cmp_years = cmp_years[:MAX_COMPARE]
    with c2:
        avg_years = st.multiselect(
            "Years in the average & band", complete, default=complete[-5:],
            key=f"seas_avg_{bsig}",
            help="Defaults to the last 5 complete crop years and rolls forward on its "
                 "own, so it cannot go stale. Incomplete years are excluded — including "
                 "one would make the average step where its data runs out.")
        show_band = st.checkbox(
            "Show percentile band", value=True, key="seas_band",
            help="25th-75th percentile and min-max envelope across the years above. "
                 "Individual crop years span a wide range of levels, so the band says "
                 "considerably more than the mean line alone.")

    d1, d2 = st.columns(2)
    with d1:
        max_dte = st.slider("Max days to expiry shown", 200, 900, 700, step=25,
                            key="seas_max_dte")
    with d2:
        table_step = st.slider("Table row step (days)", 1, 14, 7, key="seas_step",
                               help="7 matches the weekly grid of the desk sheet.")

# ── Common DTE grid, mean and band ────────────────────────────────────────────
grid    = np.arange(0, max_dte + 1)
aligned = pd.DataFrame({l: built[l].reindex(grid) for l in labels_all}, index=grid)

avg_src = aligned[avg_years] if avg_years else pd.DataFrame(index=grid)
band = pd.DataFrame({
    "mean": avg_src.mean(axis=1, skipna=True),
    "p25":  avg_src.quantile(0.25, axis=1),
    "p75":  avg_src.quantile(0.75, axis=1),
    "lo":   avg_src.min(axis=1, skipna=True),
    "hi":   avg_src.max(axis=1, skipna=True),
}, index=grid).dropna(how="all")

# ── KPIs ──────────────────────────────────────────────────────────────────────
# DTE counts DOWN as time passes, so the most recent observation is the
# series' SMALLEST days-to-expiry, not its largest. Read it off the full
# series rather than the display grid, which max_dte may have truncated.
cur_full = built[current]
cur_dte  = int(cur_full.index.min()) if len(cur_full) else None
if cur_dte is not None:
    cur_oi = cur_full.loc[cur_dte]
    ref    = band["mean"].get(cur_dte, np.nan)
    k = st.columns(5)
    k[0].metric("Crop year", current)
    k[1].metric("Basket OI", f"{cur_oi:,.0f}")
    k[2].metric("Days to back-leg expiry", f"{cur_dte}")
    k[3].metric("As of", meta[current]["last_date"].strftime("%b %d, %Y"))
    k[4].metric(f"vs {len(avg_years)}Y mean",
                "n/a" if pd.isna(ref) else f"{ref:,.0f}",
                None if pd.isna(ref) else f"{(cur_oi / ref - 1) * 100:+.1f}%")

# ── Chart ─────────────────────────────────────────────────────────────────────
fig = go.Figure()
if show_band and not band.empty and avg_years:
    fig.add_trace(go.Scatter(x=band.index, y=band["hi"], mode="lines", name="Min-Max",
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=band.index, y=band["lo"], mode="lines", name="Min-Max",
                             line=dict(width=0), fill="tonexty", fillcolor=BAND_OUTER,
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=band.index, y=band["p75"], mode="lines",
                             name="25th-75th Pct", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=band.index, y=band["p25"], mode="lines",
                             name="25th-75th Pct", line=dict(width=0), fill="tonexty",
                             fillcolor=BAND_INNER, hoverinfo="skip"))
if avg_years:
    fig.add_trace(go.Scatter(
        x=band.index, y=band["mean"], mode="lines", name=f"{len(avg_years)}Y Mean",
        line=dict(color=MEAN_COLOR, width=2.5, dash="dash"),
        hovertemplate="%{y:,.0f}<extra>Mean</extra>"))

for i, lbl in enumerate(cmp_years):
    if lbl == current:
        continue
    fig.add_trace(go.Scatter(
        x=aligned.index, y=aligned[lbl], mode="lines", name=lbl,
        line=dict(color=YEAR_COLORS[i % len(YEAR_COLORS)], width=2),
        hovertemplate="%{y:,.0f}<extra>" + lbl + "</extra>"))

fig.add_trace(go.Scatter(
    x=aligned.index, y=aligned[current], mode="lines", name=f"{current} (current)",
    line=dict(color=CURRENT_COLOR, width=3),
    hovertemplate="%{y:,.0f}<extra>" + current + "</extra>"))
if cur_dte is not None:
    fig.add_annotation(x=cur_dte, y=cur_oi, text=f" {current}",
                       showarrow=False, xanchor="left",
                       font=dict(color=CURRENT_COLOR, size=11, family="Inter, sans-serif"))

basket_txt = ", ".join(f"{k} {'+'.join(v)}" for k, v in basket.items())
fig.update_layout(
    title=dict(text=f"<b>{basket_txt}</b>  |  Total Futures OI",
               font=dict(size=15, color=C["font"]), x=0.01),
    height=620, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
    font=dict(color=C["font"], family="Inter, sans-serif"),
    margin=dict(l=70, r=60, t=55, b=70), hovermode="x unified",
    xaxis=dict(title="Days to back-leg expiry", autorange="reversed",
               showgrid=True, gridcolor=C["grid"], zeroline=False,
               tickfont=dict(size=11, color=C["font"])),
    yaxis=dict(title="Open Interest (contracts)", showgrid=True, gridcolor=C["grid"],
               zeroline=False, tickformat=",", tickfont=dict(size=11, color=C["font"])),
    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)
st.plotly_chart(fig, use_container_width=True)

# ── Diagnostics ───────────────────────────────────────────────────────────────
n_fill = sum(meta[l]["filled"] for l in labels_all)
st.caption(
    f"Anchor: the back leg of each crop year — for Z+H that is time to H expiry. Each "
    f"series stops when its front leg expires. {n_fill:,} leg-days across "
    f"{len(labels_all)} crop years fell on a date when one exchange was shut and "
    f"another open, and were carried forward rather than summed as a gap.",
    help="A raw row-wise sum drops a leg entirely on the other exchange's holidays, "
         "which halves a NY+LD basket for that date. Each leg is carried forward across "
         "those closures, only between its own first and last real print.")

# ── Table — also the relief view for the low-contrast line colours ────────────
with st.expander("Data — days to expiry by crop year", expanded=False):
    tbl_cols = [l for l in labels_all if l in set(cmp_years) | {current}]
    tbl = aligned[tbl_cols].copy()
    if avg_years:
        tbl[f"{len(avg_years)}Y Mean"] = band["mean"]
    tbl = tbl.loc[::-1].iloc[::table_step]
    tbl.index.name = "DTE"
    st.dataframe(tbl.style.format("{:,.0f}", na_rep="—"),
                 use_container_width=True, height=430)
