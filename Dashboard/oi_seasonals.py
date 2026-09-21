# -*- coding: utf-8 -*-
"""Deferred OI seasonals — total open interest across a basket of contracts
(any combination of markets and delivery months), aligned on days to the
basket's back-leg expiry, one line per crop year.

Replaces the desk's Excel sheet, which hardcodes a vendor RIC per leg and so
loses a leg silently every time a contract rolls off (an expired contract
needs a `^N` suffix the sheet does not carry), and which under-counts every
date one exchange is shut while another is open. Both are handled here
structurally rather than sheet by sheet.

A crop year is only ever compared like-for-like: if any leg of a historical
year is not in the database, that year is left out and named on the page
rather than plotted as a smaller basket that reads as a genuinely low year.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Deferred OI Seasonals", page_icon="📈",
                   layout="wide")

try:
    from common import (COMMODITIES, LOT_TONNES, MONTH_ORDER, C, _mtime, load_data,
                    _oi_heatmap_style, _oi_chg_style, render_data_freshness)
except ImportError:
    # After a deploy the running server can still hold the PREVIOUS common.py in
    # memory (Streamlit Cloud keeps imported modules across a code update), so a
    # name added to it since then is "missing" until the app is rebooted. Reload
    # the module from disk once and import again, instead of showing an error.
    import importlib
    import common
    importlib.reload(common)
    from common import (COMMODITIES, LOT_TONNES, MONTH_ORDER, C, _mtime, load_data,
                    _oi_heatmap_style, _oi_chg_style, render_data_freshness)

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
DEFAULT_PLOT = 4        # comparison years drawn on opening; more can be added, up to MAX_COMPARE

MEAN_COLOR = "#1a1a2e"          # neutral reference line, not a series hue
BAND_INNER = "rgba(99,149,237,0.28)"
BAND_OUTER = "rgba(99,149,237,0.10)"
NAV_ACCENT = C["oi_avg"]

# Each market family lists the combined basket first, then its New York leg,
# then its London leg, so the single legs can be read on their own. Cocoa is
# first, which makes CC + LCC the default. London white sugar (LSU) lists H, K, Q,
# V, Z - there is no July - so the LSU side takes Q (August), its nearest month
# to SB's July; asking for LSU N left every sugar year a silent five-leg basket.
PRESETS = {
    "CC Z+H, LCC Z+H":                   {"CC": ["Z", "H"], "LCC": ["Z", "H"]},
    "CC Z+H":                            {"CC": ["Z", "H"]},
    "LCC Z+H":                           {"LCC": ["Z", "H"]},
    "KC Z+H, RC X+F":                    {"KC": ["Z", "H"], "RC": ["X", "F"]},
    "KC Z+H":                            {"KC": ["Z", "H"]},
    "RC X+F":                            {"RC": ["X", "F"]},
    "SB K+N+V, LSU K+Q+V":               {"SB": ["K", "N", "V"], "LSU": ["K", "Q", "V"]},
    "SB K+N+V":                          {"SB": ["K", "N", "V"]},
    "LSU K+Q+V":                         {"LSU": ["K", "Q", "V"]},
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
    all_months = [m for ms in basket.values() for m in ms]
    start = _cycle_start(all_months)
    return [(c, m, crop_year + (0 if MONTH_ORDER[m] >= start else 1))
            for c, ms in basket.items() for m in ms], start


@st.cache_data(max_entries=400, show_spinner=False)
def build_crop_year(basket_key, crop_year: int, mtimes, stop_at_front: bool = True,
                    in_tonnes: bool = False):
    """One crop year of basket OI, densified onto an integer days-to-expiry grid.

    The x-axis is days to the *back* leg's expiry — for Z+H that is "time to H
    exp", which is how the desk reads it — and the series stops when the front
    leg expires, because past that point the basket is no longer the basket
    that was selected. Returns (series, meta) or None; meta["missing"] names
    any leg with no data at all (the year is then a smaller basket)."""
    basket = {c: list(ms) for c, ms in basket_key}
    legs, _ = basket_legs(basket, crop_year)

    frames, missing = [], []
    for comm, m, y in legs:
        df = load_data(comm, mtimes.get(comm, 0.0))
        sub = df[(df["month"] == m) & (df["year"] == y)]
        if sub.empty:
            missing.append(f"{comm} {m}{y % 100:02d}")
        else:
            sub = sub[["Date", "ice_symbol", "open_interest", "LTD"]].copy()
            if in_tonnes:       # convert each leg by ITS market's lot size, then add
                sub["open_interest"] = sub["open_interest"] * LOT_TONNES[comm]
            frames.append(sub)
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
                       legs=list(piv.columns), missing=missing,
                       filled=filled, trimmed=trimmed,
                       complete=front_ltd < pd.Timestamp(date.today()),
                       last_date=total.index.max())


def crop_label(basket: dict, crop_year: int) -> str:
    """"24/25" when the basket wraps a calendar year, plain "25" when it does not."""
    legs, _ = basket_legs(basket, crop_year)
    wraps = any(y != crop_year for _, _, y in legs)
    return (f"{crop_year % 100:02d}/{(crop_year + 1) % 100:02d}" if wraps
            else f"{crop_year % 100:02d}")


def _min_obs(n_years: int) -> int:
    """Fewest years that must reach a given DTE before a mean/band is drawn
    there. Years are not equally long — the front-to-back gap differs from
    year to year — so at the far end only a couple of them have data, and a
    mean of two years steps every time one of them starts. Needs 60% of the
    selected years, never fewer than 3 (or all of them, if fewer are chosen)."""
    return min(n_years, max(3, int(np.ceil(0.6 * n_years))))


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""<style>
section[data-testid="stSidebar"] h3 {{ font-size:0.82rem; font-weight:600;
    color:#6b7280; letter-spacing:.02em; margin:0 0 .3rem; }}
.kpirow{{display:flex;flex-wrap:wrap;gap:0;border:1px solid #e5e7eb;border-radius:6px;
  background:#fafbfc;margin:4px 0 10px;overflow:hidden;width:fit-content}}
.kpichip{{padding:4px 12px;border-right:1px solid #e5e7eb;white-space:nowrap;font-size:.72rem}}
.kpichip:last-child{{border-right:none}}
.kpil{{color:#9ca3af;font-size:.62rem;text-transform:uppercase;letter-spacing:.03em;margin-right:4px}}
.kpiv{{font-weight:700;color:#1a1a1a}}
/* Same underline tab strip as the OI Progression page's view row. */
.st-key-nav_view {{ margin-top:-.35rem; border-bottom:1px solid #e3e7ee; gap:0; }}
.st-key-nav_view [data-testid="stButtonGroup"] > div {{ gap:2px; flex-wrap:wrap; }}
.st-key-nav_view button[kind^="segmented_control"] {{
  border:none !important; border-radius:6px 6px 0 0 !important; margin:0 0 -1px 0 !important;
  padding:.45rem .9rem !important; min-height:0 !important;
  background:transparent !important; box-shadow:none !important;
  border-bottom:2px solid transparent !important;
  transition:color .15s ease, border-color .15s ease, background .15s ease; }}
.st-key-nav_view button[kind^="segmented_control"] p {{
  font-size:.81rem !important; font-weight:500 !important; color:#6b7280 !important; }}
.st-key-nav_view button[kind="segmented_control"]:hover {{ background:#f5f6f9 !important; }}
.st-key-nav_view button[kind="segmented_control"]:hover p {{ color:#1f2937 !important; }}
.st-key-nav_view button[kind="segmented_controlActive"] {{ border-bottom:2px solid {NAV_ACCENT} !important; }}
.st-key-nav_view button[kind="segmented_controlActive"] p {{ color:{NAV_ACCENT} !important; font-weight:600 !important; }}
</style>""", unsafe_allow_html=True)


def _kpi_row(items):
    """items = (label, value[, delta]). One compact strip of chips."""
    chips = []
    for it in items:
        delta = it[2] if len(it) > 2 else None
        dh = ""
        if delta:
            col = "#dc2626" if str(delta).strip().startswith("-") else "#16a34a"
            dh = f"<b style='color:{col};font-weight:700;margin-left:4px'>{delta}</b>"
        chips.append(f"<span class='kpichip'><span class='kpil'>{it[0]}</span> "
                     f"<span class='kpiv'>{it[1]}</span>{dh}</span>")
    st.markdown(f"<div class='kpirow'>{''.join(chips)}</div>", unsafe_allow_html=True)


MTIMES = {c: _mtime(c) for c in COMMODITIES}


# ── Sidebar: basket ───────────────────────────────────────────────────────────
# A renamed preset leaves the old name in a returning session's state, which a
# selectbox will not silently recover from.
if st.session_state.get("seas_preset") not in (None, *PRESETS):
    del st.session_state["seas_preset"]

with st.sidebar:
    st.markdown("### Basket")
    preset_name = st.selectbox("Preset", list(PRESETS), index=0,
                               key="seas_preset", label_visibility="collapsed")
    unit = st.radio("Unit", ["Lots", "Tonnes"], horizontal=True, key="seas_unit")
    in_tonnes = unit == "Tonnes"
    unit_txt = "tonnes" if in_tonnes else "lots"
    basket = {k: list(v) for k, v in PRESETS[preset_name].items()}

basket_key = tuple((c, tuple(ms)) for c, ms in sorted(basket.items()))

# ── Build every crop year the data supports ───────────────────────────────────
built_all, meta_all = {}, {}
for cy in range(2009, date.today().year + 2):
    r = build_crop_year(basket_key, cy, MTIMES, True, in_tonnes)
    if r is None:
        continue
    lbl = crop_label(basket, cy)
    built_all[lbl], meta_all[lbl] = r[0], r[1]

if not built_all:
    st.error("No contracts found for that basket.")
    st.stop()

# A finished year with a leg missing is a smaller basket, not a low year: the
# database opens at the first contract still alive in 2011, so the earliest
# seasons lack their opening legs (cocoa 10/11 has no Dec-2010 contract and
# plotted at half the true level while being labelled complete). Leave those
# out and say so. A year still in progress with a leg not yet listed is kept —
# it is the live season — but flagged below.
excluded = {l: m["missing"] for l, m in meta_all.items() if m["missing"] and m["complete"]}
built = {l: s for l, s in built_all.items() if l not in excluded}
meta = {l: meta_all[l] for l in built}

if not built:
    st.error("Every crop year for that basket is missing at least one leg.")
    st.stop()

labels_all = list(built)
complete   = [l for l in labels_all if meta[l]["complete"]]

# "Current" is the season trading now — the one whose front leg expires next —
# not simply the newest built. A crop year two seasons out is listed and
# technically incomplete, but carries a few thousand lots.
incomplete = [l for l in labels_all if not meta[l]["complete"]]
current    = (min(incomplete, key=lambda l: meta[l]["front_ltd"]) if incomplete
              else labels_all[-1])

# ── Sidebar: years & display ──────────────────────────────────────────────────
# Keyed to the basket. A keyed widget keeps its value across reruns and only
# drops entries missing from the new options — so switching from cocoa (labels
# like "21/22") to sugar ("27") silently emptied both lists instead of falling
# back to the defaults. A per-basket key gives each basket its own widget.
bsig = "_".join(f"{c}{''.join(ms)}" for c, ms in basket_key)

dte_top = max(int(np.ceil(max(float(s.index.max()) for s in built.values()) / 25.0)) * 25, 225)

with st.sidebar:
    st.markdown("### Years")
    default_cmp = [l for l in complete[-DEFAULT_PLOT:] if l != current]
    cmp_years = st.multiselect(
        "Plot", complete, default=default_cmp, key=f"seas_cmp_{bsig}")
    if len(cmp_years) > MAX_COMPARE:
        st.warning(f"Max {MAX_COMPARE} years.")
        cmp_years = cmp_years[:MAX_COMPARE]

    st.markdown("### Display")
    show_band = st.checkbox("Percentile band", value=True, key="seas_band")
    # Upper bound follows the basket: a K+N+V sugar basket lists ~1,200 days out,
    # so the old fixed 900 cap cut off the first third of every year.
    max_dte = st.slider("Max days to expiry", 200, dte_top, min(700, dte_top), step=25,
                        key=f"seas_max_dte_{bsig}")

    st.markdown("---")
    render_data_freshness(st.sidebar)

# ── Common DTE grid, mean and band ────────────────────────────────────────────
# Every complete crop year feeds the mean and band. Incomplete years are left
# out: one would make the average step where its data runs out.
avg_years = list(complete)
grid    = np.arange(0, max_dte + 1)
aligned = pd.DataFrame({l: built[l].reindex(grid) for l in labels_all}, index=grid)

avg_src = aligned[avg_years] if avg_years else pd.DataFrame(index=grid)
n_avg   = len(avg_years)
band = pd.DataFrame({
    "mean": avg_src.mean(axis=1, skipna=True),
    "p25":  avg_src.quantile(0.25, axis=1),
    "p75":  avg_src.quantile(0.75, axis=1),
    "lo":   avg_src.min(axis=1, skipna=True),
    "hi":   avg_src.max(axis=1, skipna=True),
}, index=grid)
if n_avg:
    band = band.where(avg_src.notna().sum(axis=1) >= _min_obs(n_avg))   # see _min_obs
band = band.dropna(how="all")

# DTE counts DOWN as time passes, so the latest observation is the series'
# SMALLEST days-to-expiry. Read it off the full series, not the display grid,
# which max_dte may have truncated.
cur_full = built[current]
cur_dte  = int(cur_full.index.min())
cur_oi   = float(cur_full.loc[cur_dte])

# Reference numbers come from the full series too, so they still work when
# max_dte has been dragged below the current DTE.
at_dte  = pd.Series({l: built[l].get(cur_dte, np.nan) for l in avg_years}).dropna()
mean_ref = float(at_dte.mean()) if n_avg and len(at_dte) >= _min_obs(n_avg) else np.nan
prev_lbl = complete[-1] if complete else None
prev_ref = float(built[prev_lbl].get(cur_dte, np.nan)) if prev_lbl else np.nan
pctile   = (float((at_dte < cur_oi).mean() * 100)
            if n_avg and len(at_dte) >= max(3, _min_obs(n_avg)) else np.nan)

basket_txt = ", ".join(f"{k} {'+'.join(v)}" for k, v in basket.items())
st.markdown(f"### {basket_txt} <span style='font-size:.8rem;font-weight:500;color:#6b7280'>&nbsp;as of {meta[current]['last_date']:%d %b %Y}</span>", unsafe_allow_html=True)


def _pct(a, b):
    return f"{(a / b - 1) * 100:+.1f}%" if pd.notna(b) and b > 0 else None


_kpi_row([
    (f"{current} OI, {unit_txt}", f"{cur_oi:,.0f}"),
    ("As of", meta[current]["last_date"].strftime("%b %d, %Y")),
    ("DTE", f"{cur_dte}"),
    (f"vs {n_avg}Y mean", f"{mean_ref:,.0f}" if pd.notna(mean_ref) else "—", _pct(cur_oi, mean_ref)),
    (f"vs {prev_lbl}", f"{prev_ref:,.0f}" if pd.notna(prev_ref) else "—", _pct(cur_oi, prev_ref)),
    ("Percentile", f"P{pctile:.0f} of {len(at_dte)} yrs" if pd.notna(pctile) else "—"),
])

if meta[current]["missing"]:
    st.warning(f"{current} is missing {', '.join(meta[current]['missing'])}, so it reads lower than other years.")

with st.container(key="nav_view"):
    view = st.segmented_control("View", ["Chart", "Data"], default="Chart",
                                key="seas_view", label_visibility="collapsed") or "Chart"

# ── Chart ─────────────────────────────────────────────────────────────────────
if view == "Chart":
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
            x=band.index, y=band["mean"], mode="lines", name=f"{n_avg}Y Mean",
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
        x=aligned.index, y=aligned[current], mode="lines", name=current,
        line=dict(color=CURRENT_COLOR, width=3),
        hovertemplate="%{y:,.0f}<extra>" + current + "</extra>"))
    if cur_dte <= max_dte:
        fig.add_vline(x=cur_dte, line=dict(color="rgba(0,0,0,0.18)", width=1, dash="dot"))
        fig.add_trace(go.Scatter(x=[cur_dte], y=[cur_oi], mode="markers",
                                 marker=dict(color=CURRENT_COLOR, size=9,
                                             line=dict(color="white", width=1.5)),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=cur_dte, y=cur_oi, text=f" {current}", showarrow=False,
                           xanchor="left",
                           font=dict(color=CURRENT_COLOR, size=11, family="Inter, sans-serif"))

    # Every year stops when its front leg expires, so nothing is drawn below
    # ~90 days for a Z+H basket; a plain reversed autorange still ran the axis
    # to 0 and left a fifth of the plot empty. Stop where the drawn data stops.
    x_lo = max(0, min(int(built[l].index.min()) for l in set(cmp_years) | {current} | set(avg_years)) - 5)
    fig.update_layout(
        height=620, plot_bgcolor=C["bg"], paper_bgcolor=C["bg"],
        font=dict(color=C["font"], family="Inter, sans-serif"),
        margin=dict(l=70, r=60, t=20, b=70), hovermode="x unified",
        xaxis=dict(title="Days to back-leg expiry", range=[max_dte, x_lo],
                   showgrid=True, gridcolor=C["grid"], zeroline=False,
                   tickfont=dict(size=11, color=C["font"])),
        yaxis=dict(title=f"Open Interest ({unit_txt})", showgrid=True, gridcolor=C["grid"],
                   zeroline=False, tickformat=",", tickfont=dict(size=11, color=C["font"])),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_avg:
        st.caption(f"Average and band: last {n_avg} years ({avg_years[0]} to {avg_years[-1]}).")

# ── Data ──────────────────────────────────────────────────────────────────────
SEAS_TBL_CSS = """
<style>
.seas-wrap { overflow:auto; max-height:640px; border:1px solid #e5e7eb; border-radius:6px; }
.seas-tbl { border-collapse:collapse; font-size:10px; font-family:'Inter',sans-serif;
            white-space:nowrap; width:100%; }
.seas-tbl th, .seas-tbl td { padding:2px 6px; text-align:right;
                             border-bottom:1px solid #f0f0f0; }
.seas-tbl th { position:sticky; top:0; background:#fafafa; font-weight:600; z-index:2;
               text-align:right; }
/* box-shadow rather than border-left: border-collapse drops adjacent-cell
   borders depending on which side wins the merge, box-shadow always shows. */
.seas-tbl .dte { position:sticky; left:0; background:#fff; text-align:center;
                 font-weight:600; z-index:1; box-shadow: inset -2px 0 0 0 #374151; }
.seas-tbl th.dte { background:#fafafa; z-index:3; }
.seas-tbl .cur { font-weight:700; }
.seas-tbl .mean { background:#fffbea; font-weight:600; }
.seas-tbl tbody tr:hover td { background:#f0f9ff !important; }
.seas-cap { font-size:.72rem; font-weight:600; color:#6b7280; margin:0 0 4px; }
</style>
"""


def _seas_table_html(frame, style_fn, fmt, cur_label, mean_label):
    """One HTML table: DTE down the left, one column per crop year.

    `style_fn(value)` returns the inline CSS for a cell — a heatmap tint for
    levels, a diverging bar for changes — so both tables read with the same
    conditional formatting as the comprehensive grid."""
    head = "".join(
        f'<th class="{"mean" if c == mean_label else ""}">{c}</th>' for c in frame.columns)
    rows = []
    for dte, row in frame.iterrows():
        cells = []
        for c in frame.columns:
            v = row[c]
            cls = "mean" if c == mean_label else ("cur" if c == cur_label else "")
            if pd.isna(v):
                cells.append(f'<td class="{cls}">—</td>')
            else:
                cells.append(f'<td class="{cls}" style="{style_fn(v)}">{fmt(v)}</td>')
        rows.append(f'<tr><td class="dte">{int(dte)}</td>{"".join(cells)}</tr>')
    return (f'<div class="seas-wrap"><table class="seas-tbl">'
            f'<thead><tr><th class="dte">DTE</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


if view == "Data":
    _c, _ = st.columns([1, 3])
    with _c:
        table_step = st.slider("Days per row", 1, 14, 7, key="seas_step")
    tbl_cols = [l for l in labels_all if l in set(cmp_years) | {current}]
    mean_label = f"{n_avg}Y Mean" if avg_years else None

    # DTE descending, so the table runs earliest -> latest down the page, the
    # way the desk sheet does. .iloc for both the reversal and the step: purely
    # positional, so it cannot pick up label-slicing semantics from the index.
    lvl = aligned[tbl_cols].iloc[::-1].iloc[::table_step].copy()
    if mean_label:
        # reindex onto the rows the table actually has, NOT an independent
        # reversed-and-stepped slice of `band`. band is .dropna(how="all")-ed,
        # so when the averaged years carry no data all the way out to max_dte
        # its top row is lower than the table's, and the two step sequences
        # drift out of phase: they can have zero rows in common, so the whole
        # Mean column rendered as em-dashes.
        lvl[mean_label] = band["mean"].reindex(lvl.index)

    # Change between consecutive rows, i.e. over one table step, not one day —
    # taken after the resampling so it matches what is actually on screen.
    chg = lvl.diff()

    # Scales are global across the whole table, not per column: these columns
    # are the same basket in different crop years, so per-column scaling would
    # normalise away exactly the difference being looked for (23/24 built far
    # harder than 24/25). Levels tint against the level range, changes bar
    # against the largest absolute change anywhere in the table.
    vmin = float(lvl.min().min()) if lvl.notna().any().any() else 0.0
    vmax = float(lvl.max().max()) if lvl.notna().any().any() else 1.0
    cmax = float(chg.abs().max().max()) if chg.notna().any().any() else 1.0
    cmax = cmax if cmax > 0 else 1.0

    st.download_button("Download table (CSV)", data=lvl.rename_axis("DTE").to_csv().encode("utf-8"),
                       file_name=f"oi_seasonal_{bsig}_{unit_txt}.csv", mime="text/csv")
    st.markdown(SEAS_TBL_CSS, unsafe_allow_html=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f'<div class="seas-cap">Open Interest ({unit_txt})</div>', unsafe_allow_html=True)
        st.markdown(_seas_table_html(lvl, lambda v: _oi_heatmap_style(v, vmin, vmax),
                                     lambda v: f"{v:,.0f}", current, mean_label),
                    unsafe_allow_html=True)
    with t2:
        st.markdown(f'<div class="seas-cap">OI Change, {unit_txt} (per {table_step}d step)</div>',
                    unsafe_allow_html=True)
        st.markdown(_seas_table_html(chg, lambda v: _oi_chg_style(v, cmax),
                                     lambda v: f"{v:+,.0f}", current, mean_label),
                    unsafe_allow_html=True)

