# -*- coding: utf-8 -*-
"""FND Window — what the expiring contract does into and through First Notice.

Every contract of one delivery month is aligned on trading days to its own FND
(0 = FND, negative = before), so a roll that starts on a different calendar day
each year still lines up. First visual: the expiring month's OI liquidation
curve, one line per year, a percentile band across history and the live
contract on top — "are we rolling faster or slower than usual".
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

st.set_page_config(page_title="FND Window", page_icon=":material/event:", layout="wide")


# ---------------------------------------------------------------------------------------------
# Password gate. Password lives in .streamlit/secrets.toml locally, and in this app's own
# "Secrets" box on Streamlit Cloud (Settings > Secrets) - never committed, never hardcoded.
# A correct login is remembered for 24h in the browser's localStorage (see _check_password).
# ---------------------------------------------------------------------------------------------
@st.cache_resource
def _dm_login_bridge():
    """Invisible component that keeps the signed 24h login token in the browser's localStorage and hands it
    back to Python. Works through any proxy: it only uses the normal browser <-> app connection."""
    import tempfile
    from pathlib import Path
    import streamlit.components.v1 as components
    page = """<!doctype html><html><body><script>
var KEY = "_dm_auth", last = null;
function post(m) { m.isStreamlitMessage = true; window.parent.postMessage(m, "*"); }
window.addEventListener("message", function (ev) {
  if (!ev.data || ev.data.type !== "streamlit:render") return;
  var a = ev.data.args || {};
  try { if (a.write) localStorage.setItem(KEY, a.write); } catch (e) {}
  var v = ""; try { v = localStorage.getItem(KEY) || ""; } catch (e) {}
  if (v !== last) { last = v; post({type: "streamlit:setComponentValue", value: v, dataType: "json"}); }
});
post({type: "streamlit:componentReady", apiVersion: 1});
post({type: "streamlit:setFrameHeight", height: 0});
</script></body></html>"""
    d = Path(tempfile.gettempdir()) / "dm_auth_bridge"
    d.mkdir(exist_ok=True)
    (d / "index.html").write_text(page, encoding="utf-8")
    return components.declare_component("dm_auth_bridge", path=str(d))


def _check_password() -> bool:
    if st.session_state.get("_authed"):
        return True
    import hashlib, hmac, time
    real = None
    try:
        real = st.secrets.get("auth", {}).get("password")
    except Exception:
        pass
    typed = st.session_state.get("_pw_input") or ""
    ok = bool(real) and typed == real

    def _sig(exp):
        return hmac.new(real.encode(), f"v1|{exp}".encode(), hashlib.sha256).hexdigest()

    # A correct login is remembered for 24h as a signed token (HMAC of the expiry, keyed on the password
    # itself - changing the password logs everyone out) in the browser's localStorage. The invisible
    # component below reads it back on every visit, so refresh / new tab / reconnect don't ask again.
    exp = int(time.time()) + 86400
    stored = None
    try:
        stored = _dm_login_bridge()(write=(f"{exp}.{_sig(exp)}" if ok else None), key="_dm_login_bridge", default=None)
    except Exception:
        pass
    try:
        e, _, s = (stored or "").partition(".")
        if real and e.isdigit() and int(e) > time.time() and hmac.compare_digest(s, _sig(e)):
            ok = True
    except Exception:
        pass
    if ok:
        st.session_state["_authed"] = True
        return True
    st.markdown(
        "<div style='max-width:340px;margin:120px auto 0;text-align:center'>"
        "<div style='font-size:22px;font-weight:700;color:#0a2463;margin-bottom:18px'>FND Window</div>"
        "</div>", unsafe_allow_html=True)
    _l, _c, _r = st.columns([1, 1, 1])
    with _c:
        st.text_input("Password", type="password", label_visibility="collapsed",
                      placeholder="Password", key="_pw_input")
        if typed:
            st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()

try:
    from common import (COMMODITIES, MONTH_NAMES, _mtime, load_data, load_rollex,
                        _rollex_mtime, _freshness_row)
except ImportError:
    # Streamlit Cloud can hold a stale common.py in memory after a deploy
    import importlib
    import common
    importlib.reload(common)
    from common import (COMMODITIES, MONTH_NAMES, _mtime, load_data, load_rollex,
                        _rollex_mtime, _freshness_row)

NAVY = "#0a2463"

# Certified stocks live in the Daily Miner project and are read from the sibling folder when it is
# there (local runs, freshest). The deployed app cannot see that folder, so it falls back to
# Database/certs.parquet, the slim copy the builder refreshes on every run (sync_certs).
# per_lot turns a futures lot into the certs' own unit (KC certs are in 60 kg bags,
# LRC certs are already in 10 t lots, summed over all three grades).
DM_MAIN = Path(__file__).resolve().parents[3] / "Fundamental" / "Daily Miner" / "Database" / "Main"
CERTS = {
    "KC": dict(path=DM_MAIN / "KC" / "kc_certs.parquet", cols=["KC-TOT-TOT"],
               per_lot=37_500 * 0.45359237 / 60, unit="bags", name="Arabica"),
    "RC": dict(path=DM_MAIN / "RC" / "rc_certs.parquet", cols=["LRC-TOT-VG", "LRC-TOT-NT", "LRC-TOT-CL"],
               per_lot=1.0, unit="lots", name="Robusta"),
}
CERTS_COPY = Path(__file__).resolve().parent.parent / "Database" / "certs.parquet"
for _cfg in CERTS.values():
    if not _cfg["path"].exists():
        _cfg["path"] = CERTS_COPY


@st.cache_data(show_spinner=False)
def load_certs(commodity: str, mtime: float = 0.0):
    """Total ICE certified stock for KC or RC, daily, or None if not reachable."""
    cfg = CERTS.get(commodity)
    if cfg is None or not cfg["path"].exists():
        return None
    if cfg["path"] == CERTS_COPY:
        c = pd.read_parquet(CERTS_COPY)
        c = c[c["commodity"] == commodity]
        if c.empty:
            return None
        return c.assign(Date=pd.to_datetime(c["Date"])).set_index("Date")["certs"].astype("float64").sort_index()
    c = pd.read_parquet(cfg["path"], columns=["Date"] + cfg["cols"])
    c = c.assign(Date=pd.to_datetime(c["Date"])).set_index("Date")
    vals = c[cfg["cols"]].astype("float64")
    s = vals.sum(axis=1, min_count=1)[vals[cfg["cols"][0]].notna()]
    return s.dropna().sort_index()


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def market_calendar(commodity: str, mtime: float = 0.0) -> pd.DatetimeIndex:
    """The market's own trading days (every date any contract printed), extended
    with plain weekdays past the last stored date so the live contract, whose
    FND is still ahead, gets a correct negative count."""
    df = load_data(commodity, mtime)
    days = pd.DatetimeIndex(np.sort(df["Date"].unique()))
    horizon = df["LTD"].max() + pd.Timedelta(days=10)
    return days.append(pd.bdate_range(days[-1] + pd.Timedelta(days=1), horizon))


@st.cache_data(show_spinner=False)
def fnd_aligned(commodity: str, month: str, mtime: float = 0.0) -> pd.DataFrame:
    """Every contract of `month`, with `t` = trading days from its FND, and the
    next-listed contract's settlement and volume on the same date (the leg the
    expiring month rolls into: KC Z -> H, RC X -> F, LSU Z -> H)."""
    df = load_data(commodity, mtime)
    days = market_calendar(commodity, mtime)

    ltd = df.groupby("ice_symbol")["LTD"].first().sort_values()
    next_sym = pd.Series(ltd.index[1:], index=ltd.index[:-1])

    sub = df[df["month"] == month].copy()
    sub["FND"] = pd.to_datetime(sub["FND"])
    fnd_pos = days.searchsorted(sub["FND"])       # FND, or next trading day if FND itself is missing
    sub["t"] = days.get_indexer(sub["Date"]) - fnd_pos
    sub["t_ltd"] = days.searchsorted(sub["LTD"]) - fnd_pos
    sub["next_sym"] = sub["ice_symbol"].map(next_sym)

    nxt = df[["Date", "ice_symbol", "settlement", "volume"]].rename(
        columns={"ice_symbol": "next_sym", "settlement": "next_settle", "volume": "next_vol"})
    sub = sub.merge(nxt, on=["Date", "next_sym"], how="left")
    return sub[["Date", "ice_symbol", "year", "FND", "LTD", "t", "t_ltd", "open_interest",
                "volume", "settlement", "next_sym", "next_settle", "next_vol"]]


@st.cache_data(show_spinner=False)
def rollex_aligned(commodity: str, month: str, mtime: float = 0.0, rmtime: float = 0.0) -> pd.DataFrame:
    """Rollex daily return around each `month` contract's FND (year, t, ret).
    Rollex is roll-adjusted, so a contract switch inside the window is not a
    fake price jump."""
    ret = load_rollex(commodity, rmtime)
    if ret is None:
        return pd.DataFrame(columns=["year", "t", "ret"])
    days = market_calendar(commodity, mtime)
    pos = pd.Series(days.get_indexer(ret.index), index=ret.index)
    al = fnd_aligned(commodity, month, mtime)
    out = []
    for y, fnd in al.groupby("year")["FND"].first().items():
        t = pos - days.searchsorted(fnd)
        keep = (pos >= 0) & (t.abs() <= 120)
        out.append(pd.DataFrame({"year": y, "t": t[keep].values, "ret": ret[keep].values}))
    return pd.concat(out) if out else pd.DataFrame(columns=["year", "t", "ret"])


def to_matrix(long: pd.DataFrame, col: str, t_lo: int, t_hi: int, ffill: int = 3) -> pd.DataFrame:
    """t x delivery-year grid of `col`. A missing session inside a contract's
    life is carried from the day before (max `ffill`); nothing is invented past
    its last print."""
    piv = long.pivot_table(index="t", columns="year", values=col, aggfunc="last")
    piv = piv.reindex(range(min(t_lo, piv.index.min()), max(t_hi, piv.index.max()) + 1))
    last = long.dropna(subset=[col]).groupby("year")["t"].max()
    if ffill:
        piv = piv.ffill(limit=ffill)
    for y in piv.columns:
        piv.loc[piv.index > last.get(y, -10**6), y] = np.nan
    return piv.loc[t_lo:t_hi]


def from_start(piv: pd.DataFrame, t0: int) -> pd.Series:
    """Each year's value on the window's first day, or its first print after it."""
    return piv.loc[t0].fillna(piv.bfill().loc[t0])


@st.cache_data(show_spinner=False)
def next_fnd_month(commodity: str, mtime: float = 0.0) -> str:
    """Delivery month of the next contract to reach FND (the live roll)."""
    df = load_data(commodity, mtime)
    asof = df["Date"].max()
    fnd = df.groupby("ice_symbol").agg(FND=("FND", "first"), month=("month", "first"))
    up = fnd[pd.to_datetime(fnd["FND"]) >= asof - pd.Timedelta(days=5)]
    return up.sort_values("FND")["month"].iloc[0] if len(up) else "Z"


# ── Theme: copied from LSEG/Rollex/Dashboard/ICE_Rollex.py — keep the two in step ──
for _t in ("plotly", "plotly_white"):
    pio.templates[_t].layout.font.family = "Plus Jakarta Sans, sans-serif"
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?"
            "family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');</style>",
            unsafe_allow_html=True)
st.markdown("""
<style>
  :root { color-scheme: light !important; }
  html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
    background:#fafafa !important; color:#1a1a2e !important;
  }
  [data-testid="stSidebar"] { background:#f0f2f8 !important; border-right:1px solid #dfe3ee; color:#1a1a2e; }
  [data-testid="stSidebarUserContent"] { padding-top:1rem !important; }
  [data-testid="stHeader"]  { background:#fafafa !important; }
  .block-container { padding-top:3.2rem !important; max-width:1600px; }
  div[data-testid="stExpander"] {
    border:1px solid #dfe3ee !important; border-radius:10px !important; background:#ffffff;
  }
  h1, h2, h3, h4, h5, h6 { color:#0a2463 !important; }
  div[role="radiogroup"] { background:#eef0f6; padding:4px; border-radius:999px; gap:2px; display:inline-flex; flex-wrap:wrap; }
  div[role="radiogroup"] label { background:transparent !important; border-radius:999px !important; padding:4px 12px !important; margin:0 !important; }
  div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none; }
  div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p { font-size:12px !important; color:#5a6688 !important; }
  div[role="radiogroup"] label:has(input:checked) { background:#0a2463 !important; }
  div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p { color:#ffffff !important; font-weight:600; }
  hr { border:none !important; border-top:1px solid #dfe3ee !important; margin:.5rem 0 !important; }
  [data-testid="stRadio"] label { font-size:.82rem !important; }
  h1, h2, h3, h4, h5, h6 { color: #0a2463 !important; }

  /* Pill / segmented-control tabs */
  .stTabs [data-baseweb="tab-list"] { background:#eef0f6; padding:4px; border-radius:999px; gap:4px; display:inline-flex; width:fit-content; max-width:100%; }
  .stTabs [data-baseweb="tab"] { background:transparent !important; color:#5a6688 !important; border-radius:999px !important;
      padding:8px 20px !important; font-weight:600; border:none !important; margin:0 !important; height:auto !important; }
  .stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span { font-size:0.85rem !important; font-weight:600 !important; color:inherit !important; }
  .stTabs [aria-selected="true"] { background:#0a2463 !important; color:#ffffff !important; }
  .stTabs [data-baseweb="tab-highlight"] { display:none !important; }
  .stTabs [data-baseweb="tab-border"] { display:none !important; }

  /* Radio as pill / segmented control */
  div[role="radiogroup"] { background:#eef0f6; padding:4px; border-radius:999px; gap:2px; display:inline-flex; flex-wrap:wrap; }
  div[role="radiogroup"] label { background:transparent !important; border-radius:999px !important; padding:4px 12px !important; margin:0 !important; }
  div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child { display:none; }
  div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p { font-size:12px !important; color:#5a6688 !important; }
  div[role="radiogroup"] label:has(input:checked) { background:#0a2463 !important; }
  div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p { color:#ffffff !important; font-weight:600; }

  /* Collapsed sections: same soft row tint as the section headers, lift on hover */
  div[data-testid="stExpander"] { border:1px solid rgba(43,74,138,.10) !important; border-radius:12px !important; box-shadow:none !important;
    background:#ffffff !important; overflow:hidden; margin-bottom:8px; transition:box-shadow .18s, border-color .18s; }
  div[data-testid="stExpander"] details { border:0 !important; }
  div[data-testid="stExpander"] summary { padding:10px 14px !important;
    background:linear-gradient(90deg, rgba(43,74,138,.07) 0%, rgba(43,74,138,.025) 55%, rgba(43,74,138,0) 100%) !important; transition:background .18s; }
  div[data-testid="stExpander"] summary p {
    font-family:Plus Jakarta Sans, sans-serif !important; font-size:13.5px !important; font-weight:700 !important; color:#0a2463 !important; }
  div[data-testid="stExpander"] summary svg { color:#5a6688 !important; fill:#5a6688 !important; }
  div[data-testid="stExpander"]:hover { border-color:rgba(43,74,138,.22) !important; box-shadow:0 6px 18px rgba(10,36,99,.08) !important; }
  div[data-testid="stExpander"] summary:hover { background:linear-gradient(90deg, rgba(43,74,138,.12) 0%, rgba(43,74,138,.04) 60%, rgba(43,74,138,0) 100%) !important; }
  div[data-testid="stExpander"] summary:focus, div[data-testid="stExpander"] summary:focus-visible { outline:none !important; box-shadow:none !important; }
  div[data-testid="stExpander"] details[open] > summary { border-bottom:1px solid rgba(43,74,138,.08); }
  [class*="st-key-feat_"] { background:linear-gradient(160deg, rgba(250,252,255,.98) 0%, rgba(238,243,251,.94) 55%, rgba(228,236,249,.92) 100%);
    border:1px solid rgba(255,255,255,.95); border-radius:20px; padding:18px 22px 14px; margin:14px 0 18px;
    box-shadow:0 0 0 1px rgba(43,74,138,.10), 0 14px 40px rgba(30,58,110,.12), inset 0 1px 0 #ffffff; }
  .feat-h { display:flex; align-items:center; justify-content:center; gap:10px; margin:2px 0 12px; }
  [class*="st-key-feat_"] [data-testid="stCaptionContainer"], [class*="st-key-feat_"] [data-testid="stMarkdownContainer"] > p { text-align:center; }
  [class*="st-key-feat_"] div[role="radiogroup"] { min-height:40px; align-items:center; flex-wrap:nowrap !important; }
  [class*="st-key-feat_"] [data-testid="stNumberInput"] [data-baseweb="input"] { border-radius:12px !important; background:#ffffff !important;
    border:1px solid rgba(43,74,138,.14) !important; box-shadow:0 1px 2px rgba(10,36,99,.05); }
  [class*="st-key-feat_"] [data-testid="stNumberInput"] input { background:#ffffff !important; font-weight:600; color:#0a2463; }
  .rl-wrap { display:flex; justify-content:center; overflow-x:auto; margin-top:6px; }
  .rl-wrap table { border-collapse:separate !important; border-spacing:0; background:#ffffff; border:1px solid rgba(43,74,138,.12) !important;
    border-radius:14px; overflow:hidden; box-shadow:0 1px 2px rgba(10,36,99,.04), 0 10px 28px rgba(10,36,99,.08);
    font-family:Plus Jakarta Sans, sans-serif; font-variant-numeric:tabular-nums; }
  .rl-wrap th, .rl-wrap td { border:0 !important; border-bottom:1px solid rgba(43,74,138,.07) !important; padding:8px 11px !important; white-space:nowrap; }
  .rl-wrap thead th { background:linear-gradient(180deg,#f7f9fd,#edf2fa) !important; color:#1e3a6e !important;
    font-family:Plus Jakarta Sans, sans-serif !important; font-size:10.5px !important; font-weight:700 !important; letter-spacing:.08em; text-transform:uppercase; }
  .rl-wrap th.row_heading { color:#1e3a6e !important; font-weight:600 !important; background:#fafbfe; }
  .rl-wrap tbody tr:hover td, .rl-wrap tbody tr:hover th { background:rgba(43,74,138,.05) !important; }
  .feat-t { font-family:Plus Jakarta Sans, sans-serif; font-size:17px; font-weight:800; color:#0a2463; letter-spacing:-.01em; }
  .feat-tag { font-family:Plus Jakarta Sans, sans-serif; font-size:10px; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:#1e3a6e; background:rgba(255,255,255,.75); border:1px solid rgba(43,74,138,.18); border-radius:999px; padding:2px 9px; }
  .pt-head { display:flex; align-items:center; gap:12px; margin:18px 0 10px; padding:9px 14px; border-radius:12px;
    background:linear-gradient(90deg, rgba(43,74,138,.09) 0%, rgba(43,74,138,.035) 55%, rgba(43,74,138,0) 100%);
    border:1px solid rgba(43,74,138,.08); }
  .pt-title { font-family:Plus Jakarta Sans, sans-serif; font-size:14.5px; font-weight:700; color:#0a2463; letter-spacing:-.005em; }
  .pt-sub { font-weight:500; color:#7a86a8; margin-left:6px; font-size:13px; }
  .pt-meta { margin-left:auto; display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
  .pt-chip { font-family:Plus Jakarta Sans, sans-serif; font-size:11px; font-weight:600; color:#5a6688; background:#ffffff;
    border:1px solid rgba(43,74,138,.14); border-radius:999px; padding:2px 10px; white-space:nowrap; font-variant-numeric:tabular-nums; }
  /* section titles: same soft fading band as the VaR Monitor */
  .cot-h { font-family:Plus Jakarta Sans, sans-serif; font-size:12px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#1e3a6e;
           margin:14px 0 8px; padding:5px 12px; border-radius:8px; text-align:center;
           background:linear-gradient(90deg,rgba(92,104,136,0) 0%,rgba(92,104,136,.11) 20%,rgba(92,104,136,.11) 80%,rgba(92,104,136,0) 100%); }
  /* Sidebar radios: stacked pills, like Daily Miner's selector */
  [data-testid="stSidebar"] div[role="radiogroup"] { border-radius:22px; padding:4px; gap:3px; background:#e6eaf3; }
  [data-testid="stSidebar"] div[role="radiogroup"] label { padding:6px 14px !important; }

  /* Dropdowns and multiselect: white controls, navy tags, soft pop-up */
  [data-baseweb="select"] > div { background:#ffffff !important; color:#1a1a2e !important; border-color:#dfe3ee !important; border-radius:10px !important; }
  [data-baseweb="select"] > div:focus-within { border-color:#0a2463 !important; box-shadow:0 0 0 3px rgba(10,36,99,.12) !important; }
  span[data-baseweb="tag"], span[data-baseweb="tag"] div, span[data-baseweb="tag"] span { background-color:#0a2463 !important; color:#ffffff !important; }
  span[data-baseweb="tag"] svg { fill:#ffffff !important; }
  [data-baseweb="popover"] [data-baseweb="menu"] { background:#ffffff !important; border-radius:10px !important; box-shadow:0 10px 28px rgba(10,36,99,.14) !important; }
  [data-baseweb="popover"] [data-baseweb="menu"] li, [data-baseweb="popover"] [data-baseweb="menu"] li * { color:#1a1a2e !important; }
  [data-baseweb="popover"] [data-baseweb="menu"] li:hover { background:#eef0f6 !important; }
  [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="input"] { background:#ffffff !important; border-radius:10px !important; }

  /* Sidebar title (hero) */
  .sb-title { font-family:Plus Jakarta Sans, sans-serif; font-size:1.5rem; font-weight:600; color:#0a2463 !important; margin-bottom:2px; }
  .sb-caption { font-size:11px; color:#7a86a8 !important; margin-bottom:16px; line-height:1.4; }
  /* Sidebar polish: serif title with italic accent, small-caps labels, airy spacing, clean Rollex list */
  [data-testid="stSidebarUserContent"] { padding:1.6rem 1.35rem 2rem !important; }
  [data-testid="stSidebar"] .sb-title { font-family:Plus Jakarta Sans, sans-serif !important; font-size:1.12rem !important; font-weight:800 !important;
    text-transform:none !important; letter-spacing:-.01em; line-height:1.2; color:#0a2463 !important; margin:0 0 12px; white-space:nowrap; }
  .sb-rule { height:1px; background:rgba(43,74,138,.12); margin:0 0 20px; }
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, .sb-lbl {
    font-family:Plus Jakarta Sans, sans-serif !important; font-size:10.5px !important; font-weight:700 !important;
    letter-spacing:.14em !important; text-transform:uppercase; color:#7a86a8 !important; margin:0 0 6px 2px; }
  [data-testid="stSidebar"] .st-key-hist_decimal_toggle [data-testid="stWidgetLabel"] p { text-transform:none; letter-spacing:.01em !important; font-size:.8rem !important; color:#1e3a6e !important; }
  .sb-note { font-family:Plus Jakarta Sans, sans-serif; font-size:11px; color:#8a94a8; letter-spacing:.02em; margin:2px 0 8px 4px; font-variant-numeric:tabular-nums; }
  [data-testid="stSidebar"] hr { border-top:1px solid rgba(43,74,138,.10) !important; margin:1.1rem 0 !important; }
  [data-testid="stSidebar"] [data-testid="stElementContainer"] { margin-bottom:2px; }
  .sb-rx { width:100%; border-collapse:separate; border-spacing:0; font-family:Plus Jakarta Sans, sans-serif; font-size:12px;
    background:rgba(255,255,255,.55); border:1px solid rgba(255,255,255,.9); border-radius:12px; overflow:hidden;
    box-shadow:0 1px 2px rgba(10,36,99,.05), inset 0 1px 0 rgba(255,255,255,.9); }
  .sb-rx td { border:0 !important; border-bottom:1px solid rgba(43,74,138,.07) !important; padding:5px 12px; }
  .sb-rx tr:last-child td { border-bottom:0 !important; }
  .sb-rx td.c { font-weight:700; color:#1e3a6e; letter-spacing:.04em; width:46px; }
  .sb-rx td.d { color:#5a6688; text-align:right; font-variant-numeric:tabular-nums; }
  /* Main-area widget labels ("Include Index in spec legs?" etc.) */
  [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
    font-family:Plus Jakarta Sans, sans-serif !important; font-size:12.5px !important; font-weight:600 !important;
    color:#1e3a6e !important; letter-spacing:.01em; }
  [data-testid="stMain"] [data-testid="stCheckbox"] label p { font-family:Plus Jakarta Sans, sans-serif !important; font-weight:600 !important; color:#1e3a6e !important; }

  /* ── Ice theme, same as Daily Miner: fonts, glass sidebar, glass pill tracks, glass dropdowns, icy tables ── */
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea {
    font-family:Plus Jakarta Sans, sans-serif !important; -webkit-font-smoothing:antialiased; }
  .stTabs [data-baseweb="tab"] p, div[role="radiogroup"] label p, [data-testid="stCaptionContainer"] {
    font-family:Plus Jakarta Sans, sans-serif !important; letter-spacing:.01em; }
  [data-testid="stSidebar"] { background:linear-gradient(160deg, rgba(246,248,253,.92) 0%, rgba(233,238,248,.88) 55%, rgba(240,236,246,.9) 100%) !important;
    -webkit-backdrop-filter:blur(24px) saturate(160%); backdrop-filter:blur(24px) saturate(160%);
    border-right:1px solid rgba(255,255,255,.75) !important; box-shadow:inset -1px 0 0 rgba(10,36,99,.06), 6px 0 28px rgba(10,36,99,.06); }
  .sb-title { font-family:Plus Jakarta Sans, sans-serif; }
  /* pill tracks (tabs, radios): frosted glass, navy gradient pick, soft hover */
  div[role="radiogroup"], .stTabs [data-baseweb="tab-list"] {
    background:linear-gradient(160deg, rgba(250,252,255,.97), rgba(236,242,251,.93) 60%, rgba(229,237,249,.9)) !important;
    border:1px solid rgba(43,74,138,.14) !important; box-shadow:0 4px 16px rgba(30,58,110,.07), inset 0 1px 0 rgba(255,255,255,.95) !important; }
  div[role="radiogroup"] label:has(input:checked), .stTabs [aria-selected="true"] {
    background:linear-gradient(180deg, #2b4a8a, #1e3a6e) !important; color:#ffffff !important;
    box-shadow:0 2px 8px rgba(30,58,110,.28), inset 0 1px 0 rgba(255,255,255,.18) !important; }
  div[role="radiogroup"] label, .stTabs [data-baseweb="tab"] { transition:background .15s, transform .15s; }
  div[role="radiogroup"] label:not(:has(input:checked)):hover, .stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover { background:rgba(43,74,138,.07) !important; transform:translateY(-1px); }
  /* dropdowns: glass pill, lift on hover, burgundy focus ring + caret */
  [data-testid="stMain"] [data-baseweb="select"], [data-testid="stSidebar"] [data-baseweb="select"] { background:transparent !important; box-shadow:none !important; }
  [data-baseweb="select"] > div { background:linear-gradient(180deg, rgba(255,255,255,.78), rgba(238,242,250,.62)) !important;
    -webkit-backdrop-filter:blur(14px) saturate(170%); backdrop-filter:blur(14px) saturate(170%);
    border:1px solid rgba(255,255,255,.9) !important; border-radius:999px !important; min-height:34px;
    box-shadow:0 1px 2px rgba(10,36,99,.06), 0 6px 18px rgba(10,36,99,.08), inset 0 1px 0 rgba(255,255,255,.95); transition:box-shadow .18s, transform .18s; }
  [data-baseweb="select"] > div:hover { box-shadow:0 0 0 3px rgba(10,36,99,.07), 0 8px 22px rgba(10,36,99,.14), inset 0 1px 0 rgba(255,255,255,.95); transform:translateY(-1px); }
  [data-baseweb="select"] > div:focus-within { box-shadow:0 0 0 3px rgba(123,30,58,.16), 0 8px 22px rgba(10,36,99,.14) !important; }
  [data-baseweb="select"] svg { fill:#7b1e3a !important; }
  [data-baseweb="select"] > div:has([data-baseweb="tag"]) { border-radius:16px !important; padding:4px 6px !important; }
  [data-baseweb="tag"] { height:24px !important; margin:2px 3px !important; border-radius:8px !important; }
  [data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul {
    background:rgba(255,255,255,.82) !important; -webkit-backdrop-filter:blur(18px) saturate(170%); backdrop-filter:blur(18px) saturate(170%);
    border:1px solid rgba(255,255,255,.9) !important; border-radius:14px !important; box-shadow:0 14px 40px rgba(10,36,99,.18) !important; }
  [data-baseweb="popover"] li:hover { background:rgba(43,74,138,.07) !important; }
  [data-baseweb="popover"] li[aria-selected="true"] { background:rgba(123,30,58,.10) !important; }
  /* cards and buttons */
  div[data-testid="stExpander"] { background:#ffffff; border:1px solid #e3e7f0 !important; border-radius:14px !important; box-shadow:0 1px 3px rgba(10,36,99,.06); }
  [data-testid="stMain"] [data-testid="stPlotlyChart"] { background:#ffffff; border:1px solid rgba(43,74,138,.10);
    border-radius:18px; padding:6px 8px; overflow:hidden; box-shadow:0 1px 2px rgba(10,36,99,.04), 0 4px 14px rgba(30,58,110,.06), inset 0 1px 0 #ffffff;
    transition:box-shadow .2s; }
  [data-testid="stMain"] [data-testid="stPlotlyChart"] > div, [data-testid="stMain"] [data-testid="stPlotlyChart"] .js-plotly-plot,
  [data-testid="stMain"] [data-testid="stPlotlyChart"] .main-svg { border-radius:16px; }
  [data-testid="stMain"] [data-testid="stPlotlyChart"]:hover { box-shadow:0 1px 2px rgba(10,36,99,.05), 0 8px 20px rgba(30,58,110,.09), inset 0 1px 0 #ffffff; }
  .stButton > button, .stDownloadButton > button { border-radius:999px; border:1px solid rgba(43,74,138,.18); background:linear-gradient(180deg,#ffffff,#eef2fa);
    color:#0a2463; font-weight:600; box-shadow:0 1px 2px rgba(10,36,99,.06); transition:box-shadow .18s, transform .18s; }
  .stButton > button:hover, .stDownloadButton > button:hover { transform:translateY(-1px); box-shadow:0 8px 20px rgba(10,36,99,.14); border-color:rgba(43,74,138,.35); color:#0a2463; }
  /* icy tables: frosted card, pale gradient headers (colour bands keep their own colour), zebra rows, soft hover */
  [data-testid="stMain"] div:has(> table.rtbl) { border:1px solid rgba(43,74,138,.12); border-radius:14px; background:#ffffff; }
  [data-testid="stMain"] div:has(> table.rtbl) { box-shadow:0 1px 2px rgba(10,36,99,.04), 0 8px 24px rgba(10,36,99,.06); }
  /* Rollex: use the full width */
  .block-container { max-width:100% !important; padding-left:2rem !important; padding-right:2rem !important; }
  /* FND Window additions */
  .n-note { color:#9aa3b8; font-weight:500; letter-spacing:.04em; margin-left:8px; text-transform:none; }
  .ctl-gap { height:4px; }
</style>""", unsafe_allow_html=True)

WINE, TEAL, AMBER = "#7b1e3a", "#2c8f9d", "#e8a020"
_D = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#1a1a2e", size=11),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(43,74,138,.25)",
                    font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0a2463")))
# Same heatmap finishes as Rollex: wine below zero, white at zero, teal above; one-sided white to teal
RET_CS = [[0, "#c98a9c"], [0.35, "#f1e4e8"], [0.5, "#ffffff"], [0.65, "#e3f4f6"],
          [0.82, "#a9dbe2"], [1, "#2c8f9d"]]
VOL_CS = [[0, "#ffffff"], [0.35, "#e3f4f6"], [0.7, "#a9dbe2"], [1, "#2c8f9d"]]
VOL_CS_R = [[0, "#2c8f9d"], [0.3, "#a9dbe2"], [0.65, "#e3f4f6"], [1, "#ffffff"]]
# Softer versions for the year x month grids, so black values stay easy to read
GRID_CS = [[0, "#ffffff"], [0.5, "#e6f4f6"], [1, "#9fd2da"]]
GRID_CS_R = [[0, "#9fd2da"], [0.5, "#e6f4f6"], [1, "#ffffff"]]
GRID = "rgba(10,36,99,0.05)"
AXIS_FONT = dict(size=10, color="#1e3a6e")

MARKET_NAMES = {"KC": "KC Arabica", "RC": "RC Robusta", "CC": "CC Cocoa (ICE)", "LCC": "LCC Cocoa (Liffe)",
                "SB": "SB Sugar #11", "LSU": "LSU White Sugar", "CT": "CT Cotton"}


def lbl(t: str, note: str = "") -> None:
    """Section header, same .cot-h band as Rollex, with an optional quiet note."""
    n = f"<span class='n-note'>{note}</span>" if note else ""
    st.markdown(f"<div class='cot-h'>{t}{n}</div>", unsafe_allow_html=True)


def chart(fig) -> None:
    # One decimal on every hover that does not set its own format (violin stats, unified hovers)
    for tr in fig.data:
        if "yhoverformat" in tr and tr.yhoverformat is None:
            tr.yhoverformat = ",.1f"
    # Figures that set their own width (the year x month grids) keep it; the rest fill the column
    st.plotly_chart(fig, width="content" if fig.layout.width else "stretch", config={"displayModeBar": False})


def _ord(x: float) -> str:
    n = int(round(x))
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _tl(t: int) -> str:
    return "FND" if t == 0 else f"T{t}"


def pill(store: str, label: str, options: list, default, fmt=str):
    """Pill radio whose choice survives switching views (a widget not drawn on a
    run loses its state, so the value is kept under its own key)."""
    cur = st.session_state.get(store, default)
    idx = options.index(cur) if cur in options else options.index(default)
    v = st.radio(label, options, index=idx, horizontal=True, format_func=fmt, key=f"w_{store}_{view}")
    st.session_state[store] = v
    return v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sb-title'>FND Window</div><div class='sb-rule'></div>", unsafe_allow_html=True)
    comm = st.selectbox("Market", list(COMMODITIES), format_func=lambda c: MARKET_NAMES[c], key="sb_comm")
    mt = _mtime(comm)
    raw = load_data(comm, mt)
    months = sorted(raw["month"].unique(), key=lambda m: "FGHJKMNQUVXZ".index(m))
    default_m = next_fnd_month(comm, mt)
    month = st.selectbox("Delivery month", months, index=months.index(default_m),
                         format_func=lambda m: f"{m}  ·  {MONTH_NAMES[m]}", key=f"sb_month_{comm}")
    sb_ctl = st.container()          # filled once the open view is known

    st.markdown("---")
    _rows = ""
    for _c in COMMODITIES:
        try:
            _px, _oi = _freshness_row(_c, _mtime(_c))
        except Exception:
            continue
        if pd.notna(_oi):
            _rows += f"<tr><td class='c'>{_c}</td><td class='d'>{_oi:%d %b %Y}</td></tr>"
    st.markdown(f"<div class='sb-lbl'>Open interest as of</div><table class='sb-rx'>{_rows}</table>",
                unsafe_allow_html=True)
    st.markdown("<div class='sb-note' style='margin-top:12px'>Aligned on trading days to FND · Source: LSEG, ICE certs</div>",
                unsafe_allow_html=True)

al = fnd_aligned(comm, month, mt)
if al.empty:
    st.warning("No contracts for this month.")
    st.stop()

cfg = CERTS.get(comm)
certs = load_certs(comm, cfg["path"].stat().st_mtime) if cfg and cfg["path"].exists() else None

# ── Navigation ────────────────────────────────────────────────────────────────
VIEWS = ["FND Window"] + (["Comprehensive View", "Study"]
                          if certs is not None else [])
_v = st.session_state.get("view_store", VIEWS[0])
if "rh_on" not in st.session_state:
    st.session_state["rh_on"] = False          # hover panels off until asked for: the page loads light


def _rh_flip():
    st.session_state["rh_on"] = not st.session_state["rh_on"]


_nav = st.columns([5, 1.4], vertical_alignment="center")
with _nav[0]:
    view = st.radio("View", VIEWS, index=VIEWS.index(_v) if _v in VIEWS else 0, horizontal=True,
                    key=f"nav_{len(VIEWS)}", label_visibility="collapsed")
st.session_state["view_store"] = view
if view == "Comprehensive View":
    with _nav[1]:
        st.markdown("""<style>.st-key-rh_toggle button { background:transparent !important; border:0 !important; box-shadow:none !important; }
          .st-key-rh_toggle button p { color:#ffffff !important; font-weight:700 !important; letter-spacing:.02em; white-space:nowrap; }
          .st-key-rh_toggle { border:1px solid rgba(42,160,150,.45); border-radius:999px; padding:0 8px; width:auto !important;
            margin-left:auto; background:linear-gradient(135deg, #22396f 0%, #2b6e69 100%);
            box-shadow:0 0 0 1px rgba(42,160,150,.18), 0 0 12px rgba(42,160,150,.25); }
          .st-key-rh_toggle button { min-height:32px !important; padding:2px 10px !important; }</style>""",
                    unsafe_allow_html=True)
        with st.container(key="rh_toggle"):
            st.button("Deactivate Hover" if st.session_state["rh_on"] else "Activate Hover", on_click=_rh_flip,
                      help="Charts behind every grid cell: hover to preview, click to pin")
st.markdown("<div class='ctl-gap'></div>", unsafe_allow_html=True)

# ── Controls: in the sidebar, only what the open view uses ─────────────────────
t_lo, t_hi, unit = 40, 20, "% of start"
t_chk, t_exit = -5, -1
with sb_ctl:
    st.markdown("---")
    if view == "FND Window":
        # Kept under their own keys so the values survive switching views
        t_lo = st.slider("Days before FND", 10, 80, st.session_state.get("t_lo_n", 40), 5, key="w_t_lo")
        t_hi = st.slider("Days after FND", 0, 30, st.session_state.get("t_hi_n", 20), key="w_t_hi")
        st.session_state["t_lo_n"], st.session_state["t_hi_n"] = t_lo, t_hi
        unit = pill("unit", "Open interest as", ["% of start", "Lots"], "% of start")
    else:
        n_chk = st.slider("Certs measured at (days before FND)", 1, 80,
                          st.session_state.get("t_chk_n", 5), key=f"w_t_chk_{view}")
        n_exit = st.slider("Spread move ends at (days before FND)", 0, 40,
                           st.session_state.get("t_exit_n", 1), key=f"w_t_exit_{view}")
        st.session_state["t_chk_n"], st.session_state["t_exit_n"] = n_chk, n_exit
        t_chk, t_exit = -n_chk, -n_exit
        if view == "Comprehensive View":
            yrs = pill("yrs", "Years in the grids", ["Last 10", "All"], "Last 10")
if t_exit <= t_chk:
    st.info("The spread move has to end after the day certs are measured.")
    st.stop()


# ── Shared context ────────────────────────────────────────────────────────────
asof = raw["Date"].max()
years = sorted(al["year"].unique())
live_year = al.loc[al["LTD"] >= asof, "year"].min()      # NaN once every contract has expired
cur_year = int(live_year) if pd.notna(live_year) else years[-1]
hist_years = [y for y in years if y != cur_year and al.loc[al["year"] == y, "LTD"].iloc[0] < asof]
prev_year = max([y for y in hist_years if y < cur_year], default=None)
med_ltd = int(al.loc[al["year"].isin(hist_years)].groupby("year")["t_ltd"].first().median())
nxt_m = raw.loc[raw["ice_symbol"] == al["next_sym"].dropna().iloc[-1], "month"].iloc[0]
cur_lbl = f"{month}{cur_year % 100:02d}"
past_note = f"vs {len(hist_years)} past {MONTH_NAMES[month]} contracts"

# Plain spread in the market's own price units: expiring minus next
al["spread"] = al["settlement"] - al["next_settle"]
sp_plot = to_matrix(al, "spread", -t_lo, t_hi)
if certs is not None:
    # Open positions as a multiple of certified stock: 20x = open positions are
    # twenty times what is available to deliver against them
    al["cover"] = ((al["open_interest"] * cfg["per_lot"]).to_numpy()
                   / certs.reindex(al["Date"], method="ffill").to_numpy())
    _cm = al.loc[al["year"] == cur_year].sort_values("Date")["cover"].dropna()
    now_mult = _cm.iloc[-1] if len(_cm) else np.nan


# ── Charts ────────────────────────────────────────────────────────────────────
def band_chart(plot: pd.DataFrame, y_title: str, y_fmt: str, y_suf: str = "",
               zero_line: bool = False, tozero: bool = False, x_max: int | None = None,
               height: int = 400) -> go.Figure:
    """Rollex seasonality look: past years as 10-90 / 25-75 bands with thin
    min/max, dotted teal median, last year navy, live contract wine."""
    hy = [y for y in hist_years if y in plot]
    hist = plot[hy]
    n = hist.notna().sum(axis=1)
    # Values go to the chart at one decimal: the unified hover reads the data, not the hovertemplate
    q = hist.quantile([0.1, 0.25, 0.5, 0.75, 0.9], axis=1).T[n >= 3].round(1)
    lo_, hi_ = hist.min(axis=1)[n >= 3].round(1), hist.max(axis=1)[n >= 3].round(1)
    plot = plot.round(1)
    x = q.index
    hf = ("+" if y_fmt.startswith("+") else "") + ",.1f"      # hover always shows one decimal
    hv = lambda nm: f"{nm} <b>%{{y:{hf}}}{y_suf}</b><extra></extra>"

    fig = go.Figure()
    for lo, hi, a, nm in ((0.1, 0.9, 0.12, "10-90th pct"), (0.25, 0.75, 0.24, "25-75th pct")):
        fig.add_trace(go.Scatter(x=x, y=q[hi], line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=q[lo], line=dict(width=0), fill="tonexty",
                                 fillcolor=f"rgba(44,143,157,{a})", name=nm, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=hi_, name="Max", line=dict(color="rgba(26,122,26,0.30)", width=1),
                             hovertemplate=hv("max")))
    fig.add_trace(go.Scatter(x=x, y=lo_, name="Min", line=dict(color="rgba(192,57,43,0.30)", width=1),
                             hovertemplate=hv("min")))
    fig.add_trace(go.Scatter(x=x, y=q[0.5], name="Median", line=dict(color=TEAL, width=2, dash="dot"),
                             hovertemplate=hv("median")))
    if prev_year in plot:
        s = plot[prev_year].dropna()
        nm = f"{month}{prev_year % 100:02d}"
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=nm, line=dict(color=NAVY, width=1.4),
                                 hovertemplate=hv(nm)))
    cur = plot[cur_year].dropna() if cur_year in plot else pd.Series(dtype=float)
    if len(cur):
        fig.add_trace(go.Scatter(x=cur.index, y=cur.values, name=cur_lbl, line=dict(color=WINE, width=2.4),
                                 hovertemplate=hv(cur_lbl)))
        fig.add_trace(go.Scatter(x=cur.index[-1:], y=cur.values[-1:], mode="markers+text", showlegend=False,
                                 marker=dict(color=WINE, size=8), text=[f"{cur.iloc[-1]:{y_fmt}}{y_suf}"],
                                 textposition="middle right", textfont=dict(size=11, color=WINE),
                                 hoverinfo="skip"))

    x_hi = t_hi if x_max is None else x_max
    fig.add_vline(x=0, line=dict(color="rgba(10,36,99,0.35)", width=1, dash="dot"))
    fig.add_annotation(x=0, y=1, yref="paper", yshift=12, text="<b>FND</b>", showarrow=False,
                       font=dict(size=10, color=NAVY))
    if med_ltd != 0 and -t_lo <= med_ltd <= x_hi:
        fig.add_vline(x=med_ltd, line=dict(color="rgba(10,36,99,0.18)", width=1, dash="dot"))
        fig.add_annotation(x=med_ltd, y=1, yref="paper", yshift=12, text="LTD", showarrow=False,
                           font=dict(size=10, color="#7a86a8"))
    fig.update_traces(yhoverformat=("+" if y_fmt.startswith("+") else "") + ",.1f")
    fig.update_layout(
        height=height, margin=dict(t=30, b=60, l=50, r=16), hovermode="x unified",
        legend=dict(orientation="h", y=-0.16, yanchor="top", x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(range=[-t_lo, x_hi], tickformat="+d", dtick=5, showgrid=False, tickfont=AXIS_FONT,
                   title=dict(text="Trading days to FND", font=dict(size=10, color="#7a86a8")),
                   hoverformat="+d"),
        yaxis=dict(title=dict(text=y_title, font=dict(size=10, color="#7a86a8")), ticksuffix=y_suf or None,
                   showgrid=True, gridcolor=GRID, zeroline=zero_line, zerolinecolor="rgba(10,36,99,0.2)",
                   rangemode="tozero" if tozero else "normal", tickfont=dict(size=10, color="#7a86a8"),
                   hoverformat=hf),
        **_D)
    return fig


def _live_marks(fig, piv, pts) -> None:
    """Orange dashed outline on the live contract's cell: its values run only to its latest session."""
    if "live" not in pts:
        return
    cols, rows = list(piv.columns), list(piv.index)
    for r in pts[pts["live"]].itertuples():
        if r.m in cols and str(r.y) in rows:
            j, i = cols.index(r.m), rows.index(str(r.y))
            fig.add_shape(type="rect", x0=j - 0.47, x1=j + 0.47, y0=i - 0.44, y1=i + 0.44,
                          line=dict(color="#ea580c", width=1.6, dash="dot"))


def expiry_heatmap(pts, col, fmt, cs, suffix="", diverging=False):
    """Delivery year (rows, newest first) x contract month, Rollex heatmap finish.
    The colour range is clipped at the 90th percentile so one wild year does not
    wash out the rest."""
    order = "FGHJKMNQUVXZ"
    piv = pts.pivot_table(index="y", columns="m", values=col, aggfunc="last")
    piv = piv[sorted(piv.columns, key=order.index)].sort_index()
    piv.index = piv.index.astype(str)
    hist = pts[~pts["live"]] if "live" in pts else pts
    piv.loc["Avg"] = hist.groupby("m")[col].mean().reindex(piv.columns)    # mean of finished expiries only
    text = piv.map(lambda v: "" if pd.isna(v) else f"{v:{fmt}}{suffix}")
    vals = pts[col].dropna()
    if diverging:
        zmax = float(vals.abs().quantile(0.9)) or 1.0
        rng = dict(zmin=-zmax, zmax=zmax, zmid=0)
    else:
        rng = dict(zmin=float(vals.quantile(0.05)), zmax=float(vals.quantile(0.95)))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=list(piv.columns), y=[str(y) for y in piv.index], text=text.values,
        texttemplate="%{text}", textfont=dict(size=10.5, color="#111111"), colorscale=cs,
        showscale=False, xgap=3, ygap=3, hovertemplate="%{x}%{y}  <b>%{text}</b><extra></extra>", **rng))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=len(piv) - 1.5, y1=len(piv) - 1.5,
                  line=dict(color="rgba(10,36,99,.35)", width=1.2))
    _live_marks(fig, piv, pts)
    fig.update_layout(
        height=24 * len(piv) + 40, margin=dict(t=24, b=6, l=36, r=2),
        xaxis=dict(side="top", showgrid=False, tickfont=dict(size=11, color="#1e3a6e")),
        yaxis=dict(showgrid=False, type="category", autorange="reversed", tickfont=dict(size=9.5, color="#1e3a6e")),
        **_D)
    return fig


def bar_heatmap(pts, col, fmt) -> go.Figure:
    """Same grid and sizing as expiry_heatmap, but each pale cell carries a mini
    bar from its centre: wine to the left for a fall, teal to the right for a
    rise. Bars scale to the 90th percentile of absolute moves, so one outlier
    does not flatten the rest. Drawn as five traces (no per-cell shapes) so
    the chart redraws quickly."""
    order = "FGHJKMNQUVXZ"
    piv = pts.pivot_table(index="y", columns="m", values=col, aggfunc="last")
    piv = piv[sorted(piv.columns, key=order.index)].sort_index()
    piv.index = piv.index.astype(str)
    hist = pts[~pts["live"]] if "live" in pts else pts
    piv.loc["Avg"] = hist.groupby("m")[col].mean().reindex(piv.columns)    # mean of finished expiries only
    nr, nc = piv.shape
    vmax = float(pts[col].abs().quantile(0.9)) or 1.0
    ii, jj = np.meshgrid(np.arange(nr), np.arange(nc), indexing="ij")
    v = piv.to_numpy().ravel()
    ii, jj = ii.ravel(), jj.ravel()
    ok = ~np.isnan(v)
    w = np.minimum(np.abs(v) / vmax, 1.0) * 0.46
    keys = np.array([f"{m}{y}" for y in piv.index for m in piv.columns])
    txt = np.array(["" if np.isnan(x) else f"{x:{fmt}}" for x in v])

    fig = go.Figure(go.Heatmap(z=np.zeros((nr, nc)), x=list(range(nc)), y=list(range(nr)),
                               colorscale=[[0, "#f4f6fb"], [1, "#f4f6fb"]], showscale=False,
                               xgap=3, ygap=3, hoverinfo="skip"))
    for sel, colr in ((ok & (v >= 0), "rgba(44,143,157,.28)"), (ok & (v < 0), "rgba(123,30,58,.20)")):
        neg = colr.startswith("rgba(123")
        fig.add_trace(go.Bar(x=w[sel], y=ii[sel], base=(jj[sel] - w[sel]) if neg else jj[sel], orientation="h",
                             width=0.68, marker=dict(color=colr, line=dict(width=0)), hoverinfo="skip",
                             showlegend=False))
    # thin centre tick per column, one trace with breaks between columns
    cx = np.repeat(np.arange(nc), 3).astype(float)
    cy = np.tile([-0.5, nr - 0.5, np.nan], nc)
    cx[2::3] = np.nan
    fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color="rgba(10,36,99,.14)", width=1),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=jj[ok], y=ii[ok], mode="text", text=txt[ok], customdata=keys[ok],
                             textfont=dict(size=10.5, color="#111111", family="Plus Jakarta Sans, sans-serif"),
                             texttemplate="%{text}", showlegend=False,
                             hovertemplate="%{customdata}  <b>%{text}</b><extra></extra>"))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=nr - 1.5, y1=nr - 1.5,
                  line=dict(color="rgba(10,36,99,.35)", width=1.2))
    _live_marks(fig, piv, pts)
    fig.update_layout(
        height=24 * nr + 40, margin=dict(t=24, b=6, l=36, r=2),
        barmode="overlay", bargap=0,
        xaxis=dict(side="top", showgrid=False, zeroline=False, tickvals=list(range(nc)), ticktext=list(piv.columns),
                   range=[-0.5, nc - 0.5], tickfont=dict(size=11, color="#1e3a6e")),
        yaxis=dict(showgrid=False, zeroline=False, tickvals=list(range(nr)), ticktext=[str(y) for y in piv.index],
                   range=[nr - 0.5, -0.5], tickfont=dict(size=9.5, color="#1e3a6e")), **_D)
    return fig


# Hover panels: same mechanism as the COT_ALL recap (LSEG/COT_ALL/Dashboard/cot_app.py, _RH_CSS):
# every cell holds a hidden glass panel with an inline SVG, shown by pure CSS on hover.
SP_CSS = """<style>
.sp-wrap { overflow-x:auto; background:#ffffff; border:1px solid rgba(43,74,138,.10); padding:4px 4px 2px; }
.sp-tbl { border-collapse:separate; border-spacing:3px; font-family:Plus Jakarta Sans, sans-serif; font-variant-numeric:tabular-nums; }
.sp-tbl { border-spacing:2px !important; }
.sp-tbl th { font-size:11px; font-weight:600; color:#1e3a6e; padding:2px 2px; text-align:center; background:transparent; border:0; }
.sp-tbl th.yr { font-size:9.5px; font-weight:500; text-align:right; padding-right:4px; }
.sp-tbl { width:100%; table-layout:fixed; }
.sp-tbl th:first-child { width:34px; }
.sp-tbl td { position:relative; height:21px; padding:0; border:0; background:#f4f6fb; overflow:hidden;
  text-align:center; font-size:10.5px; color:#111111; white-space:nowrap; }
.sp-tbl td .bar { position:absolute; top:4px; bottom:4px; }
.sp-tbl td .mid { position:absolute; left:50%; top:2px; bottom:2px; width:1px; background:rgba(10,36,99,.14); }
.sp-tbl td .v { position:relative; z-index:1; }
.sp-tbl tr.avg th, .sp-tbl tr.avg td { font-weight:800; color:#0a2463; }
.rh-leg { display:flex; gap:16px; font-size:11px; color:#5a6688; margin:0 0 4px 4px; }
.rh-leg span { display:inline-flex; align-items:center; gap:6px; }
.rh-leg i { display:inline-block; }
.rh-leg i.ln { width:20px; height:0; border-top:2px solid #ea580c; }
.rh-leg i.ln.dash { border-top:2px dashed #4f46e5; }
.rh-leg i.bx { width:14px; height:10px; border-radius:3px; }
.rh-leg i.bx.in { background:rgba(99,102,241,.34); } .rh-leg i.bx.out { background:rgba(99,102,241,.14); }
.sp-tbl tr.avg td, .sp-tbl tr.avg th { border-top:2px solid rgba(10,36,99,.30); }
.sp-tbl td.rh-hov { cursor:pointer; transition:box-shadow .15s; }
.sp-tbl td.rh-hov:hover { box-shadow:inset 0 0 0 1.5px rgba(42,160,150,.8); }
.sp-tbl .rh-pop { display:none; }     /* each cell's panel is copied into the strip under the grids */
.rh-t { font-size:14px; font-weight:800; color:#0a2463; margin:0 0 6px; }
.rh-stats { display:flex; flex-wrap:wrap; gap:8px 12px; margin:0 0 8px; font-size:12px; }
.rh-s { background:#ffffff; border:1px solid rgba(43,74,138,.12); border-radius:999px; padding:3px 11px; }
.rh-s b { color:#7a86a8; font-weight:600; margin-right:5px; font-size:10.5px; text-transform:uppercase; letter-spacing:.04em; }
.rh-s i { font-style:normal; font-weight:800; color:#0a2463; }
.rh-s i.up { color:#1f8a9c; } .rh-s i.dn { color:#7b1e3a; }
/* strip under the grids (click helper below): the hovered cell as a preview, pinned cells side by side */
.sp-tbl td.rh-pinned, .sp-tbl td.rh-pinned:hover { outline:2px solid #8b5cf6; outline-offset:-2px;
  box-shadow:0 0 4px rgba(139,92,246,.28) !important; background-color:#f4f0ff !important; z-index:2; }
.sp-tbl td.rh-pinned::after { content:attr(data-pin); position:absolute; top:-1px; right:-1px; min-width:12px; height:12px;
  padding:0 2px; border-radius:6px; background:#7c3aed; color:#ffffff; font-size:8.5px; font-weight:800; line-height:12px;
  text-align:center; z-index:3; }
.rh-num { display:inline-block; min-width:18px; height:18px; margin-right:8px; border-radius:9px; background:#7c3aed; color:#ffffff;
  font-size:11px; font-weight:800; line-height:18px; text-align:center; vertical-align:1px; }
.sp-tbl td.live { outline:1.6px dashed #ea580c; outline-offset:-2px; font-style:italic; }
.rh-s.live { border-color:rgba(10,36,99,.35); background:#f3f6fc; }
.rh-s em { font-style:normal; font-weight:500; color:#7a86a8; font-size:10.5px; }
.rh-leg i.ln.live { border-top:3px solid #0a2463; }
.sp-tbl td.rh-prev { box-shadow:inset 0 0 0 1.5px rgba(79,70,229,.85); }
/* glass windows, as the COT_ALL recap panels: one or two keep their natural size, from four on they share the screen */
#rh-dock { display:grid; gap:14px; margin:12px 0 0; color:#1a1a2e; align-items:start; }
#rh-dock.c1 { grid-template-columns:minmax(0, 940px); } #rh-dock.c2 { grid-template-columns:repeat(2, minmax(0, 760px)); }
#rh-dock.c3 { grid-template-columns:repeat(3, minmax(0, 1fr)); }
#rh-dock.fill { align-items:stretch; grid-template-rows:repeat(2, minmax(0, 1fr)); }
@keyframes rhRise { from { opacity:0; transform:translateY(14px) scale(.985); } to { opacity:1; transform:none; } }
.rh-card { position:relative; min-width:0; min-height:0; display:flex; flex-direction:column; overflow:hidden; border-radius:22px;
  padding:14px 18px 8px; color:#1a1a2e;
  background:linear-gradient(135deg, rgba(255,255,255,.84), rgba(238,244,251,.70)); -webkit-backdrop-filter:blur(26px) saturate(185%);
  backdrop-filter:blur(26px) saturate(185%); border:1px solid rgba(42,160,150,.42);
  box-shadow:0 0 0 1px rgba(42,160,150,.18), 0 0 24px rgba(42,160,150,.20), 0 16px 40px rgba(10,36,99,.14), inset 0 1px 0 rgba(255,255,255,.92); }
.rh-card.enter { animation:rhRise .34s cubic-bezier(.2,.8,.2,1); }
.rh-card > svg { width:100%; height:auto; }
#rh-dock.fill .rh-card > svg { flex:1 1 0; min-height:0; height:auto; }
.rh-card.pin { border-color:rgba(139,92,246,.40);
  box-shadow:0 0 0 1px rgba(139,92,246,.14), 0 0 10px rgba(139,92,246,.10), 0 14px 34px rgba(10,36,99,.12), inset 0 1px 0 rgba(255,255,255,.92); }
.rh-card.prev { border-style:dashed; border-color:rgba(79,70,229,.45);
  box-shadow:0 0 0 1px rgba(79,70,229,.14), 0 0 24px rgba(79,70,229,.16), 0 16px 40px rgba(10,36,99,.14), inset 0 1px 0 rgba(255,255,255,.92); }
.rh-tag { position:absolute; right:16px; top:12px; font-size:9.5px; font-weight:800; letter-spacing:.12em; text-transform:uppercase; color:#4f46e5; }
.rh-empty { grid-column:1 / -1; padding:22px; text-align:center; color:#8a94a8; font-size:12.5px; border-radius:18px;
  border:1px dashed rgba(42,160,150,.40); background:linear-gradient(135deg, rgba(255,255,255,.75), rgba(238,244,251,.6)); }
#rh-dock.dense .rh-t { font-size:12px; margin-bottom:4px; }
#rh-dock.dense .rh-stats { font-size:10.5px; gap:4px 6px; margin-bottom:4px; }
#rh-dock.dense .rh-s { padding:1px 8px; }
#rh-dock.dense .rh-leg { gap:8px; font-size:10px; }
/* focus mode while hover is on: grids pinned to the top of the page, charts below take the rest of the screen */
/* focus window while hover is on (idea from the COT_ALL recap): only the grids and the chart strip stay,
   sidebar and header slide away, a Back button on the left snaps out of it */
.fnd-hide { display:none !important; }
body.fnd-focus [data-testid="stHeader"] { display:none !important; }
body.fnd-focus [data-testid="stMain"] { overflow-x:hidden !important; }
body.fnd-focus .st-key-grids [data-testid="stColumn"] { min-width:0 !important; }
body.fnd-focus [data-testid="stSidebar"] { transform:translateX(-105%) !important; opacity:0 !important; width:0 !important;
  min-width:0 !important; max-width:0 !important; overflow:hidden !important; pointer-events:none !important; }
body.fnd-focus [data-testid="stSidebarCollapsedControl"], body.fnd-focus [data-testid="stExpandSidebarButton"] { display:none !important; }
body.fnd-focus .block-container { max-width:100% !important; padding-top:.6rem !important; padding-bottom:.4rem !important;
  padding-left:116px !important; padding-right:1.2rem !important; }
#fnd-back { position:fixed; left:14px; top:10px; width:86px; height:108px; z-index:1000006; cursor:pointer; display:flex;
  flex-direction:column; align-items:center; justify-content:center; gap:5px; color:#6d28d9; text-align:center;
  background:linear-gradient(160deg, rgba(245,243,255,.96), rgba(237,233,254,.88)); border:1px solid rgba(139,92,246,.32);
  box-shadow:0 2px 8px rgba(109,40,217,.08); transition:transform .16s ease, box-shadow .16s ease; }
#fnd-back:hover { transform:translateY(-1px); background:#ede9fe; border-color:rgba(139,92,246,.55); box-shadow:0 6px 16px rgba(109,40,217,.16); }
#fnd-back:active { transform:scale(.96); }
#fnd-back .a { font-size:19px; line-height:1; font-weight:600; }
#fnd-back .t { font-family:Plus Jakarta Sans, sans-serif; font-size:10px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; line-height:1.25; }
.rh-x { position:absolute; right:14px; top:9px; cursor:pointer; font-size:17px; line-height:1; color:#7a86a8; padding:0 2px; z-index:2; }
.rh-x:hover { color:#6d28d9; }
.st-key-grids [data-testid="stPlotlyChart"], .st-key-grids [data-testid="stPlotlyChart"] > div,
.st-key-grids [data-testid="stPlotlyChart"] .js-plotly-plot, .st-key-grids [data-testid="stPlotlyChart"] .main-svg { border-radius:0 !important; }
.st-key-grids [data-testid="stPlotlyChart"] { padding:2px !important; box-shadow:none !important; }
</style>"""


def _svg_yaxis(lo, hi, Y, x0, x1, n=5):
    """Round gridlines with labels, and a stronger zero line when 0 is in range (as in cot_app._rh_yaxis)."""
    import math
    span = (hi - lo) or abs(hi) or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    fmt = "{:,.0f}" if step >= 1 else "{:,.1f}"
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + 1e-9:
        y = Y(v)
        zero = abs(v) < step / 1e6
        out.append(f"<line x1='{x0}' x2='{x1}' y1='{y:.1f}' y2='{y:.1f}' "
                   + ("stroke='#8a94ad' stroke-width='1.2'/>" if zero else "stroke='rgba(10,36,99,.08)' stroke-width='1'/>")
                   + f"<text x='{x0 - 6}' y='{y + 3:.1f}' text-anchor='end' font-size='10' fill='#7a86a8'>"
                   + f"{fmt.format(0 if zero else v)}</text>")
        v += step
    return "".join(out)


def _svg_xaxis(X, x0, x1, ax, t_a, t_b, dates=None, w=880) -> str:
    """Hover-chart x-axis: grey day ticks with their calendar dates, FND as a bold tick, and the
    measured window's two ends labelled in orange (merged into one label when they sit close)."""
    out = [f"<line x1='{X(x0):.1f}' x2='{X(x1):.1f}' y1='{ax}' y2='{ax}' stroke='rgba(10,36,99,.18)'/>"]
    a, b = max(t_a, x0), min(t_b, x1)
    xa, xb = X(a), X(b)
    d = (lambda t: dates.get(t)) if dates else (lambda t: None)
    fmt_d = lambda t, f: f"{d(t):{f}}" if d(t) is not None else ""
    close = xb - xa < 110
    lit_x = [(xa + xb) / 2] if close else [xa, xb]
    for t in range(int(np.ceil(x0 / 5) * 5), int(x1) + 1, 5):
        if any(abs(X(t) - u) < (60 if close else 42) for u in lit_x):
            continue
        out.append(f"<text x='{X(t):.1f}' y='{ax + 15}' text-anchor='middle' font-size='10' "
                   + ("font-weight='800' fill='#0a2463'>FND" if t == 0 else f"fill='#7a86a8'>{t:+d}") + "</text>")
        if d(t) is not None:
            out.append(f"<text x='{X(t):.1f}' y='{ax + 27}' text-anchor='middle' font-size='9' fill='#a3abbd'>{fmt_d(t, '%d %b')}</text>")
    lab = lambda x, top, bot: (f"<text x='{min(max(x, 40), w - 40):.1f}' y='{ax + 15}' text-anchor='middle' font-size='10.5' "
                               f"font-weight='800' fill='#c2410c'>{top}</text>"
                               + (f"<text x='{min(max(x, 40), w - 40):.1f}' y='{ax + 27}' text-anchor='middle' font-size='9.5' "
                                  f"font-weight='600' fill='#dd7a3e'>{bot}</text>" if bot else ""))
    if close:
        bot = (f"{fmt_d(a, '%d')}–{fmt_d(b, '%d %b %y')}" if d(a) is not None and d(a).month == d(b).month
               else f"{fmt_d(a, '%d %b')} – {fmt_d(b, '%d %b %y')}") if d(a) is not None and d(b) is not None else ""
        out.append(lab((xa + xb) / 2, f"{_tl(a)} → {_tl(b)}", bot))
    else:
        out.append(lab(xa, _tl(a), fmt_d(a, "%d %b %y")) + lab(xb, _tl(b), fmt_d(b, "%d %b %y")))
    return "".join(out)


def _svg_frame(lo, hi, x0, x1, w, h):
    pl, pr, pt, pb = 56, 24, 20, 38
    iw, ih = w - pl - pr, h - pt - pb
    X = lambda t: pl + (t - x0) / ((x1 - x0) or 1) * iw
    Y = lambda v: pt + (hi - v) / ((hi - lo) or 1) * ih
    return X, Y, pl, pr, pt, pt + ih


def _svg_window(X, Y, pt, ax, x0, x1, t_a, t_b) -> str:
    out = "".join(f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{pt}' y2='{ax}' stroke='#ea580c' stroke-width='1.3' "
                  f"stroke-dasharray='2 3' opacity='.85'/>" for t in sorted({t_a, t_b}) if x0 <= t <= x1)
    if x0 <= 0 <= x1:
        out += f"<line x1='{X(0):.1f}' x2='{X(0):.1f}' y1='{pt}' y2='{ax}' stroke='rgba(10,36,99,.40)' stroke-dasharray='4 3'/>"
    return out


def path_svg(sr: pd.Series, t_a: int, t_b: int, dates, w: int = 880, h: int = 260) -> str:
    """One contract's series against trading days to FND, measured window shaded and
    marked on the axis with its calendar dates; values live in the chips above."""
    sr = sr.dropna()
    if len(sr) < 2:
        return ""
    lo, hi = float(sr.min()), float(sr.max())
    pad = (hi - lo) * 0.12 or 1.0
    lo, hi = lo - pad, hi + pad
    x0, x1 = sr.index.min(), sr.index.max()
    X, Y, pl, pr, pt, ax = _svg_frame(lo, hi, x0, x1, w, h)
    pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in sr.items())
    dots = "".join(f"<circle cx='{X(t):.1f}' cy='{Y(float(sr[t])):.1f}' r='3.4' fill='#ea580c' stroke='#ffffff' stroke-width='1.2'/>"
                   for t in (t_a, t_b) if t in sr.index)
    return (f"<svg width='100%' viewBox='0 0 {w} {h}' preserveAspectRatio='xMidYMid meet'>"
            f"{_svg_yaxis(lo, hi, Y, pl, w - pr)}{_svg_window(X, Y, pt, ax, x0, x1, t_a, t_b)}"
            f"<polyline points='{pts}' fill='none' stroke='#1e3a6e' stroke-width='2' stroke-linejoin='round'/>{dots}"
            f"{_svg_xaxis(X, x0, x1, ax, t_a, t_b, dates, w)}</svg>")


def band_svg(paths: pd.DataFrame, last_lbl: str, t_a: int, t_b: int, live=None, dates=None,
             w: int = 880, h: int = 260) -> str:
    """Seasonality of one contract month around FND: 10-90 and 25-75 bands across past
    expiries, the mean dotted teal, the most recent expiry in wine (Rollex look)."""
    n = paths.notna().sum(axis=1)
    paths = paths[n >= 3]
    if len(paths) < 2:
        return ""
    q = paths.quantile([0.1, 0.25, 0.75, 0.9], axis=1).T
    mean = paths.mean(axis=1)
    last = paths.iloc[:, -1].dropna()
    extra = [x for x in (last, live) if x is not None and len(x)]
    lo = float(min([q[0.1].min()] + [x.min() for x in extra]))
    hi = float(max([q[0.9].max()] + [x.max() for x in extra]))
    pad = (hi - lo) * 0.10 or 1.0
    lo, hi = lo - pad, hi + pad
    x0, x1 = paths.index.min(), paths.index.max()
    X, Y, pl, pr, pt, ax = _svg_frame(lo, hi, x0, x1, w, h)
    live_svg = ""
    if live is not None and len(live):
        lpts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in live.items())
        live_svg = (f"<polyline points='{lpts}' fill='none' stroke='#0a2463' stroke-width='2.6' stroke-linejoin='round'/>"
                    f"<circle cx='{X(live.index[-1]):.1f}' cy='{Y(float(live.iloc[-1])):.1f}' r='4' fill='#0a2463' "
                    f"stroke='#ffffff' stroke-width='1.5'/>")
    poly = lambda up, dn: (" ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in up.items()) + " "
                           + " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in dn[::-1].items()))
    line = lambda sr: " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in sr.dropna().items())
    return (f"<svg width='100%' viewBox='0 0 {w} {h}' preserveAspectRatio='xMidYMid meet'>"
            f"{_svg_yaxis(lo, hi, Y, pl, w - pr)}{_svg_window(X, Y, pt, ax, x0, x1, t_a, t_b)}"
            f"<polygon points='{poly(q[0.9], q[0.1])}' fill='rgba(99,102,241,.12)'/>"
            f"<polygon points='{poly(q[0.75], q[0.25])}' fill='rgba(99,102,241,.24)'/>"
            f"<polyline points='{line(mean)}' fill='none' stroke='#4f46e5' stroke-width='2' stroke-dasharray='5 4'/>"
            f"<polyline points='{line(last)}' fill='none' stroke='#ea580c' stroke-width='2'/>"
            f"{live_svg}"
            f"{_svg_xaxis(X, x0, x1, ax, t_a, t_b, dates, w)}</svg>")


def _pin_click_helper() -> None:
    """Inline handlers are stripped from st.markdown HTML, so a 0-height component installs delegated
    listeners on the parent page (same mechanism as the COT_ALL recap). The strip sits right under the
    grid row and, in focus mode, fills the rest of the screen: hovering a cell previews its chart there,
    clicking pins it (six at most, laid out 1 / 2 / 3 / 2x2 / 3x2, the oldest drops off), clicking it
    again or its x unpins it."""
    import json
    import streamlit.components.v1 as components
    code = """
(function () {
  var w = window, d = document, VER = 8, MAX = 6;
  if (w.__fndPin === VER) return;
  try { if (w.__fndPinOff) w.__fndPinOff(); } catch (e) {}
  w.__fndPin = VER;
  var pins = [], prev = null, shown = [];
  function host() { var h = d.querySelector('.st-key-grids'); return h && h.querySelector('td.rh-hov') ? h : null; }
  function main() { return d.querySelector('[data-testid="stMain"]'); }
  function focus(on) {                                       // hide everything on the page but the grid row and the strip
    d.body.classList.toggle('fnd-focus', on);
    var bb = d.getElementById('fnd-back');
    if (!on) {
      d.querySelectorAll('.fnd-hide').forEach(function (el) { el.classList.remove('fnd-hide'); });
      if (bb) bb.remove(); return;
    }
    var m = main(), node = host();
    while (node && node !== m && node.parentElement) {
      Array.prototype.forEach.call(node.parentElement.children, function (sib) {
        if (sib !== node && sib.id !== 'rh-dock' && sib.tagName !== 'STYLE' && !sib.classList.contains('fnd-hide')) sib.classList.add('fnd-hide');
      });
      node = node.parentElement;
    }
    if (!bb) {
      bb = d.createElement('div'); bb.id = 'fnd-back';
      bb.innerHTML = "<span class='a'>&larr;</span><span class='t'>Back to<br>page</span>"; d.body.appendChild(bb);
      w.scrollTo(0, 0); if (m) m.scrollTop = 0;
    }
  }
  function dockEl() {
    var h = host(), dock = d.getElementById('rh-dock');
    focus(!!h);
    if (!h) { if (dock) dock.remove(); return null; }
    if (!dock) { dock = d.createElement('div'); dock.id = 'rh-dock'; }
    if (dock.previousElementSibling !== h) h.parentNode.insertBefore(dock, h.nextSibling);
    return dock;
  }
  function fit() {                                           // from four windows on, fill the screen to the bottom edge
    var dock = d.getElementById('rh-dock'); if (!dock) return;
    if (!dock.classList.contains('fill')) { dock.style.height = ''; return; }
    var m = main(), top = dock.getBoundingClientRect().top + (m ? m.scrollTop : 0) + w.scrollY;
    dock.style.height = Math.max(260, w.innerHeight - top - 12) + 'px';
  }
  function card(td, cls, i) {
    var pop = td.querySelector('.rh-pop'); if (!pop) return '';
    var head = i >= 0 ? "<span class='rh-x' data-i='" + i + "'>&times;</span>" : "<span class='rh-tag'>Preview · click to pin</span>";
    var body = pop.innerHTML;
    if (i >= 0) body = body.replace("<div class='rh-t'>", "<div class='rh-t'><span class='rh-num'>" + (i + 1) + "</span>");
    var fresh = shown.indexOf(td) < 0 ? ' enter' : '';     // only windows that just appeared rise in
    return "<div class='rh-card " + cls + fresh + "'>" + head + body + "</div>";
  }
  function rebuild() {
    pins = pins.filter(function (el) { return el.isConnected && el.classList.contains('rh-pinned'); });
    if (prev && !prev.isConnected) prev = null;
    d.querySelectorAll('td.rh-prev').forEach(function (el) { if (el !== prev) el.classList.remove('rh-prev'); });
    var dock = dockEl(); if (!dock) return;
    var html = '', n = pins.length;
    d.querySelectorAll('td[data-pin]').forEach(function (el) { el.removeAttribute('data-pin'); });
    pins.forEach(function (td, i) { td.setAttribute('data-pin', i + 1); html += card(td, 'pin', i); });
    if (prev && pins.indexOf(prev) < 0 && pins.length < MAX) { html += card(prev, 'prev', -1); n++; prev.classList.add('rh-prev'); }
    if (!n) html = "<div class='rh-empty'>Hover any cell to see its chart here &middot; click to pin it, up to six &middot; the Avg row shows each month's seasonality</div>";
    var cols = n <= 1 ? 1 : n === 2 ? 2 : n === 4 ? 2 : 3, rows = n <= 3 ? 1 : 2;
    dock.innerHTML = html;
    dock.className = 'c' + cols + (rows > 1 ? ' fill' : '') + (n >= 3 ? ' dense' : '');
    shown = pins.slice(); if (prev && pins.indexOf(prev) < 0) shown.push(prev);
    fit();
  }
  function onOver(e) {
    if (!e.target.closest) return;
    var td = e.target.closest('td.rh-hov');
    if (!td) {
      if (prev && !e.target.closest('.sp-tbl')) { prev.classList.remove('rh-prev'); prev = null; rebuild(); }
      return;
    }
    if (td === prev) return;
    if (prev) prev.classList.remove('rh-prev');
    prev = td; rebuild();
  }
  function onClick(e) {
    if (e.target.closest && e.target.closest('#fnd-back')) {  // Back: unpin, leave focus, switch hover off
      pins.forEach(function (el) { el.classList.remove('rh-pinned'); }); pins = []; prev = null;
      var off = Array.prototype.find.call(d.querySelectorAll('button'), function (b) { return /Deactivate Hover/.test(b.innerText); });
      focus(false); var dk = d.getElementById('rh-dock'); if (dk) dk.remove();
      if (off) off.click();
      return;
    }
    var x = e.target.closest && e.target.closest('.rh-x');
    if (x) { var el = pins[+x.getAttribute('data-i')]; if (el) el.classList.remove('rh-pinned'); rebuild(); return; }
    var td = e.target.closest && e.target.closest('td.rh-hov');
    if (!td) return;
    if (td.classList.contains('rh-pinned')) { td.classList.remove('rh-pinned'); }
    else { if (pins.length >= MAX) pins.shift().classList.remove('rh-pinned'); td.classList.add('rh-pinned'); pins.push(td); }
    rebuild();
  }
  var timer = setInterval(function () {                       // app reruns replace the tables: re-seat the strip
    var h = host(), dock = d.getElementById('rh-dock');
    if ((h && (!dock || dock.previousElementSibling !== h)) || (!h && (dock || d.body.classList.contains('fnd-focus')))
        || pins.some(function (el) { return !el.isConnected; }) || (prev && !prev.isConnected)) rebuild();
    else { if (h) focus(true); fit(); }                       // reruns re-render the page: keep it hidden
  }, 600);
  d.addEventListener('mouseover', onOver, true);
  d.addEventListener('click', onClick, true);
  w.addEventListener('resize', fit);
  w.__fndPinOff = function () {
    d.removeEventListener('mouseover', onOver, true); d.removeEventListener('click', onClick, true);
    w.removeEventListener('resize', fit); clearInterval(timer); focus(false);
  };
  rebuild();
})();
"""
    components.html("<script>(function () { var d = window.parent.document, s = d.createElement('script');"
                    f" s.textContent = {json.dumps(code)}; d.head.appendChild(s); }})();</script>", height=0)


def hover_table(pts: pd.DataFrame, col: str, fmt: str, series: pd.DataFrame, title: str,
                s_fmt: str, s_suf: str = "", style: str = "bar", rebase: bool = False,
                avg_mode: str = "change", hist_keys: dict | None = None, live_keys: dict | None = None,
                fnd_of: dict | None = None) -> str:
    """Year x contract-month table, COT-recap style: hovering a cell previews that contract's
    series around FND, clicking pins it. style "bar" = centre mini bars (as bar_heatmap);
    "heat" / "heat_r" = teal shading that deepens with the value (or as it falls).
    rebase turns each series into % change from the measured day (for the Rollex index)."""
    order = "FGHJKMNQUVXZ"
    months = sorted(pts["m"].unique(), key=order.index)
    vals = pts[col].dropna()
    vmax = float(vals.abs().quantile(0.9)) or 1.0
    q_lo, q_hi = float(vals.quantile(0.05)), float(vals.quantile(0.95))
    by = {(r.y, r.m): r for r in pts.itertuples()}
    lo_t, hi_t = min(t_chk - 15, -40), max(t_exit + 10, 5)
    days = market_calendar(comm, mt)
    span = np.arange(lo_t, hi_t + 1)
    change_mode = rebase or avg_mode == "change"

    def win_val(sr, ta, tb):
        """What a cell shows, from a contract's own series: level at ta, or the move ta -> tb."""
        a_, b_ = sr.get(ta, np.nan), sr.get(tb, np.nan)
        if not change_mode:
            return a_
        if rebase:
            return (b_ / a_ - 1) * 100 if a_ else np.nan
        return b_ - a_

    def pct_chip(v, m, ta, tb, label):
        """Where v ranks among every finished expiry of month m over the same days (real readings only)."""
        past = pd.Series([win_val(series[k], ta, tb) for k, _ in (hist_keys or {}).get(m, []) if k in series]).dropna()
        if len(past) < 3 or pd.isna(v):
            return ""
        return (f"<span class='rh-s live'><b>{label}</b><i>{_ord((past < v).mean() * 100)} pct</i>"
                f"<em> of {len(past)} past</em></span>")

    rows = []
    for y in sorted(pts["y"].unique()):
        tds = []
        for m in months:
            r = by.get((y, m))
            v = getattr(r, col) if r is not None else np.nan
            if r is None or pd.isna(v):
                tds.append("<td></td>")
                continue
            if style == "bar":
                h_ = min(abs(v) / vmax, 1.0) * 46
                left, colr = (50, "rgba(44,143,157,.28)") if v >= 0 else (50 - h_, "rgba(123,30,58,.20)")
                cell = (f"<div class='mid'></div><div class='bar' style='left:{left:.1f}%;width:{h_:.1f}%;"
                        f"background:{colr}'></div>")
                bg = ""
            else:
                f = min(max((v - q_lo) / ((q_hi - q_lo) or 1), 0), 1)
                f = 1 - f if style == "heat_r" else f
                cell, bg = "", f" style='background:rgba(44,143,157,{0.04 + 0.42 * f:.2f})'"
            k = days.searchsorted(r.fnd) + span
            ok = (k >= 0) & (k < len(days))
            cal = dict(zip(span[ok].tolist(), days[k[ok]]))
            t_b = int(r.t_end)                                  # T`t_exit`, or the live contract's latest session
            sr = series[r.key]
            sr = sr[(sr.index >= lo_t) & (sr.index <= hi_t)]
            if rebase and t_chk in sr.index and sr[t_chk]:
                sr = (sr / sr[t_chk] - 1) * 100
            v_a, v_b = sr.get(t_chk, np.nan), sr.get(t_b, np.nan)
            chg = v_b - v_a
            live_txt = f" · live, to {_tl(t_b)}" if r.live else ""
            pop = (f"<div class='rh-pop'><div class='rh-t'>{comm} {r.lbl} · {title}{live_txt}</div>"
                   f"<div class='rh-stats'><span class='rh-s'><b>At {_tl(t_chk)}</b><i>{v_a:{s_fmt}}{s_suf}</i></span>"
                   f"<span class='rh-s'><b>At {_tl(t_b)}{' (latest)' if r.live else ''}</b><i>{v_b:{s_fmt}}{s_suf}</i></span>"
                   f"<span class='rh-s'><b>Change</b><i class='{'up' if chg >= 0 else 'dn'}'>{chg:+{s_fmt.lstrip('+')}}{s_suf}</i></span>"
                   f"<span class='rh-s'><b>FND</b><i>{r.fnd:%d %b %Y}</i></span>"
                   + (pct_chip(v, m, t_chk, t_b, f"vs past {m}, same days") if r.live else "") + "</div>"
                   f"{path_svg(sr, t_chk, t_b, cal)}</div>")
            tds.append(f"<td class='rh-hov{' live' if r.live else ''}'{bg}>{cell}<span class='v'>{v:{fmt}}</span>{pop}</td>")
        rows.append(f"<tr><th class='yr'>{y}</th>{''.join(tds)}</tr>")

    # Avg row: the mean cell, and behind it the month's seasonality across every past expiry.
    # Change grids show each path as change since the measured day (so years line up); level grids show levels.
    tds = []
    for m in months:
        sub = pts[(pts["m"] == m) & ~pts["live"]].sort_values("y")     # finished expiries only
        v = sub[col].mean()
        if not len(sub) or pd.isna(v):
            tds.append("<td></td>")
            continue
        paths = {}
        for k_, lbl_ in (hist_keys or {}).get(m, []):          # all years, oldest first
            if k_ not in series:
                continue
            sr = series[k_]
            sr = sr[(sr.index >= lo_t) & (sr.index <= hi_t)]
            if rebase or avg_mode == "change":
                if t_chk not in sr.index or pd.isna(sr[t_chk]) or (rebase and not sr[t_chk]):
                    continue
                sr = (sr / sr[t_chk] - 1) * 100 if rebase else sr - sr[t_chk]
            paths[lbl_] = sr
        paths = pd.DataFrame(paths)
        # The live contract of this month on top of the bands, real readings up to its latest session
        live_sr, live_lbl, live_chip = None, "", ""
        if live_keys and m in live_keys:
            lk, live_lbl, t_now = live_keys[m]
            lsr = series[lk]
            lsr = lsr[(lsr.index >= lo_t) & (lsr.index <= min(hi_t, t_now))].dropna()
            if change_mode:
                base_v = lsr.get(t_chk, np.nan)
                lsr = ((lsr / base_v - 1) * 100 if rebase else lsr - base_v) if pd.notna(base_v) and base_v else None
            if lsr is not None and len(lsr):
                live_sr = lsr
                tn = int(lsr.index.max())
                ref = paths.loc[tn].dropna() if tn in paths.index else pd.Series(dtype=float)
                if len(ref) >= 3:
                    live_chip = (f"<span class='rh-s live'><b>{live_lbl} now ({_tl(tn)})</b><i>{lsr[tn]:{s_fmt}}{s_suf}</i>"
                                 f"<em> {_ord((ref < lsr[tn]).mean() * 100)} pct of {len(ref)} past</em></span>")
        fnd_ref = None
        if live_keys and m in live_keys:
            lf = (fnd_of or {}).get(live_keys[m][0])
            fnd_ref = lf if lf is not None and pd.notna(lf) else None
        if fnd_ref is None and len(sub):
            fnd_ref = sub["fnd"].max()
        avg_cal = None
        if fnd_ref is not None:
            k = days.searchsorted(fnd_ref) + span
            ok = (k >= 0) & (k < len(days))
            avg_cal = dict(zip(span[ok].tolist(), days[k[ok]]))
        up = (sub[col] > 0).mean() * 100
        what = f"change since {_tl(t_chk)}" if (rebase or avg_mode == "change") else "level"
        pop = (f"<div class='rh-pop'><div class='rh-t'>{comm} {m} · {title} · seasonality ({what})</div>"
               f"<div class='rh-stats'><span class='rh-s'><b>Expiries</b><i>{len(sub)}</i></span>"
               f"<span class='rh-s'><b>Average</b><i>{v:{fmt}}</i></span>"
               f"<span class='rh-s'><b>Median</b><i>{sub[col].median():{fmt}}</i></span>"
               + (f"<span class='rh-s'><b>Up</b><i>{up:.0f}%</i></span>" if style == "bar" else "")
               + f"<span class='rh-s'><b>Range</b><i>{sub[col].min():{fmt}} to {sub[col].max():{fmt}}</i></span>"
               + live_chip + "</div>"
               f"<div class='rh-leg'><span><i class='ln dash'></i>Mean</span>"
               f"<span><i class='ln wine'></i>{paths.columns[-1] if len(paths.columns) else ''} (latest finished)</span>"
               + (f"<span><i class='ln live'></i>{live_lbl} live</span>" if live_sr is not None else "")
               + f"<span><i class='bx in'></i>25–75%</span><span><i class='bx out'></i>10–90%</span>"
               f"<span>bands: all {paths.shape[1]} finished {m} expiries · dates of "
               f"{live_lbl or (paths.columns[-1] if len(paths.columns) else '')}</span></div>"
               f"{band_svg(paths, '', t_chk, t_exit, live_sr, avg_cal)}</div>")
        if style == "bar":
            h_ = min(abs(v) / vmax, 1.0) * 46
            left, colr = (50, "rgba(44,143,157,.28)") if v >= 0 else (50 - h_, "rgba(123,30,58,.20)")
            cell, bg = (f"<div class='mid'></div><div class='bar' style='left:{left:.1f}%;width:{h_:.1f}%;"
                        f"background:{colr}'></div>"), ""
        else:
            f = min(max((v - q_lo) / ((q_hi - q_lo) or 1), 0), 1)
            f = 1 - f if style == "heat_r" else f
            cell, bg = "", f" style='background:rgba(44,143,157,{0.04 + 0.42 * f:.2f})'"
        tds.append(f"<td class='rh-hov'{bg}>{cell}<span class='v'>{v:{fmt}}</span>{pop}</td>")
    rows.append(f"<tr class='avg'><th class='yr'>Avg</th>{''.join(tds)}</tr>")
    head_ = "".join(f"<th>{m}</th>" for m in months)
    return (f"<div class='sp-wrap'><table class='sp-tbl{' m6' if len(months) > 5 else ''}'><thead><tr><th></th>{head_}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def month_violin(pts, col, unit_lbl, now_v=np.nan) -> go.Figure:
    """One violin per contract month (Rollex seasonality violin): distribution
    of past expiries with box and average line; the live contract's change so
    far as a wine diamond on its own month."""
    order = sorted(pts["m"].unique(), key="FGHJKMNQUVXZ".index)
    fig = go.Figure()
    for m in order:
        d = pts[pts["m"] == m]
        fig.add_trace(go.Violin(
            x=[m] * len(d), y=d[col].round(1), name=m, customdata=d["lbl"], showlegend=False, hoveron="points",
            fillcolor="rgba(44,143,157,0.10)", line=dict(color="rgba(44,143,157,0.55)", width=0.9),
            box=dict(visible=True, width=0.22, fillcolor="rgba(255,255,255,0.85)",
                     line=dict(color="rgba(44,143,157,0.8)", width=1)),
            meanline=dict(visible=True, color="#7c3aed", width=1.8), points="all", pointpos=0, jitter=0.25,
            marker=dict(size=4, color="rgba(11,58,66,0.35)", line=dict(width=0)), spanmode="hard",
            hovertemplate="%{customdata}: <b>%{y:+,.1f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Average", line=dict(color="#7c3aed", width=1.8)))
    if pd.notna(now_v) and month in order:
        fig.add_trace(go.Scatter(
            x=[month], y=[now_v], mode="markers", name=f"{cur_lbl} so far",
            marker=dict(symbol="diamond", size=10, color=WINE, line=dict(color="#ffffff", width=1.5)),
            hovertemplate=f"{cur_lbl} so far: <b>%{{y:+,.1f}}</b><extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(10,36,99,0.2)", width=1))
    fig.update_layout(
        height=400, margin=dict(t=30, b=60, l=50, r=16), violingap=0.3,
        legend=dict(orientation="h", y=-0.12, yanchor="top", x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(categoryorder="array", categoryarray=order, showgrid=False, tickfont=dict(size=11, color="#1e3a6e")),
        yaxis=dict(title=dict(text=f"Change in certs ({unit_lbl})", font=dict(size=10, color="#7a86a8")),
                   showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=10, color="#7a86a8"),
                   hoverformat="+,.1f"),
        **_D)
    return fig


def certs_scatter(pts, x, x_title, x_fmt, now_x, now_txt):
    """Past contracts as teal dots (KC's 2024+ low-certs era in wine), today's
    reading as a wine marker line with a Rollex-style tag."""
    fig = go.Figure()
    if comm == "KC":
        recent = pts["fnd"] >= "2024-01-01"
        groups = [(~recent, "rgba(44,143,157,0.55)", 8, "Before 2024"), (recent, WINE, 9, "Since 2024")]
    else:
        groups = [(pts["fnd"].notna(), "rgba(44,143,157,0.55)", 8, "Past contracts")]
    for sel, color, size, name in groups:
        p = pts[sel]
        fig.add_trace(go.Scatter(
            x=p[x], y=p["dsp"], mode="markers", name=name, text=p["lbl"],
            marker=dict(color=color, size=size, line=dict(color="#ffffff", width=1)),
            hovertemplate=f"<b>%{{text}}</b><br>{x_title} %{{x:{x_fmt}}}<br>Spread %{{y:+,.2f}}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="rgba(10,36,99,0.2)", width=1))
    if pd.notna(now_x):
        fig.add_vline(x=now_x, line=dict(color=WINE, width=1.4, dash="dot"))
        fig.add_annotation(x=now_x, y=1, yref="paper", yshift=4, showarrow=False, text=now_txt,
                           font=dict(size=10, color="#ffffff"), bgcolor=WINE, bordercolor="#ffffff",
                           borderwidth=1.5, borderpad=5, opacity=.96)
    fig.update_layout(
        height=420, margin=dict(t=30, b=60, l=50, r=16), hovermode="closest",
        legend=dict(orientation="h", y=-0.16, yanchor="top", x=0.5, xanchor="center", font=dict(size=10)),
        xaxis=dict(title=dict(text=(f"{x_title}, {_tl(t_chk)} to {_tl(t_exit)}" if x.startswith("cchg")
                                    else f"{x_title} at {_tl(t_chk)}"), font=dict(size=10, color="#7a86a8")),
                   showgrid=True, gridcolor=GRID, zeroline=False, tickfont=AXIS_FONT),
        yaxis=dict(title=dict(text=f"Spread change {_tl(t_chk)} to {_tl(t_exit)}",
                              font=dict(size=10, color="#7a86a8")),
                   showgrid=True, gridcolor=GRID, zeroline=False, tickfont=dict(size=10, color="#7a86a8")),
        **_D)
    return fig


# ── Certs data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def certs_base(commodity: str, mt: float, cmt: float) -> dict:
    """Every contract of every delivery month as t x contract grids (date, spread,
    open interest) plus the certs reading on each date; live contracts are flagged
    with t_now, their last session so far. Built once per market, so moving the
    sliders is only a lookup."""
    raw = load_data(commodity, mt)
    asof = raw["Date"].max()
    a = pd.concat([fnd_aligned(commodity, m, mt).assign(m=m) for m in raw["month"].unique()])
    a = a.drop_duplicates(["ice_symbol", "t"])
    a["key"] = a["ice_symbol"]
    a["spread"] = a["settlement"] - a["next_settle"]
    a["certs"] = load_certs(commodity, cmt).reindex(a["Date"], method="ffill").to_numpy()
    ret = load_rollex(commodity, _rollex_mtime())
    a["rx_idx"] = (1 + ret).cumprod().reindex(a["Date"], method="ffill").to_numpy() if ret is not None else np.nan
    grid = {c: a.pivot(index="t", columns="key", values=c) for c in ("spread", "open_interest", "certs", "rx_idx")}
    meta = a.groupby("key").agg(m=("m", "first"), y=("year", "first"), fnd=("FND", "first"),
                                nxt=("next_sym", "first"))
    sym_lbl = raw.groupby("ice_symbol").agg(m=("month", "first"), y=("year", "first"))
    sym_lbl = sym_lbl["m"] + (sym_lbl["y"] % 100).map("{:02d}".format)
    meta["nxt_lbl"] = meta["nxt"].map(sym_lbl)
    meta["live"] = a.groupby("key")["LTD"].first() >= asof
    meta["t_now"] = a.dropna(subset=["open_interest"]).groupby("key")["t"].max()
    dates = a.pivot(index="t", columns="key", values="Date")
    return dict(grid=grid, meta=meta, dates=dates)


def _pick(grid: pd.DataFrame, t: pd.Series) -> pd.Series:
    """grid[t[key], key] for every key of t (NaN / NaT where the contract has no reading)."""
    r, c = grid.index.get_indexer(t.to_numpy()), grid.columns.get_indexer(t.index)
    arr = grid.to_numpy()
    return pd.Series([arr[i, j] if i >= 0 and j >= 0 else None for i, j in zip(r, c)], index=t.index)


def certs_points(commodity: str, t_chk: int, t_exit: int, mt: float, cmt: float) -> pd.DataFrame:
    """One row per contract (every delivery month): the reading at T`t_chk` and what
    certs, open interest, the expiring-minus-next spread and Rollex did from there to
    T`t_exit`. A live contract appears once it has reached T`t_chk`, measured only up
    to its latest session (t_end) and flagged live."""
    b = certs_base(commodity, mt, cmt)
    g, meta = b["grid"], b["meta"]
    meta = meta[~meta["live"] | (meta["t_now"] >= t_chk)]
    t0 = pd.Series(t_chk, index=meta.index)
    t1 = pd.Series(t_exit, index=meta.index)
    t1[meta["live"]] = np.minimum(t_exit, meta.loc[meta["live"], "t_now"])
    t1 = t1.astype(int)
    num = lambda name, t: pd.to_numeric(_pick(g[name], t), errors="coerce")
    lvl, lvl_end = num("certs", t0), num("certs", t1)
    oi0, oi1 = num("open_interest", t0), num("open_interest", t1)
    d0, d1 = pd.to_datetime(_pick(b["dates"], t0)), pd.to_datetime(_pick(b["dates"], t1))
    out = pd.DataFrame({
        "m": meta["m"], "y": meta["y"].astype(int), "fnd": meta["fnd"], "date": d0,
        "dsp": num("spread", t1) - num("spread", t0), "level": lvl,
        "mult": oi0 * CERTS[commodity]["per_lot"] / lvl, "cchg": lvl_end - lvl,
        "oi": oi0, "oichg": oi1 - oi0, "live": meta["live"], "t_end": t1})
    # Rollex move over the same dates, from its roll-adjusted return index
    ret = load_rollex(commodity, _rollex_mtime())
    if ret is not None:
        idx = (1 + ret).cumprod()
        out["rx"] = (idx.reindex(pd.DatetimeIndex(d1), method="ffill").to_numpy()
                     / idx.reindex(pd.DatetimeIndex(d0), method="ffill").to_numpy() - 1) * 100
    else:
        out["rx"] = np.nan
    out = out[out["dsp"].notna() & out["level"].gt(0)]
    out["lbl"] = out["m"] + (out["y"] % 100).map("{:02d}".format)
    out["nxt_lbl"] = meta["nxt_lbl"].reindex(out.index)
    out["key"] = out.index
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def certs_around(commodity: str, month: str, t_lo: int, t_hi: int, t_base: int,
                 mt: float, cmt: float) -> pd.DataFrame:
    """t x delivery-year grid of certified stock around each `month` contract's
    FND: the level, or with t_base the change in stock since T`t_base`. Certs keep printing after the
    contract expires, so this runs the full window."""
    certs = load_certs(commodity, cmt)
    days = market_calendar(commodity, mt)
    ts = np.arange(-t_lo, t_hi + 1)
    cols = {}
    for y, fnd in fnd_aligned(commodity, month, mt).groupby("year")["FND"].first().items():
        idx = days.searchsorted(fnd) + ts
        ok = (idx >= 0) & (idx < len(days))
        dates = days[idx[ok]]
        vals = certs.reindex(dates, method="ffill").to_numpy()
        vals[(dates < certs.index[0]) | (dates > certs.index[-1])] = np.nan
        cols[y] = pd.Series(vals, index=ts[ok])
    piv = pd.DataFrame(cols).reindex(ts)
    return piv if t_base is None else piv - from_start(piv, t_base)


# ── Views ─────────────────────────────────────────────────────────────────────
if view == "FND Window":
    oi = to_matrix(al, "open_interest", -t_lo, t_hi)
    if unit.startswith("%"):
        oi_plot, oi_fmt, oi_suf, oi_title = oi.div(from_start(oi, -t_lo)) * 100, ".1f", "%", f"% of OI at T-{t_lo}"
    else:
        oi_plot, oi_fmt, oi_suf, oi_title = oi, ",.0f", "", "Lots"
    # Expiring month's share of the two legs' volume, 3-session sums to calm single days
    al = al.sort_values(["year", "Date"])
    roll3 = lambda c: al.groupby("year")[c].transform(lambda s: s.rolling(3, min_periods=1).sum())
    al["vol_share"] = roll3("volume") / (roll3("volume") + roll3("next_vol")) * 100
    vs_plot = to_matrix(al, "vol_share", -t_lo, t_hi)

    rx = rollex_aligned(comm, month, mt, _rollex_mtime())
    rx_plot = None
    if len(rx):
        rp = to_matrix(rx, "ret", -t_lo, t_hi, ffill=0)
        rp.loc[-t_lo, rp.columns[rp.notna().any()]] = 0.0     # performance measured from the window start
        live = rp.notna()
        rx_plot = ((1 + rp.fillna(0)).cumprod() - 1).where(live.cummax() & live[::-1].cummax()[::-1]) * 100
    since = f"since T-{t_lo}"
    H = 360

    # Open interest: what is left, how much has gone, and whether trading has moved on
    c = st.columns(3, gap="medium")
    with c[0]:
        lbl(f"Open interest · {cur_lbl}", past_note)
        chart(band_chart(oi_plot, oi_title, oi_fmt, oi_suf, tozero=True, height=H))
    with c[1]:
        oi_k = oi / 1e3
        lbl("Open interest change", f"{since}, '000 lots")
        chart(band_chart(oi_k - from_start(oi_k, -t_lo), "Change in OI ('000 lots)", "+,.1f", zero_line=True, height=H))
    with c[2]:
        lbl(f"Volume share · {month} vs {nxt_m}", "3-session sums")
        chart(band_chart(vs_plot, f"{month} share of volume", ".0f", "%", tozero=True, height=H))

    # Price: the spread's level, its move, and the outright
    c = st.columns(3, gap="medium")
    with c[0]:
        lbl(f"Spread · {month} minus {nxt_m}", "price units")
        chart(band_chart(sp_plot, "Price difference", "+,.2f", zero_line=True, height=H))
    with c[1]:
        lbl("Spread change", f"{since}, price units")
        chart(band_chart(sp_plot - from_start(sp_plot, -t_lo), "Change in spread", "+,.2f", zero_line=True, height=H))
    if rx_plot is not None:
        with c[2]:
            lbl(f"Price · Rollex {since}", "roll-adjusted, %")
            chart(band_chart(rx_plot, "Return", "+.1f", "%", zero_line=True, height=H))

    # Certs (KC / RC): the stock, its move, and how much open interest sits on it
    if certs is not None:
        unit_c = cfg["unit"]
        cmt = cfg["path"].stat().st_mtime
        sc, u_lbl = (1e3, "'000 bags") if unit_c == "bags" else (1.0, "lots")
        c = st.columns(3, gap="medium")
        with c[0]:
            lbl(f"Certs · {cur_lbl}", u_lbl)
            chart(band_chart(certs_around(comm, month, t_lo, t_hi, None, mt, cmt) / sc,
                             f"Certs ({u_lbl})", ",.0f", height=H))
        with c[1]:
            lbl("Certs change", f"{since}, {u_lbl}")
            chart(band_chart(certs_around(comm, month, t_lo, t_hi, -t_lo, mt, cmt) / sc,
                             f"Change in certs ({u_lbl})", "+,.0f", zero_line=True, height=H))
        with c[2]:
            lbl("Open positions ÷ certs", "to FND")
            cv_plot = to_matrix(al, "cover", -t_lo, t_hi).loc[:0]     # past FND the month is all but closed
            chart(band_chart(cv_plot, f"Open {unit_c} ÷ certified {unit_c}", ".0f", "×", tozero=True, x_max=0, height=H))

else:
    unit_c = cfg["unit"]
    cmt = cfg["path"].stat().st_mtime
    # KC certs run to hundreds of thousands of bags, so they are shown in thousands
    kc_k = unit_c == "bags"
    sc = 1e3 if kc_k else 1.0
    u_lbl = "'000 bags" if kc_k else "lots"
    pts = certs_points(comm, t_chk, t_exit, mt, cmt)
    pts["level_show"], pts["cchg_show"] = pts["level"] / sc, pts["cchg"] / sc
    pts["oi_k"], pts["oichg_k"] = pts["oi"] / 1e3, pts["oichg"] / 1e3
    win = f"{_tl(t_chk)} to {_tl(t_exit)}"
    hover = st.session_state["rh_on"]

    if view == "Study":
        pts = pts[~pts["live"]]                                  # the study is about finished expiries
        # Live contract's certs change so far, from its own T`t_chk` reading to today
        cur_rows = al[(al["year"] == cur_year) & (al["t"] == t_chk)]
        now_c = (certs.iloc[-1] - certs.asof(cur_rows["Date"].iloc[0])) / sc if len(cur_rows) else np.nan

        st.markdown(f"<div class='pt-head'><span class='pt-title'>Certs flow by contract month"
                    f"<span class='pt-sub'>{win} · every past expiry</span></span></div>", unsafe_allow_html=True)
        with st.columns([1, 2, 1])[1]:
            lbl("Certs change by contract month", f"{win}, {u_lbl}")
            chart(month_violin(pts, "cchg_show", u_lbl, now_c))

        st.markdown(f"<div class='pt-head'><span class='pt-title'>Certs vs spread move"
                    f"<span class='pt-sub'>{win} · {len(pts)} past contracts</span></span></div>",
                    unsafe_allow_html=True)
        c = st.columns(3, gap="medium")
        with c[0]:
            lbl("Open positions ÷ certs")
            chart(certs_scatter(pts, "mult", f"Open {unit_c} ÷ certs", ".0f", now_mult, f"{cur_lbl} now {now_mult:.0f}x"))
        with c[1]:
            lbl("Certs level", u_lbl)
            chart(certs_scatter(pts, "level_show", f"Certs ({u_lbl})", ",.0f",
                                certs.iloc[-1] / sc, f"Now {certs.iloc[-1] / sc:,.0f}"))
        with c[2]:
            lbl("Certs change", u_lbl)
            chart(certs_scatter(pts, "cchg_show", f"Certs change ({u_lbl})", "+,.0f",
                                now_c, f"{cur_lbl} so far {now_c:+,.0f}"))

    else:
        if st.session_state.get("yrs", "Last 10") == "Last 10":
            pts = pts[pts["y"] >= pts["y"].max() - 9]           # Avg rows follow the years shown
        grids = [
            # (title, unit, column, cell format, chart builder, hover series, series label, series format, suffix, style, rebase)
            (f"OI {_tl(t_chk)}", "'000 lots", "oi_k", ",.1f", lambda: expiry_heatmap(pts, "oi_k", ",.1f", GRID_CS),
             "open_interest", "open interest ('000 lots)", ",.1f", "", "heat", False),
            ("OI change", "'000 lots", "oichg_k", "+,.1f", lambda: bar_heatmap(pts, "oichg_k", "+,.1f"),
             "open_interest", "open interest ('000 lots)", ",.1f", "", "bar", False),
            (f"Certs {_tl(t_chk)}", u_lbl, "level_show", ",.0f", lambda: expiry_heatmap(pts, "level_show", ",.0f", GRID_CS_R),
             "certs", f"certs ({u_lbl})", ",.0f", "", "heat_r", False),
            ("Certs change", u_lbl, "cchg_show", "+,.0f", lambda: bar_heatmap(pts, "cchg_show", "+,.0f"),
             "certs", f"certs ({u_lbl})", ",.0f", "", "bar", False),
            ("Spread move", "price", "dsp", "+,.1f", lambda: bar_heatmap(pts, "dsp", "+,.1f"),
             "spread", "spread vs next month", "+,.2f", "", "bar", False),
            ("Rollex move", "%", "rx", "+.1f", lambda: bar_heatmap(pts.dropna(subset=["rx"]), "rx", "+.1f"),
             "rx_idx", f"Rollex, % from {_tl(t_chk)}", "+.1f", "%", "bar", True),
        ]
        scale = {"certs": sc, "open_interest": 1e3}
        st.markdown(f"<div class='pt-head'><span class='pt-title'>Expiry by expiry"
                    f"<span class='pt-sub'>levels at {_tl(t_chk)} · changes {win}"
                    + (f" · orange dotted = {', '.join(pts.loc[pts['live'], 'lbl'])} live, real data to its latest session"
                       if pts["live"].any() else "")
                    + (" · hover a cell to preview, click to pin" if hover else "") + "</span></span></div>",
                    unsafe_allow_html=True)
        st.markdown(SP_CSS, unsafe_allow_html=True)
        if hover:
            cb = certs_base(comm, mt, cmt)
            base, meta_ = cb["grid"], cb["meta"]
            done = meta_[~meta_["live"]].sort_values("fnd")
            hist_keys = {m_: [(k_, f"{m_}{int(y_) % 100:02d}") for k_, y_ in d_["y"].items()] for m_, d_ in done.groupby("m")}
            lv = meta_[meta_["live"] & meta_["t_now"].notna()].sort_values("fnd").groupby("m").head(1)
            live_keys = {r_.m: (k_, f"{r_.m}{int(r_.y) % 100:02d}", int(r_.t_now)) for k_, r_ in lv.iterrows()}
        # All six side by side, in reading order: open interest, certs, then what the market did
        row = st.container(key="grids")
        for col_, (title, note, colname, fmt, build, ser, s_title, s_fmt, s_suf, style, rebase) in zip(
                row.columns(6, gap="small"), grids):
            with col_:
                lbl(title, note)
                if hover:
                    st.markdown(hover_table(pts, colname, fmt, base[ser] / scale.get(ser, 1.0), s_title,
                                            s_fmt, s_suf, style, rebase,
                                            "level" if style.startswith("heat") else "change",
                                            hist_keys, live_keys, meta_["fnd"].to_dict()),
                                unsafe_allow_html=True)
                else:
                    chart(build())
        if hover:
            _pin_click_helper()
