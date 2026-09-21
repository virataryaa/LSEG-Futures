# Futures — Interim Migration (LSEG)

Interim replacement for `ICEBREAKER/Futures`, rebuilt against the **LSEG
Data API** (`lseg.data`) instead of ICE Connect (`icepython`), for the period
while ICE API access is unavailable. Per-contract-month OHLCV + Open
Interest history — not a continuous/rolled series — for every individual
delivery month of 7 ICE softs commodities (KC, CC, CT, SB, RC, LCC, LSU)
back to 2011.

## What's here

- **`Code/futures_builder_lseg.py`** — the builder. All FND/LTD contract-
  calendar and holiday-calendar logic is carried over **unchanged** from the
  ICE source (it's the same vendor-agnostic math already used in the Rollex
  migration — same rules, same US/UK business-day calendars). Only the price
  fetch and symbol construction changed.
- **`Database/{comm}_futures.parquet`** — one file per commodity, same 13-
  column schema as the ICE source: `Date, commodity, ice_symbol, month, year,
  FND, LTD, Open, High, Low, settlement, volume, open_interest`.
  `settlement` is LSEG's `SETTLE` field (official settlement). It used to be
  `TRDPRC_1`, the last trade, which is null on any no-trade day and sits off
  the settlement otherwise; rebuilt 2026-09-21. `Open/High/Low` are null on a
  day nothing traded, and `volume` is 0 there rather than interpolated.
- **`Database/total_oi.parquet`** — LSEG's own whole-market futures open
  interest (`TOTCNTROI`, queried on `<root>c2`) per commodity, `Date,
  commodity, total_oi`, stored raw back to 2000 (US) / 2008 (RC); the dashboard uses it from 2009 only (see below). Refreshed every run with
  a 10-day overlap. Matches the CFTC 'Futures Only' total exactly and is what
  the Total OI charts plot; summing the per-contract table is only the fallback.
  **Reliable only from 2009:** KC's Tuesday values sit ~25-30% off the CFTC total in
  2006-07 and match it exactly from 2009 (Robusta prints single digits in 2008), so
  `common.load_total_oi` starts there and drops isolated one-day glitches (a session
  far off two neighbours that agree, e.g. CT 24 Dec 2007). The stored file stays raw.
- **`Database/rollex.parquet`** — the desk's Rollex price index (`rollex_px`) and its daily
  roll-adjusted return (`rollex_ret`) for all 7 markets, `Date, commodity, rollex_px,
  rollex_ret`, 2010 onward. **A copy, not a source:** every builder run (`sync_rollex`)
  refreshes it from `../Rollex/Database`, because the deployed dashboard cannot see the
  sibling folder; Rollex's own builder runs on its own schedule, so the copy is only as
  fresh as that. Slim on purpose (~0.6 MB vs 3.1 MB for the full files) and rewritten only
  when its content changes. Feeds the Monthly Rollex Price Change matrix.
- **`Dashboard/oi_progression.py`** — copied from the ICE source (pure parquet
  consumer, zero API calls). OI Progression, Recap, Charts, Spreads, Volume,
  Flow and grid tabs, DTE-aligned seasonality banding. Single-commodity by
  construction. **Its own Streamlit app.**
- **`Dashboard/oi_seasonals.py`** — deferred OI seasonals, **a separate
  Streamlit app** (deliberately not a page of the one above: it works on a
  basket spanning several markets at once, which has no single commodity to
  select, and keeping the deploys separate means neither app re-runs the
  other's computations). Total OI across a basket of contracts spanning
  several markets and delivery months (CC+LCC Z+H, SB+LSU K+N+V, or an
  arbitrary combination such as KC Z+H together with RC X+F), aligned on days
  to the basket's back-leg expiry, one line per crop year plus a rolling mean
  and percentile band. Replaces the desk's Excel sheet — see "Two bugs it
  fixes" below.
- **`Dashboard/common.py`** — shared constants and cached parquet loaders used
  by both apps.
- **`Automator/`** — `run.bat` (daily incremental update + git push + email),
  `run_updater.py`.

## LSEG symbol convention (the part that had to be figured out)

LSEG's outright-contract RICs follow `<root><month_code><last digit of
year>`, with a `^N` suffix disambiguating repeat occurrences of the same
month-code+year-digit pair across decades — e.g. `KCH1^1` = March 2011,
`KCH1^2` = March 2021. Currently-listed, not-yet-expired contracts resolve
with the bare symbol (no suffix); everything else needs the right `^N`.
Rather than guess a formula for which suffix applies, the builder tries the
bare symbol then `^1`/`^2`/`^3` in order against the real fetch and keeps
whichever one actually returns data — reliable across the whole 2011-2028
span this project covers.

Root mapping (LSEG root differs from the ICE source's `ice_root` for the two
LIFFE-coded commodities — ICE uses the confusing single-letter root `C` for
LCC and `W` for LSU; LSEG uses the same tickers as its continuation RICs):

| Commodity | LSEG root | Calendar |
|---|---|---|
| KC | `KC` | US |
| CC | `CC` | US |
| CT | `CT` | US |
| SB | `SB` | US |
| RC | `LRC` | UK |
| LCC | `LCC` | UK |
| LSU | `LSU` | UK |

## Data-completeness treatment and the one real gap

Same issue found and fixed in every prior migration in this series: LSEG's
raw per-contract series have real gaps on days the exchange was open — a
representative contract checked at ~80% density before any fix. Each
resolved contract is reindexed onto its own exchange business-day calendar
and linearly interpolated for strictly-internal gaps only (never
extrapolating past its own first/last real print).

**Cotton (CT) is a genuine exception, not a bug.** Checked systematically
across all 7 commodities (median per-contract row-count ratio vs. the ICE
archive):

| Commodity | Median density vs ICE | Contracts <80% dense |
|---|---|---|
| SB | 1.00 | 0% |
| CC | 0.98 | 0% |
| LCC | 0.98 | 2% |
| LSU | 0.98 | 10% |
| KC | 0.96 | 7% |
| RC | 0.92 | 15% |
| **CT** | **0.70** | **66%** |

Traced to ground on a specific contract (CT V2020): ICE's archive starts
2017-11-01, LSEG's starts 2019-11-07 — LSEG appears to simply not carry the
first ~2 years of a Cotton contract's quiet, barely-traded early life the
way ICE's archive does (ICE even shows a `-0.24` placeholder settlement on
day one with zero volume, suggesting ICE backfills a symbolic price before
real trading starts; LSEG doesn't). This isn't fixable by widening the fetch
window or interpolating harder — the data isn't there to interpolate across
a 2-year span credibly. Treat early-life CT contract history (particularly
anything more than ~1.5 years before a contract's LTD) as less reliable than
the other six commodities.

## Running it

```bash
python Code/futures_builder_lseg.py              # all 7, incremental
python Code/futures_builder_lseg.py --full        # all 7, full rebuild
python Code/futures_builder_lseg.py KC RC         # specific commodities
python Code/futures_builder_lseg.py CT --full     # single commodity, full rebuild
streamlit run Dashboard/oi_progression.py    # main OI dashboard
streamlit run Dashboard/oi_seasonals.py      # deferred OI seasonals
```

Requires an authenticated LSEG Workspace/Eikon session on the host running
the builder.

## Two bugs the seasonals app fixes

The Excel sheet it replaces built each crop year by summing four hardcoded
vendor RICs. Two failure modes were found in it and are handled structurally
here:

1. **A leg goes missing when a contract rolls off.** LSEG needs a `^N` suffix
   once a contract expires (see the symbol convention above), but the sheet
   hardcodes bare tickers, so `LCCH6`/`CCH6` returned `Invalid val` and the
   25/26 column silently became Z-only — understating by 66-82% wherever both
   legs should have been live. `26/27` is written bare too and would break the
   same way. The builder already resolves suffixes by trying each against the
   real fetch, so the dashboard cannot inherit this.
2. **Cross-exchange holidays halve a NY+LD basket.** On a UK bank holiday the
   London legs have no row at all, so a raw row-wise sum drops them for that
   date (and the reverse on US holidays) — 26% of dates in a CC+LCC Z+H basket
   are affected. Each leg is carried forward across the other exchange's
   closures, only between its own first and last real print. The same applies
   to the newest date, where one market's feed can land before another's: the
   series ends at the last date on which every leg actually printed, rather
   than emitting a half-basket that reads as a collapse.

Reconstruction was validated against the desk sheet's own cells — 24/25 at
DTE 182 matches to the digit (214,073), and the other complete years agree
within 0.3%, the residual being the sheet's 7-day sampling grid versus the
daily DTE grid used here.
