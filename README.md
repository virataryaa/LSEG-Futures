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
- **`Dashboard/fnd_window.py`** — FND Window, **a third, separate Streamlit app**: every
  contract lined up on trading days to its own First Notice Day (T-N before, T+N after).
  Five views:
  - *FND Window* (all 7 markets): bands across past years, live contract on top, for open
    interest (level and change), the expiring month's share of whole-market volume, the spread
    to the next month (level and change), Rollex, and on KC / RC certs (level, change) and
    open positions per certified unit.
  - *Daily Flow*: day-by-day range bars (min-max, 10-90%, 25-75%, median) of OI, volume,
    spread and certs, switchable between daily change and level; live days as diamonds.
  - *Roll & Positioning*: where the expiring month's OI goes each day (next month, other
    months, or closed) with the index roll window marked.
  - *COT Link*: weekly change in the chosen positions (any trader group long, short or net)
    around FND as violins, one plot each, in '000 lots or % of the prior week's total OI; history from the chosen contract month or all months
    pooled. Long and short are outright, as reported, spreads apart. Below, a year x
    ISO-week matrix of one position with Avg / Std / ICV.
  - *Comprehensive View* (KC / RC): year x contract-month grids of OI, OI change, certs
    level, certs change, spread move and Rollex move over a chosen window, an Avg row, and an
    **Activate Hover** focus mode (every cell opens its chart, clicks pin up to six,
    COT_ALL-recap style; the Avg row opens that month's seasonality across all years).
  - *Study* (KC / RC): certs change by contract month (violin) and certs level vs spread move.

  Real data only: the live contract appears once it reaches the measured day and runs only
  to its latest session; nothing is projected.
- **`Database/certs.parquet`** — total ICE certified stock, `Date, commodity, certs` (KC in
  60 kg bags, RC in lots over its three grades). **A copy, not a source:** every builder run
  (`sync_certs`) refreshes it from `Fundamental/Daily Miner/Database/Main`, because the deployed
  app cannot see that folder; locally the app reads Daily Miner directly.
- **`Database/cot.parquet`** — weekly COT per market from `LSEG/COT_ALL/Database`: total OI,
  each trader group's share of it ((long + short + 2 x spread) / 2, so the groups add up to
  total OI), managed-money net and CIT index long (US markets only). Same arrangement as the
  certs copy: refreshed by the builder (`sync_cot`).
- **`Studies/certs_squeeze_backtest.py`** — KC bull-spread-into-FND backtest conditioned on
  certs (signal from the session before entry, past-only thresholds, costs deducted). Result:
  no edge over the full history.
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

## FND Window: what the data shows (KC, as of Nov 2026)

Measured on every finished KC contract since 2011 (about 78 expiries, 15 per delivery month),
real settlements, open interest, certs and COT only. These are averages and ranges of the past,
not forecasts; check them again as new expiries come in.

**When the roll happens**
- About **half of the expiring month's OI drop from T-40 to T-1 happens in six sessions**: the
  5th-10th business days of the FND month, the GSCI (5th-9th) / BCOM (6th-10th) index roll
  window, which falls at about **T-10 to T-5**. OI leaves about **3x faster** there (~8,400
  lots a day vs ~2,800 on other days of the last 20). Same in every delivery month (48-53%).
- Trading moves with it: the expiring month's share of whole-market volume falls from ~60% at
  T-40 to ~15% at FND.

**Where the OI goes**
- Over those six days the expiring month loses a median ~47k lots; the next month gains ~27k
  (about 56%), months further out take part of the rest, and **whole-market OI falls ~11k**:
  those positions are closed, not rolled.
- COT (weekly, all months together): **index long barely moves** around FND (about -1k lots a
  week) — index money rolls, it does not leave. In the week before FND the drop comes from
  managed money (about half of it spread positions), other reportables and swap dealers; in
  the FND week mostly from producers / merchants.

**Spread and certs**
- On a typical day the expiring-minus-next spread barely moves into FND (median daily change ~0).
- **Certs do not predict the spread's direction.** Certs level, certs change, open positions
  per cert, managed-money net and index share all show rank correlations near 0 with the move
  into FND, and a bull spread bought on low certs (T-5 to T-1, costs deducted) lost money over
  the full history (`Studies/certs_squeeze_backtest.py`).
- **Low certs go with bigger spread swings, both ways**: every large move (beyond about ±10c)
  came with certs under ~1 million bags, almost all since 2024. Read low certs as a risk
  flag, not a signal. Part of the post-2024 effect is the higher price level itself, since
  spreads are in cents.

**Z26 (FND 19 Nov 2026), at about T-15**
- Roll pace, OI and volume: inside the usual range.
- Daily spread moves: outside the past 10-90% range on several days.
- Certs: about 225k bags, the lowest reading for a December contract in the data; open
  positions about 80x certified stock.
- Reading: the roll should run through the usual index window, but with record-low certs and
  an already jumpy spread, positions left in the expiring month near FND carry more spread risk
  than usual. Risk, not direction.

**Limits**: COT is weekly (Tuesday) and sums all contract months, so it cannot attribute a move
to a single expiry or day; the certs / spread pattern since 2024 rests on a handful of
contracts; index roll dates are taken as the 5th-10th business days of the FND month, not from
the index providers' published calendars.
