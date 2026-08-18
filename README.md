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
- **`Dashboard/oi_progression.py`** — copied verbatim from the ICE source
  (pure parquet consumer, zero API calls, confirmed by grep). OI Progression
  and Volume tabs, DTE-aligned seasonality banding.
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
streamlit run Dashboard/oi_progression.py
```

Requires an authenticated LSEG Workspace/Eikon session on the host running
the builder.
