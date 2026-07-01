# Data Ingestion — Vaayu AI

## Sources per city

| City | Live AQI | Historical AQI | Weather |
|---|---|---|---|
| Delhi | CPCB (`data.gov.in`), OpenAQ fallback | OpenAQ only (see caveat below) | Open-Meteo |
| Bengaluru | CPCB (`data.gov.in`), OpenAQ fallback | OpenAQ only | Open-Meteo |
| Indore | CPCB (`data.gov.in`), OpenAQ fallback | OpenAQ only | Open-Meteo |

**CPCB's `data.gov.in` resource is a live-snapshot feed, not a historical
archive.** The "Real time Air Quality Index from various locations" resource
(id `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`) has no date-range filter — every
call returns whatever CPCB's dashboard currently holds per station. It is
used as the primary, authoritative source for **live** AQI (`fetch_live_aqi`),
and it contributes a "today" slice to `fetch_historical_aqi` when the
requested range includes the current date. All other historical backfill
comes from **OpenAQ v3**, which is the only one of the two sources with a
genuine historical query API (`/v3/sensors/{id}/hours` with `date_from`/
`date_to`). This is expected behavior, not a bug — see the module docstring
in `cpcb_openaq.py` for the full reasoning.

Neither source reports a ready-made single AQI number consistently (CPCB
gives per-pollutant concentrations per station; OpenAQ gives raw pollutant
concentrations only). Both are converted to an approximate AQI using the
published CPCB PM2.5/PM10 sub-index breakpoints, taking the max sub-index
— see `estimate_aqi()` in `cpcb_openaq.py`. This is documented as an
approximation: official CPCB AQI can also weigh in NO2/SO2/O3/CO/NH3/Pb,
which are not ingested here.

## API keys required

| Key | Needed for | Where to get it |
|---|---|---|
| `CPCB_API_KEY` | Live AQI (primary) | Register at [data.gov.in](https://www.data.gov.in), find the "Real time Air Quality Index" resource, generate an API key |
| `OPENAQ_API_KEY` | Historical AQI (the only source with real date-range queries) | Free signup at [explore.openaq.org/register](https://explore.openaq.org/register) — **required** for all v3 endpoints, despite being labeled "optional" in an earlier draft of `.env.example` |
| — | Weather (historical + forecast) | None — Open-Meteo's `/v1/archive` and `/v1/forecast` are fully free, no key |

## What was actually run and found (as of this commit, no keys configured)

Ran against the live internet with the venv in this repo (`pandas`, `pyarrow`,
`requests`, `python-dotenv`), with **no `.env` file present** (i.e. no real
API keys yet):

- **Weather module — genuinely verified working.** `fetch_historical_weather`
  for Delhi's coordinates over the last 7 days returned 192 real hourly rows
  (8 calendar days inclusive) with real temperature/humidity/wind/
  precipitation/`boundary_layer_height` values — no fallback needed, the
  full variable list was accepted by the archive endpoint on the first try.
  `fetch_weather_forecast` for the next 24h returned 24 real forecast rows.
- **CPCB/OpenAQ module — degrades exactly as designed, no crashes.** With
  `CPCB_API_KEY` and `OPENAQ_API_KEY` both unset:
  - `fetch_live_aqi()` logs a clear warning for each missing key and returns
    an empty, correctly-schema'd DataFrame (`(0, 9)`) rather than raising.
  - `fetch_historical_aqi()` for Bengaluru over the last 30 days behaves the
    same way — logs why CPCB can't help (no historical query support) and
    why OpenAQ can't be tried (no key), then returns an empty DataFrame.
- **Full pipeline (`run_ingestion.py`), all 3 cities, 540-day window
  (18 months)**: completed without errors in under a second, correctly
  reported `0 AQI rows` for Delhi, Bengaluru, and Indore, and wrote a
  **correctly-schema'd but empty** `data/db/unified_history.parquet`
  (16 columns, 0 rows) rather than crashing or silently writing garbage.
- **AQI estimation formula — spot-checked by hand.** `estimate_aqi(100, 80)`
  returned `233.8`: PM2.5=100 falls in the (90.1–120.0 → 201–300) band,
  interpolating to 233.8; PM10=80 falls in the (50.1–100.0 → 51–100) band,
  interpolating to ~80.4; `max(233.8, 80.4) = 233.8` — matches the code's
  output exactly.
- **Caching verified.** Only the two real Open-Meteo calls produced cache
  files under `data/db/raw_cache/`; the skipped CPCB/OpenAQ calls (no key)
  correctly produced zero cache entries, since no HTTP request was ever made
  for them.

**No data quality issues could be observed yet** because no real AQI data
was retrieved — that requires adding real `CPCB_API_KEY` and `OPENAQ_API_KEY`
values to `.env`. Once those are added, re-run and watch for: sparse/missing
OpenAQ station coverage in Tier-2 cities (Indore in particular is likely to
have far fewer OpenAQ sensors than Delhi), gaps where a CPCB station reports
under a slightly different `city` label than expected (only "Bengaluru" and
"Bangalore" aliases are currently handled), and any station whose lat/lon
comes back null (skips the weather join, logged explicitly).

## Re-running ingestion

```bash
# from the project root, with a venv active
cp .env.example .env        # then fill in CPCB_API_KEY and OPENAQ_API_KEY
pip install pandas pyarrow requests python-dotenv
python data/ingest/run_ingestion.py
```

Re-running is safe and fast on repeat: every raw API response is cached to
`data/db/raw_cache/` keyed by request parameters, so only new date ranges or
new stations trigger real network calls. Delete `data/db/raw_cache/` to force
a full refetch.
