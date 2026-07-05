# Data Ingestion — Vaayu AI

Five sources feed the pipeline. Run everything with:

```bash
# from the project root, with a venv active
cp .env.example .env        # then fill in the keys below
pip install pandas pyarrow requests python-dotenv numpy
python data/ingest/run_ingestion.py
```

Outputs written under `data/db/`:

| File | Produced by | Contents |
|---|---|---|
| `unified_history.parquet` | `cpcb_openaq` + `weather` | hourly AQI joined to weather, per station |
| `fire_daily.parquet` | `firms_fire` | daily count of active fires within 100 km of each station (+ mean/max FRP) |
| `city_layers/{city}_layers.geojson` | `osm_overpass` | road-density grid, industrial zones, vulnerability POIs |
| `raw_cache/` | all modules | raw API responses, keyed by request — safe to delete to force a refetch |

## Sources per city

| City | Live AQI | Historical AQI | Weather | Fire | OSM layers |
|---|---|---|---|---|---|
| Delhi | CPCB (`data.gov.in`), OpenAQ fallback | OpenAQ only (see caveat) | Open-Meteo | NASA FIRMS | Overpass |
| Bengaluru | CPCB, OpenAQ fallback | OpenAQ only | Open-Meteo | NASA FIRMS | Overpass |
| Indore | CPCB, OpenAQ fallback | OpenAQ only | Open-Meteo | NASA FIRMS | Overpass |

**CPCB's `data.gov.in` resource is a live-snapshot feed, not a historical
archive.** The "Real time Air Quality Index from various locations" resource
(id `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`) has no date-range filter — every
call returns whatever CPCB's dashboard currently holds per station. It is the
primary source for **live** AQI (`fetch_live_aqi`) and contributes a "today"
slice to `fetch_historical_aqi` when the range includes the current date. All
other historical backfill comes from **OpenAQ v3** (`/v3/sensors/{id}/hours`
with `date_from`/`date_to`) — the only source with a genuine historical query
API. This is expected behaviour, not a bug — see the `cpcb_openaq.py` docstring.

Neither source reports a ready-made AQI number; both are converted from
PM2.5/PM10 using the published CPCB sub-index breakpoints (`estimate_aqi()`),
taking the max sub-index. This is an approximation — official CPCB AQI can also
weigh in NO2/SO2/O3/CO/NH3/Pb, which are not ingested here.

## Fire detections (`firms_fire.py`)

`fetch_fire_detections(bbox, start_date, end_date)` pulls active-fire hotspots
from the [NASA FIRMS area API](https://firms.modaps.eosdis.nasa.gov/api/)
(CSV) and returns `lat, lon, detection_date, confidence, frp`.
`aggregate_daily_fire_counts()` then collapses these into the model feature:
**count of fires within 100 km of each station, per day** (plus mean/max fire
radiative power as an intensity proxy).

- The per-city bounding box is **buffered by ~150 km** before querying, so the
  Punjab/Haryana stubble-burning belt upwind of Delhi is captured — that
  regional biomass burning is the dominant driver of Delhi's autumn PM2.5 and
  the whole reason this feature exists (forecast leading signal + attribution).
- FIRMS caps a single request at a **5-day range** for the VIIRS_SNPP sources
  used here (confirmed directly against the live API — a 10-day request
  returns HTTP 400 "Invalid day range. Expects [1..5]."), so long windows are
  chunked in 5-day steps. The 375 m **VIIRS S-NPP** product is used: the
  **SP** (archive) stream for chunks older than ~60 days and **NRT** for
  recent chunks, since NRT only retains ~2 months and SP lags the present by
  a couple of months.
- Uses `VIIRS confidence` verbatim (`low`/`nominal`/`high`); duplicate hotspots
  from overlapping chunks/passes are deduped on rounded (lat, lon, date).

Requires `FIRMS_API_KEY` (free). Without it the module logs a warning and
returns an empty, correctly-typed frame — the pipeline does not crash.

## OSM layers (`osm_overpass.py`)

`fetch_city_layers(city_bbox)` issues **three independent Overpass queries** and
returns a dict:

- `road_density_grid` — total `highway=*` length per **1 km grid cell** (a
  traffic/combustion-exposure proxy), with `road_length_km` and
  `road_density_km_per_km2` per cell.
- `industrial_zones` — `landuse=industrial` ways/relations, with centroid,
  area (km²) and the polygon ring for GeoJSON.
- `vulnerability_pois` — hospitals/clinics (`hospital`), schools/colleges/
  kindergartens (`school`), and elderly-care (`elderly_care`, from
  `social_facility`), each with lat/lon.

Outputs are plain **pandas DataFrames**, not GeoDataFrames: the operations we
need (great-circle distance, polyline length, gridding, shoelace area) are a
few lines of numpy each, and pulling in the geopandas/GEOS/GDAL stack would be
a heavy, brittle dependency for a hackathon pipeline. Geometry travels as
coordinate lists and is serialised to one **GeoJSON FeatureCollection** per
city (each feature tagged with its `layer`). Raw Overpass JSON is cached to
`data/db/raw_cache/`, so the slow, rate-limited queries are paid for once.

### Overpass reliability (the free tier is genuinely flaky)

Handling built into `osm_overpass.py`, in the order it was actually needed:

1. **406 Not Acceptable** — `overpass-api.de` rejects the bare
   `python-requests` User-Agent outright. Fixed by sending a descriptive
   `User-Agent` with a contact (required by its usage policy). This was the
   first blocker and is easy to miss.
2. **429 / 504 rate-limiting & gateway timeouts** — the free tier allows ~2
   slots per IP and 429s aggressively when hammered. We retry up to 5× while
   **honouring the `Retry-After` header** (in `_http_utils`) and pause
   `OVERPASS_COOLDOWN_S` between the three queries.
3. **Smaller-bbox fallback** — if a layer still fails on the full bbox after
   retries, it is retried once against a **centre-50% (¼-area) bbox**; the
   layer's status is then recorded as `fallback`/`fallback_partial` in the
   returned `_meta` (and echoed in the run summary) so a partial extract is
   never mistaken for a complete one.
4. Public mirrors (`overpass.kumi.systems`, `overpass.private.coffee`) were
   evaluated as failovers but were unreachable from the test network, so the
   pipeline stays on the main endpoint with the retry/fallback ladder above.

If a layer ends up `failed`, it is written out **empty** and clearly flagged —
re-running later picks up where it left off thanks to the cache.

## API keys

| Key | Needed for | Where to get it |
|---|---|---|
| `CPCB_API_KEY` | Live AQI (primary) | Register at [data.gov.in](https://www.data.gov.in), find the "Real time Air Quality Index" resource, generate a key |
| `OPENAQ_API_KEY` | Historical AQI (only source with real date-range queries) | Free signup at [explore.openaq.org/register](https://explore.openaq.org/register) — **required** on all v3 endpoints |
| `FIRMS_API_KEY` | Active-fire detections | Free map key at [firms.modaps.eosdis.nasa.gov/api/map_key](https://firms.modaps.eosdis.nasa.gov/api/map_key/) |
| — | Weather + Overpass | None — Open-Meteo and Overpass need no key (Overpass needs a descriptive User-Agent, already set) |

## What was actually run and found

Full pipeline run from the project root against the live internet
(2026-07-03, 18-month window `2025-01-09 .. 2026-07-03`), venv with
`pandas`/`pyarrow`/`requests`/`python-dotenv`/`numpy`. `CPCB_API_KEY` and
`OPENAQ_API_KEY` were configured; **`FIRMS_API_KEY` was not set**.

### OSM / Overpass — fully successful for all three cities

All three layers returned `status: ok` for every city (no fallback needed once
the User-Agent + retry/`Retry-After` handling above was in place):

| City | Road cells (total km) | Industrial zones | Vulnerability POIs (hospital / school / elderly) |
|---|---|---|---|
| **Delhi** | 3,543 (42,644 km) | 382 | 3,360 (1,696 / 1,648 / 16) |
| **Bengaluru** | 1,801 (28,216 km) | 1,285 | 4,675 (2,143 / 2,500 / 32) |
| **Indore** | 1,145 (8,443 km) | 51 | 1,524 (1,169 / 350 / 5) |

Notes: the `hospital` bucket includes `amenity=clinic`, which inflates the
count relative to large hospitals; `elderly_care` counts are small because
`social_facility` senior-care tagging is sparse in Indian OSM data. Getting to
these numbers required the User-Agent fix (initial requests returned 406) and
riding out repeated 429/504 responses — expect the same on a cold cache.

### NASA FIRMS — skipped cleanly (no key)

With `FIRMS_API_KEY` unset, `fetch_fire_detections` logged the missing-key
warning and returned an empty frame for every city; `fire_daily.parquet` was
written **empty but correctly-schema'd** (`city, station_id, detection_date,
fire_count, mean_frp, max_frp`) rather than crashing. Add a free FIRMS map key
to `.env` and re-run to populate it — the 150 km-buffered Delhi bbox will then
pick up Punjab/Haryana stubble fires in the Oct–Nov window.

### AQI + weather — slow, and gated by two upstream problems

Two independent upstream issues showed up on this run:

- **CPCB `data.gov.in` was down/timing out** — every request to the resource
  read-timed-out after the full 3× retry ladder (30s each). Live "today"
  slices were therefore unavailable and all AQI had to come from OpenAQ.
- **OpenAQ v3 deep pagination is unreliable and effectively ignores the
  date window.** Each sensor's `/hours` endpoint keeps returning full 1000-row
  pages well past the requested `date_to`, so the client paginates through the
  sensor's *entire* history until a page finally `408 Request Timeout`s
  (around page 11–18). Across all of Delhi's OpenAQ sensors this makes an
  18-month backfill run for **hours**, and even a 30-day `VAAYU_HISTORY_DAYS`
  window does not shorten the pagination. This is pre-existing `cpcb_openaq`
  behaviour, not introduced by the fire/OSM work.

Because of that, `HISTORY_DAYS` is now overridable via the
`VAAYU_HISTORY_DAYS` env var (default 540) so a full AQI backfill can be run
deliberately/off-line rather than blocking every pipeline run. The
fire-daily and OSM-layer outputs below were generated on their own (they do
not depend on the AQI leg), so `unified_history.parquet` on this machine
currently holds only whatever partial OpenAQ slice a completed backfill
leaves — **run `python data/ingest/run_ingestion.py` and let it finish to
populate the full AQI table.**

### Output files actually written this run

- `data/db/city_layers/Delhi_layers.geojson` — 7,285 features (3,543 road
  cells + 382 industrial + 3,360 POIs)
- `data/db/city_layers/Bengaluru_layers.geojson` — 7,761 features (1,801 +
  1,285 + 4,675)
- `data/db/city_layers/Indore_layers.geojson` — 2,720 features (1,145 + 51 +
  1,524)
- `data/db/fire_daily.parquet` — 0 rows (no `FIRMS_API_KEY`), correct schema
  `city, station_id, detection_date, fire_count, mean_frp, max_frp`

## Update (2026-07-05) — the AQI/FIRMS blockers above are now fixed

The two upstream problems in "AQI + weather — slow, and gated by two upstream
problems" above turned out to have real, fixable root causes, not just
upstream flakiness to wait out. All three were found and fixed once a real
`FIRMS_API_KEY` and `OPENAQ_API_KEY` were exercised end-to-end:

1. **`cache_key_for()` never actually sanitized the cache-file prefix**, despite
   its docstring's claim of being "filesystem-safe." OpenAQ's ISO timestamps
   (`2025-01-10T00:00:00Z`) embed colons, which Windows rejects in filenames —
   every `/hours` cache write raised `OSError 22` before it could return data,
   silently breaking pagination. Fixed in `_http_utils.py`.
2. **OpenAQ v3's `/hours` endpoint ignores `date_from`/`date_to` server-side**
   and always paginates from a sensor's absolute first reading — this, not
   flakiness, is why it "paginated through the sensor's entire history." Fixed
   by checking each row's own timestamp client-side and stopping once past
   `date_to`, plus skipping sensors whose `datetimeFirst`/`datetimeLast`
   metadata shows no overlap with the requested window at all (some Bengaluru
   stations had stopped reporting as early as 2018).
3. **`firms_fire.py`'s `MAX_DAY_RANGE` was hardcoded to 10**, but NASA FIRMS'
   VIIRS_SNPP sources only accept a 1–5 day range (HTTP 400 otherwise) — every
   fire request was failing until this was caught and fixed to 5.

**Result after the fix**, full 3-city backfill:

| City | AQI rows | Stations | Fire station-days | Unique FIRMS detections |
|---|---:|---:|---:|---:|
| Delhi | 372,094 | 24 | 10,398 | 21,941 |
| Bengaluru | 138,377 | 18 | 5,695 | 37,520 |
| Indore | 45,331 | 5 | 1,914 | 52,289 |
| **Total** | **555,802** | **47** (46 trained — see `docs/BENCHMARKS.md`) | **18,007** | **111,750** |

`unified_history.parquet` now holds all three cities (300,349 rows with a
weather join). The forecast model trained on this data beats persistence at
every horizon in all three cities — see `docs/BENCHMARKS.md` §1. Indore's
backfill alone took ~5.4 hours due to real network drops (DNS resolution
failures, connection aborts) rather than the pagination bug, so expect a cold
Indore run to still take a while even with the fixes in place.

## Re-running

Re-running is safe and fast on repeat: every raw API response is cached to
`data/db/raw_cache/` keyed by request parameters, so only new date ranges,
stations, or bounding boxes trigger real network calls. Delete
`data/db/raw_cache/` to force a full refetch.

One caveat on cache reuse: the ingestion window is a **rolling** `[today -
VAAYU_HISTORY_DAYS, today]`, so re-running on a later date shifts every chunk
boundary by that many days and busts most of the OpenAQ/FIRMS cache even
though the actual window barely changed — expect a re-run days later to
re-fetch most of the AQI and fire history rather than hitting cache.
