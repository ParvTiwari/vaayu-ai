# Vaayu AI — Benchmarks

All numbers below are **measured**, not estimated. The forecast metrics are
reproduced by an automated test (`tests/test_all.py::test_forecast_reproduces_documented_metrics`),
which rebuilds the features and re-runs the held-out evaluation rather than reading
the JSON's own claims. The auto-generated model card lives at `models/BENCHMARKS.md`;
this document is the curated submission version with data coverage, latency, and
language coverage added.

## 1. Forecast accuracy vs. persistence baseline

- **Model:** one XGBoost regressor per horizon (24/48/72h), pooled across cities.
- **Target:** approximate Indian AQI (from PM2.5/PM10 CPCB sub-indices, capped at 500).
- **Split:** time-based — the most recent **15% of each city's timeline** is held out
  (no shuffle), so every test origin is later than every training origin. Delhi train
  ends 2025-12-22; test spans to 2026-07-03.
- **Baseline:** persistence, `forecast(t+h) = actual(t)`, evaluated on the **same test
  rows** as the model.
- **Data:** 211,205 hourly feature rows from 23 Delhi stations (2016–2026).

| City · Horizon | n (test) | Persistence RMSE | **Model RMSE** | **RMSE ↓** | Persistence MAE | Model MAE |
|---|---:|---:|---:|:--:|---:|---:|
| Delhi · 24h | 28,847 | 109.49 | **91.29** | **+16.6%** | 77.13 | 69.13 |
| Delhi · 48h | 28,051 | 118.28 | **94.92** | **+19.8%** | 86.00 | 72.76 |
| Delhi · 72h | 27,402 | 118.95 | **97.15** | **+18.3%** | 87.27 | 73.34 |

RMSE/MAE are in AQI points. **The model beats persistence at every horizon**, and the
gap widens at longer horizons (48h/72h) where "carry today forward" degrades faster.

**Top features (24h model, XGBoost gain-importance):**
`aqi_current` 0.537 · `aqi_roll_6h` 0.172 · `is_winter_pollution_season` 0.074 ·
`aqi_roll_24h` 0.062 · `aqi_lag_1h` 0.042 · `month` 0.018 · `aqi_lag_48h` 0.013 ·
`aqi_lag_24h` 0.012. The Oct–Feb winter-season flag ranking 3rd confirms the model
learned Delhi's seasonal inversion/stubble signal.

Hyperparameters: `n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0`, early stopping (40 rounds)
on a time-ordered validation tail carved from the training set only.

### Honest caveats

- **Only Delhi has forecast results** — Bengaluru/Indore AQI backfill is pending
  (OpenAQ deep-pagination is slow); the pipeline supports them the moment their
  history is ingested.
- **Fire feature was inactive** (`nearby_fire_count = 0` — no `FIRMS_API_KEY` at
  ingestion). Expect gains on Delhi's Oct–Nov spikes once FIRMS data is present.
- **Weather is observed-at-t**, not forecast weather — a mild optimism, documented
  rather than hidden. Production should use Open-Meteo *forecast* weather at t+h.

## 2. Data source coverage

| Source | Used for | Coverage in this build |
|---|---|---|
| **OpenAQ v3** (historical) + **CPCB** `data.gov.in` (live) | AQI | 372,094 rows, **24 Delhi stations**, 2016–2026 (AQI capped 0–500) |
| **Open-Meteo** archive | Weather (temp, humidity, wind, precip, boundary-layer height) | 128,343 rows joined within 1h of an AQI reading |
| **NASA FIRMS** (VIIRS) | Active-fire count within 100 km/station/day | 0 (no `FIRMS_API_KEY`; feature scaffolded and tested) |
| **OpenStreetMap** / Overpass | Road-density grid, industrial land-use, vulnerability POIs | Delhi 3,543 cells / 382 zones / 3,360 POIs · Bengaluru 1,801 / 1,285 / 4,675 · Indore 1,145 / 51 / 1,524 |

Data-integrity guarantees (enforced by `tests/test_all.py` §6): no duplicate
station+timestamp rows, all AQI in [0, 500], ≥ 5 stations per ingested city.

## 3. End-to-end API latency

Measured over HTTP (`requests` → local Uvicorn, single process, warm cache; median of
5 calls after a warm-up) on a dev laptop. All routes are **sub-second**.

| Endpoint | Method | Median latency | Notes |
|---|---|---:|---|
| `/advisory` | POST | **~2 ms** | deterministic guidance (no LLM key set) |
| `/attribution/{city}` | GET | **~9 ms** | in-memory geojson, point-in-polygon |
| `/stations/{city}` | GET | **~95 ms** | parquet scan for latest-per-station |
| `/forecast/{city}` | GET | **~130 ms** | feature build + 3 XGBoost predicts |
| `/heatmap/{city}` | GET | **~132 ms** | IDW over the city grid (3,465 cells) |
| `/query` (full pipeline) | POST | **~139 ms** | forecast → attribution → advisory + assemble |
| `/enforcement-priorities/{city}` | GET | **~640 ms** | POI + road-density counts for all 24 stations |

Notes: city-layer data is loaded/cached at first use, so repeated calls stay fast;
the Streamlit UI additionally caches responses with `st.cache_data`. `/enforcement`
is the heaviest (per-station spatial counts over ~3,360 POIs) and is the first
candidate for precomputation if needed. LLM narration, when enabled via
`LLM_API_KEY`, adds one Claude round-trip to `/advisory` and the `full` path.

## 4. Language coverage

| Language | Code | Deterministic CPCB guidance | LLM localization | Voice (gTTS) | Smoke-tested |
|---|---|---|---|---|---|
| English | `en` | ✅ full (6 CPCB categories × general/sensitive) | ✅ | ✅ | ✅ |
| Hindi | `hi` | ✅ full (hand-translated) | ✅ | ✅ | ✅ (Devanagari asserted) |
| Kannada | `kn` | via English fallback | ✅ | ✅ | ✅ |

Advisory smoke tests (`tests/test_all.py` §4) assert non-empty output **and** a
populated `sources_cited` for every language. The health facts are always sourced
from the hardcoded CPCB table; the LLM only rephrases/localizes and is instructed
never to invent AQI numbers or health claims. Without an `LLM_API_KEY` the
deterministic path is used (English/Hindi verbatim from the table; Kannada falls back
to the English guidance text, clearly flagged).

## 5. Test suite

`pytest -q` → **55 passed** across six areas: forecast regression (re-run held-out
evaluation), attribution scoring, enforcement ranking, advisory (all languages), API
endpoints (happy + failure per route), and data integrity. Data/model-dependent tests
skip cleanly when artifacts are absent.

## Reproduce

```bash
python data/ingest/run_ingestion.py     # populate data/db/
python models/train_forecast_model.py   # trains, evaluates, regenerates models/BENCHMARKS.md
uvicorn main:app &                       # then re-run the latency probe against localhost:8000
pytest -q                                # verifies the accuracy numbers reproduce
```
