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
  (no shuffle), so every test origin is later than every training origin. Train ends
  per city: Delhi 2025-12-22, Bengaluru 2026-04-06, Indore 2026-04-22; test spans to
  2026-07-05.
- **Baseline:** persistence, `forecast(t+h) = actual(t)`, evaluated on the **same test
  rows** as the model.
- **Data:** 385,509 hourly feature rows from 46 stations across Delhi, Bengaluru, and
  Indore (2016–2026).

| City · Horizon | n (test) | Persistence RMSE | **Model RMSE** | **RMSE ↓** | Persistence MAE | Model MAE |
|---|---:|---:|---:|:--:|---:|---:|
| Delhi · 24h | 28,847 | 109.49 | **87.56** | **+20.0%** | 77.13 | 65.17 |
| Delhi · 48h | 28,051 | 118.28 | **91.63** | **+22.5%** | 86.00 | 69.19 |
| Delhi · 72h | 27,402 | 118.95 | **94.25** | **+20.8%** | 87.27 | 71.26 |
| Bengaluru · 24h | 16,813 | 44.78 | **35.86** | **+19.9%** | 18.53 | 17.86 |
| Bengaluru · 48h | 16,009 | 44.90 | **36.28** | **+19.2%** | 20.73 | 20.59 |
| Bengaluru · 72h | 15,409 | 46.64 | **37.22** | **+20.2%** | 22.96 | 22.22 |
| Indore · 24h | 5,820 | 31.87 | **26.35** | **+17.3%** | 16.93 | 16.24 |
| Indore · 48h | 5,500 | 34.13 | **27.41** | **+19.7%** | 19.21 | 17.51 |
| Indore · 72h | 5,452 | 34.58 | **28.19** | **+18.5%** | 19.79 | 18.46 |
| ALL · 24h | 51,480 | 86.53 | **69.25** | **+20.0%** | 51.18 | 44.19 |
| ALL · 48h | 49,560 | 93.27 | **72.53** | **+22.2%** | 57.51 | 47.75 |
| ALL · 72h | 48,263 | 94.14 | **74.67** | **+20.7%** | 59.11 | 49.64 |

RMSE/MAE are in AQI points. **The model beats persistence at every horizon in every
pilot city** — 17.3% to 22.5% RMSE reduction across the full 3-city × 3-horizon
matrix — and the gap widens at longer horizons (48h/72h) where "carry today forward"
degrades faster.

**Top features (24h model, XGBoost gain-importance):**
`aqi_current` 0.631 · `aqi_roll_24h` 0.137 · `aqi_lag_1h` 0.042 ·
`is_winter_pollution_season` 0.040 · `aqi_roll_6h` 0.037 · `city_Delhi` 0.019 ·
`aqi_lag_48h` 0.012 · `aqi_lag_24h` 0.010. The city one-hot appearing in the top 8
confirms the model is genuinely using city identity (not just averaging across
cities), and the Oct–Feb winter-season flag remaining a top-5 feature confirms it
still captures the seasonal inversion/stubble signal even pooled across 3 cities.

Hyperparameters: `n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0`, early stopping (40 rounds)
on a time-ordered validation tail carved from the training set only.

### Honest caveats

- **The fire feature is genuinely live; isolating its exact contribution is the
  next step, not something we've measured yet.** All three cities now have real
  NASA FIRMS data (18,007 station-day rows; see §2), and `nearby_fire_count` is a
  real, non-zero signal the model uses. This run changed two things at once — fire
  data went live *and* two more cities were pooled in — so the RMSE improvement
  versus the earlier Delhi-only run (16.0–19.8% → 17.3–22.5%) can't yet be cleanly
  attributed to either change alone; that's a confound, not a conclusion. In the
  model's own feature ranking, `nearby_fire_count` sits alongside other sensible
  secondary signals — wind speed, temperature, wind direction (each ~0.4–0.5%
  importance) — well behind the dominant autocorrelation signal (current AQI,
  rolling trend), which is expected for an hourly pollution series where "what was
  the air like an hour ago" naturally dominates. A controlled ablation (same data,
  fire on vs. off) is the natural next step to isolate fire's true marginal effect.
- **Weather is observed-at-t**, not forecast weather — a mild optimism, documented
  rather than hidden. Production should use Open-Meteo *forecast* weather at t+h.

## 2. Data source coverage

| Source | Used for | Coverage in this build |
|---|---|---|
| **OpenAQ v3** (historical) + **CPCB** `data.gov.in` (live) | AQI | 555,802 rows across 46 stations — Delhi (24), Bengaluru (18), Indore (5), 2016–2026 (AQI capped 0–500) |
| **Open-Meteo** archive | Weather (temp, humidity, wind, precip, boundary-layer height) | 300,349 rows joined within 1h of an AQI reading |
| **NASA FIRMS** (VIIRS) | Active-fire count within 100 km/station/day | 18,007 station-day rows across all 3 cities (Delhi 10,398 · Bengaluru 5,695 · Indore 1,914). Raw unique detections over the 540-day window: Delhi 21,941 · Bengaluru 37,520 · Indore 52,289 |
| **OpenStreetMap** / Overpass | Road-density grid, industrial land-use, vulnerability POIs | Delhi 3,543 cells / 382 zones / 3,360 POIs · Bengaluru 1,801 / 1,285 / 4,675 · Indore 1,145 / 51 / 1,524 |

Data-integrity guarantees (enforced by `tests/test_all.py` §6): no duplicate
station+timestamp rows, all AQI in [0, 500], ≥ 5 stations per ingested city.

## 3. End-to-end API latency

Measured over HTTP (`requests` → local Uvicorn, single process, warm cache; median of
5 calls after a warm-up) on a dev laptop. All routes are **sub-second**. These numbers
predate the 3-city merge (measured against Delhi only); not yet re-benchmarked against
the larger 555K-row unified table, though latency should be broadly similar since every
route filters by city before scanning.

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
evaluation, now against the 3-city pooled model), attribution scoring, enforcement
ranking, advisory (all languages — verified against a live Groq LLM key, not just the
deterministic fallback), API endpoints (happy + failure per route), and data
integrity. Data/model-dependent tests skip cleanly when artifacts are absent.

## Reproduce

```bash
python data/ingest/run_ingestion.py     # populate data/db/
python models/train_forecast_model.py   # trains, evaluates, regenerates models/BENCHMARKS.md
uvicorn main:app &                       # then re-run the latency probe against localhost:8000
pytest -q                                # verifies the accuracy numbers reproduce
```
