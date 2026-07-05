# Vaayu AI — Forecast Model Card & Benchmarks

_Generated 2026-07-05T02:57:47.239298+00:00 by `models/train_forecast_model.py`._

## Task

Forecast the (approximate Indian) **AQI** at each monitoring station **24h, 48h
and 72h ahead**, and beat a naive persistence baseline.

## Model

- **Algorithm:** XGBoostRegressor (gradient-boosted trees), one model per
  horizon.
- **Design choice:** one model per horizon, pooled across cities (city one-hot + lat/lon). We chose per-horizon models over a
  single horizon-as-feature model because each model then predicts AQI at
  exactly t+h, making the target, evaluation and persistence comparison direct
  and leak-free. Cities are pooled (not one model each) so sparse cities borrow
  signal from dense ones; city identity and location remain available via the
  one-hot city columns and `lat`/`lon`.
- **Hyperparameters:** `{"n_estimators": 600, "max_depth": 6, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 1.0, "objective": "reg:squarederror", "eval_metric": "rmse", "early_stopping_rounds": 40, "n_jobs": 4, "random_state": 42}` (early stopping on a
  time-ordered validation tail carved from the training set only).

## Features

Lag & trend: `aqi_current` (value at t), `aqi_lag_1h/24h/48h`,
`aqi_roll_6h/24h`. Weather at t: `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`, `precipitation`, `boundary_layer_height`,
plus wind direction as `wind_dir_sin`/`wind_dir_cos`. Fire:
`nearby_fire_count` (FIRMS detections within 100 km that day) — **active in this
run: True**. Calendar: `hour`, `day_of_week`, `month`,
`is_winter_pollution_season` (Oct–Feb, North-India inversion + stubble season).
Location/identity: `lat`, `lon`, one-hot `city_*`.

Full ordered feature list is in `models/checkpoints/forecast_metadata.json`.

### Top features (24h model, by XGBoost gain-importance)
- `aqi_current` — 0.63123
- `aqi_roll_24h` — 0.13749
- `aqi_lag_1h` — 0.04229
- `is_winter_pollution_season` — 0.04032
- `aqi_roll_6h` — 0.03713
- `city_Delhi` — 0.019
- `aqi_lag_48h` — 0.01174
- `aqi_lag_24h` — 0.01046

## Data & split

- **Rows:** 385,509 hourly feature rows from
  46 station(s) across 3 city/cities
  (Bengaluru, Delhi, Indore).
- **Timeline:** 2016-01-29T23:30:00+00:00 → 2026-07-04T21:30:00+00:00.
- **Split:** time-based, **most recent 15%
  of each city's timeline held out for test** (no random shuffle). Every test
  origin is later than every training origin for that city, so there is no
  temporal leakage.
- **Persistence baseline:** `forecast(t+h) = actual(t)` (naive carry-forward),
  evaluated on the exact same test rows as the model.

## Results — RMSE & MAE, per city × horizon

| City | Horizon | n (test) | Persistence RMSE | Model RMSE | RMSE ↓ | Persistence MAE | Model MAE | Beats baseline? |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| Bengaluru | 24h | 16813 | 44.78 | 35.86 | +19.9% | 18.53 | 17.86 | ✅ yes |
| Bengaluru | 48h | 16009 | 44.9 | 36.28 | +19.2% | 20.73 | 20.59 | ✅ yes |
| Bengaluru | 72h | 15409 | 46.64 | 37.22 | +20.2% | 22.96 | 22.22 | ✅ yes |
| Delhi | 24h | 28847 | 109.49 | 87.56 | +20.0% | 77.13 | 65.17 | ✅ yes |
| Delhi | 48h | 28051 | 118.28 | 91.63 | +22.5% | 86.0 | 69.19 | ✅ yes |
| Delhi | 72h | 27402 | 118.95 | 94.25 | +20.8% | 87.27 | 71.26 | ✅ yes |
| Indore | 24h | 5820 | 31.87 | 26.35 | +17.3% | 16.93 | 16.24 | ✅ yes |
| Indore | 48h | 5500 | 34.13 | 27.41 | +19.7% | 19.21 | 17.51 | ✅ yes |
| Indore | 72h | 5452 | 34.58 | 28.19 | +18.5% | 19.79 | 18.46 | ✅ yes |
| ALL | 24h | 51480 | 86.53 | 69.25 | +20.0% | 51.18 | 44.19 | ✅ yes |
| ALL | 48h | 49560 | 93.27 | 72.53 | +22.2% | 57.51 | 47.75 | ✅ yes |
| ALL | 72h | 48263 | 94.14 | 74.67 | +20.7% | 59.11 | 49.64 | ✅ yes |

_RMSE/MAE are in AQI points. "RMSE ↓" is the percentage RMSE reduction of the
model versus persistence (positive = model better). "ALL" pools every city's
test rows._

## Verdict — does the model beat persistence?

- **Bengaluru / 24h** beats persistence by **+19.9% RMSE**.
- **Bengaluru / 48h** beats persistence by **+19.2% RMSE**.
- **Bengaluru / 72h** beats persistence by **+20.2% RMSE**.
- **Delhi / 24h** beats persistence by **+20.0% RMSE**.
- **Delhi / 48h** beats persistence by **+22.5% RMSE**.
- **Delhi / 72h** beats persistence by **+20.8% RMSE**.
- **Indore / 24h** beats persistence by **+17.3% RMSE**.
- **Indore / 48h** beats persistence by **+19.7% RMSE**.
- **Indore / 72h** beats persistence by **+18.5% RMSE**.

## Honest caveats

- **Persistence is a strong baseline at short horizons.** AQI is highly
  autocorrelated hour-to-hour, so at 24h persistence is hard to beat; the gap
  should widen in the model's favour at 48h/72h where "just carry today
  forward" degrades faster.
- **Weather is observed-at-t, not forecast weather.** In production the 24–72h
  features should use Open-Meteo *forecast* weather; using observed-at-t weather
  here is a mild optimism and is documented rather than hidden.
- **AQI is an approximation** derived from PM2.5/PM10 sub-indices only (no
  NO₂/SO₂/O₃/CO), inherited from the ingestion layer.
- **Fire feature:** `nearby_fire_count` is genuinely active this run — real NASA FIRMS data, not zeros. Its exact marginal contribution isn't isolated yet, though: if this run also changed city coverage versus whatever you're comparing it to, that's a confound, not a clean measurement. A controlled ablation (identical data, fire on vs. off) is what would isolate its own effect.
- **Coverage:** only cities with sufficient cached/ingested history appear
  above; add the others by completing their AQI backfill and re-running.

## Reproduce

```bash
python data/ingest/run_ingestion.py     # populate unified_history + fire_daily
python models/train_forecast_model.py   # trains, evaluates, writes this file
```
