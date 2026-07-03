# Vaayu AI — Forecast Model Card & Benchmarks

_Generated 2026-07-03T09:50:04.509228+00:00 by `models/train_forecast_model.py`._

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
run: False**. Calendar: `hour`, `day_of_week`, `month`,
`is_winter_pollution_season` (Oct–Feb, North-India inversion + stubble season).
Location/identity: `lat`, `lon`, one-hot `city_*`.

Full ordered feature list is in `models/checkpoints/forecast_metadata.json`.

### Top features (24h model, by XGBoost gain-importance)
- `aqi_current` — 0.55052
- `aqi_roll_6h` — 0.1058
- `aqi_roll_24h` — 0.08064
- `is_winter_pollution_season` — 0.0613
- `aqi_lag_1h` — 0.04323
- `month` — 0.02467
- `aqi_lag_48h` — 0.01696
- `aqi_lag_24h` — 0.0144

## Data & split

- **Rows:** 211,205 hourly feature rows from
  23 station(s) across 1 city/cities
  (Delhi).
- **Timeline:** 2016-01-29T23:30:00+00:00 → 2026-07-03T06:30:00+00:00.
- **Split:** time-based, **most recent 15%
  of each city's timeline held out for test** (no random shuffle). Every test
  origin is later than every training origin for that city, so there is no
  temporal leakage.
- **Persistence baseline:** `forecast(t+h) = actual(t)` (naive carry-forward),
  evaluated on the exact same test rows as the model.

## Results — RMSE & MAE, per city × horizon

| City | Horizon | n (test) | Persistence RMSE | Model RMSE | RMSE ↓ | Persistence MAE | Model MAE | Beats baseline? |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| Delhi | 24h | 28847 | 132.26 | 106.47 | +19.5% | 84.53 | 72.41 | ✅ yes |
| Delhi | 48h | 28051 | 140.85 | 108.87 | +22.7% | 93.39 | 75.34 | ✅ yes |
| Delhi | 72h | 27402 | 139.6 | 111.25 | +20.3% | 94.39 | 77.83 | ✅ yes |
| ALL | 24h | 28847 | 132.26 | 106.47 | +19.5% | 84.53 | 72.41 | ✅ yes |
| ALL | 48h | 28051 | 140.85 | 108.87 | +22.7% | 93.39 | 75.34 | ✅ yes |
| ALL | 72h | 27402 | 139.6 | 111.25 | +20.3% | 94.39 | 77.83 | ✅ yes |

_RMSE/MAE are in AQI points. "RMSE ↓" is the percentage RMSE reduction of the
model versus persistence (positive = model better). "ALL" pools every city's
test rows._

## Verdict — does the model beat persistence?

- **Delhi / 24h** beats persistence by **+19.5% RMSE**.
- **Delhi / 48h** beats persistence by **+22.7% RMSE**.
- **Delhi / 72h** beats persistence by **+20.3% RMSE**.

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
- **Fire feature:** if `fire_feature_active` is false above, `nearby_fire_count`
  was uniformly 0 (no `FIRMS_API_KEY` at ingestion time) and contributed
  nothing — expect gains on Delhi's Oct–Nov spikes once FIRMS data is present.
- **Coverage:** only cities with sufficient cached/ingested history appear
  above; add the others by completing their AQI backfill and re-running.

## Reproduce

```bash
python data/ingest/run_ingestion.py     # populate unified_history + fire_daily
python models/train_forecast_model.py   # trains, evaluates, writes this file
```
