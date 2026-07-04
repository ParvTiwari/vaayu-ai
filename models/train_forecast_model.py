"""
Train AQI forecasting models (XGBoost) from data/db/unified_history.parquet,
join in the FIRMS daily fire feature, evaluate against a naive persistence
baseline, and persist the models + metadata to models/checkpoints/.

MODELLING CHOICE — one model per horizon, pooled across cities
--------------------------------------------------------------
We train ONE XGBoost regressor per forecast horizon (24h, 48h, 72h): three
models total, each pooled across all cities with the city one-hot-encoded and
station lat/lon included as features. Rationale:
  * It is the simpler of the two options the brief offered ("per-horizon" vs
    "single model with horizon as a feature") to implement *well*: each model
    predicts AQI at exactly t+h, so the target, the evaluation, and the
    persistence comparison are all direct — no row replication, no horizon
    column, no risk of the horizon feature interacting oddly with lags.
  * Pooling cities (rather than a separate model per city) shares signal across
    stations, which matters when some cities have far sparser coverage than
    Delhi. City identity and location are still available to the trees via the
    one-hot city columns and lat/lon.
Metrics are reported per city AND per horizon regardless, and the train/test
split is done PER CITY on the timeline (most recent ~15% held out) so there is
no temporal leakage within any city.

LEAKAGE CONTROL
---------------
  * Features at time t use only information available at t: the current AQI,
    lags (t-1h/-24h/-48h), trailing rolling means, current weather, current-day
    fire count, and calendar fields. The target is AQI at t+h. Weather at t is
    "known now", not future weather, so it is not leakage.
  * Each station is reindexed to a complete hourly grid so lags/targets align by
    real timestamp across gaps; rows are only used as a forecast origin when the
    current-hour AQI is a real observation (not a gap-fill).
  * The test set is strictly the most recent slice of each city's timeline, so
    every test origin is later than every training origin for that city.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The console report below prints "↓" — Windows terminals default to a cp1252
# stdout encoding that can't represent it and crashes on print(). Force UTF-8
# with a safe fallback rather than hunting down every non-ASCII character.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import xgboost as xgb  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

UNIFIED_PATH = ROOT / "data" / "db" / "unified_history.parquet"
FIRE_PATH = ROOT / "data" / "db" / "fire_daily.parquet"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METADATA_PATH = CHECKPOINT_DIR / "forecast_metadata.json"
BENCHMARKS_PATH = ROOT / "models" / "BENCHMARKS.md"

HORIZONS = [24, 48, 72]
TEST_FRACTION = 0.15          # most-recent share of each city's timeline held out
MIN_STATION_ROWS = 200        # skip stations with too little history to lag
WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "boundary_layer_height",
]
# Feature columns fed to the model (city one-hot columns are appended at runtime).
BASE_FEATURES = [
    "aqi_current",
    "aqi_lag_1h",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_roll_6h",
    "aqi_roll_24h",
    *WEATHER_COLS,
    "wind_dir_sin",
    "wind_dir_cos",
    "nearby_fire_count",
    "hour",
    "day_of_week",
    "month",
    "is_winter_pollution_season",
    "lat",
    "lon",
]

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    early_stopping_rounds=40,
    n_jobs=4,
    random_state=42,
)


# --- feature engineering -----------------------------------------------------


def _fire_lookup(fire_df: pd.DataFrame) -> dict:
    """(station_id, date) -> nearby fire count, from fire_daily.parquet."""
    if fire_df.empty or "detection_date" not in fire_df.columns:
        return {}
    lut: dict = {}
    for _, r in fire_df.dropna(subset=["detection_date"]).iterrows():
        d = pd.to_datetime(r["detection_date"]).date()
        lut[(r["station_id"], d)] = lut.get((r["station_id"], d), 0) + int(r["fire_count"])
    return lut


def _build_station_features(g: pd.DataFrame, station_id, fire_lut: dict) -> pd.DataFrame | None:
    g = g.dropna(subset=["aqi"]).drop_duplicates("timestamp", keep="last")
    if len(g) < MIN_STATION_ROWS:
        return None
    g = g.set_index("timestamp").sort_index()

    # Reindex to a dense hourly grid so shift(k) means "k hours", not "k rows".
    full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
    g = g.reindex(full_idx)
    g.index.name = "timestamp"

    for col in ("city", "lat", "lon"):
        g[col] = g[col].ffill().bfill()
    for col in WEATHER_COLS + ["wind_direction_10m"]:
        if col in g:
            g[col] = pd.to_numeric(g[col], errors="coerce").ffill(limit=3)
        else:
            g[col] = np.nan

    aqi = pd.to_numeric(g["aqi"], errors="coerce")
    g["aqi_current"] = aqi
    g["aqi_lag_1h"] = aqi.shift(1)
    g["aqi_lag_24h"] = aqi.shift(24)
    g["aqi_lag_48h"] = aqi.shift(48)
    g["aqi_roll_6h"] = aqi.rolling(6, min_periods=3).mean()
    g["aqi_roll_24h"] = aqi.rolling(24, min_periods=6).mean()

    wd = np.radians(pd.to_numeric(g.get("wind_direction_10m"), errors="coerce"))
    g["wind_dir_sin"] = np.sin(wd)
    g["wind_dir_cos"] = np.cos(wd)

    g["hour"] = g.index.hour
    g["day_of_week"] = g.index.dayofweek
    g["month"] = g.index.month
    # North-India winter pollution season: Oct–Feb (inversion + stubble smoke).
    g["is_winter_pollution_season"] = g["month"].isin([10, 11, 12, 1, 2]).astype(int)

    if fire_lut:
        dates = g.index.date
        g["nearby_fire_count"] = [fire_lut.get((station_id, d), 0) for d in dates]
    else:
        g["nearby_fire_count"] = 0

    for h in HORIZONS:
        g[f"target_{h}h"] = aqi.shift(-h)

    g["station_id"] = station_id
    # Only rows whose current-hour AQI is a real observation can be a forecast
    # origin (persistence needs actual(t); the model uses aqi_current too).
    g = g[g["aqi_current"].notna()]
    return g.reset_index()


def build_features(unified: pd.DataFrame, fire_df: pd.DataFrame) -> pd.DataFrame:
    fire_lut = _fire_lookup(fire_df)
    frames = []
    for (city, station_id), g in unified.groupby(["city", "station_id"], sort=False):
        feats = _build_station_features(g, station_id, fire_lut)
        if feats is not None:
            feats["city"] = city
            frames.append(feats)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df


# --- split / metrics ---------------------------------------------------------


def per_city_time_split(df: pd.DataFrame) -> pd.DataFrame:
    """Flag the most-recent TEST_FRACTION of each city's timeline as test."""
    df = df.copy()
    df["is_test"] = False
    for city, g in df.groupby("city"):
        cutoff = g["timestamp"].quantile(1 - TEST_FRACTION)
        df.loc[(df["city"] == city) & (df["timestamp"] > cutoff), "is_test"] = True
    return df


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# --- main --------------------------------------------------------------------


def main() -> dict:
    if not UNIFIED_PATH.exists():
        raise SystemExit(f"{UNIFIED_PATH} not found — run data/ingest/run_ingestion.py first.")
    unified = pd.read_parquet(UNIFIED_PATH)
    fire_df = pd.read_parquet(FIRE_PATH) if FIRE_PATH.exists() else pd.DataFrame()
    logger.info(f"Loaded {len(unified)} unified rows, {len(fire_df)} fire-daily rows")

    if unified.empty:
        raise SystemExit(
            "unified_history.parquet is empty — no AQI data to train on. Run a full "
            "AQI backfill (data/ingest/run_ingestion.py) before training."
        )

    unified["timestamp"] = pd.to_datetime(unified["timestamp"], utc=True)
    feats = build_features(unified, fire_df)
    if feats.empty:
        raise SystemExit("No station had enough history to build features (min "
                         f"{MIN_STATION_ROWS} hourly observations).")

    # bool(...) wraps the whole expression (not just the first operand) because
    # `and` short-circuits to whichever operand decided the result — if that's
    # the pandas/numpy comparison, it's a numpy.bool_, which json.dumps() can't
    # serialize. Only ever surfaced once real fire data made this True.
    fire_active = bool(fire_df.shape[0] and feats["nearby_fire_count"].sum() > 0)
    logger.info(
        f"Built {len(feats)} feature rows across {feats['city'].nunique()} cities / "
        f"{feats['station_id'].nunique()} stations "
        f"(nearby_fire_count active: {fire_active})"
    )

    # One-hot the city; keep a stable, sorted column order. Cast to int so
    # XGBoost sees a plain numeric matrix (get_dummies returns bool in pandas 2).
    city_dummies = pd.get_dummies(feats["city"], prefix="city").astype(int)
    feats = pd.concat([feats, city_dummies], axis=1)
    feature_cols = BASE_FEATURES + sorted(city_dummies.columns)

    feats = per_city_time_split(feats)
    cities = sorted(feats["city"].unique())
    train_all = feats[~feats["is_test"]]
    test_all = feats[feats["is_test"]]

    date_range = {
        "start": feats["timestamp"].min().isoformat(),
        "end": feats["timestamp"].max().isoformat(),
        "train_end_per_city": {
            c: feats[(feats["city"] == c) & (~feats["is_test"])]["timestamp"].max().isoformat()
            for c in cities
        },
    }

    models: dict[int, xgb.XGBRegressor] = {}
    importances: dict[int, dict] = {}
    metrics: dict = {c: {} for c in cities}
    metrics["ALL"] = {}

    for h in HORIZONS:
        tgt = f"target_{h}h"
        tr = train_all.dropna(subset=[tgt])
        if tr.empty:
            logger.warning(f"[{h}h] no training rows with a target — skipping horizon")
            continue

        # Carve a time-ordered validation tail from TRAIN only (for early stopping).
        tr_sorted = tr.sort_values("timestamp")
        n_val = max(1, int(len(tr_sorted) * 0.1))
        tr_fit, tr_val = tr_sorted.iloc[:-n_val], tr_sorted.iloc[-n_val:]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            tr_fit[feature_cols], tr_fit[tgt],
            eval_set=[(tr_val[feature_cols], tr_val[tgt])],
            verbose=False,
        )
        models[h] = model
        importances[h] = {
            f: round(float(v), 5)
            for f, v in sorted(
                zip(feature_cols, model.feature_importances_), key=lambda kv: kv[1], reverse=True
            )
        }
        logger.info(f"[{h}h] trained on {len(tr_fit)} rows (best_iteration={model.best_iteration})")

        # Evaluate per city + pooled, model vs persistence, on identical rows.
        pooled_true, pooled_model, pooled_persist = [], [], []
        for c in cities:
            te = test_all[(test_all["city"] == c)].dropna(subset=[tgt])
            if te.empty:
                metrics[c][f"{h}h"] = {"n_test": 0, "note": "no test rows with target"}
                continue
            y_true = te[tgt].to_numpy()
            y_model = model.predict(te[feature_cols])
            y_persist = te["aqi_current"].to_numpy()  # persistence: forecast(t+h)=actual(t)

            m_rmse, m_mae = _rmse(y_true, y_model), float(mean_absolute_error(y_true, y_model))
            p_rmse, p_mae = _rmse(y_true, y_persist), float(mean_absolute_error(y_true, y_persist))
            metrics[c][f"{h}h"] = {
                "n_test": int(len(te)),
                "model_rmse": round(m_rmse, 2),
                "model_mae": round(m_mae, 2),
                "persistence_rmse": round(p_rmse, 2),
                "persistence_mae": round(p_mae, 2),
                "rmse_reduction_pct": round((p_rmse - m_rmse) / p_rmse * 100, 1) if p_rmse else None,
                "beats_persistence": bool(m_rmse < p_rmse),
            }
            pooled_true.append(y_true)
            pooled_model.append(y_model)
            pooled_persist.append(y_persist)

        if pooled_true:
            yt = np.concatenate(pooled_true)
            ym = np.concatenate(pooled_model)
            yp = np.concatenate(pooled_persist)
            pm_rmse, pp_rmse = _rmse(yt, ym), _rmse(yt, yp)
            metrics["ALL"][f"{h}h"] = {
                "n_test": int(len(yt)),
                "model_rmse": round(pm_rmse, 2),
                "model_mae": round(float(mean_absolute_error(yt, ym)), 2),
                "persistence_rmse": round(pp_rmse, 2),
                "persistence_mae": round(float(mean_absolute_error(yt, yp)), 2),
                "rmse_reduction_pct": round((pp_rmse - pm_rmse) / pp_rmse * 100, 1) if pp_rmse else None,
                "beats_persistence": bool(pm_rmse < pp_rmse),
            }

    # --- persist models + metadata ------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for h, model in models.items():
        model.save_model(CHECKPOINT_DIR / f"forecast_{h}h.json")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "XGBoostRegressor",
        "design": "one model per horizon, pooled across cities (city one-hot + lat/lon)",
        "horizons_hours": HORIZONS,
        "target": "aqi (approximate Indian AQI from PM2.5/PM10 sub-indices)",
        "feature_columns": feature_cols,
        "weather_features": WEATHER_COLS,
        "fire_feature_active": fire_active,
        "xgb_params": {k: v for k, v in XGB_PARAMS.items()},
        "test_fraction_recent_per_city": TEST_FRACTION,
        "training_date_range": date_range,
        "cities": cities,
        "n_feature_rows": int(len(feats)),
        "n_stations": int(feats["station_id"].nunique()),
        "metrics": metrics,
        "feature_importance_by_horizon": importances,
        "checkpoints": {f"{h}h": f"forecast_{h}h.json" for h in models},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved {len(models)} model(s) + metadata to {CHECKPOINT_DIR}")

    _write_benchmarks(metadata)
    _print_report(metadata)
    return metadata


def _fmt_metric_table(metrics: dict, cities: list, horizons: list) -> str:
    lines = [
        "| City | Horizon | n (test) | Persistence RMSE | Model RMSE | RMSE ↓ | Persistence MAE | Model MAE | Beats baseline? |",
        "|---|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for c in cities + ["ALL"]:
        for h in horizons:
            m = metrics.get(c, {}).get(f"{h}h")
            if not m or m.get("n_test", 0) == 0:
                lines.append(f"| {c} | {h}h | 0 | — | — | — | — | — | n/a |")
                continue
            red = m.get("rmse_reduction_pct")
            red_s = f"{red:+.1f}%" if red is not None else "—"
            beat = "✅ yes" if m.get("beats_persistence") else "❌ no"
            lines.append(
                f"| {c} | {h}h | {m['n_test']} | {m['persistence_rmse']} | {m['model_rmse']} | "
                f"{red_s} | {m['persistence_mae']} | {m['model_mae']} | {beat} |"
            )
    return "\n".join(lines)


def _write_benchmarks(meta: dict) -> None:
    cities = meta["cities"]
    dr = meta["training_date_range"]
    table = _fmt_metric_table(meta["metrics"], cities, meta["horizons_hours"])

    # Top features for the 24h model, if available.
    imp24 = meta["feature_importance_by_horizon"].get(24, {})
    top_feats = "\n".join(f"- `{k}` — {v}" for k, v in list(imp24.items())[:8]) or "- (n/a)"

    verdicts = []
    for c in cities:
        for h in meta["horizons_hours"]:
            m = meta["metrics"].get(c, {}).get(f"{h}h", {})
            if m.get("n_test"):
                if m.get("beats_persistence"):
                    verdicts.append(f"- **{c} / {h}h** beats persistence by **{m['rmse_reduction_pct']:+.1f}% RMSE**.")
                else:
                    verdicts.append(f"- **{c} / {h}h** does **NOT** beat persistence ({m['rmse_reduction_pct']:+.1f}% RMSE) — see caveats.")
    verdict_block = "\n".join(verdicts) if verdicts else "- No horizon had usable test data."

    content = f"""# Vaayu AI — Forecast Model Card & Benchmarks

_Generated {meta['created_utc']} by `models/train_forecast_model.py`._

## Task

Forecast the (approximate Indian) **AQI** at each monitoring station **24h, 48h
and 72h ahead**, and beat a naive persistence baseline.

## Model

- **Algorithm:** {meta['model_type']} (gradient-boosted trees), one model per
  horizon.
- **Design choice:** {meta['design']}. We chose per-horizon models over a
  single horizon-as-feature model because each model then predicts AQI at
  exactly t+h, making the target, evaluation and persistence comparison direct
  and leak-free. Cities are pooled (not one model each) so sparse cities borrow
  signal from dense ones; city identity and location remain available via the
  one-hot city columns and `lat`/`lon`.
- **Hyperparameters:** `{json.dumps(meta['xgb_params'])}` (early stopping on a
  time-ordered validation tail carved from the training set only).

## Features

Lag & trend: `aqi_current` (value at t), `aqi_lag_1h/24h/48h`,
`aqi_roll_6h/24h`. Weather at t: {", ".join(f"`{w}`" for w in meta['weather_features'])},
plus wind direction as `wind_dir_sin`/`wind_dir_cos`. Fire:
`nearby_fire_count` (FIRMS detections within 100 km that day) — **active in this
run: {meta['fire_feature_active']}**. Calendar: `hour`, `day_of_week`, `month`,
`is_winter_pollution_season` (Oct–Feb, North-India inversion + stubble season).
Location/identity: `lat`, `lon`, one-hot `city_*`.

Full ordered feature list is in `models/checkpoints/forecast_metadata.json`.

### Top features (24h model, by XGBoost gain-importance)
{top_feats}

## Data & split

- **Rows:** {meta['n_feature_rows']:,} hourly feature rows from
  {meta['n_stations']} station(s) across {len(cities)} city/cities
  ({", ".join(cities)}).
- **Timeline:** {dr['start']} → {dr['end']}.
- **Split:** time-based, **most recent {int(meta['test_fraction_recent_per_city']*100)}%
  of each city's timeline held out for test** (no random shuffle). Every test
  origin is later than every training origin for that city, so there is no
  temporal leakage.
- **Persistence baseline:** `forecast(t+h) = actual(t)` (naive carry-forward),
  evaluated on the exact same test rows as the model.

## Results — RMSE & MAE, per city × horizon

{table}

_RMSE/MAE are in AQI points. "RMSE ↓" is the percentage RMSE reduction of the
model versus persistence (positive = model better). "ALL" pools every city's
test rows._

## Verdict — does the model beat persistence?

{verdict_block}

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
"""
    BENCHMARKS_PATH.write_text(content, encoding="utf-8")
    logger.info(f"Wrote {BENCHMARKS_PATH}")


def _print_report(meta: dict) -> None:
    print("\n================ FORECAST vs PERSISTENCE ================")
    print(_fmt_metric_table(meta["metrics"], meta["cities"], meta["horizons_hours"]))
    print("\nVerdict:")
    for c in meta["cities"]:
        for h in meta["horizons_hours"]:
            m = meta["metrics"].get(c, {}).get(f"{h}h", {})
            if not m.get("n_test"):
                continue
            tag = "BEATS" if m.get("beats_persistence") else "does NOT beat"
            print(f"  {c} {h}h: {tag} persistence ({m['rmse_reduction_pct']:+.1f}% RMSE, n={m['n_test']})")
    print("========================================================\n")


if __name__ == "__main__":
    main()
