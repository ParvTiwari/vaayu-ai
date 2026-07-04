"""
Forecast Agent — serves 24/48/72h AQI forecasts from the trained XGBoost models
in models/checkpoints/, with CPCB category mapping and a persistence fallback
for stations with insufficient history or when the models are unavailable.

The features are rebuilt for the target station exactly as in
models/train_forecast_model.py (lags, rolling means, weather, fire, calendar),
and the most recent feature row is fed to each per-horizon model. If anything is
missing (no model, no history, no feature row) the agent degrades to a naive
persistence forecast (forecast(t+h) = latest observed AQI) so the graph never
breaks.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from data.ingest._geo_utils import haversine_km

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
UNIFIED_PATH = ROOT / "data" / "db" / "unified_history.parquet"
FIRE_PATH = ROOT / "data" / "db" / "fire_daily.parquet"
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
METADATA_PATH = CHECKPOINT_DIR / "forecast_metadata.json"

# CPCB AQI category breakpoints — deterministic lookup, not LLM-derived.
AQI_CATEGORIES = {
    "good": (0, 50),
    "satisfactory": (51, 100),
    "moderate": (101, 200),
    "poor": (201, 300),
    "very_poor": (301, 400),
    "severe": (401, 500),
}

HORIZONS = [24, 48, 72]

_MODELS_CACHE: dict | None = None


def aqi_category(aqi: float | None) -> str | None:
    if aqi is None:
        return None
    for name, (_lo, hi) in AQI_CATEGORIES.items():
        if aqi <= hi:
            return name
    return list(AQI_CATEGORIES)[-1]


def _load_models() -> dict | None:
    """Load the per-horizon XGBoost models + feature list (cached). None if absent."""
    global _MODELS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE or None
    if not METADATA_PATH.exists():
        _MODELS_CACHE = {}
        return None
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed — forecast will use persistence fallback.")
        _MODELS_CACHE = {}
        return None
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    models: dict[int, Any] = {}
    for h in HORIZONS:
        path = CHECKPOINT_DIR / f"forecast_{h}h.json"
        if path.exists():
            m = xgb.XGBRegressor()
            m.load_model(str(path))
            models[h] = m
    if not models:
        _MODELS_CACHE = {}
        return None
    _MODELS_CACHE = {"models": models, "feature_columns": meta["feature_columns"], "cities": meta.get("cities", [])}
    return _MODELS_CACHE


def _resolve_station(city: str, lat, lon, station_id) -> dict | None:
    if not UNIFIED_PATH.exists():
        return None
    u = pd.read_parquet(UNIFIED_PATH, columns=["city", "station_id", "station_name", "lat", "lon"])
    u = u[(u["city"] == city)].dropna(subset=["lat", "lon"]).drop_duplicates("station_id")
    if u.empty:
        return None
    if station_id and (u["station_id"] == station_id).any():
        row = u[u["station_id"] == station_id].iloc[0]
    elif lat is not None and lon is not None:
        u = u.assign(_d=u.apply(lambda r: haversine_km(lat, lon, r["lat"], r["lon"]), axis=1))
        row = u.sort_values("_d").iloc[0]
    else:
        row = u.iloc[0]
    return {"station_id": row["station_id"], "station_name": row.get("station_name") or row["station_id"],
            "lat": float(row["lat"]), "lon": float(row["lon"]), "city": city}


def _latest_feature_row(city: str, station_id: str):
    """Build features for one station and return its most recent row (or None)."""
    try:
        from models.train_forecast_model import build_features
    except Exception as exc:  # pragma: no cover
        logger.warning(f"could not import feature builder: {exc}")
        return None
    u = pd.read_parquet(UNIFIED_PATH)
    u = u[(u["city"] == city) & (u["station_id"] == station_id)]
    if u.empty:
        return None
    u["timestamp"] = pd.to_datetime(u["timestamp"], utc=True)
    fire = pd.read_parquet(FIRE_PATH) if FIRE_PATH.exists() else pd.DataFrame()
    feats = build_features(u, fire)
    if feats.empty:
        return None
    return feats.sort_values("timestamp").iloc[-1]


def _predict(bundle: dict, row, city: str) -> dict[int, float]:
    """Run each horizon model on the feature row, aligning to the trained columns."""
    import numpy as np

    cols = bundle["feature_columns"]
    values = {}
    for c in cols:
        if c.startswith("city_"):
            values[c] = 1 if c == f"city_{city}" else 0
        elif c in row.index:
            v = row[c]
            values[c] = float(v) if pd.notna(v) else np.nan
        else:
            values[c] = np.nan
    X = pd.DataFrame([values], columns=cols)
    return {h: float(m.predict(X)[0]) for h, m in bundle["models"].items()}


def run_forecast(state: dict[str, Any]) -> dict[str, Any]:
    """Predict 24/48/72h AQI for a city/station. Writes state['forecast'].

    Input state: city, and (lat & lon) or station_id.
    Output state: forecast (dict), status.
    """
    city = state.get("city")
    if not city:
        state["status"] = "error"
        state["error"] = "missing city"
        return state

    station = _resolve_station(city, state.get("lat"), state.get("lon"), state.get("station_id"))
    if station is None:
        state["status"] = "error"
        state["error"] = f"no station history for {city}"
        return state

    row = _latest_feature_row(city, station["station_id"])
    bundle = _load_models()
    method = "persistence"
    reference_aqi = None
    ref_ts = None
    preds: dict[int, float] = {}

    if row is not None:
        reference_aqi = float(row["aqi_current"]) if pd.notna(row.get("aqi_current")) else None
        ref_ts = pd.to_datetime(row["timestamp"])
        if bundle is not None and city in bundle.get("cities", []):
            try:
                preds = _predict(bundle, row, city)
                method = "xgboost"
            except Exception as exc:
                logger.warning(f"model prediction failed ({exc}); using persistence.")

    # Persistence fallback: carry the latest observed AQI forward.
    if not preds:
        if reference_aqi is None:
            reference_aqi = _persistence_aqi(city, station["station_id"])
            ref_ts = ref_ts or pd.Timestamp.utcnow()
        if reference_aqi is None:
            state["status"] = "error"
            state["error"] = f"no AQI observations for {station['station_id']}"
            return state
        preds = {h: reference_aqi for h in HORIZONS}

    horizons = {
        f"{h}h": {"aqi": round(preds[h], 1), "category": aqi_category(preds[h]),
                  "valid_at": (ref_ts + timedelta(hours=h)).isoformat() if ref_ts is not None else None}
        for h in sorted(preds)
    }
    primary = min(preds)  # 24h is the headline forecast (fed to the advisory)
    state["forecast"] = {
        **station,
        "reference_aqi": round(reference_aqi, 1) if reference_aqi is not None else None,
        "reference_timestamp": ref_ts.isoformat() if ref_ts is not None else None,
        "horizons": horizons,
        "aqi": round(preds[primary], 1),
        "category": aqi_category(preds[primary]),
        "horizon_hours": primary,
        "timestamp": horizons[f"{primary}h"]["valid_at"],
        "model": method,
    }
    state["status"] = "ok"
    return state


def _persistence_aqi(city: str, station_id: str) -> float | None:
    if not UNIFIED_PATH.exists():
        return None
    u = pd.read_parquet(UNIFIED_PATH, columns=["city", "station_id", "timestamp", "aqi"])
    u = u[(u["city"] == city) & (u["station_id"] == station_id)].dropna(subset=["aqi", "timestamp"])
    if u.empty:
        return None
    return float(u.sort_values("timestamp")["aqi"].iloc[-1])


__all__ = ["run_forecast", "aqi_category", "AQI_CATEGORIES", "HORIZONS"]
