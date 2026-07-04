"""
Vaayu AI — full pytest suite.

Sections
  1. Forecast regression  — re-run held-out evaluation and assert the model
     beats persistence and reproduces the documented BENCHMARKS numbers.
  2. Attribution scoring   — deterministic traffic/industrial/fire + IDW.
  3. Enforcement ranking   — severity, weights, priority ordering.
  4. Advisory generation   — smoke tests across all supported languages.
  5. API endpoints         — TestClient, happy path + a failure mode per route.
  6. Data integrity        — unified_history.parquet sanity checks.
  (plus orchestrator/output unit tests.)

Data/model-dependent tests skip cleanly when the artifacts are absent (fresh
clone / CI without ingestion), so the pure-logic tests always run.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.advisory_agent import HEALTH_GUIDANCE, aqi_category_name, run_advisory
from agents.attribution_agent import (
    INDUSTRIAL_MAX_KM,
    compute_attribution,
    idw_interpolate,
    score_fire,
    score_industrial,
    score_traffic,
)
from agents.enforcement_agent import aqi_severity, rank_zones, vulnerability_from_pois
from agents.orchestrator_agent import run_orchestrator
from agents.output_agent import run_output

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "db"
UNIFIED = DB / "unified_history.parquet"
DELHI_GEOJSON = DB / "city_layers" / "Delhi_layers.geojson"
METADATA = ROOT / "models" / "checkpoints" / "forecast_metadata.json"


def _data_available() -> bool:
    if not UNIFIED.exists():
        return False
    try:
        return len(pd.read_parquet(UNIFIED, columns=["aqi"])) > 0
    except Exception:
        return False


DATA_AVAILABLE = _data_available()
GEOJSON_AVAILABLE = DELHI_GEOJSON.exists()
MODEL_AVAILABLE = METADATA.exists() and (ROOT / "models" / "checkpoints" / "forecast_24h.json").exists()

needs_data = pytest.mark.skipif(not DATA_AVAILABLE, reason="unified_history.parquet missing/empty")
needs_geojson = pytest.mark.skipif(not GEOJSON_AVAILABLE, reason="Delhi city-layers geojson missing")
needs_model = pytest.mark.skipif(not (DATA_AVAILABLE and MODEL_AVAILABLE), reason="model/data missing")


def test_project_scaffold_sanity():
    assert True


# =============================================================================
# 1 · FORECAST REGRESSION — re-run evaluation on the held-out test set
# =============================================================================


@pytest.fixture(scope="module")
def forecast_eval():
    """Rebuild features + the per-city time split exactly as training did, load the
    SAVED models, and recompute model-vs-persistence RMSE on the held-out test set.
    This genuinely re-runs the evaluation — it does not read the JSON's own claims."""
    import json

    import xgboost as xgb

    from models import train_forecast_model as T

    unified = pd.read_parquet(T.UNIFIED_PATH)
    unified["timestamp"] = pd.to_datetime(unified["timestamp"], utc=True)
    fire = pd.read_parquet(T.FIRE_PATH) if T.FIRE_PATH.exists() else pd.DataFrame()
    feats = T.build_features(unified, fire)
    dummies = pd.get_dummies(feats["city"], prefix="city").astype(int)
    feats = pd.concat([feats, dummies], axis=1)
    feature_cols = T.BASE_FEATURES + sorted(dummies.columns)
    feats = T.per_city_time_split(feats)

    meta = json.loads(T.METADATA_PATH.read_text(encoding="utf-8"))
    results: dict[int, dict[str, dict]] = {}
    for h in T.HORIZONS:
        tgt = f"target_{h}h"
        model = xgb.XGBRegressor()
        model.load_model(str(T.CHECKPOINT_DIR / f"forecast_{h}h.json"))
        per_city = {}
        for city in meta["cities"]:
            te = feats[(feats["city"] == city) & feats["is_test"]].dropna(subset=[tgt])
            if te.empty:
                continue
            y_true = te[tgt].to_numpy()
            y_model = model.predict(te[feature_cols])
            y_persist = te["aqi_current"].to_numpy()
            per_city[city] = {
                "model_rmse": T._rmse(y_true, y_model),
                "persist_rmse": T._rmse(y_true, y_persist),
                "n": int(len(te)),
            }
        results[h] = per_city
    return results, meta


@needs_model
def test_forecast_beats_persistence(forecast_eval):
    """The re-run evaluation must show the model beating persistence everywhere."""
    results, meta = forecast_eval
    checked = 0
    for h, per_city in results.items():
        for city, r in per_city.items():
            assert r["n"] > 100, f"{city} {h}h test set too small ({r['n']})"
            assert r["model_rmse"] < r["persist_rmse"], (
                f"{city} {h}h: model RMSE {r['model_rmse']:.2f} !< persistence {r['persist_rmse']:.2f}")
            checked += 1
    assert checked >= 3  # at least the three Delhi horizons


@needs_model
def test_forecast_reproduces_documented_metrics(forecast_eval):
    """The recomputed RMSE + reduction% must match BENCHMARKS/metadata (± tolerance)."""
    results, meta = forecast_eval
    for h, per_city in results.items():
        for city, r in per_city.items():
            claimed = meta["metrics"][city][f"{h}h"]
            assert abs(r["model_rmse"] - claimed["model_rmse"]) < 1.0
            assert abs(r["persist_rmse"] - claimed["persistence_rmse"]) < 1.0
            recomputed_red = (r["persist_rmse"] - r["model_rmse"]) / r["persist_rmse"] * 100
            assert abs(recomputed_red - claimed["rmse_reduction_pct"]) < 1.0
            assert claimed["beats_persistence"] is True


# =============================================================================
# 2 · ATTRIBUTION SCORING
# =============================================================================

_INDUSTRIAL_RING = [[77.195, 28.595], [77.205, 28.595], [77.205, 28.605], [77.195, 28.605], [77.195, 28.595]]
_INDUSTRIAL = [{"ring": _INDUSTRIAL_RING, "name": "Test Industrial Estate"}]
_ROAD_CELLS = (
    [{"lat": 28.70 + i * 0.001, "lon": 77.10, "density": 40.0} for i in range(5)]
    + [{"lat": 28.50 + i * 0.001, "lon": 77.30, "density": 2.0} for i in range(5)]
)
_REF_DENSITY = 40.0
_FIRE_STATS = {"Delhi": {"by_date": {date(2025, 11, 5): 400, date(2025, 11, 6): 100}, "ref_max": 400.0}}


def test_industrial_inside_polygon_scores_one():
    inside = score_industrial(28.600, 77.200, _INDUSTRIAL)
    assert inside["score"] == 1.0 and inside["nearest_km"] == 0.0


def test_industrial_inside_beats_far_point():
    inside = score_industrial(28.600, 77.200, _INDUSTRIAL)["score"]
    far = score_industrial(28.900, 77.600, _INDUSTRIAL)["score"]
    assert inside > far and far == 0.0


def test_industrial_proximity_is_monotonic():
    near = score_industrial(28.590, 77.200, _INDUSTRIAL)["score"]
    mid = score_industrial(28.580, 77.200, _INDUSTRIAL)["score"]
    assert 0.0 < mid < near < 1.0
    beyond = score_industrial(28.560, 77.200, _INDUSTRIAL)
    assert beyond["score"] == 0.0 and beyond["nearest_km"] > INDUSTRIAL_MAX_KM


def test_traffic_dense_beats_sparse():
    dense = score_traffic(28.702, 77.100, _ROAD_CELLS, _REF_DENSITY)
    sparse = score_traffic(28.502, 77.300, _ROAD_CELLS, _REF_DENSITY)
    assert dense > sparse and math.isclose(dense, 1.0, abs_tol=1e-6) and 0.0 < sparse < 0.2


def test_traffic_zero_when_no_cells_within_radius():
    assert score_traffic(29.50, 78.50, _ROAD_CELLS, _REF_DENSITY) == 0.0
    assert score_traffic(28.702, 77.100, _ROAD_CELLS, 0.0) == 0.0


def test_fire_zero_out_of_season():
    off = score_fire("Delhi", date(2025, 6, 5), _FIRE_STATS)
    assert off["score"] == 0.0 and off["in_season"] is False


def test_fire_normalised_in_season():
    peak = score_fire("Delhi", date(2025, 11, 5), _FIRE_STATS)
    lower = score_fire("Delhi", date(2025, 11, 6), _FIRE_STATS)
    assert peak["score"] == 1.0 and math.isclose(lower["score"], 0.25) and peak["score"] > lower["score"]


def test_fire_zero_when_no_data():
    assert score_fire("Delhi", date(2025, 11, 5), {})["score"] == 0.0


def test_attribution_reports_all_three_and_picks_winner():
    layers = {"road_cells": _ROAD_CELLS, "industrial": _INDUSTRIAL, "ref_density": _REF_DENSITY, "available": True}
    res = compute_attribution(28.600, 77.200, "Delhi", date(2025, 11, 5), layers, _FIRE_STATS)
    for key in ("traffic_score", "industrial_score", "fire_score", "overall_source_estimate", "confidence", "methodology_note"):
        assert key in res
    assert res["industrial_score"] == 1.0
    assert res["overall_source_estimate"] == "industrial"
    assert res["confidence"] in ("low", "medium")


def test_attribution_indeterminate_when_all_zero():
    layers = {"road_cells": [], "industrial": [], "ref_density": 0.0, "available": False}
    res = compute_attribution(1.0, 1.0, "Delhi", date(2025, 6, 1), layers, {})
    assert res["traffic_score"] == res["industrial_score"] == res["fire_score"] == 0.0
    assert res["overall_source_estimate"] == "indeterminate" and res["confidence"] == "low"


def test_idw_at_station_returns_station_value():
    s_lat, s_lon, s_aqi = np.array([28.6, 28.8]), np.array([77.2, 77.0]), np.array([100.0, 300.0])
    out = idw_interpolate(s_lat, s_lon, s_lat, s_lon, s_aqi)
    assert abs(out[0] - 100.0) < 1.0 and abs(out[1] - 300.0) < 1.0


def test_idw_midpoint_is_between_and_weighted():
    s_lat, s_lon, s_aqi = np.array([28.6, 28.8]), np.array([77.1, 77.1]), np.array([100.0, 300.0])
    mid = idw_interpolate(np.array([28.7]), np.array([77.1]), s_lat, s_lon, s_aqi)[0]
    assert 100.0 < mid < 300.0 and abs(mid - 200.0) < 1.0
    assert idw_interpolate(np.array([28.78]), np.array([77.1]), s_lat, s_lon, s_aqi)[0] > 200.0


# =============================================================================
# 3 · ENFORCEMENT RANKING
# =============================================================================


def test_aqi_severity_breakpoints():
    assert [aqi_severity(x) for x in (30, 80, 150, 250, 350, 450, 999)] == [1, 2, 3, 4, 5, 6, 6]
    assert aqi_severity(None) == 0


def test_vulnerability_weights_are_explicit():
    assert vulnerability_from_pois({"hospital": 1, "school": 2}, norm_density=0.5) == 7.5
    assert vulnerability_from_pois({"elderly_care": 2, "park": 9}) == 6.0


def test_enforcement_ranking_matches_hand_calculation():
    zones = [
        {"station_id": "A", "risk_score": 4, "vulnerability_score": 10.0},   # raw 40 → 100
        {"station_id": "B", "risk_score": 6, "vulnerability_score": 5.0},    # raw 30 → 75
    ]
    ranked = rank_zones(zones)
    assert [z["station_id"] for z in ranked] == ["A", "B"]
    assert ranked[0]["priority_score"] == 100.0 and ranked[1]["priority_score"] == 75.0


def test_enforcement_ranking_is_stable_and_deterministic():
    zones = [
        {"station_id": "z2", "risk_score": 3, "vulnerability_score": 4.0},
        {"station_id": "z1", "risk_score": 2, "vulnerability_score": 6.0},
        {"station_id": "z3", "risk_score": 6, "vulnerability_score": 4.0},
    ]
    ranked = rank_zones(zones)
    assert [z["station_id"] for z in ranked] == ["z3", "z2", "z1"]
    assert ranked[1]["priority_score"] == ranked[2]["priority_score"] == 50.0
    assert [z["station_id"] for z in rank_zones(list(reversed(zones)))] == ["z3", "z2", "z1"]


# =============================================================================
# 4 · ADVISORY GENERATION — all supported languages
# =============================================================================


def test_aqi_category_name_mapping():
    assert [aqi_category_name(x) for x in (30, 150, 320, 999)] == ["good", "moderate", "very_poor", "severe"]
    assert aqi_category_name(None) is None


def test_health_guidance_table_is_complete():
    cats = {"good", "satisfactory", "moderate", "poor", "very_poor", "severe"}
    for lang in ("en", "hi"):
        assert set(HEALTH_GUIDANCE[lang]) == cats
        for cat in cats:
            assert HEALTH_GUIDANCE[lang][cat]["general"] and HEALTH_GUIDANCE[lang][cat]["sensitive"]


def _advisory_state(lang):
    return {
        "city": "Delhi", "lat": 28.63, "lon": 77.22, "station_id": "openaq_test",
        "station_name": "ITO, Delhi", "lang": lang,
        "forecast": {"aqi": 320, "horizon_hours": 24, "timestamp": "2025-11-05T12:00:00Z"},
    }


@pytest.mark.parametrize("lang", ["en", "hi", "kn"])
def test_advisory_all_languages_non_empty_and_cited(lang):
    """Every supported language returns non-empty text with sources_cited populated."""
    out = run_advisory(_advisory_state(lang))
    assert out["status"] == "ok"
    assert isinstance(out["advisory_text"], str) and out["advisory_text"].strip()
    src = out["sources_cited"]
    assert src is not None
    assert src["station_name"] == "ITO, Delhi"
    assert src["aqi"] == 320.0 and src["aqi_category"] == "very_poor"
    assert src["reading_timestamp"] == "2025-11-05T12:00:00Z"
    assert src["language"] == lang
    assert src["guidance_source"].startswith("CPCB")


def test_advisory_hindi_is_devanagari():
    out = run_advisory(_advisory_state("hi"))
    assert any("ऀ" <= ch <= "ॿ" for ch in out["advisory_text"])


def test_advisory_errors_without_any_aqi():
    assert run_advisory({"city": "Nowhere-Ville", "lang": "en"})["status"] == "error"


# =============================================================================
# 5 · API ENDPOINTS (TestClient) — happy path + a failure mode per route
# =============================================================================

from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app)


def test_api_root():
    r = client.get("/")
    assert r.status_code == 200 and r.json()["status"] == "ok"


# --- /attribution ---
@needs_geojson
def test_api_attribution_happy():
    r = client.get("/attribution/Delhi", params={"lat": 28.63, "lon": 77.22, "date": "2025-11-05"})
    assert r.status_code == 200
    a = r.json()["attribution"]
    assert set(("traffic_score", "industrial_score", "fire_score", "overall_source_estimate")) <= set(a)
    assert a["confidence"] in ("low", "medium")


def test_api_attribution_missing_point_400():
    assert client.get("/attribution/Delhi").status_code == 400


def test_api_attribution_unknown_city_400():
    assert client.get("/attribution/Atlantis", params={"lat": 1, "lon": 1}).status_code == 400


def test_api_attribution_out_of_range_coords_422():
    assert client.get("/attribution/Delhi", params={"lat": 999, "lon": 999}).status_code == 422


# --- /heatmap ---
@needs_data
def test_api_heatmap_happy():
    r = client.get("/heatmap/Delhi")
    assert r.status_code == 200 and r.json()["type"] == "FeatureCollection"
    assert r.json()["properties"]["n_cells"] > 0


def test_api_heatmap_unknown_city_404():
    assert client.get("/heatmap/Atlantis").status_code == 404


def test_api_heatmap_out_of_range_resolution_422():
    assert client.get("/heatmap/Delhi", params={"grid_resolution_km": 50}).status_code == 422


# --- /enforcement-priorities ---
@needs_data
def test_api_enforcement_happy():
    r = client.get("/enforcement-priorities/Delhi")
    assert r.status_code == 200
    zones = r.json()["priority_zones"]
    assert len(zones) > 0
    scores = [z["priority_score"] for z in zones]
    assert scores == sorted(scores, reverse=True)  # descending


def test_api_enforcement_unknown_city_400():
    assert client.get("/enforcement-priorities/Atlantis").status_code == 400


# --- /stations ---
@needs_data
def test_api_stations_happy():
    r = client.get("/stations/Delhi")
    assert r.status_code == 200 and r.json()["count"] > 0


def test_api_stations_unknown_city_empty():
    r = client.get("/stations/Atlantis")
    assert r.status_code == 200 and r.json()["count"] == 0  # graceful, not an error


# --- /forecast ---
@needs_data
def test_api_forecast_happy():
    r = client.get("/forecast/Delhi", params={"lat": 28.63, "lon": 77.22})
    assert r.status_code == 200
    f = r.json()["forecast"]
    assert set(f["horizons"]) == {"24h", "48h", "72h"}
    assert f["model"] in ("xgboost", "persistence")


def test_api_forecast_unknown_city_400():
    assert client.get("/forecast/Atlantis", params={"lat": 1, "lon": 1}).status_code == 400


def test_api_forecast_out_of_range_coords_422():
    assert client.get("/forecast/Delhi", params={"lat": 999, "lon": 0}).status_code == 422


# --- /advisory (POST) ---
def test_api_advisory_happy():
    r = client.post("/advisory", json={"city": "Delhi", "lang": "hi", "forecast": {"aqi": 90}})
    assert r.status_code == 200
    assert r.json()["advisory_text"] and r.json()["sources_cited"]["aqi_category"] == "satisfactory"


def test_api_advisory_missing_city_422():
    assert client.post("/advisory", json={"lang": "en"}).status_code == 422


def test_api_advisory_out_of_range_coords_422():
    assert client.post("/advisory", json={"city": "Delhi", "lat": 200, "lon": 0}).status_code == 422


# --- /query (POST) ---
@needs_data
def test_api_query_full_happy():
    r = client.post("/query", json={"city": "Delhi", "lat": 28.63, "lon": 77.22, "query_type": "full"})
    assert r.status_code == 200
    fr = r.json()
    assert fr["query_type"] == "full" and "components_run" in fr and "map" in fr and "chat" in fr


def test_api_query_missing_city_422():
    assert client.post("/query", json={"lang": "en"}).status_code == 422


def test_api_query_out_of_range_coords_422():
    assert client.post("/query", json={"city": "Delhi", "lat": -999, "lon": 0}).status_code == 422


# =============================================================================
# 6 · DATA INTEGRITY — unified_history.parquet
# =============================================================================

MIN_STATIONS_PER_CITY = 5  # Delhi ingested 24; a real monitoring network has several


@needs_data
def test_no_duplicate_station_timestamp_rows():
    u = pd.read_parquet(UNIFIED, columns=["station_id", "timestamp"]).dropna(subset=["timestamp"])
    assert u.duplicated(subset=["station_id", "timestamp"]).sum() == 0


@needs_data
def test_aqi_within_plausible_range():
    aqi = pd.read_parquet(UNIFIED, columns=["aqi"])["aqi"].dropna()
    assert len(aqi) > 0
    assert (aqi >= 0).all(), f"negative AQI present (min {aqi.min()})"
    assert (aqi <= 500).all(), f"AQI above the CPCB 0-500 ceiling present (max {aqi.max()})"


@needs_data
def test_min_stations_per_ingested_city():
    u = pd.read_parquet(UNIFIED, columns=["city", "station_id", "aqi"]).dropna(subset=["aqi"])
    counts = u.groupby("city")["station_id"].nunique()
    present = counts[counts > 0]
    assert len(present) >= 1, "no city has any AQI stations"
    for city, n in present.items():
        assert n >= MIN_STATIONS_PER_CITY, f"{city} has only {n} stations (< {MIN_STATIONS_PER_CITY})"


# =============================================================================
# ORCHESTRATOR + OUTPUT (pipeline glue)
# =============================================================================


def test_orchestrator_explicit_query_type_wins():
    assert run_orchestrator({"city": "Delhi", "query_type": "enforcement", "query": "forecast tomorrow"})["query_type"] == "enforcement"


def test_orchestrator_intent_detection():
    assert run_orchestrator({"city": "Delhi", "query": "who is to blame for this smog?"})["query_type"] == "attribution"
    assert run_orchestrator({"city": "Delhi", "query": "what will AQI be tomorrow?"})["query_type"] == "forecast"
    assert run_orchestrator({"city": "Delhi", "query": "which zones should we inspect first?"})["query_type"] == "enforcement"
    assert run_orchestrator({"city": "Delhi", "query": "is it safe to go for a jog?"})["query_type"] == "advisory"
    assert run_orchestrator({"city": "Delhi", "query": "is it safe to jog tomorrow morning?"})["query_type"] == "advisory"
    assert run_orchestrator({"city": "Delhi", "query": "give me the full report"})["query_type"] == "full"
    assert run_orchestrator({"city": "Delhi"})["query_type"] == "full"


def test_orchestrator_lang_normalization():
    assert run_orchestrator({"city": "Delhi", "lang": "Hindi"})["lang"] == "hi"
    assert run_orchestrator({"city": "Delhi", "lang": "ENGLISH"})["lang"] == "en"
    assert run_orchestrator({"city": "Delhi"})["lang"] == "en"


def test_output_assembles_map_and_chat_from_partial_state():
    state = {
        "city": "Delhi", "lang": "en", "query_type": "full", "lat": 28.6, "lon": 77.2,
        "forecast": {"aqi": 310.0, "category": "very_poor", "horizon_hours": 24,
                     "reference_aqi": 320.0, "model": "xgboost", "lat": 28.6, "lon": 77.2,
                     "station_id": "s1", "station_name": "ITO", "horizons": {}},
        "attribution": {"traffic_score": 0.9, "industrial_score": 0.1, "fire_score": 0.0,
                        "overall_source_estimate": "traffic", "confidence": "medium",
                        "details": {"point": {"lat": 28.6, "lon": 77.2}}},
        "advisory_text": "Stay indoors today.",
    }
    fr = run_output(state)["final_response"]
    assert fr["components_run"] == ["forecast", "attribution", "advisory"]
    assert fr["map"]["forecast"]["aqi"] == 310.0
    assert fr["map"]["attribution"]["overall_source_estimate"] == "traffic"
    assert "improving" in fr["chat"]["lines"][0]
    assert fr["chat"]["advisory"] == "Stay indoors today."
    assert "/heatmap/Delhi" in fr["map"]["heatmap_hint"]["endpoint"]


def test_output_empty_when_nothing_ran():
    fr = run_output({"city": "Delhi", "query_type": "forecast"})["final_response"]
    assert fr["status"] == "empty" and fr["components_run"] == []
