"""
Vaayu AI — test suite.

Covers the deterministic Attribution Agent scoring formulae, the IDW
interpolation, and the FastAPI attribution route. Forecast regression,
enforcement, and advisory tests will be added as those agents are implemented.
"""
from __future__ import annotations

import math
from datetime import date

import numpy as np

from agents.attribution_agent import (
    INDUSTRIAL_MAX_KM,
    compute_attribution,
    idw_interpolate,
    score_fire,
    score_industrial,
    score_traffic,
)


def test_project_scaffold_sanity():
    """Confirms pytest is wired up correctly."""
    assert True


# --- synthetic fixtures ------------------------------------------------------

# A ~1km square industrial polygon around (28.60, 77.20), ring as [lon, lat].
_INDUSTRIAL_RING = [
    [77.195, 28.595],
    [77.205, 28.595],
    [77.205, 28.605],
    [77.195, 28.605],
    [77.195, 28.595],
]
_INDUSTRIAL = [{"ring": _INDUSTRIAL_RING, "name": "Test Industrial Estate"}]

# Dense road cells clustered near (28.70, 77.10), sparse ones near (28.50, 77.30).
_ROAD_CELLS = (
    [{"lat": 28.70 + i * 0.001, "lon": 77.10, "density": 40.0} for i in range(5)]
    + [{"lat": 28.50 + i * 0.001, "lon": 77.30, "density": 2.0} for i in range(5)]
)
_REF_DENSITY = 40.0  # 95th-pct-like reference


# --- industrial_score --------------------------------------------------------


def test_industrial_inside_polygon_scores_one():
    inside = score_industrial(28.600, 77.200, _INDUSTRIAL)
    assert inside["score"] == 1.0
    assert inside["nearest_km"] == 0.0


def test_industrial_inside_beats_far_point():
    """A point inside a known industrial polygon must out-score a far one."""
    inside = score_industrial(28.600, 77.200, _INDUSTRIAL)["score"]
    far = score_industrial(28.900, 77.600, _INDUSTRIAL)["score"]  # tens of km away
    assert inside > far
    assert far == 0.0


def test_industrial_proximity_is_monotonic():
    """Closer to the polygon ⇒ higher score, and 0 beyond INDUSTRIAL_MAX_KM."""
    near = score_industrial(28.590, 77.200, _INDUSTRIAL)["score"]   # ~0.5 km south of edge
    mid = score_industrial(28.580, 77.200, _INDUSTRIAL)["score"]    # ~1.7 km south of edge
    assert 0.0 < mid < near < 1.0
    beyond = score_industrial(28.560, 77.200, _INDUSTRIAL)          # >3 km away
    assert beyond["score"] == 0.0
    assert beyond["nearest_km"] > INDUSTRIAL_MAX_KM


# --- traffic_score -----------------------------------------------------------


def test_traffic_dense_beats_sparse():
    dense = score_traffic(28.702, 77.100, _ROAD_CELLS, _REF_DENSITY)
    sparse = score_traffic(28.502, 77.300, _ROAD_CELLS, _REF_DENSITY)
    assert dense > sparse
    assert math.isclose(dense, 1.0, abs_tol=1e-6)  # local mean == ref density
    assert 0.0 < sparse < 0.2


def test_traffic_zero_when_no_cells_within_radius():
    # Far from every cell (>2 km) → no cells contribute.
    assert score_traffic(29.50, 78.50, _ROAD_CELLS, _REF_DENSITY) == 0.0
    # No reference density → 0.
    assert score_traffic(28.702, 77.100, _ROAD_CELLS, 0.0) == 0.0


# --- fire_score --------------------------------------------------------------

_FIRE_STATS = {"Delhi": {"by_date": {date(2025, 11, 5): 400, date(2025, 11, 6): 100}, "ref_max": 400.0}}


def test_fire_zero_out_of_season():
    """Fire signal is forced to 0 outside the stubble season, even with fires present."""
    off = score_fire("Delhi", date(2025, 6, 5), _FIRE_STATS)
    assert off["score"] == 0.0 and off["in_season"] is False


def test_fire_normalised_in_season():
    peak = score_fire("Delhi", date(2025, 11, 5), _FIRE_STATS)
    lower = score_fire("Delhi", date(2025, 11, 6), _FIRE_STATS)
    assert peak["score"] == 1.0            # 400/400
    assert math.isclose(lower["score"], 0.25)  # 100/400
    assert peak["score"] > lower["score"]


def test_fire_zero_when_no_data():
    assert score_fire("Delhi", date(2025, 11, 5), {})["score"] == 0.0


# --- compute_attribution (integration of the three) --------------------------


def test_attribution_reports_all_three_and_picks_winner():
    layers = {"road_cells": _ROAD_CELLS, "industrial": _INDUSTRIAL, "ref_density": _REF_DENSITY, "available": True}
    # Point inside the industrial polygon, away from dense roads, in fire season.
    res = compute_attribution(28.600, 77.200, "Delhi", date(2025, 11, 5), layers, _FIRE_STATS)
    for key in ("traffic_score", "industrial_score", "fire_score", "overall_source_estimate", "confidence", "methodology_note"):
        assert key in res
    assert res["industrial_score"] == 1.0
    assert res["overall_source_estimate"] == "industrial"  # the highest of the three
    # Correlational heuristic: confidence must never claim more than "medium".
    assert res["confidence"] in ("low", "medium")


def test_attribution_indeterminate_when_all_zero():
    layers = {"road_cells": [], "industrial": [], "ref_density": 0.0, "available": False}
    res = compute_attribution(1.0, 1.0, "Delhi", date(2025, 6, 1), layers, {})
    assert res["traffic_score"] == res["industrial_score"] == res["fire_score"] == 0.0
    assert res["overall_source_estimate"] == "indeterminate"
    assert res["confidence"] == "low"


# --- IDW interpolation -------------------------------------------------------


def test_idw_at_station_returns_station_value():
    s_lat = np.array([28.6, 28.8])
    s_lon = np.array([77.2, 77.0])
    s_aqi = np.array([100.0, 300.0])
    # Grid point exactly on station 1 → ~100; on station 2 → ~300.
    out = idw_interpolate(s_lat, s_lon, s_lat, s_lon, s_aqi)
    assert abs(out[0] - 100.0) < 1.0
    assert abs(out[1] - 300.0) < 1.0


def test_idw_midpoint_is_between_and_weighted():
    s_lat = np.array([28.6, 28.8])
    s_lon = np.array([77.1, 77.1])
    s_aqi = np.array([100.0, 300.0])
    mid = idw_interpolate(np.array([28.7]), np.array([77.1]), s_lat, s_lon, s_aqi)[0]
    assert 100.0 < mid < 300.0
    assert abs(mid - 200.0) < 1.0  # equidistant → average
    # Closer to the high station → pulled above the mean.
    near_high = idw_interpolate(np.array([28.78]), np.array([77.1]), s_lat, s_lon, s_aqi)[0]
    assert near_high > 200.0
