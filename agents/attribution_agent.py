"""
Attribution Agent — DETERMINISTIC, rule-based pollution-source scoring.

NO LLM is involved anywhere in this module. Like AgriBloom's non-LLM
compliance_agent, source attribution is a decision that must be *auditable*:
a reviewer has to be able to re-derive every number by hand from the same
inputs. An LLM's plausible-sounding but unverifiable "this is 60% traffic"
is exactly what we refuse to ship here. Every score below comes from an
explicit, documented formula over the OSM + FIRMS layers ingested earlier.

SCORING FORMULAE (all scores are in [0, 1]; higher = stronger local signal)
---------------------------------------------------------------------------
traffic_score
    Road density around the point as a proxy for vehicular/combustion load.
    Take every 1 km road-density grid cell (from OSM `highway=*`, persisted in
    data/db/city_layers/{city}_layers.geojson) whose centre lies within
    TRAFFIC_RADIUS_KM (2 km) of the point. Let `local` be the mean of those
    cells' `road_density_km_per_km2`. Normalise against the city's own road
    network so the score is "how dense is here vs. the densest parts of town":
        traffic_score = clip(local / ref_density, 0, 1)
    where ref_density is the 95th percentile of all cell densities in the city
    (a robust "very dense" reference that ignores a few freak cells). No cells
    within 2 km ⇒ 0.

industrial_score
    Proximity to OSM `landuse=industrial` polygons.
        inside any polygon                  → 1.0
        else, nearest edge within 3 km       → 1 - dist_km / INDUSTRIAL_MAX_KM
        else                                 → 0.0
    Distance is the great-circle point-to-polygon-edge distance (0 if the point
    is inside), computed in a local equirectangular projection.

fire_score
    Regional biomass/stubble-burning signal from FIRMS (data/db/fire_daily.parquet).
    OUTSIDE the stubble-burning season (months not in STUBBLE_MONTHS = Oct–Nov)
    it is forced to 0.0 — fires elsewhere in the year are not the stubble
    phenomenon this signal is meant to capture. In season:
        fire_score = clip(city_fire_count(date) / ref_fire_max, 0, 1)
    where city_fire_count(date) is the max `nearby_fire_count` across the city's
    stations on that date and ref_fire_max is that city's historical daily max.
    (With no FIRMS_API_KEY at ingestion time this is uniformly 0 — documented,
    not hidden.)

overall_source_estimate / confidence
    overall_source_estimate = the single highest-scoring of the three (the
    "primary likely contributor"), or "indeterminate" if all three are 0. ALL
    THREE scores are always returned, never just the winner. `confidence` is
    "medium" only when the top score ≥ 0.5, otherwise "low" — and NEVER "high":
    this is a correlational spatial heuristic, not a validated causal
    attribution model, and it must not claim more certainty than that.
"""
from __future__ import annotations

import json
import math
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.ingest._geo_utils import haversine_km, haversine_km_vec

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "data" / "db"
LAYERS_DIR = DB_DIR / "city_layers"
FIRE_PATH = DB_DIR / "fire_daily.parquet"
UNIFIED_PATH = DB_DIR / "unified_history.parquet"

TRAFFIC_RADIUS_KM = 2.0
INDUSTRIAL_MAX_KM = 3.0
STUBBLE_MONTHS = {10, 11}  # peak Punjab/Haryana crop-residue burning
IDW_POWER = 2.0

# City bounding boxes (min_lon, min_lat, max_lon, max_lat) for the heatmap grid.
# Imported lazily from the ingestion config to stay in sync.
try:  # pragma: no cover - trivial import guard
    from data.ingest.cpcb_openaq import CITY_CONFIG
except Exception:  # pragma: no cover
    CITY_CONFIG = {}

_LAYERS_CACHE: dict[str, dict] = {}
_FIRE_CACHE: dict[str, dict] | None = None
_STATIONS_CACHE: pd.DataFrame | None = None


# --- geometry helpers --------------------------------------------------------


def _ring_centroid(ring: list[list[float]]) -> tuple[float, float]:
    """(lat, lon) centroid of a GeoJSON ring [[lon,lat],...] (ignores closing pt)."""
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    lons = [p[0] for p in pts]
    lats = [p[1] for p in pts]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def _point_in_ring(lat: float, lon: float, ring: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon on a [[lon,lat],...] ring."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _point_to_ring_km(lat: float, lon: float, ring: list[list[float]]) -> float:
    """Great-circle distance (km) from a point to a polygon; 0 if inside.

    Distances use a local equirectangular projection centred on the ring so the
    point-to-segment maths is planar and cheap.
    """
    if _point_in_ring(lat, lon, ring):
        return 0.0
    mean_lat = sum(p[1] for p in ring) / len(ring)
    kx = 111.320 * math.cos(math.radians(mean_lat))
    ky = 111.0
    px, py = lon * kx, lat * ky
    best = math.inf
    for a, b in zip(ring, ring[1:]):
        ax, ay = a[0] * kx, a[1] * ky
        bx, by = b[0] * kx, b[1] * ky
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, math.hypot(px - cx, py - cy))
    return best


# --- data loading ------------------------------------------------------------


def load_city_layers(city: str) -> dict:
    """Parse {city}_layers.geojson into road cells + industrial polygons (cached)."""
    if city in _LAYERS_CACHE:
        return _LAYERS_CACHE[city]

    path = LAYERS_DIR / f"{city}_layers.geojson"
    road_cells: list[dict] = []
    industrial: list[dict] = []
    pois: list[dict] = []
    if path.exists():
        gj = json.loads(path.read_text(encoding="utf-8"))
        for ft in gj.get("features", []):
            layer = ft.get("properties", {}).get("layer")
            geom = ft.get("geometry", {})
            if layer == "road_density_grid" and geom.get("type") == "Polygon":
                clat, clon = _ring_centroid(geom["coordinates"][0])
                road_cells.append(
                    {
                        "lat": clat,
                        "lon": clon,
                        "density": float(ft["properties"].get("road_density_km_per_km2", 0.0)),
                    }
                )
            elif layer == "industrial_zones" and geom.get("type") == "Polygon":
                ring = geom["coordinates"][0]
                clat, clon = _ring_centroid(ring)
                industrial.append(
                    {
                        "ring": ring,
                        "lat": clat,
                        "lon": clon,
                        "name": ft["properties"].get("name"),
                        "area_km2": ft["properties"].get("area_km2"),
                    }
                )
            elif layer == "vulnerability_pois" and geom.get("type") == "Point":
                lon_c, lat_c = geom["coordinates"]
                pois.append(
                    {
                        "category": ft["properties"].get("category"),
                        "name": ft["properties"].get("name"),
                        "lat": lat_c,
                        "lon": lon_c,
                    }
                )

    densities = np.array([c["density"] for c in road_cells], dtype=float)
    ref_density = float(np.percentile(densities, 95)) if densities.size else 0.0
    layers = {
        "road_cells": road_cells,
        "industrial": industrial,
        "pois": pois,
        "ref_density": ref_density,
        "available": path.exists(),
    }
    _LAYERS_CACHE[city] = layers
    return layers


def load_fire_stats() -> dict[str, dict]:
    """Per-city daily fire signal + historical max, from fire_daily.parquet (cached)."""
    global _FIRE_CACHE
    if _FIRE_CACHE is not None:
        return _FIRE_CACHE
    stats: dict[str, dict] = {}
    if FIRE_PATH.exists():
        fdf = pd.read_parquet(FIRE_PATH)
        fdf = fdf.dropna(subset=["detection_date"]) if "detection_date" in fdf else fdf.iloc[0:0]
        for city, g in fdf.groupby("city"):
            # City-level daily signal = worst-affected station that day.
            by_date = g.groupby("detection_date")["fire_count"].max()
            by_date.index = [pd.to_datetime(d).date() for d in by_date.index]
            ref_max = float(by_date.max()) if len(by_date) else 0.0
            stats[str(city)] = {"by_date": by_date.to_dict(), "ref_max": ref_max}
    _FIRE_CACHE = stats
    return stats


def _load_stations() -> pd.DataFrame:
    global _STATIONS_CACHE
    if _STATIONS_CACHE is not None:
        return _STATIONS_CACHE
    if UNIFIED_PATH.exists():
        u = pd.read_parquet(UNIFIED_PATH, columns=["city", "station_id", "lat", "lon"])
        _STATIONS_CACHE = u.dropna(subset=["lat", "lon"]).drop_duplicates("station_id").reset_index(drop=True)
    else:
        _STATIONS_CACHE = pd.DataFrame(columns=["city", "station_id", "lat", "lon"])
    return _STATIONS_CACHE


def clear_caches() -> None:
    """Reset module caches (used by tests after writing synthetic fixtures)."""
    global _FIRE_CACHE, _STATIONS_CACHE
    _LAYERS_CACHE.clear()
    _FIRE_CACHE = None
    _STATIONS_CACHE = None


# --- pure scoring functions (unit-testable) ----------------------------------


def score_traffic(lat: float, lon: float, road_cells: list[dict], ref_density: float) -> float:
    """Normalised road density within TRAFFIC_RADIUS_KM of the point (see module docstring)."""
    if not road_cells or ref_density <= 0:
        return 0.0
    near = [c["density"] for c in road_cells if haversine_km(lat, lon, c["lat"], c["lon"]) <= TRAFFIC_RADIUS_KM]
    if not near:
        return 0.0
    local = sum(near) / len(near)
    return float(min(max(local / ref_density, 0.0), 1.0))


def score_industrial(lat: float, lon: float, industrial: list[dict]) -> dict:
    """Proximity score to the nearest industrial polygon (see module docstring)."""
    if not industrial:
        return {"score": 0.0, "nearest_name": None, "nearest_km": None}
    best_km = math.inf
    best_name = None
    for poly in industrial:
        d = _point_to_ring_km(lat, lon, poly["ring"])
        if d < best_km:
            best_km, best_name = d, poly.get("name")
        if best_km == 0.0:
            break
    if best_km == 0.0:
        score = 1.0
    elif best_km <= INDUSTRIAL_MAX_KM:
        score = 1.0 - best_km / INDUSTRIAL_MAX_KM
    else:
        score = 0.0
    return {"score": float(score), "nearest_name": best_name, "nearest_km": round(best_km, 3) if math.isfinite(best_km) else None}


def score_fire(city: str, when: date_cls, fire_stats: dict[str, dict]) -> dict:
    """Normalised stubble-burning signal for city/date (0 out of season)."""
    if when.month not in STUBBLE_MONTHS:
        return {"score": 0.0, "in_season": False, "fire_count": 0}
    city_stats = fire_stats.get(city)
    if not city_stats or city_stats["ref_max"] <= 0:
        return {"score": 0.0, "in_season": True, "fire_count": 0}
    count = float(city_stats["by_date"].get(when, 0))
    score = min(max(count / city_stats["ref_max"], 0.0), 1.0)
    return {"score": float(score), "in_season": True, "fire_count": int(count)}


def compute_attribution(lat: float, lon: float, city: str, when: date_cls,
                        layers: dict, fire_stats: dict[str, dict]) -> dict:
    """Combine the three scores into the attribution result dict."""
    traffic = round(score_traffic(lat, lon, layers["road_cells"], layers["ref_density"]), 3)
    ind = score_industrial(lat, lon, layers["industrial"])
    fire = score_fire(city, when, fire_stats)
    industrial = round(ind["score"], 3)
    fire_score = round(fire["score"], 3)

    scores = {"traffic": traffic, "industrial": industrial, "fire": fire_score}
    top_label = max(scores, key=scores.get)
    top_value = scores[top_label]
    primary = top_label if top_value > 0 else "indeterminate"
    confidence = "medium" if top_value >= 0.5 else "low"

    note = (
        "Correlational spatial heuristic, not validated causal attribution. "
        f"traffic=normalised road density within {TRAFFIC_RADIUS_KM:.0f}km; "
        f"industrial=proximity to OSM landuse=industrial within {INDUSTRIAL_MAX_KM:.0f}km "
        f"(1.0 if inside); fire=normalised FIRMS stubble signal (0 outside Oct-Nov). "
        "Confidence is capped at 'medium' by design."
    )
    return {
        "traffic_score": traffic,
        "industrial_score": industrial,
        "fire_score": fire_score,
        "overall_source_estimate": primary,
        "confidence": confidence,
        "methodology_note": note,
        "details": {
            "point": {"lat": lat, "lon": lon},
            "date": when.isoformat(),
            "nearest_industrial_name": ind["nearest_name"],
            "nearest_industrial_km": ind["nearest_km"],
            "fire_in_season": fire["in_season"],
            "fire_count": fire["fire_count"],
            "layers_available": layers["available"],
        },
    }


# --- LangGraph node ----------------------------------------------------------


def _resolve_point(state: dict[str, Any]) -> tuple[float, float] | None:
    if state.get("lat") is not None and state.get("lon") is not None:
        return float(state["lat"]), float(state["lon"])
    sid = state.get("station_id")
    if sid:
        st = _load_stations()
        row = st[st["station_id"] == sid]
        if not row.empty:
            return float(row["lat"].iloc[0]), float(row["lon"].iloc[0])
    return None


def _resolve_date(state: dict[str, Any]) -> date_cls:
    raw = state.get("date") or state.get("timestamp")
    if raw:
        try:
            return pd.to_datetime(raw).date()
        except (ValueError, TypeError):
            pass
    return datetime.utcnow().date()


def run_attribution(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic source-attribution node. See module docstring for the formulae.

    Input state: city, and (lat & lon) OR station_id; optional date/timestamp.
    Output state: attribution (dict), status.
    """
    city = state.get("city")
    if not city or (CITY_CONFIG and city not in CITY_CONFIG):
        state["status"] = "error"
        state["error"] = f"unknown or missing city: {city!r}"
        return state

    point = _resolve_point(state)
    if point is None:
        state["status"] = "error"
        state["error"] = "provide lat & lon, or a known station_id"
        return state

    lat, lon = point
    when = _resolve_date(state)
    layers = load_city_layers(city)
    fire_stats = load_fire_stats()

    state["attribution"] = compute_attribution(lat, lon, city, when, layers, fire_stats)
    state["status"] = "ok"
    return state


# --- IDW heatmap interpolation ----------------------------------------------
#
# NOTE ON RETURN TYPE: the brief asks for a GeoDataFrame. This project
# deliberately avoids a geopandas/GEOS/GDAL dependency (see data/ingest —
# geometry is handled with numpy + hand-written GeoJSON). interpolate_city_aqi
# therefore returns a plain pandas DataFrame of grid points (lat, lon, aqi);
# grid_to_geojson() below turns it into a GeoJSON FeatureCollection for the API
# / map layer, which is what a heatmap consumer actually needs.


def _station_readings_at(city: str, when_ts: pd.Timestamp, tolerance_hours: int = 3) -> pd.DataFrame:
    """Nearest-in-time AQI reading per station for `city` around `when_ts`."""
    if not UNIFIED_PATH.exists():
        return pd.DataFrame(columns=["station_id", "lat", "lon", "aqi"])
    u = pd.read_parquet(UNIFIED_PATH, columns=["city", "station_id", "lat", "lon", "timestamp", "aqi"])
    u = u[(u["city"] == city)].dropna(subset=["lat", "lon", "aqi", "timestamp"])
    if u.empty:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "aqi"])
    u["timestamp"] = pd.to_datetime(u["timestamp"], utc=True)
    tol = pd.Timedelta(hours=tolerance_hours)
    rows = []
    for sid, g in u.groupby("station_id"):
        g = g.assign(dt=(g["timestamp"] - when_ts).abs())
        nearest = g.loc[g["dt"].idxmin()]
        if nearest["dt"] <= tol:
            rows.append({"station_id": sid, "lat": nearest["lat"], "lon": nearest["lon"], "aqi": float(nearest["aqi"])})
    return pd.DataFrame(rows)


def interpolate_city_aqi(city: str, timestamp: str, grid_resolution_km: float = 1.0) -> pd.DataFrame:
    """IDW-interpolate station AQI onto a city-wide grid for heatmap rendering.

    Returns a DataFrame with columns (lat, lon, aqi) — one row per grid cell
    centre. Empty if the city is unknown or no station reported near `timestamp`.
    (See the note above on why this is a DataFrame, not a GeoDataFrame.)
    """
    if city not in CITY_CONFIG:
        return pd.DataFrame(columns=["lat", "lon", "aqi"])
    when_ts = pd.to_datetime(timestamp, utc=True)
    stations = _station_readings_at(city, when_ts)
    if stations.empty:
        return pd.DataFrame(columns=["lat", "lon", "aqi"])

    min_lon, min_lat, max_lon, max_lat = CITY_CONFIG[city]["bbox"]
    mid_lat = (min_lat + max_lat) / 2
    dlat = grid_resolution_km / 111.0
    dlon = grid_resolution_km / (111.320 * max(math.cos(math.radians(mid_lat)), 0.01))
    lats = np.arange(min_lat, max_lat + dlat, dlat)
    lons = np.arange(min_lon, max_lon + dlon, dlon)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    flat_lat = grid_lat.ravel()
    flat_lon = grid_lon.ravel()

    aqi = idw_interpolate(
        flat_lat, flat_lon,
        stations["lat"].to_numpy(), stations["lon"].to_numpy(), stations["aqi"].to_numpy(),
    )
    return pd.DataFrame({"lat": np.round(flat_lat, 6), "lon": np.round(flat_lon, 6), "aqi": np.round(aqi, 1)})


def idw_interpolate(grid_lat, grid_lon, s_lat, s_lon, s_aqi, power: float = IDW_POWER):
    """Inverse-distance-weighted AQI at each grid point from station readings.

    Pure/vectorised: weight w_i = 1/dist_i^power (great-circle km); returns the
    weighted mean per grid point. A grid point coincident with a station takes
    that station's value (distance is floored to avoid div-by-zero).
    """
    grid_lat = np.asarray(grid_lat, dtype=float)
    grid_lon = np.asarray(grid_lon, dtype=float)
    s_lat = np.asarray(s_lat, dtype=float)
    s_lon = np.asarray(s_lon, dtype=float)
    s_aqi = np.asarray(s_aqi, dtype=float)
    d = np.stack([haversine_km_vec(grid_lat, grid_lon, la, lo) for la, lo in zip(s_lat, s_lon)], axis=1)
    d = np.maximum(d, 1e-6)
    w = 1.0 / (d ** power)
    return (w * s_aqi).sum(axis=1) / w.sum(axis=1)


def latest_reading_timestamp(city: str) -> str | None:
    """Most recent AQI reading timestamp for a city (used as the heatmap default)."""
    if not UNIFIED_PATH.exists():
        return None
    u = pd.read_parquet(UNIFIED_PATH, columns=["city", "timestamp", "aqi"])
    u = u[(u["city"] == city)].dropna(subset=["aqi", "timestamp"])
    if u.empty:
        return None
    return pd.to_datetime(u["timestamp"], utc=True).max().isoformat()


def city_station_snapshot(city: str) -> list[dict]:
    """Latest AQI + CPCB category per station for a city — powers the map markers."""
    from agents.forecast_agent import aqi_category  # local import avoids import-time coupling

    if not UNIFIED_PATH.exists():
        return []
    u = pd.read_parquet(UNIFIED_PATH, columns=["city", "station_id", "station_name", "lat", "lon", "timestamp", "aqi"])
    u = u[(u["city"] == city)].dropna(subset=["lat", "lon", "aqi", "timestamp"])
    if u.empty:
        return []
    u["timestamp"] = pd.to_datetime(u["timestamp"], utc=True)
    latest = u.sort_values("timestamp").groupby("station_id").tail(1)
    return [
        {
            "station_id": r.station_id,
            "station_name": r.station_name or r.station_id,
            "lat": round(float(r.lat), 6),
            "lon": round(float(r.lon), 6),
            "aqi": round(float(r.aqi), 1),
            "category": aqi_category(float(r.aqi)),
            "timestamp": r.timestamp.isoformat(),
        }
        for r in latest.itertuples(index=False)
    ]


def grid_to_geojson(grid: pd.DataFrame) -> dict:
    """Turn an interpolate_city_aqi() grid into a GeoJSON point FeatureCollection."""
    features = [
        {
            "type": "Feature",
            "properties": {"aqi": float(r.aqi)},
            "geometry": {"type": "Point", "coordinates": [float(r.lon), float(r.lat)]},
        }
        for r in grid.itertuples(index=False)
    ]
    return {"type": "FeatureCollection", "features": features}


__all__ = [
    "run_attribution",
    "interpolate_city_aqi",
    "idw_interpolate",
    "grid_to_geojson",
    "city_station_snapshot",
    "latest_reading_timestamp",
    "compute_attribution",
    "score_traffic",
    "score_industrial",
    "score_fire",
    "load_city_layers",
    "load_fire_stats",
    "clear_caches",
]
