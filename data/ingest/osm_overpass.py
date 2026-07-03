"""
Fetch OSM layers via the Overpass API (https://overpass-api.de/) for a pilot
city bounding box:

  (a) road_density_grid   — total highway length per 1 km grid cell (a traffic
                            / combustion-exposure proxy for the Attribution Agent)
  (b) industrial_zones    — landuse=industrial polygons (with centroid + area)
  (c) vulnerability_pois  — hospitals, clinics, schools, colleges and
                            elderly-care facilities, as points

Design notes:
- The three layers are fetched with three independent Overpass queries so a
  timeout on one (Overpass' free tier is genuinely flaky) does not take out the
  others — each degrades to an empty frame and is flagged in the returned
  '_meta'.
- On failure/timeout we retry once against a shrunk (centre 50%) bounding box
  before giving up, and record that the layer is a partial/fallback extract.
- Raw Overpass JSON is cached to data/db/raw_cache/ (via _http_utils) so slow,
  rate-limited queries are only paid for once.
- Outputs are plain pandas DataFrames, not GeoDataFrames, to avoid a
  geopandas/GEOS dependency; geometry travels as coordinate lists and is
  serialised to GeoJSON by run_ingestion.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import pandas as pd

from ._geo_utils import haversine_km
from ._http_utils import cached_post_json

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 180  # seconds, passed both to Overpass ([timeout:]) and requests
# overpass-api.de returns 406 Not Acceptable to the bare python-requests
# User-Agent — a descriptive UA (with contact) is required by its usage policy.
OVERPASS_HEADERS = {"User-Agent": "vaayu-ai/1.0 (ET AI Hackathon air-quality project; parvtiwari1@gmail.com)"}
# The free tier gives ~2 slots/IP and 429s aggressively when hammered, so we
# retry generously (honouring Retry-After) and pause between queries.
OVERPASS_MAX_RETRIES = 5
OVERPASS_COOLDOWN_S = 5.0

GRID_CELL_KM = 1.0

# amenity/social_facility tag -> the vulnerability category we bucket it into.
POI_CATEGORIES = {
    "hospital": "hospital",
    "clinic": "hospital",
    "school": "school",
    "college": "school",
    "university": "school",
    "kindergarten": "school",
}


# --- Overpass query building / calling ---------------------------------------


def _bbox_to_overpass(bbox: tuple[float, float, float, float]) -> str:
    """(min_lon, min_lat, max_lon, max_lat) -> Overpass 'south,west,north,east'."""
    min_lon, min_lat, max_lon, max_lat = bbox
    return f"{min_lat},{min_lon},{max_lat},{max_lon}"


def _shrink_bbox(bbox: tuple[float, float, float, float], factor: float = 0.5) -> tuple[float, float, float, float]:
    """Shrink a bbox toward its centre (factor=0.5 -> half width/height, ~1/4 area)."""
    min_lon, min_lat, max_lon, max_lat = bbox
    clon, clat = (min_lon + max_lon) / 2, (min_lat + max_lat) / 2
    hw, hh = (max_lon - min_lon) * factor / 2, (max_lat - min_lat) * factor / 2
    return (clon - hw, clat - hh, clon + hw, clat + hh)


def _run_query(query: str, prefix: str) -> dict[str, Any] | None:
    """
    POST an Overpass QL query; return parsed JSON or None on hard failure.
    A JSON body carrying a 'remark' about a timeout/error is surfaced by the
    caller (it usually means partial results).
    """
    try:
        return cached_post_json(
            OVERPASS_URL, prefix=prefix, data={"data": query},
            headers=OVERPASS_HEADERS, timeout=OVERPASS_TIMEOUT + 30,
            max_retries=OVERPASS_MAX_RETRIES,
        )
    except Exception as exc:
        logger.warning(f"Overpass query '{prefix}' failed: {exc}")
        return None


def _fetch_layer(build_query, bbox: tuple[float, float, float, float], name: str):
    """
    Run one layer query with a smaller-bbox fallback. Returns
    (elements, status) where status is one of: 'ok', 'partial', 'fallback',
    'fallback_partial', 'failed'.
    """
    data = _run_query(build_query(bbox), prefix=f"overpass_{name}")
    if data is not None:
        remark = data.get("remark", "")
        if remark and ("timed out" in remark.lower() or "error" in remark.lower()):
            logger.warning(f"Overpass '{name}': partial results — remark: {remark.strip()}")
            return data.get("elements", []), "partial"
        return data.get("elements", []), "ok"

    # Hard failure (timeout/5xx after retries) — try a smaller bbox once.
    small = _shrink_bbox(bbox)
    logger.warning(f"Overpass '{name}': retrying against a shrunk bbox {tuple(round(v, 3) for v in small)}")
    data = _run_query(build_query(small), prefix=f"overpass_{name}_small")
    if data is None:
        logger.error(f"Overpass '{name}': failed on both full and shrunk bbox — layer will be empty")
        return [], "failed"
    remark = data.get("remark", "")
    status = "fallback_partial" if (remark and "timed out" in remark.lower()) else "fallback"
    return data.get("elements", []), status


# --- Layer queries -----------------------------------------------------------


def _q_roads(bbox):
    b = _bbox_to_overpass(bbox)
    return f'[out:json][timeout:{OVERPASS_TIMEOUT}];way["highway"]({b});out geom;'


def _q_industrial(bbox):
    b = _bbox_to_overpass(bbox)
    return (
        f'[out:json][timeout:{OVERPASS_TIMEOUT}];'
        f'(way["landuse"="industrial"]({b});relation["landuse"="industrial"]({b}););out geom;'
    )


def _q_pois(bbox):
    b = _bbox_to_overpass(bbox)
    amenities = "|".join(POI_CATEGORIES)
    return (
        f'[out:json][timeout:{OVERPASS_TIMEOUT}];'
        f'(nwr["amenity"~"^({amenities})$"]({b});'
        # elderly / senior care lives under social_facility, not amenity
        f'nwr["social_facility"~"nursing_home|assisted_living|group_home"]({b});'
        f'nwr["amenity"="social_facility"]["social_facility:for"~"senior|elderly"]({b}););'
        f'out center;'
    )


# --- Element parsing ---------------------------------------------------------


def _elem_point(el: dict) -> tuple[float, float] | None:
    """Extract a representative (lat, lon) from a node ('lat'/'lon') or a
    way/relation ('center')."""
    if el.get("type") == "node" and "lat" in el:
        return el["lat"], el["lon"]
    c = el.get("center")
    if c:
        return c["lat"], c["lon"]
    return None


def _build_road_grid(elements: list, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = (min_lat + max_lat) / 2
    dlat = GRID_CELL_KM / 111.0
    dlon = GRID_CELL_KM / (111.320 * max(math.cos(math.radians(mid_lat)), 0.01))

    cells: dict[tuple[int, int], float] = {}
    for el in elements:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        pts = [(g["lat"], g["lon"]) for g in geom]
        for (lat1, lon1), (lat2, lon2) in zip(pts, pts[1:]):
            seg_km = haversine_km(lat1, lon1, lat2, lon2)
            if seg_km == 0:
                continue
            mlat, mlon = (lat1 + lat2) / 2, (lon1 + lon2) / 2
            col = int((mlon - min_lon) / dlon)
            row = int((mlat - min_lat) / dlat)
            cells[(row, col)] = cells.get((row, col), 0.0) + seg_km

    rows = []
    for (row, col), length_km in cells.items():
        c_min_lon = min_lon + col * dlon
        c_min_lat = min_lat + row * dlat
        c_max_lon = c_min_lon + dlon
        c_max_lat = c_min_lat + dlat
        center_lat = c_min_lat + dlat / 2
        center_lon = c_min_lon + dlon / 2
        # Actual cell footprint (km^2) so density is comparable across latitudes.
        w_km = haversine_km(center_lat, c_min_lon, center_lat, c_max_lon)
        h_km = haversine_km(c_min_lat, center_lon, c_max_lat, center_lon)
        area_km2 = max(w_km * h_km, 1e-6)
        rows.append(
            {
                "cell_id": f"r{row}_c{col}",
                "row": row,
                "col": col,
                "min_lon": round(c_min_lon, 6),
                "min_lat": round(c_min_lat, 6),
                "max_lon": round(c_max_lon, 6),
                "max_lat": round(c_max_lat, 6),
                "center_lat": round(center_lat, 6),
                "center_lon": round(center_lon, 6),
                "road_length_km": round(length_km, 4),
                "road_density_km_per_km2": round(length_km / area_km2, 4),
            }
        )
    cols = ["cell_id", "row", "col", "min_lon", "min_lat", "max_lon", "max_lat",
            "center_lat", "center_lon", "road_length_km", "road_density_km_per_km2"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("road_length_km", ascending=False).reset_index(drop=True)


def _polygon_from_geom(geom: list) -> list[list[float]]:
    """OSM 'geometry' [{lat,lon},...] -> GeoJSON ring [[lon,lat],...] (closed)."""
    ring = [[g["lon"], g["lat"]] for g in geom]
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring


def _ring_area_km2(geom: list) -> float:
    """Shoelace area of a ring, projected equirectangularly at its mean lat."""
    if len(geom) < 3:
        return 0.0
    mean_lat = sum(g["lat"] for g in geom) / len(geom)
    kx = 111.320 * math.cos(math.radians(mean_lat))
    ky = 111.0
    pts = [(g["lon"] * kx, g["lat"] * ky) for g in geom]
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _build_industrial(elements: list) -> pd.DataFrame:
    cols = ["osm_type", "osm_id", "name", "centroid_lat", "centroid_lon", "area_km2", "geometry"]
    rows = []
    for el in elements:
        geom = el.get("geometry")
        if not geom:
            continue
        lats = [g["lat"] for g in geom]
        lons = [g["lon"] for g in geom]
        tags = el.get("tags", {}) or {}
        rows.append(
            {
                "osm_type": el.get("type"),
                "osm_id": el.get("id"),
                "name": tags.get("name"),
                "centroid_lat": round(sum(lats) / len(lats), 6),
                "centroid_lon": round(sum(lons) / len(lons), 6),
                "area_km2": round(_ring_area_km2(geom), 5),
                "geometry": _polygon_from_geom(geom),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values("area_km2", ascending=False).reset_index(drop=True)


def _build_pois(elements: list) -> pd.DataFrame:
    cols = ["category", "name", "lat", "lon", "osm_type", "osm_id"]
    rows = []
    for el in elements:
        pt = _elem_point(el)
        if pt is None:
            continue
        tags = el.get("tags", {}) or {}
        amenity = tags.get("amenity", "")
        if amenity in POI_CATEGORIES:
            category = POI_CATEGORIES[amenity]
        elif tags.get("social_facility") or tags.get("social_facility:for"):
            category = "elderly_care"
        elif amenity == "social_facility":
            category = "elderly_care"
        else:
            continue
        rows.append(
            {
                "category": category,
                "name": tags.get("name"),
                "lat": round(pt[0], 6),
                "lon": round(pt[1], 6),
                "osm_type": el.get("type"),
                "osm_id": el.get("id"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].drop_duplicates(subset=["osm_type", "osm_id"]).reset_index(drop=True)


# --- Public API --------------------------------------------------------------


def fetch_city_layers(city_bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    """
    Fetch road-density grid, industrial zones and vulnerability POIs for a city
    bounding box (min_lon, min_lat, max_lon, max_lat).

    Returns a dict with keys:
      'road_density_grid'  -> DataFrame of 1 km cells with road length/density
      'industrial_zones'   -> DataFrame of industrial polygons (+ area, geometry)
      'vulnerability_pois' -> DataFrame of hospital/school/elderly-care points
      '_meta'              -> per-layer fetch status ('ok' | 'partial' |
                              'fallback' | 'fallback_partial' | 'failed') and
                              the bbox used, for run_ingestion to report on.
    """
    meta: dict[str, Any] = {"bbox": city_bbox, "status": {}}

    road_elems, road_status = _fetch_layer(_q_roads, city_bbox, "roads")
    time.sleep(OVERPASS_COOLDOWN_S)
    ind_elems, ind_status = _fetch_layer(_q_industrial, city_bbox, "industrial")
    time.sleep(OVERPASS_COOLDOWN_S)
    poi_elems, poi_status = _fetch_layer(_q_pois, city_bbox, "pois")
    meta["status"] = {"roads": road_status, "industrial": ind_status, "pois": poi_status}

    return {
        "road_density_grid": _build_road_grid(road_elems, city_bbox),
        "industrial_zones": _build_industrial(ind_elems),
        "vulnerability_pois": _build_pois(poi_elems),
        "_meta": meta,
    }


__all__ = ["fetch_city_layers"]
