"""
Enforcement Agent — DETERMINISTIC, rule-based ranking of monitoring zones by
enforcement/response priority.

NO LLM is involved in this ranking. Auditable formulae are explicitly favoured
over opaque LLM judgment for
high-stakes recommendations, so the priority order here is a pure function of the
inputs. The LLM's only later role (in the Output Agent) is to *narrate* this
ranking — never to invent or re-order it.

FORMULA (explicit, documented)
------------------------------
For each zone (one per monitoring station in the city):

  risk_score  ∈ {1..6}
      Severity of the station's current AQI, using the exact CPCB category
      breakpoints from the Forecast Agent (good=1, satisfactory=2, moderate=3,
      poor=4, very_poor=5, severe=6). "Current" = the most recent observed AQI
      for that station; swap in the Forecast Agent's t+24h value once wired.

  vulnerability_score
      Weighted count of vulnerable OSM POIs within POI_RADIUS_KM (1 km), plus a
      small generic-residential-density term. Explicit weights — most
      vulnerable populations weighted highest:
          hospital       = 3.0   (patients, least able to avoid exposure)
          elderly_care   = 3.0   (elderly — same rationale)
          school         = 2.0   (children)
          generic density= 1.0 × normalised road density in [0,1]  (a proxy for
                                 residential/pedestrian presence; deliberately
                                 below a single school so POIs dominate)

  priority_score  ∈ [0, 100]
      priority_score = 100 × (risk_score × vulnerability_score) / max(risk × vuln
      over all of the city's zones). Normalised to 0–100 for readability; the
      single worst zone scores 100. Zones are returned sorted descending.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.attribution_agent import load_city_layers
from agents.forecast_agent import AQI_CATEGORIES
from data.ingest._geo_utils import haversine_km_vec

ROOT = Path(__file__).resolve().parent.parent
UNIFIED_PATH = ROOT / "data" / "db" / "unified_history.parquet"

POI_RADIUS_KM = 1.0
POI_WEIGHTS = {"hospital": 3.0, "elderly_care": 3.0, "school": 2.0}
GENERIC_DENSITY_WEIGHT = 1.0

try:  # keep in sync with the ingestion config; degrade gracefully if absent
    from data.ingest.cpcb_openaq import CITY_CONFIG
except Exception:  # pragma: no cover
    CITY_CONFIG = {}


# --- pure scoring (unit-testable) --------------------------------------------


def aqi_severity(aqi: float | None) -> int:
    """Map an AQI value to CPCB category severity 1..6 (good..severe)."""
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return 0
    for severity, (_name, (_lo, hi)) in enumerate(AQI_CATEGORIES.items(), start=1):
        if aqi <= hi:
            return severity
    return len(AQI_CATEGORIES)  # above the top breakpoint → severe


def vulnerability_from_pois(counts: dict[str, int], norm_density: float = 0.0) -> float:
    """Weighted vulnerability from nearby-POI counts + a generic-density term."""
    v = sum(POI_WEIGHTS.get(cat, 0.0) * n for cat, n in counts.items())
    v += GENERIC_DENSITY_WEIGHT * min(max(norm_density, 0.0), 1.0)
    return float(v)


def rank_zones(zones: list[dict]) -> list[dict]:
    """Attach priority_score (0–100, normalised to the worst zone) and sort desc.

    Deterministic: ties break by risk_score, then vulnerability_score, then id.
    Each input zone must carry `risk_score` and `vulnerability_score`.
    """
    raws = [z["risk_score"] * z["vulnerability_score"] for z in zones]
    mx = max(raws) if raws else 0.0
    out = []
    for z, raw in zip(zones, raws):
        z = dict(z)
        z["priority_score"] = round(100.0 * raw / mx, 2) if mx > 0 else 0.0
        out.append(z)
    return sorted(
        out,
        key=lambda z: (z["priority_score"], z["risk_score"], z["vulnerability_score"], str(z.get("station_id", ""))),
        reverse=True,
    )


# --- data loading ------------------------------------------------------------


def _latest_aqi_by_station(city: str) -> pd.DataFrame:
    """Most recent observed AQI per station for `city` (station_id, lat, lon, aqi)."""
    if not UNIFIED_PATH.exists():
        return pd.DataFrame(columns=["station_id", "lat", "lon", "aqi"])
    u = pd.read_parquet(UNIFIED_PATH, columns=["city", "station_id", "lat", "lon", "timestamp", "aqi"])
    u = u[(u["city"] == city)].dropna(subset=["lat", "lon", "aqi", "timestamp"])
    if u.empty:
        return pd.DataFrame(columns=["station_id", "lat", "lon", "aqi"])
    u = u.sort_values("timestamp")
    latest = u.groupby("station_id").tail(1)
    return latest[["station_id", "lat", "lon", "aqi"]].reset_index(drop=True)


def _count_pois_within(lat: float, lon: float, poi_lat, poi_lon, poi_cat, radius_km: float) -> dict[str, int]:
    if poi_lat.size == 0:
        return {}
    d = haversine_km_vec(lat, lon, poi_lat, poi_lon)
    within = d <= radius_km
    if not within.any():
        return {}
    cats, counts = np.unique(poi_cat[within], return_counts=True)
    return {str(c): int(n) for c, n in zip(cats, counts)}


def _norm_density_within(lat: float, lon: float, road_cells: list[dict], ref_density: float, radius_km: float) -> float:
    if not road_cells or ref_density <= 0:
        return 0.0
    near = [c["density"] for c in road_cells if haversine_km_vec(lat, lon, c["lat"], c["lon"]) <= radius_km]
    if not near:
        return 0.0
    return min(max((sum(near) / len(near)) / ref_density, 0.0), 1.0)


# --- LangGraph node ----------------------------------------------------------


def run_enforcement(state: dict[str, Any]) -> dict[str, Any]:
    """Deterministic enforcement-priority ranking for a city. See module docstring."""
    city = state.get("city")
    if not city or (CITY_CONFIG and city not in CITY_CONFIG):
        state["status"] = "error"
        state["error"] = f"unknown or missing city: {city!r}"
        return state

    layers = load_city_layers(city)
    pois = layers["pois"]
    poi_lat = np.array([p["lat"] for p in pois], dtype=float)
    poi_lon = np.array([p["lon"] for p in pois], dtype=float)
    poi_cat = np.array([p["category"] for p in pois], dtype=object)

    stations = _latest_aqi_by_station(city)
    zones: list[dict] = []
    for st in stations.itertuples(index=False):
        risk = aqi_severity(float(st.aqi))
        if risk == 0:
            continue
        counts = _count_pois_within(st.lat, st.lon, poi_lat, poi_lon, poi_cat, POI_RADIUS_KM)
        norm_density = _norm_density_within(st.lat, st.lon, layers["road_cells"], layers["ref_density"], POI_RADIUS_KM)
        vuln = vulnerability_from_pois(counts, norm_density)
        zones.append(
            {
                "station_id": st.station_id,
                "lat": round(float(st.lat), 6),
                "lon": round(float(st.lon), 6),
                "current_aqi": round(float(st.aqi), 1),
                "risk_score": risk,
                "vulnerability_score": round(vuln, 3),
                "nearby_vulnerable_pois": counts,
            }
        )

    priority_zones = rank_zones(zones)
    state["priority_zones"] = priority_zones
    state["methodology_note"] = (
        "Deterministic, non-LLM ranking. risk_score = CPCB AQI severity 1-6 (current "
        f"observed AQI); vulnerability_score = weighted POIs within {POI_RADIUS_KM:.0f}km "
        f"(hospital/elderly-care={POI_WEIGHTS['hospital']:.0f}, school={POI_WEIGHTS['school']:.0f}) "
        "+ generic road-density proxy (weight 1, normalised 0-1); "
        "priority_score = 100 x risk x vulnerability / worst-zone value. "
        "The LLM narrates this ranking downstream but never computes or reorders it."
    )
    state["status"] = "ok"
    return state


__all__ = [
    "run_enforcement",
    "rank_zones",
    "aqi_severity",
    "vulnerability_from_pois",
    "POI_WEIGHTS",
    "POI_RADIUS_KM",
]
