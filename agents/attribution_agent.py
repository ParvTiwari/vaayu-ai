"""
Attribution Agent — DETERMINISTIC, rule-based pollution-source scoring
(traffic density, industrial land-use proximity, nearby active-fire counts)
plus IDW interpolation for the city AQI heatmap.

No LLM involved in scoring — auditability over black-box claims.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

from typing import Any


def run_attribution(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: score likely pollution sources for a location. TODO: implement."""
    raise NotImplementedError("run_attribution is not yet implemented")


def interpolate_city_aqi(city: str, timestamp: str, grid_resolution_km: float = 1.0):
    """IDW-interpolate station readings into a city-wide grid for heatmap rendering. TODO: implement."""
    raise NotImplementedError("interpolate_city_aqi is not yet implemented")


__all__ = ["run_attribution", "interpolate_city_aqi"]
