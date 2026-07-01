"""
Forecast Agent — serves 24/48/72h AQI forecasts from the trained model in
models/checkpoints/, with CPCB category mapping and fallback handling for
stations with insufficient history.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

from typing import Any

# CPCB AQI category breakpoints — deterministic lookup, not LLM-derived.
AQI_CATEGORIES = {
    "good": (0, 50),
    "satisfactory": (51, 100),
    "moderate": (101, 200),
    "poor": (201, 300),
    "very_poor": (301, 400),
    "severe": (401, 500),
}


def run_forecast(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: predict AQI for a city/station over the requested horizon. TODO: implement."""
    raise NotImplementedError("run_forecast is not yet implemented")


__all__ = ["run_forecast", "AQI_CATEGORIES"]
