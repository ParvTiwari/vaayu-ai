"""
Vaayu AI — FastAPI backend entrypoint.

Routes for /forecast, /enforcement-priorities, /advisory, and /query will be
added here as each remaining agent is implemented. /attribution and /heatmap
are served by the deterministic Attribution Agent.
"""
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# Load .env from the project root before any agent module reads os.getenv().
load_dotenv(Path(__file__).resolve().parent / ".env")

from agents.advisory_agent import run_advisory  # noqa: E402
from agents.attribution_agent import (  # noqa: E402
    grid_to_geojson,
    interpolate_city_aqi,
    run_attribution,
)
from agents.enforcement_agent import run_enforcement  # noqa: E402

app = FastAPI(
    title="Vaayu AI",
    description="Urban Air Quality Intelligence — ET AI Hackathon 2026, Problem Statement 5",
    version="0.1.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "vaayu-ai"}


@app.get("/attribution/{city}")
def attribution(
    city: str,
    lat: Optional[float] = Query(None, description="Latitude of the point to score"),
    lon: Optional[float] = Query(None, description="Longitude of the point to score"),
    station_id: Optional[str] = Query(None, description="Station id (alternative to lat/lon)"),
    date: Optional[str] = Query(None, description="ISO date; affects fire seasonality"),
) -> dict:
    """Deterministic, rule-based source attribution for a point in `city`."""
    state = {"city": city, "lat": lat, "lon": lon, "station_id": station_id, "date": date}
    result = run_attribution(state)
    if result.get("status") != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "attribution failed"))
    return {"city": city, "attribution": result["attribution"], "status": "ok"}


@app.get("/heatmap/{city}")
def heatmap(
    city: str,
    timestamp: str = Query(..., description="ISO timestamp to interpolate AQI at"),
    grid_resolution_km: float = Query(1.0, gt=0.05, le=10.0),
) -> dict:
    """IDW-interpolated AQI grid over the city bbox, as a GeoJSON point layer."""
    grid = interpolate_city_aqi(city, timestamp, grid_resolution_km)
    if grid.empty:
        raise HTTPException(
            status_code=404,
            detail=f"no station readings near {timestamp} for {city} (or unknown city)",
        )
    fc = grid_to_geojson(grid)
    fc["properties"] = {"city": city, "timestamp": timestamp, "n_cells": len(grid)}
    return fc


@app.get("/enforcement-priorities/{city}")
def enforcement_priorities(city: str) -> dict:
    """Deterministic ranking of monitoring zones by enforcement/response priority."""
    result = run_enforcement({"city": city})
    if result.get("status") != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "enforcement ranking failed"))
    return {
        "city": city,
        "priority_zones": result["priority_zones"],
        "methodology_note": result["methodology_note"],
        "status": "ok",
    }


class AdvisoryRequest(BaseModel):
    city: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    lang: str = "en"
    station_id: Optional[str] = None
    forecast: Optional[dict[str, Any]] = None
    attribution: Optional[dict[str, Any]] = None


@app.post("/advisory")
def advisory(req: AdvisoryRequest) -> dict:
    """Localized, source-cited citizen advisory (deterministic guidance + LLM narration)."""
    result = run_advisory(req.model_dump())
    if result.get("status") != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "advisory failed"))
    return {
        "advisory_text": result["advisory_text"],
        "sources_cited": result["sources_cited"],
        "voice_output_path": result.get("voice_output_path"),
        "status": "ok",
    }
