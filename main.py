"""
Vaayu AI — FastAPI backend entrypoint.

Routes for /forecast, /enforcement-priorities, /advisory, and /query will be
added here as each remaining agent is implemented. /attribution and /heatmap
are served by the deterministic Attribution Agent.
"""
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

# Load .env from the project root before any agent module reads os.getenv().
load_dotenv(Path(__file__).resolve().parent / ".env")

from agents.attribution_agent import (  # noqa: E402
    grid_to_geojson,
    interpolate_city_aqi,
    run_attribution,
)

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
