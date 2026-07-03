"""
Run full historical ingestion for the pilot cities:
  1. AQI (CPCB live-slice + OpenAQ historical) joined to weather (Open-Meteo),
     written to data/db/unified_history.parquet.
  2. NASA FIRMS active fire detections, aggregated to a daily
     fires-within-radius feature per station, written to
     data/db/fire_daily.parquet.
  3. OSM/Overpass city layers (road-density grid, industrial zones,
     vulnerability POIs), written to data/db/city_layers/{city}_layers.geojson.

Run from the project root as:
    python data/ingest/run_ingestion.py
or:
    python -m data.ingest.run_ingestion
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load .env from the project root before anything reads os.getenv() —
# without this, CPCB_API_KEY/OPENAQ_API_KEY/FIRMS_API_KEY/etc. are never
# actually visible to the process even if they're set in .env.
load_dotenv(ROOT / ".env")

from data.ingest.cpcb_openaq import CITY_CONFIG, fetch_historical_aqi  # noqa: E402
from data.ingest.firms_fire import aggregate_daily_fire_counts, fetch_fire_detections  # noqa: E402
from data.ingest.osm_overpass import fetch_city_layers  # noqa: E402
from data.ingest.weather import fetch_historical_weather  # noqa: E402
from data.ingest._geo_utils import buffer_bbox  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CITIES = list(CITY_CONFIG.keys())
# Optionally restrict to a subset, e.g. VAAYU_CITIES="Delhi" or "Delhi,Indore",
# useful when only some cities' AQI is cached/available (OpenAQ backfill is slow).
_cities_env = os.getenv("VAAYU_CITIES", "").strip()
if _cities_env:
    _requested = [c.strip() for c in _cities_env.split(",") if c.strip()]
    CITIES = [c for c in _requested if c in CITY_CONFIG]
# ~18 months by default. Overridable with VAAYU_HISTORY_DAYS because OpenAQ v3's
# deep pagination is unreliable over long windows (later pages 408-timeout), so
# a full 18-month backfill across all of Delhi's sensors can run for hours; a
# shorter window is useful for a quick end-to-end run or CI.
HISTORY_DAYS = int(os.getenv("VAAYU_HISTORY_DAYS", str(18 * 30)))
DB_DIR = ROOT / "data" / "db"
OUTPUT_PATH = DB_DIR / "unified_history.parquet"
FIRE_OUTPUT_PATH = DB_DIR / "fire_daily.parquet"
LAYERS_DIR = DB_DIR / "city_layers"

# How far to buffer each city bbox when searching for active fires. ~150 km
# captures the Punjab/Haryana stubble-burning belt upwind of Delhi, which is
# the whole point of this feature.
FIRE_BUFFER_KM = 150.0
# Count fires within this radius of each individual station, per day.
FIRE_RADIUS_KM = 100.0

UNIFIED_SCHEMA = [
    "city",
    "station_id",
    "station_name",
    "lat",
    "lon",
    "timestamp",
    "pm25",
    "pm10",
    "aqi",
    "source",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "boundary_layer_height",
]

FIRE_DAILY_SCHEMA = ["city", "station_id", "detection_date", "fire_count", "mean_frp", "max_frp"]


def _nearest_hour_join(aqi_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join AQI readings to the nearest weather reading within 1 hour."""
    if aqi_df.empty:
        return aqi_df
    aqi_sorted = aqi_df.sort_values("timestamp")
    if weather_df.empty:
        merged = aqi_sorted.copy()
        for col in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation", "boundary_layer_height"):
            merged[col] = None
        return merged
    weather_sorted = weather_df.drop(columns=["lat", "lon"]).sort_values("timestamp")
    merged = pd.merge_asof(
        aqi_sorted,
        weather_sorted,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    return merged


def _stations_from_unified(unified: pd.DataFrame) -> pd.DataFrame:
    """One row per station with its coordinates, for fire aggregation."""
    if unified.empty:
        return pd.DataFrame(columns=["city", "station_id", "station_name", "lat", "lon"])
    return (
        unified.dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=["station_id"])[["city", "station_id", "station_name", "lat", "lon"]]
        .reset_index(drop=True)
    )


def ingest_aqi_weather(start_date, end_date) -> pd.DataFrame:
    """Fetch + join AQI and weather for every city into the unified schema."""
    all_frames: list[pd.DataFrame] = []
    for city in CITIES:
        logger.info(f"=== {city}: fetching historical AQI ===")
        aqi_df = fetch_historical_aqi(city, start_date.isoformat(), end_date.isoformat())

        n_stations = aqi_df["station_id"].nunique() if not aqi_df.empty else 0
        logger.info(f"{city}: {len(aqi_df)} AQI rows across {n_stations} station(s)")

        if aqi_df.empty:
            logger.warning(f"{city}: no AQI data retrieved from CPCB or OpenAQ - skipping weather join for this city")
            continue

        city_frames = []
        for station_id, station_group in aqi_df.groupby("station_id"):
            lat = station_group["lat"].iloc[0]
            lon = station_group["lon"].iloc[0]
            if pd.isna(lat) or pd.isna(lon):
                logger.warning(f"{city}/{station_id}: missing coordinates, skipping weather join for this station")
                joined = station_group.copy()
                for col in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m", "precipitation", "boundary_layer_height"):
                    joined[col] = None
                city_frames.append(joined)
                continue

            logger.info(f"{city}/{station_id}: fetching weather for ({lat:.4f}, {lon:.4f})")
            try:
                weather_df = fetch_historical_weather(lat, lon, start_date.isoformat(), end_date.isoformat())
            except Exception as exc:
                logger.warning(f"{city}/{station_id}: weather fetch failed ({exc}), leaving weather columns null")
                weather_df = pd.DataFrame(columns=["lat", "lon", "timestamp"])

            city_frames.append(_nearest_hour_join(station_group, weather_df))

        city_unified = pd.concat(city_frames, ignore_index=True)
        city_unified["city"] = city
        all_frames.append(city_unified)

    if not all_frames:
        logger.error("No AQI data retrieved for any city - writing an empty parquet file with schema only.")
        unified = pd.DataFrame(columns=UNIFIED_SCHEMA)
    else:
        unified = pd.concat(all_frames, ignore_index=True)
        for col in UNIFIED_SCHEMA:
            if col not in unified.columns:
                unified[col] = None
        unified = unified[UNIFIED_SCHEMA]

    DB_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {len(unified)} rows to {OUTPUT_PATH}")
    return unified


def ingest_fires(stations: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """
    Fetch FIRMS detections around each city (buffered bbox) and aggregate to a
    per-station, per-day fire-count feature. Fire detections are shared across
    all stations in a city, so we fetch once per city bbox.
    """
    per_city_daily: list[pd.DataFrame] = []
    summary: dict[str, int] = {}

    for city in CITIES:
        bbox = buffer_bbox(CITY_CONFIG[city]["bbox"], FIRE_BUFFER_KM)
        logger.info(f"=== {city}: fetching FIRMS fire detections (bbox buffered ~{FIRE_BUFFER_KM:.0f}km) ===")
        fires = fetch_fire_detections(bbox, start_date.isoformat(), end_date.isoformat())
        summary[city] = len(fires)

        city_stations = stations[stations["city"] == city]
        if city_stations.empty:
            logger.warning(f"{city}: no stations with coordinates - cannot aggregate fires to stations")
            continue

        daily = aggregate_daily_fire_counts(fires, city_stations, radius_km=FIRE_RADIUS_KM)
        if not daily.empty:
            per_city_daily.append(daily)

    if per_city_daily:
        fire_daily = pd.concat(per_city_daily, ignore_index=True)
    else:
        fire_daily = pd.DataFrame(columns=FIRE_DAILY_SCHEMA)

    DB_DIR.mkdir(parents=True, exist_ok=True)
    fire_daily.to_parquet(FIRE_OUTPUT_PATH, index=False)
    logger.info(f"Wrote {len(fire_daily)} station-day fire rows to {FIRE_OUTPUT_PATH}")
    fire_daily.attrs["raw_detection_counts"] = summary
    return fire_daily


def _layers_to_geojson(layers: dict) -> dict:
    """Serialise the three city layers into one GeoJSON FeatureCollection,
    tagging each feature with its 'layer'."""
    features: list[dict] = []

    for _, cell in layers["road_density_grid"].iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "road_density_grid",
                    "cell_id": cell["cell_id"],
                    "road_length_km": cell["road_length_km"],
                    "road_density_km_per_km2": cell["road_density_km_per_km2"],
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [cell["min_lon"], cell["min_lat"]],
                        [cell["max_lon"], cell["min_lat"]],
                        [cell["max_lon"], cell["max_lat"]],
                        [cell["min_lon"], cell["max_lat"]],
                        [cell["min_lon"], cell["min_lat"]],
                    ]],
                },
            }
        )

    for _, zone in layers["industrial_zones"].iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "industrial_zones",
                    "osm_type": zone["osm_type"],
                    "osm_id": int(zone["osm_id"]) if pd.notna(zone["osm_id"]) else None,
                    "name": zone["name"],
                    "area_km2": zone["area_km2"],
                },
                "geometry": {"type": "Polygon", "coordinates": [zone["geometry"]]},
            }
        )

    for _, poi in layers["vulnerability_pois"].iterrows():
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": "vulnerability_pois",
                    "category": poi["category"],
                    "name": poi["name"],
                    "osm_type": poi["osm_type"],
                    "osm_id": int(poi["osm_id"]) if pd.notna(poi["osm_id"]) else None,
                },
                "geometry": {"type": "Point", "coordinates": [poi["lon"], poi["lat"]]},
            }
        )

    return {"type": "FeatureCollection", "properties": {"_meta": layers.get("_meta", {})}, "features": features}


def ingest_city_layers() -> dict[str, dict]:
    """Fetch and persist OSM/Overpass layers for every city."""
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    for city in CITIES:
        bbox = CITY_CONFIG[city]["bbox"]
        logger.info(f"=== {city}: fetching OSM/Overpass city layers ===")
        layers = fetch_city_layers(bbox)

        counts = {
            "road_cells": len(layers["road_density_grid"]),
            "road_length_km": round(float(layers["road_density_grid"]["road_length_km"].sum()), 1)
            if not layers["road_density_grid"].empty else 0.0,
            "industrial_zones": len(layers["industrial_zones"]),
            "vulnerability_pois": len(layers["vulnerability_pois"]),
            "status": layers["_meta"]["status"],
        }
        report[city] = counts
        logger.info(f"{city}: {counts}")

        geojson = _layers_to_geojson(layers)
        out_path = LAYERS_DIR / f"{city}_layers.geojson"
        out_path.write_text(json.dumps(geojson), encoding="utf-8")
        logger.info(f"{city}: wrote {len(geojson['features'])} features to {out_path}")

    return report


def run() -> None:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    logger.info(f"Ingestion window: {start_date} .. {end_date} ({HISTORY_DAYS} days)")

    unified = ingest_aqi_weather(start_date, end_date)
    stations = _stations_from_unified(unified)
    logger.info(f"{len(stations)} unique geolocated station(s) across all cities")

    fire_daily = ingest_fires(stations, start_date, end_date)
    layers_report = ingest_city_layers()

    # --- Final human-readable summary ---------------------------------------
    logger.info("================ INGESTION SUMMARY ================")
    logger.info(f"Unified AQI+weather rows: {len(unified)} ({stations['station_id'].nunique()} stations)")
    fire_counts = fire_daily.attrs.get("raw_detection_counts", {})
    for city in CITIES:
        logger.info(f"[{city}] raw fire detections (buffered bbox): {fire_counts.get(city, 0)}")
    if not fire_daily.empty:
        by_city = fire_daily.groupby("city")["fire_count"].sum().to_dict()
        logger.info(f"Station-day fire feature rows: {len(fire_daily)}; total fires-near-station by city: {by_city}")
    for city, counts in layers_report.items():
        logger.info(
            f"[{city}] OSM: {counts['road_cells']} road cells "
            f"({counts['road_length_km']} km), {counts['industrial_zones']} industrial zones, "
            f"{counts['vulnerability_pois']} vulnerability POIs; status={counts['status']}"
        )
    logger.info("==================================================")


if __name__ == "__main__":
    run()
