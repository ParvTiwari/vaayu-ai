"""
Run full historical ingestion for both pilot cities, join AQI + weather by
station + nearest-hour, and write the unified dataset to
data/db/unified_history.parquet.

Run from the project root as:
    python data/ingest/run_ingestion.py
or:
    python -m data.ingest.run_ingestion
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingest.cpcb_openaq import CITY_CONFIG, fetch_historical_aqi  # noqa: E402
from data.ingest.weather import fetch_historical_weather  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CITIES = list(CITY_CONFIG.keys())
HISTORY_DAYS = 18 * 30  # ~18 months
OUTPUT_PATH = ROOT / "data" / "db" / "unified_history.parquet"

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


def run() -> pd.DataFrame:
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)
    logger.info(f"Ingestion window: {start_date} .. {end_date} ({HISTORY_DAYS} days)")

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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Wrote {len(unified)} rows to {OUTPUT_PATH}")
    return unified


if __name__ == "__main__":
    run()