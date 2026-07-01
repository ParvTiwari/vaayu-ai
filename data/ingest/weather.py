"""
Fetch historical and forecast weather from Open-Meteo — free, no API key
required for either endpoint used here.

Historical: https://archive-api.open-meteo.com/v1/archive
Forecast:   https://api.open-meteo.com/v1/forecast

boundary_layer_height is a confirmed valid hourly variable on the archive
endpoint; the forecast endpoint's support for it is less certain across
Open-Meteo's underlying weather models, so requests fall back to dropping it
(and logging why) rather than failing the whole call on a 400.
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from ._http_utils import cached_json_get

logger = logging.getLogger(__name__)

ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "boundary_layer_height",
]

SCHEMA_COLUMNS = ["lat", "lon", "timestamp", *HOURLY_VARS]


def _hourly_response_to_df(data: dict, lat: float, lon: float, hourly_vars: list[str]) -> pd.DataFrame:
    hourly = data.get("hourly", {}) or {}
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    df = pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True)})
    for var in HOURLY_VARS:
        df[var] = hourly.get(var, [None] * len(times)) if var in hourly_vars else None
    df["lat"] = lat
    df["lon"] = lon
    return df[SCHEMA_COLUMNS]


def _fetch_hourly(base_url: str, prefix: str, lat: float, lon: float, extra_params: dict) -> pd.DataFrame:
    """
    Call an Open-Meteo hourly endpoint, retrying once without
    boundary_layer_height if the full variable list gets rejected (some
    Open-Meteo model backends don't expose it for forecast-horizon data).
    """
    hourly_vars = list(HOURLY_VARS)
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(hourly_vars), "timezone": "UTC", **extra_params}
    try:
        data = cached_json_get(base_url, prefix=f"{prefix}_full", params=params)
        return _hourly_response_to_df(data, lat, lon, hourly_vars)
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", None)
        if status == 400 and "boundary_layer_height" in hourly_vars:
            logger.warning(
                f"[{prefix}] request with boundary_layer_height was rejected (HTTP 400) - "
                f"retrying without it. This field is marked 'if available' for exactly this reason."
            )
            hourly_vars.remove("boundary_layer_height")
            params["hourly"] = ",".join(hourly_vars)
            data = cached_json_get(base_url, prefix=f"{prefix}_no_pbl", params=params)
            return _hourly_response_to_df(data, lat, lon, hourly_vars)
        raise


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical hourly weather for a coordinate between start_date and end_date (YYYY-MM-DD)."""
    prefix = f"openmeteo_archive_{lat}_{lon}_{start_date}_{end_date}"
    return _fetch_hourly(ARCHIVE_BASE_URL, prefix, lat, lon, {"start_date": start_date, "end_date": end_date})


def fetch_weather_forecast(lat: float, lon: float, hours_ahead: int = 72) -> pd.DataFrame:
    """Fetch forecast hourly weather for a coordinate, `hours_ahead` hours from now."""
    forecast_days = max(1, -(-hours_ahead // 24))  # ceil division
    prefix = f"openmeteo_forecast_{lat}_{lon}_{hours_ahead}"
    df = _fetch_hourly(FORECAST_BASE_URL, prefix, lat, lon, {"forecast_days": forecast_days})
    return df.head(hours_ahead).reset_index(drop=True)


__all__ = ["fetch_historical_weather", "fetch_weather_forecast", "HOURLY_VARS"]
