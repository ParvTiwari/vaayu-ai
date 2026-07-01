"""
Fetch historical and forecast weather from Open-Meteo (free, no API key).

Not yet implemented — scaffold only.
"""
from __future__ import annotations

import pandas as pd

# Expected output schema for both functions below:
# lat, lon, timestamp, temperature_2m, relative_humidity_2m, wind_speed_10m,
# wind_direction_10m, precipitation, boundary_layer_height


def fetch_historical_weather(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical hourly weather for a coordinate. TODO: implement."""
    raise NotImplementedError("fetch_historical_weather is not yet implemented")


def fetch_weather_forecast(lat: float, lon: float, hours_ahead: int = 72) -> pd.DataFrame:
    """Fetch forecast weather for a coordinate. TODO: implement."""
    raise NotImplementedError("fetch_weather_forecast is not yet implemented")


__all__ = ["fetch_historical_weather", "fetch_weather_forecast"]
