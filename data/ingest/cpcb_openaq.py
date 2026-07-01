"""
Fetch historical and live AQI station readings.

Primary source: CPCB via data.gov.in (CPCB_API_KEY).
Fallback source: OpenAQ (api.openaq.org), used when CPCB has no data for a
station/date range.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

import pandas as pd

# Expected output schema for both functions below:
# station_id, station_name, lat, lon, timestamp, pm25, pm10, aqi, source


def fetch_historical_aqi(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical hourly AQI readings for all stations in `city`. TODO: implement."""
    raise NotImplementedError("fetch_historical_aqi is not yet implemented")


def fetch_live_aqi(city: str) -> pd.DataFrame:
    """Fetch current AQI readings for all stations in `city`. TODO: implement."""
    raise NotImplementedError("fetch_live_aqi is not yet implemented")


__all__ = ["fetch_historical_aqi", "fetch_live_aqi"]
