"""
Fetch active fire detections from NASA FIRMS (FIRMS_API_KEY required) —
used as a stubble-burning signal for the Attribution Agent and as a
forecast model feature.

Not yet implemented — scaffold only.
"""
from __future__ import annotations

import pandas as pd

# Expected output schema:
# lat, lon, detection_date, confidence, frp


def fetch_fire_detections(bbox: tuple[float, float, float, float], start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch active fire detections within a bounding box and date range. TODO: implement."""
    raise NotImplementedError("fetch_fire_detections is not yet implemented")


__all__ = ["fetch_fire_detections"]
