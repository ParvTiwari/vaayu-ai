"""
Fetch active fire detections from NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/api/)
and turn them into a daily "fires near station" feature.

Why this matters for Vaayu AI: crop-residue (stubble) burning in Punjab/Haryana
is a dominant, highly seasonal driver of Delhi's autumn PM2.5. Counting active
fire detections within a radius of each monitoring station gives the Forecast
Agent a leading feature and the Attribution Agent a defensible, auditable
"how much of today's load is regional biomass burning" signal.

Requires a free FIRMS map key in FIRMS_API_KEY (register at
https://firms.modaps.eosdis.nasa.gov/api/map_key/).

FIRMS area/CSV endpoint:
    /api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{START_DATE}
- DAY_RANGE is capped at 5 days per request for the VIIRS_SNPP sources used here
  (verified directly against the live API — requesting 10 returns HTTP 400
  "Invalid day range. Expects [1..5]."), so long ranges are chunked in 5-day steps.
- NRT ("near real-time") products only retain roughly the last ~2 months;
  older data lives in the SP ("standard processing") archive, which in turn
  lags the present by a couple of months. We therefore pick SP vs NRT per
  chunk based on how far back the chunk starts (see _source_for_chunk).
"""
from __future__ import annotations

import io
import logging
import os
from datetime import date, datetime, timedelta

import pandas as pd

from ._geo_utils import haversine_km_vec
from ._http_utils import cached_text_get

logger = logging.getLogger(__name__)

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# VIIRS 375 m is the workhorse for fire/hotspot work (much finer than MODIS
# 1 km and better at picking up small agricultural fires). SP = archive,
# NRT = last ~2 months.
SOURCE_ARCHIVE = "VIIRS_SNPP_SP"
SOURCE_NRT = "VIIRS_SNPP_NRT"
# Chunks starting within this many days of "today" are served by the NRT
# stream; older chunks come from the SP archive.
NRT_WINDOW_DAYS = 60

MAX_DAY_RANGE = 5  # FIRMS hard limit per request for VIIRS_SNPP sources (verified live)

SCHEMA_COLUMNS = ["lat", "lon", "detection_date", "confidence", "frp"]
DAILY_SCHEMA = ["city", "station_id", "detection_date", "fire_count", "mean_frp", "max_frp"]


def _firms_map_key() -> str:
    return os.getenv("FIRMS_API_KEY", "").strip()


def _empty_schema_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


def _date_chunks(start_date: str, end_date: str, size: int = MAX_DAY_RANGE):
    """Yield (chunk_start_date, day_span) tuples covering [start_date, end_date]."""
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    cur = start
    while cur <= end:
        span = min(size, (end - cur).days + 1)
        yield cur, span
        cur += timedelta(days=span)


def _source_for_chunk(chunk_start: date, source_override: str | None) -> str:
    if source_override:
        return source_override
    age_days = (datetime.utcnow().date() - chunk_start).days
    return SOURCE_NRT if age_days <= NRT_WINDOW_DAYS else SOURCE_ARCHIVE


def _parse_firms_csv(text: str) -> pd.DataFrame:
    """
    Parse a FIRMS area/CSV payload into the module schema. FIRMS signals bad
    requests (invalid key, unavailable source/date) with a short plain-text
    message rather than an HTTP error, so anything without a real CSV header
    is treated as "no data" and logged.
    """
    head = text.lstrip()[:200].lower()
    if "latitude" not in head:
        logger.warning(f"FIRMS returned a non-CSV response (likely an error): {text.strip()[:200]!r}")
        return _empty_schema_df()

    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return _empty_schema_df()

    out = pd.DataFrame(
        {
            "lat": pd.to_numeric(df.get("latitude"), errors="coerce"),
            "lon": pd.to_numeric(df.get("longitude"), errors="coerce"),
            "detection_date": pd.to_datetime(df.get("acq_date"), errors="coerce").dt.date,
            # VIIRS reports confidence as low/nominal/high, MODIS as 0-100;
            # keep it verbatim so downstream code can filter either way.
            "confidence": df.get("confidence").astype(str) if "confidence" in df else None,
            "frp": pd.to_numeric(df.get("frp"), errors="coerce"),
        }
    )
    return out.dropna(subset=["lat", "lon", "detection_date"]).reset_index(drop=True)


def fetch_fire_detections(
    bbox: tuple[float, float, float, float],
    start_date: str,
    end_date: str,
    source_override: str | None = None,
) -> pd.DataFrame:
    """
    Fetch active fire detections within `bbox` = (min_lon, min_lat, max_lon,
    max_lat) between start_date and end_date (YYYY-MM-DD, inclusive).

    Returns a DataFrame with columns: lat, lon, detection_date, confidence, frp.
    Returns an empty (correctly-typed) frame if FIRMS_API_KEY is unset or no
    detections are found — never raises on "no data".
    """
    map_key = _firms_map_key()
    if not map_key:
        logger.warning(
            "FIRMS_API_KEY not set — skipping fire ingestion. Get a free key at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/"
        )
        return _empty_schema_df()

    west, south, east, north = bbox
    area = f"{west},{south},{east},{north}"

    frames: list[pd.DataFrame] = []
    for chunk_start, span in _date_chunks(start_date, end_date):
        source = _source_for_chunk(chunk_start, source_override)
        url = f"{FIRMS_BASE_URL}/{map_key}/{source}/{area}/{span}/{chunk_start.isoformat()}"
        # Cache/log key deliberately excludes the secret map key.
        prefix = f"firms_{source}_{area}_{chunk_start.isoformat()}_{span}"
        try:
            text = cached_text_get(url, prefix=prefix, timeout=60)
        except Exception as exc:
            logger.warning(f"FIRMS fetch failed for {source} {chunk_start} (+{span}d): {exc}")
            continue
        chunk_df = _parse_firms_csv(text)
        if not chunk_df.empty:
            frames.append(chunk_df)

    if not frames:
        logger.info(f"FIRMS: no fire detections for bbox {bbox} over {start_date}..{end_date}")
        return _empty_schema_df()

    combined = pd.concat(frames, ignore_index=True)
    # The same hotspot can appear in overlapping chunks/passes; dedup on the
    # rounded location + date so daily counts aren't double-inflated.
    combined["_k"] = (
        combined["lat"].round(4).astype(str)
        + "_"
        + combined["lon"].round(4).astype(str)
        + "_"
        + combined["detection_date"].astype(str)
    )
    combined = combined.drop_duplicates(subset="_k").drop(columns="_k").reset_index(drop=True)
    logger.info(f"FIRMS: {len(combined)} unique detections for bbox {bbox} over {start_date}..{end_date}")
    return combined[SCHEMA_COLUMNS]


def aggregate_daily_fire_counts(
    fires_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    radius_km: float = 100.0,
) -> pd.DataFrame:
    """
    Collapse raw detections into a per-station, per-day feature: how many fires
    were detected within `radius_km` of each station on each day, plus the
    mean/max fire radiative power of those fires (an intensity proxy).

    `stations_df` must have columns: city, station_id, lat, lon.
    Returns a DataFrame with columns: city, station_id, detection_date,
    fire_count, mean_frp, max_frp.
    """
    if stations_df.empty:
        return pd.DataFrame(columns=DAILY_SCHEMA)

    stations = stations_df.dropna(subset=["lat", "lon"]).drop_duplicates(subset=["station_id"])
    all_dates = sorted(fires_df["detection_date"].unique()) if not fires_df.empty else []

    rows: list[dict] = []
    fire_lat = fires_df["lat"].to_numpy() if not fires_df.empty else None
    fire_lon = fires_df["lon"].to_numpy() if not fires_df.empty else None

    for _, st in stations.iterrows():
        if fires_df.empty:
            # Still emit an explicit zero-history row so the feature table has
            # an entry per station even when no fires were retrieved at all.
            rows.append(
                {"city": st.get("city"), "station_id": st["station_id"], "detection_date": None,
                 "fire_count": 0, "mean_frp": None, "max_frp": None}
            )
            continue

        dists = haversine_km_vec(st["lat"], st["lon"], fire_lat, fire_lon)
        within = dists <= radius_km
        if not within.any():
            continue
        near = fires_df.loc[within]
        for det_date, grp in near.groupby("detection_date"):
            rows.append(
                {
                    "city": st.get("city"),
                    "station_id": st["station_id"],
                    "detection_date": det_date,
                    "fire_count": int(len(grp)),
                    "mean_frp": round(float(grp["frp"].mean()), 3) if grp["frp"].notna().any() else None,
                    "max_frp": round(float(grp["frp"].max()), 3) if grp["frp"].notna().any() else None,
                }
            )

    if not rows:
        return pd.DataFrame(columns=DAILY_SCHEMA)
    return pd.DataFrame(rows)[DAILY_SCHEMA].sort_values(["city", "station_id", "detection_date"]).reset_index(drop=True)


__all__ = ["fetch_fire_detections", "aggregate_daily_fire_counts", "SCHEMA_COLUMNS", "DAILY_SCHEMA"]
