"""
Fetch historical and live AQI station readings for Indian cities.

DATA SOURCE REALITY CHECK (read before modifying anything below):

- CPCB's public data.gov.in resource ("Real time Air Quality Index from
  various locations", resource id 3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69) is a
  LIVE SNAPSHOT feed. It has no date-range filter and cannot serve historical
  hourly data — every call returns whatever CPCB's dashboard currently holds
  per station. It IS the primary, authoritative source for fetch_live_aqi().
  For fetch_historical_aqi(), it will only ever contribute rows for "today"
  (if today falls inside the requested range) — that is expected behavior,
  not a bug. Genuine historical backfill comes from OpenAQ, which is the
  only one of these two sources with a real historical query API
  (/v3/sensors/{id}/hours with date_from/date_to).
- Neither source reports a single ready-made AQI number consistently: CPCB
  gives one row per station+pollutant with a Pollutant_Avg concentration;
  OpenAQ gives raw pollutant concentrations only. We derive an approximate
  AQI from PM2.5/PM10 using the published CPCB sub-index breakpoints (see
  estimate_aqi() below) and take the max sub-index, mirroring CPCB's own
  combination rule. This is documented as an approximation — official CPCB
  AQI can also weigh in NO2/SO2/O3/CO/NH3/Pb, which we are not ingesting.
- OpenAQ API v3 requires an X-API-Key header for every request (no
  key-optional tier). If OPENAQ_API_KEY is unset, OpenAQ calls are skipped
  with a clear warning rather than failing loudly.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta

import pandas as pd

from ._http_utils import cached_json_get

logger = logging.getLogger(__name__)

SCHEMA_COLUMNS = ["station_id", "station_name", "lat", "lon", "timestamp", "pm25", "pm10", "aqi", "source"]

CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_BASE_URL = f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"

OPENAQ_BASE_URL = "https://api.openaq.org/v3"

# bbox = (min_lon, min_lat, max_lon, max_lat), buffered a few km around the
# official city boundary. CPCB's "city" filter uses whatever label its own
# station metadata carries, which for Bengaluru has historically appeared as
# either "Bengaluru" or "Bangalore" — we query both aliases and merge results.
CITY_CONFIG = {
    "Delhi": {
        "cpcb_aliases": ["Delhi"],
        "bbox": (76.80, 28.35, 77.35, 28.90),
    },
    "Bengaluru": {
        "cpcb_aliases": ["Bengaluru", "Bangalore"],
        "bbox": (77.45, 12.75, 77.75, 13.20),
    },
    "Indore": {
        "cpcb_aliases": ["Indore"],
        "bbox": (75.70, 22.55, 76.00, 22.85),
    },
}

# --- CPCB / official Indian AQI sub-index breakpoints ------------------------
# (range_low, range_high, index_low, index_high), µg/m3 -> AQI points.
_PM25_BREAKPOINTS = [
    (0.0, 30.0, 0, 50),
    (30.1, 60.0, 51, 100),
    (60.1, 90.0, 101, 200),
    (90.1, 120.0, 201, 300),
    (120.1, 250.0, 301, 400),
    (250.1, 380.0, 401, 500),
]
_PM10_BREAKPOINTS = [
    (0.0, 50.0, 0, 50),
    (50.1, 100.0, 51, 100),
    (100.1, 250.0, 101, 200),
    (250.1, 350.0, 201, 300),
    (350.1, 430.0, 301, 400),
    (430.1, 510.0, 401, 500),
]


def _linear_subindex(conc: float | None, breakpoints: list[tuple[float, float, int, int]]) -> float | None:
    if conc is None or pd.isna(conc):
        return None
    for lo, hi, ilo, ihi in breakpoints:
        if lo <= conc <= hi:
            return round(ilo + (ihi - ilo) * (conc - lo) / (hi - lo), 1)
    lo, hi, ilo, ihi = breakpoints[-1]
    if conc > hi:
        # Beyond the top published band — extend linearly rather than
        # silently dropping the reading. Flagged as an approximation.
        return round(ihi + (conc - hi), 1)
    return None


def estimate_aqi(pm25: float | None, pm10: float | None) -> float | None:
    """
    Approximate the Indian AQI from PM2.5/PM10 concentrations using CPCB's
    published sub-index breakpoints, taking the max sub-index (CPCB's own
    combination rule across pollutants). This is an approximation: official
    CPCB AQI can also incorporate NO2/SO2/O3/CO/NH3/Pb, not ingested here.

    The result is capped at 500 — the top of the official CPCB AQI scale.
    Beyond the top published PM band the sub-index is extended linearly (see
    _linear_subindex), which for the occasional garbage/extreme sensor reading
    (PM2.5 in the thousands µg/m³) would otherwise yield implausible AQI values
    well above 500. Capping keeps every AQI on the official 0-500 scale.
    """
    sub_indices = [
        s
        for s in (_linear_subindex(pm25, _PM25_BREAKPOINTS), _linear_subindex(pm10, _PM10_BREAKPOINTS))
        if s is not None
    ]
    return min(max(sub_indices), 500.0) if sub_indices else None


def _safe_float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _slugify_station_id(name: str, prefix: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return f"{prefix}_{slug or 'unknown'}"


def _empty_schema_df() -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


# --- CPCB (data.gov.in) ------------------------------------------------------


def _cpcb_api_key() -> str:
    return os.getenv("CPCB_API_KEY", "").strip()


def _cpcb_fetch_records(city_alias: str, api_key: str, page_limit: int = 500) -> list[dict]:
    """Paginate through the CPCB resource for a single city alias."""
    records: list[dict] = []
    offset = 0
    while True:
        params = {
            "api-key": api_key,
            "format": "json",
            "filters[city]": city_alias,
            "limit": page_limit,
            "offset": offset,
        }
        data = cached_json_get(
            CPCB_BASE_URL, prefix=f"cpcb_{city_alias}_{offset}", params=params
        )
        batch = data.get("records", [])
        records.extend(batch)
        total = int(data.get("total", len(records)) or len(records))
        offset += len(batch)
        if len(batch) < page_limit or offset >= total or not batch:
            break
    return records


def _parse_cpcb_records(records: list[dict]) -> pd.DataFrame:
    """Pivot raw CPCB (station, pollutant) rows into one row per station+timestamp."""
    parsed = []
    for r in records:
        pollutant = str(r.get("pollutant_id", "")).strip().upper().replace(".", "")
        if pollutant not in ("PM25", "PM10"):
            continue
        avg = _safe_float(r.get("pollutant_avg"))
        if avg is None:
            continue
        ts_raw = r.get("last_update")
        if not ts_raw:
            continue
        try:
            # CPCB's last_update is IST wall-clock time with no tz marker.
            ts_naive = pd.to_datetime(ts_raw, dayfirst=True)
            ts_utc = (ts_naive - timedelta(hours=5, minutes=30)).tz_localize("UTC")
        except (ValueError, TypeError):
            logger.debug(f"Could not parse CPCB timestamp: {ts_raw!r}")
            continue
        parsed.append(
            {
                "station_name": r.get("station", "unknown"),
                "lat": _safe_float(r.get("latitude")),
                "lon": _safe_float(r.get("longitude")),
                "timestamp": ts_utc,
                "pollutant": "pm25" if pollutant == "PM25" else "pm10",
                "value": avg,
            }
        )

    if not parsed:
        return _empty_schema_df()

    df = pd.DataFrame(parsed)
    pivot = df.pivot_table(
        index=["station_name", "lat", "lon", "timestamp"],
        columns="pollutant",
        values="value",
        aggfunc="mean",
    ).reset_index()
    for col in ("pm25", "pm10"):
        if col not in pivot.columns:
            pivot[col] = None

    pivot["station_id"] = pivot["station_name"].apply(lambda n: _slugify_station_id(n, "cpcb"))
    pivot["aqi"] = pivot.apply(lambda row: estimate_aqi(row.get("pm25"), row.get("pm10")), axis=1)
    pivot["source"] = "cpcb_datagovin"
    return pivot[SCHEMA_COLUMNS]


# --- OpenAQ v3 ----------------------------------------------------------------


def _openaq_headers() -> dict | None:
    key = os.getenv("OPENAQ_API_KEY", "").strip()
    if not key:
        logger.warning(
            "OPENAQ_API_KEY not set - OpenAQ v3 requires a free API key "
            "(sign up at https://explore.openaq.org/register). Skipping OpenAQ."
        )
        return None
    return {"X-API-Key": key}


def _openaq_locations_for_bbox(bbox: tuple[float, float, float, float], limit: int = 100) -> list[dict]:
    headers = _openaq_headers()
    if headers is None:
        return []
    params = {"bbox": ",".join(str(v) for v in bbox), "limit": limit}
    try:
        data = cached_json_get(
            f"{OPENAQ_BASE_URL}/locations", prefix="openaq_locations", params=params, headers=headers
        )
    except Exception as exc:
        logger.warning(f"OpenAQ locations fetch failed for bbox {bbox}: {exc}")
        return []
    return data.get("results", [])


def _openaq_sensors_for_location(location_id: int, headers: dict) -> list[dict]:
    try:
        data = cached_json_get(
            f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors",
            prefix=f"openaq_sensors_{location_id}",
            headers=headers,
        )
    except Exception as exc:
        logger.warning(f"OpenAQ sensors fetch failed for location {location_id}: {exc}")
        return []
    return data.get("results", [])


def _openaq_sensor_hours(
    sensor_id: int, date_from: str, date_to: str, headers: dict, limit: int = 1000, max_pages: int = 100
) -> list[dict]:
    """
    Paginate a sensor's /hours results within [date_from, date_to].

    OpenAQ v3's /hours endpoint has been observed to honour date_from but NOT
    date_to server-side: it keeps returning full pages of a sensor's entire
    remaining history well past the requested window, until an eventual
    (slow) 408. len(batch) < limit therefore may never fire before we've
    walked years of unwanted data. Fix: check each row's own timestamp and
    stop as soon as we see data past date_to, rather than trusting page size
    alone to signal "done". This is a per-row check (not an order
    assumption), so it's correct even if a single page straddles the boundary.
    """
    results: list[dict] = []
    page = 1
    date_to_ts = pd.Timestamp(date_to)
    while page <= max_pages:
        params = {"date_from": date_from, "date_to": date_to, "limit": limit, "page": page}
        try:
            data = cached_json_get(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
                prefix=f"openaq_hours_{sensor_id}_{date_from}_{date_to}_p{page}",
                params=params,
                headers=headers,
            )
        except Exception as exc:
            logger.warning(f"OpenAQ hours fetch failed for sensor {sensor_id} page {page}: {exc}")
            break
        batch = data.get("results", [])
        if not batch:
            break

        past_end = False
        for row in batch:
            ts = ((row.get("period") or {}).get("datetimeFrom") or {}).get("utc")
            if ts and pd.Timestamp(ts) > date_to_ts:
                past_end = True
                continue
            results.append(row)

        if past_end:
            logger.debug(f"sensor {sensor_id}: page {page} crossed date_to — stopping pagination early")
            break
        if len(batch) < limit:
            break
        page += 1
    if page > max_pages:
        logger.warning(f"OpenAQ sensor {sensor_id} hit pagination safety cap ({max_pages} pages)")
    return results


def _fetch_openaq_historical(bbox: tuple[float, float, float, float], start_date: str, end_date: str) -> pd.DataFrame:
    headers = _openaq_headers()
    if headers is None:
        return _empty_schema_df()

    locations = _openaq_locations_for_bbox(bbox)
    if not locations:
        logger.info(f"OpenAQ: no locations found for bbox {bbox}")
        return _empty_schema_df()

    date_from = f"{start_date}T00:00:00Z"
    date_to = f"{end_date}T23:59:59Z"
    date_from_ts = pd.Timestamp(date_from)
    date_to_ts = pd.Timestamp(date_to)
    rows = []
    skipped_dead = 0
    for loc in locations:
        loc_id = loc.get("id")
        loc_name = loc.get("name") or f"openaq_{loc_id}"
        coords = loc.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")

        sensors = _openaq_sensors_for_location(loc_id, headers)
        pm_by_ts: dict[str, dict[str, float]] = {}
        for sensor in sensors:
            param_name = ((sensor.get("parameter") or {}).get("name") or "").lower()
            if param_name not in ("pm25", "pm10"):
                continue
            sensor_id = sensor.get("id")

            # /hours ignores date_from/date_to server-side (confirmed live) and
            # always paginates from the sensor's absolute first reading, so a
            # sensor whose datetimeLast predates our window would otherwise
            # burn pages walking its entire (irrelevant) history before
            # returning nothing. Sensor metadata already tells us this up
            # front — skip it without making a single /hours request.
            last_seen = ((sensor.get("datetimeLast") or {}).get("utc"))
            if last_seen and pd.Timestamp(last_seen) < date_from_ts:
                skipped_dead += 1
                logger.debug(f"sensor {sensor_id} ({loc_name}/{param_name}): last reading {last_seen} "
                             f"predates the window — skipping (dead sensor)")
                continue
            first_seen = ((sensor.get("datetimeFirst") or {}).get("utc"))
            if first_seen and pd.Timestamp(first_seen) > date_to_ts:
                skipped_dead += 1
                logger.debug(f"sensor {sensor_id} ({loc_name}/{param_name}): first reading {first_seen} "
                             f"is after the window — skipping")
                continue

            for h in _openaq_sensor_hours(sensor_id, date_from, date_to, headers):
                ts = ((h.get("period") or {}).get("datetimeFrom") or {}).get("utc")
                if not ts:
                    continue
                pm_by_ts.setdefault(ts, {})[param_name] = h.get("value")

        for ts, vals in pm_by_ts.items():
            pm25, pm10 = vals.get("pm25"), vals.get("pm10")
            rows.append(
                {
                    "station_id": f"openaq_{loc_id}",
                    "station_name": loc_name,
                    "lat": lat,
                    "lon": lon,
                    "timestamp": pd.to_datetime(ts, utc=True),
                    "pm25": pm25,
                    "pm10": pm10,
                    "aqi": estimate_aqi(pm25, pm10),
                    "source": "openaq",
                }
            )

    if not rows:
        return _empty_schema_df()
    return pd.DataFrame(rows)[SCHEMA_COLUMNS]


def _fetch_openaq_latest(bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    headers = _openaq_headers()
    if headers is None:
        return _empty_schema_df()

    locations = _openaq_locations_for_bbox(bbox)
    rows = []
    for loc in locations:
        loc_id = loc.get("id")
        loc_name = loc.get("name") or f"openaq_{loc_id}"
        coords = loc.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        try:
            data = cached_json_get(
                f"{OPENAQ_BASE_URL}/locations/{loc_id}/latest",
                prefix=f"openaq_latest_{loc_id}",
                headers=headers,
            )
        except Exception as exc:
            logger.warning(f"OpenAQ latest fetch failed for location {loc_id}: {exc}")
            continue

        pm: dict[str, float] = {}
        ts = None
        for item in data.get("results", []):
            param_name = ((item.get("parameter") or {}).get("name") or "").lower()
            if param_name in ("pm25", "pm10"):
                pm[param_name] = item.get("value")
                ts = ts or (item.get("datetime") or {}).get("utc")
        if not pm:
            continue

        rows.append(
            {
                "station_id": f"openaq_{loc_id}",
                "station_name": loc_name,
                "lat": lat,
                "lon": lon,
                "timestamp": pd.to_datetime(ts, utc=True) if ts else pd.Timestamp.now(tz="UTC"),
                "pm25": pm.get("pm25"),
                "pm10": pm.get("pm10"),
                "aqi": estimate_aqi(pm.get("pm25"), pm.get("pm10")),
                "source": "openaq",
            }
        )
    if not rows:
        return _empty_schema_df()
    return pd.DataFrame(rows)[SCHEMA_COLUMNS]


# --- Public API ---------------------------------------------------------------


def fetch_live_aqi(city: str) -> pd.DataFrame:
    """Fetch current AQI readings for all known stations in `city`."""
    if city not in CITY_CONFIG:
        raise ValueError(f"Unknown city '{city}'. Configured cities: {list(CITY_CONFIG)}")
    cfg = CITY_CONFIG[city]

    api_key = _cpcb_api_key()
    records: list[dict] = []
    if api_key:
        for alias in cfg["cpcb_aliases"]:
            try:
                records.extend(_cpcb_fetch_records(alias, api_key))
            except Exception as exc:
                logger.warning(f"[{city}] CPCB fetch failed for alias '{alias}': {exc}")
    else:
        logger.warning("CPCB_API_KEY not set - skipping CPCB, falling back to OpenAQ 'latest' for live data.")

    if records:
        df = _parse_cpcb_records(records)
        if not df.empty:
            logger.info(f"[{city}] CPCB live: {len(df)} station readings")
            return df
        logger.info(f"[{city}] CPCB returned no usable PM2.5/PM10 records, falling back to OpenAQ latest.")

    openaq_df = _fetch_openaq_latest(cfg["bbox"])
    logger.info(f"[{city}] OpenAQ latest: {len(openaq_df)} station readings")
    return openaq_df


def fetch_historical_aqi(city: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical hourly AQI readings for `city` between start_date and
    end_date (YYYY-MM-DD). See the module docstring for why CPCB only ever
    contributes "today" and OpenAQ carries the real historical backfill.
    """
    if city not in CITY_CONFIG:
        raise ValueError(f"Unknown city '{city}'. Configured cities: {list(CITY_CONFIG)}")
    cfg = CITY_CONFIG[city]
    frames = []

    today = datetime.utcnow().date().isoformat()
    if start_date <= today <= end_date:
        try:
            live = fetch_live_aqi(city)
            if not live.empty:
                frames.append(live)
                logger.info(f"[{city}] CPCB/OpenAQ-latest contributed {len(live)} rows for {today}")
        except Exception as exc:
            logger.warning(f"[{city}] live fetch failed while backfilling today's slice: {exc}")
    else:
        logger.info(
            f"[{city}] Requested range {start_date}..{end_date} does not include today - "
            f"CPCB's data.gov.in feed has no historical query capability, skipping it and "
            f"relying on OpenAQ for this range."
        )

    openaq_hist = _fetch_openaq_historical(cfg["bbox"], start_date, end_date)
    if not openaq_hist.empty:
        frames.append(openaq_hist)
        logger.info(f"[{city}] OpenAQ historical: {len(openaq_hist)} rows for {start_date}..{end_date}")

    if not frames:
        logger.warning(f"[{city}] No historical AQI data retrieved from any source for {start_date}..{end_date}")
        return _empty_schema_df()

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.drop_duplicates(subset=["station_id", "timestamp"])
        .sort_values(["station_id", "timestamp"])
        .reset_index(drop=True)
    )
    return combined


__all__ = ["fetch_historical_aqi", "fetch_live_aqi", "estimate_aqi", "CITY_CONFIG"]
