"""
Small dependency-light geospatial helpers shared by the ingestion modules.

We deliberately avoid geopandas/shapely here: the operations we need (great-
circle distance, polyline length, point-in-bbox gridding) are a few lines of
numpy each, and pulling in the GEOS/GDAL stack would be a heavy, brittle
dependency for a hackathon pipeline. GeoJSON is emitted by hand with `json`.
"""
from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two scalar (lat, lon) points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def haversine_km_vec(lat1, lon1, lat2, lon2):
    """Vectorised haversine: any argument may be a numpy array (broadcast)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    dphi = lat2 - lat1
    dlmb = lon2 - lon1
    a = np.sin(dphi / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def polyline_length_km(coords: list[tuple[float, float]]) -> float:
    """Total length in km of a polyline given as [(lat, lon), ...]."""
    total = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(coords, coords[1:]):
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total


def buffer_bbox(bbox: tuple[float, float, float, float], km: float) -> tuple[float, float, float, float]:
    """
    Expand a (min_lon, min_lat, max_lon, max_lat) bbox outward by ~`km` in
    every direction. Longitude degrees are scaled by cos(latitude) so the
    buffer is roughly `km` on the ground rather than `km` of raw degrees.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    dlat = km / 111.0
    mid_lat = (min_lat + max_lat) / 2
    dlon = km / (111.320 * max(math.cos(math.radians(mid_lat)), 0.01))
    return (min_lon - dlon, min_lat - dlat, max_lon + dlon, max_lat + dlat)


__all__ = ["haversine_km", "haversine_km_vec", "polyline_length_km", "buffer_bbox", "EARTH_RADIUS_KM"]
