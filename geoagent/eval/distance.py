"""Coordinate distance metrics for geolocation evaluation."""

from __future__ import annotations

import math

# Distance thresholds (meters) used for coordinate-level accuracy.
DISTANCE_THRESHOLDS: tuple[int, ...] = (10, 20, 50, 100, 1000, 5000, 25000)

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points in meters."""

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return _EARTH_RADIUS_M * c


def within_threshold(distance_m: float, threshold_m: float) -> bool:
    return distance_m <= threshold_m
