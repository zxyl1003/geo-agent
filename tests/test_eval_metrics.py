"""Tests for coordinate-distance evaluation metrics."""

from __future__ import annotations

from geoagent.eval import (
    DISTANCE_THRESHOLDS,
    haversine_m,
)


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def test_haversine_known_distance():
    # Xiamen (~24.48, 118.11) to Quanzhou (~24.87, 118.67) is ~60-70 km.
    d = haversine_m(24.4798, 118.0894, 24.8741, 118.6757)
    assert 55_000 < d < 75_000


def test_haversine_zero_distance():
    assert haversine_m(31.23, 121.47, 31.23, 121.47) == 0.0


def test_distance_thresholds_ordering():
    assert DISTANCE_THRESHOLDS == (10, 20, 50, 100, 1000, 5000, 25000)
