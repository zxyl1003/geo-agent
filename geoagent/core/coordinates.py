"""Coordinate and tile conversion helpers for map imagery tools."""

from __future__ import annotations

import math


Point = tuple[float, float]  # lon, lat

_PI = math.pi
_A = 6378245.0
_EE = 0.006693421622965823
_X_PI = _PI * 3000.0 / 180.0
_MERCATOR_MAX = 20037508.3427892

LLBAND = [75, 60, 45, 30, 15, 0]
LL2MC = [
    [
        -0.0015702102444,
        111320.7020616939,
        1704480524535203.0,
        -10338987376042340.0,
        26112667856603880.0,
        -35149669176653700.0,
        26595700718403920.0,
        -10725012454188240.0,
        1800819912950474.0,
        82.5,
    ],
    [
        0.0008277824516172526,
        111320.7020463578,
        647795574.6671607,
        -4082003173.641316,
        10774905663.51142,
        -15171875531.51559,
        12053065338.62167,
        -5124939663.577472,
        913311935.9512032,
        67.5,
    ],
    [
        0.00337398766765,
        111320.7020202162,
        4481351.045890365,
        -23393751.19931662,
        79682215.47186455,
        -115964993.2797253,
        97236711.15602145,
        -43661946.33752821,
        8477230.501135234,
        52.5,
    ],
    [
        0.00220636496208,
        111320.7020209128,
        51751.86112841131,
        3796837.749470245,
        992013.7397791013,
        -1221952.21711287,
        1340652.697009075,
        -620943.6990984312,
        144416.9293806241,
        37.5,
    ],
    [
        -0.0003441963504368392,
        111320.7020576856,
        278.2353980772752,
        2485758.690035394,
        6070.750963243378,
        54821.18345352118,
        9540.606633304236,
        -2710.55326746645,
        1405.483844121726,
        22.5,
    ],
    [
        -0.0003218135878613132,
        111320.7020701615,
        0.00369383431289,
        823725.6402795718,
        0.46104986909093,
        2351.343141331292,
        1.58060784298199,
        8.77738589078284,
        0.37238884252424,
        7.45,
    ],
]
MCBAND = [12890594.86, 8362377.87, 5591021.0, 3481989.83, 1678043.12, 0]
MC2LL = [
    [
        1.410526172116255e-8,
        0.00000898305509648872,
        -1.9939833816331,
        200.9824383106796,
        -187.2403703815547,
        91.6087516669843,
        -23.38765649603339,
        2.57121317296198,
        -0.03801003308653,
        17337981.2,
    ],
    [
        -7.435856389565537e-9,
        0.000008983055097726239,
        -0.78625201886289,
        96.32687599759846,
        -1.85204757529826,
        -59.36935905485877,
        47.40033549296737,
        -16.50741931063887,
        2.28786674699375,
        10260144.86,
    ],
    [
        -3.030883460898826e-8,
        0.00000898305509983578,
        0.30071316287616,
        59.74293618442277,
        7.357984074871,
        -25.38371002664745,
        13.45380521110908,
        -3.29883767235584,
        0.32710905363475,
        6856817.37,
    ],
    [
        -1.981981304930552e-8,
        0.000008983055099779535,
        0.03278182852591,
        40.31678527705744,
        0.65659298677277,
        -4.44255534477492,
        0.85341911805263,
        0.12923347998204,
        -0.04625736007561,
        4482777.06,
    ],
    [
        3.09191371068437e-9,
        0.000008983055096812155,
        0.00006995724062,
        23.10934304144901,
        -0.00023663490511,
        -0.6321817810242,
        -0.00663494467273,
        0.03430082397953,
        -0.00466043876332,
        2555164.4,
    ],
    [
        2.890871144776878e-9,
        0.000008983055095805407,
        -3.068298e-8,
        7.47137025468032,
        -0.00000353937994,
        -0.02145144861037,
        -0.00001234426596,
        0.00010322952773,
        -0.00000323890364,
        826088.5,
    ],
]


def out_of_china(lon: float, lat: float) -> bool:
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(lon: float, lat: float) -> float:
    x = lon - 105.0
    y = lat - 35.0
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320.0 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(lon: float, lat: float) -> float:
    x = lon - 105.0
    y = lat - 35.0
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon: float, lat: float) -> Point:
    lon = float(lon)
    lat = float(lat)
    if out_of_china(lon, lat):
        return lon, lat
    dlat = _transform_lat(lon, lat)
    dlon = _transform_lon(lon, lat)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon: float, lat: float) -> Point:
    lon = float(lon)
    lat = float(lat)
    if out_of_china(lon, lat):
        return lon, lat
    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    return lon * 2 - gcj_lon, lat * 2 - gcj_lat


def gcj02_to_bd09(lon: float, lat: float) -> Point:
    z = math.hypot(lon, lat) + 0.00002 * math.sin(lat * _X_PI)
    theta = math.atan2(lat, lon) + 0.000003 * math.cos(lon * _X_PI)
    return z * math.cos(theta) + 0.0065, z * math.sin(theta) + 0.006


def bd09_to_gcj02(lon: float, lat: float) -> Point:
    x = lon - 0.0065
    y = lat - 0.006
    z = math.hypot(x, y) - 0.00002 * math.sin(y * _X_PI)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * _X_PI)
    return z * math.cos(theta), z * math.sin(theta)


def wgs84_to_bd09(lon: float, lat: float) -> Point:
    gcj_lon, gcj_lat = wgs84_to_gcj02(lon, lat)
    return gcj02_to_bd09(gcj_lon, gcj_lat)


def bd09_to_wgs84(lon: float, lat: float) -> Point:
    gcj_lon, gcj_lat = bd09_to_gcj02(lon, lat)
    return gcj02_to_wgs84(gcj_lon, gcj_lat)


def bd09ll_to_bd09mc(lon: float, lat: float) -> Point:
    factor = LL2MC[-1]
    abs_lat = abs(lat)
    for idx, band in enumerate(LLBAND):
        if abs_lat >= band:
            factor = LL2MC[idx]
            break
    x = abs(lon)
    y = abs_lat / factor[9]
    xt = factor[0] + factor[1] * x
    yt = (
        factor[2]
        + factor[3] * y
        + factor[4] * y**2
        + factor[5] * y**3
        + factor[6] * y**4
        + factor[7] * y**5
        + factor[8] * y**6
    )
    return (-xt if lon < 0 else xt), (-yt if lat < 0 else yt)


def bd09mc_to_bd09ll(x: float, y: float) -> Point:
    factor = MC2LL[-1]
    abs_y = abs(y)
    for idx, band in enumerate(MCBAND):
        if abs_y >= band:
            factor = MC2LL[idx]
            break
    x_sign = 1 if x >= 0 else -1
    y_sign = 1 if y >= 0 else -1
    x = abs(x)
    y_norm = abs_y / factor[9]
    lon = factor[0] + factor[1] * x
    lat = (
        factor[2]
        + factor[3] * y_norm
        + factor[4] * y_norm**2
        + factor[5] * y_norm**3
        + factor[6] * y_norm**4
        + factor[7] * y_norm**5
        + factor[8] * y_norm**6
    )
    return x_sign * lon, y_sign * lat


def wgs84_to_bd09mc(lon: float, lat: float) -> Point:
    bd_lon, bd_lat = wgs84_to_bd09(lon, lat)
    return bd09ll_to_bd09mc(bd_lon, bd_lat)


def bd09mc_to_wgs84(x: float, y: float) -> Point:
    bd_lon, bd_lat = bd09mc_to_bd09ll(x, y)
    return bd09_to_wgs84(bd_lon, bd_lat)


def normalize_to_wgs84(lon: float, lat: float, coordinate_system: str = "wgs84") -> Point:
    coord = coordinate_system.lower().replace("-", "").replace("_", "")
    if coord in {"wgs84", "gps"}:
        return float(lon), float(lat)
    if coord in {"gcj02", "gcj02ll"}:
        return gcj02_to_wgs84(float(lon), float(lat))
    if coord in {"bd09", "bd09ll", "bd09latlon"}:
        return bd09_to_wgs84(float(lon), float(lat))
    if coord in {"bd09mc", "baidumc"}:
        return bd09mc_to_wgs84(float(lon), float(lat))
    raise ValueError(f"Unsupported coordinate_system: {coordinate_system}")


def canonical_coordinate_system(coordinate_system: str = "wgs84") -> str:
    coord = coordinate_system.lower().replace("-", "").replace("_", "")
    if coord in {"wgs84", "gps"}:
        return "WGS84"
    if coord in {"gcj02", "gcj02ll"}:
        return "GCJ-02"
    if coord in {"bd09", "bd09ll", "bd09latlon"}:
        return "BD-09"
    if coord in {"bd09mc", "baidumc"}:
        return "BD-09MC"
    return coordinate_system


def normalize_coordinate_fields(
    lat: object,
    lon: object,
    coordinate_system: str = "wgs84",
) -> dict[str, float | str | None]:
    """Return public WGS84 lat/lon fields plus raw provider coordinates."""

    raw_lat = _optional_float(lat)
    raw_lon = _optional_float(lon)
    fields: dict[str, float | str | None] = {
        "lat": None,
        "lon": None,
        "coordinate_system": "WGS84",
        "raw_lat": raw_lat,
        "raw_lon": raw_lon,
        "raw_coordinate_system": canonical_coordinate_system(coordinate_system),
    }
    if raw_lat is None or raw_lon is None:
        return fields
    wgs_lon, wgs_lat = normalize_to_wgs84(raw_lon, raw_lat, coordinate_system)
    fields["lat"] = wgs_lat
    fields["lon"] = wgs_lon
    return fields


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def wgs84_to_web_mercator(lon: float, lat: float) -> Point:
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    lon = float(lon)
    x = lon * _MERCATOR_MAX / 180.0
    y = math.log(math.tan((90.0 + lat) * _PI / 360.0)) / (_PI / 180.0)
    y = y * _MERCATOR_MAX / 180.0
    return x, y


def web_mercator_to_wgs84(x: float, y: float) -> Point:
    lon = x / _MERCATOR_MAX * 180.0
    lat = y / _MERCATOR_MAX * 180.0
    lat = 180.0 / _PI * (2.0 * math.atan(math.exp(lat * _PI / 180.0)) - _PI / 2.0)
    return lon, lat


def google_xyz_from_wgs84(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(float(lat), 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2.0**int(zoom)
    x = int((float(lon) + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / _PI) / 2.0 * n)
    max_tile = int(n - 1)
    return max(0, min(max_tile, x)), max(0, min(max_tile, y))


def google_tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2.0**int(zoom)
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(_PI * (1 - 2 * y / n))))
    lat_min = math.degrees(math.atan(math.sinh(_PI * (1 - 2 * (y + 1) / n))))
    return lon_min, lat_min, lon_max, lat_max


def baidu_tile_position_from_wgs84(lon: float, lat: float, zoom: int) -> tuple[int, int, float, float]:
    mc_x, mc_y = wgs84_to_bd09mc(lon, lat)
    resolution = 2 ** (18 - int(zoom))
    tile_span = 256 * resolution
    tile_x = int(math.floor(mc_x / tile_span))
    tile_y = int(math.floor(mc_y / tile_span))
    pixel_x = (mc_x / tile_span - tile_x) * 256
    pixel_y = (1 - (mc_y / tile_span - tile_y)) * 256
    return tile_x, tile_y, pixel_x, pixel_y


def baidu_tile_from_wgs84(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    tile_x, tile_y, _, _ = baidu_tile_position_from_wgs84(lon, lat, zoom)
    return tile_x, tile_y
