"""Geometry and raster utility helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from shapely.geometry import Polygon, box

try:
    from pyproj import Geod
except ModuleNotFoundError:
    Geod = None

try:
    from osgeo import osr
except ModuleNotFoundError:
    osr = None


WGS84_GEOD = Geod(ellps="WGS84") if Geod is not None else None


def km_to_deg_lat(km: float) -> float:
    return km / 110.574


def km_to_deg_lon(km: float, latitude_deg: float) -> float:
    cos_lat = max(1e-6, math.cos(math.radians(latitude_deg)))
    return km / (111.320 * cos_lat)


def make_grid(bounds: Iterable[float], tile_km: float) -> list[Polygon]:
    minx, miny, maxx, maxy = [float(v) for v in bounds]
    tiles: list[Polygon] = []
    lat = miny
    while lat < maxy:
        dlat = km_to_deg_lat(tile_km)
        lon = minx
        while lon < maxx:
            dlon = km_to_deg_lon(tile_km, lat + (dlat / 2.0))
            tiles.append(box(lon, lat, min(maxx, lon + dlon), min(maxy, lat + dlat)))
            lon += dlon
        lat += dlat
    return tiles


def pixel_size_m(gt: tuple[float, float, float, float, float, float], lat_deg: float) -> float:
    px_deg_x = abs(float(gt[1]))
    px_deg_y = abs(float(gt[5]))
    cos_lat = max(1e-6, math.cos(math.radians(lat_deg)))
    mx = px_deg_x * 111320.0 * cos_lat
    my = px_deg_y * 110540.0
    return float((mx + my) / 2.0)


def circle_overlap_ratio(x1: float, y1: float, r1: float, x2: float, y2: float, r2: float) -> float:
    d = math.hypot(x1 - x2, y1 - y2)
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        min_area = math.pi * min(r1, r2) ** 2
        max_area = math.pi * max(r1, r2) ** 2
        return float(min_area / max_area) if max_area > 0 else 0.0

    r1_sq = r1 * r1
    r2_sq = r2 * r2
    alpha = math.acos((d * d + r1_sq - r2_sq) / (2.0 * d * r1))
    beta = math.acos((d * d + r2_sq - r1_sq) / (2.0 * d * r2))
    area = (r1_sq * alpha + r2_sq * beta -
            0.5 * math.sqrt(
                max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
            ))
    union = math.pi * r1_sq + math.pi * r2_sq - area
    return float(area / union) if union > 0 else 0.0


def circle_polygon_wgs84(lon: float, lat: float, radius_m: float, n_points: int = 64) -> Polygon:
    if WGS84_GEOD is not None:
        points = []
        for azimuth in np.linspace(0.0, 360.0, num=n_points, endpoint=False):
            out_lon, out_lat, _ = WGS84_GEOD.fwd(lon, lat, azimuth, radius_m)
            if math.isfinite(out_lon) and math.isfinite(out_lat):
                points.append((float(out_lon), float(out_lat)))
        if len(points) >= 3:
            points.append(points[0])
            return Polygon(points)
        # Fallback if geodesic conversion produced invalid coordinates.
        radius_m = max(1.0, float(radius_m))

    # Fallback approximation when pyproj is unavailable.
    lat_scale = 1.0 / 110540.0
    lon_scale = 1.0 / max(1e-6, (111320.0 * math.cos(math.radians(lat))))
    points = []
    for theta in np.linspace(0.0, 2.0 * math.pi, num=n_points, endpoint=False):
        dx = radius_m * math.cos(theta)
        dy = radius_m * math.sin(theta)
        px = lon + dx * lon_scale
        py = lat + dy * lat_scale
        if math.isfinite(px) and math.isfinite(py):
            points.append((px, py))
    if len(points) < 3:
        # Last-resort small triangle to avoid total failure.
        d = 1e-5
        points = [(lon, lat), (lon + d, lat), (lon, lat + d)]
    points.append(points[0])
    return Polygon(points)


def raster_xy_to_lonlat(gt: tuple[float, float, float, float, float, float], x: float, y: float) -> tuple[float, float]:
    lon = gt[0] + (x * gt[1]) + (y * gt[2])
    lat = gt[3] + (x * gt[4]) + (y * gt[5])
    return float(lon), float(lat)


def _set_traditional_axis_order(srs) -> None:
    if srs is None:
        return
    if hasattr(srs, "SetAxisMappingStrategy") and hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)


def _normalize_lon_lat(x: float, y: float) -> tuple[float, float]:
    if abs(x) <= 90 and abs(y) > 90:
        return float(y), float(x)
    return float(x), float(y)


def raster_xy_to_wgs84(ds, x: float, y: float) -> tuple[float, float]:
    gt = ds.GetGeoTransform()
    mx = gt[0] + (x * gt[1]) + (y * gt[2])
    my = gt[3] + (x * gt[4]) + (y * gt[5])
    wkt = ds.GetProjectionRef()
    if osr is None or not wkt:
        return float(mx), float(my)

    src = osr.SpatialReference()
    src.ImportFromWkt(wkt)
    _set_traditional_axis_order(src)
    if bool(src.IsGeographic()):
        return _normalize_lon_lat(float(mx), float(my))

    dst = osr.SpatialReference()
    dst.ImportFromEPSG(4326)
    _set_traditional_axis_order(dst)

    ct = osr.CoordinateTransformation(src, dst)
    try:
        out = ct.TransformPoint(mx, my, 0.0)
        return _normalize_lon_lat(float(out[0]), float(out[1]))
    except Exception:
        return _normalize_lon_lat(float(mx), float(my))


def raster_pixel_size_m(ds) -> float:
    gt = ds.GetGeoTransform()
    # Pixel size magnitude in source units.
    px_x = math.hypot(float(gt[1]), float(gt[4]))
    px_y = math.hypot(float(gt[2]), float(gt[5]))
    px_units = (px_x + px_y) / 2.0
    if px_units <= 0:
        return 1.0

    wkt = ds.GetProjectionRef()
    if osr is None or not wkt:
        # Assume degrees if projection unknown; fall back to lat-aware conversion.
        center_lon, center_lat = raster_xy_to_lonlat(
            gt,
            x=0.5 * ds.RasterXSize,
            y=0.5 * ds.RasterYSize,
        )
        return pixel_size_m(gt, center_lat)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    _set_traditional_axis_order(srs)

    # Projected CRS: convert source linear units to meters.
    if bool(srs.IsProjected()):
        unit_m = float(srs.GetLinearUnits() or 1.0)
        return float(px_units * unit_m)

    # Geographic CRS: units are degrees.
    if bool(srs.IsGeographic()):
        center_lon, center_lat = raster_xy_to_wgs84(
            ds,
            x=0.5 * ds.RasterXSize,
            y=0.5 * ds.RasterYSize,
        )
        mx = px_units * 111320.0 * max(1e-6, math.cos(math.radians(center_lat)))
        my = px_units * 110540.0
        return float((mx + my) / 2.0)

    # Fallback.
    return float(px_units)
