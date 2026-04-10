"""Lightweight point-in-region filter for GeoJSON masks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    inside = False
    n = len(ring)
    if n < 3:
        return False

    x1, y1 = float(ring[0][0]), float(ring[0][1])
    for i in range(1, n + 1):
        x2, y2 = float(ring[i % n][0]), float(ring[i % n][1])
        if (y1 > y) != (y2 > y):
            xin = ((x2 - x1) * (y - y1) / ((y2 - y1) + 1e-300)) + x1
            if x < xin:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def _point_in_polygon(x: float, y: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon:
        return False
    if not _point_in_ring(x, y, polygon[0]):
        return False
    for hole in polygon[1:]:
        if _point_in_ring(x, y, hole):
            return False
    return True


@dataclass(frozen=True)
class RegionMask:
    """GeoJSON MultiPolygon/Polygon mask with bbox acceleration."""

    polygons: list[tuple[float, float, float, float, list[list[list[float]]]]]

    @property
    def polygon_count(self) -> int:
        return len(self.polygons)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if not self.polygons:
            return (0.0, 0.0, 0.0, 0.0)
        minx = min(p[0] for p in self.polygons)
        miny = min(p[1] for p in self.polygons)
        maxx = max(p[2] for p in self.polygons)
        maxy = max(p[3] for p in self.polygons)
        return (float(minx), float(miny), float(maxx), float(maxy))

    def contains(self, lon: float, lat: float) -> bool:
        x = float(lon)
        y = float(lat)
        for minx, miny, maxx, maxy, poly in self.polygons:
            if x < minx or x > maxx or y < miny or y > maxy:
                continue
            if _point_in_polygon(x, y, poly):
                return True
        return False

    def contains_many(self, coords: Iterable[tuple[float, float]]) -> list[bool]:
        return [self.contains(lon, lat) for lon, lat in coords]

    @classmethod
    def from_geojson(cls, path: str | Path) -> "RegionMask":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Region mask not found: {p}")

        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        features = payload.get("features", [])
        polygons: list[list[list[list[float]]]] = []
        for feat in features:
            geom = feat.get("geometry") or {}
            gtype = str(geom.get("type") or "")
            coords = geom.get("coordinates")
            if not coords:
                continue
            if gtype == "Polygon":
                polygons.append(coords)
            elif gtype == "MultiPolygon":
                polygons.extend(coords)

        packed: list[tuple[float, float, float, float, list[list[list[float]]]]] = []
        for poly in polygons:
            if not poly or not poly[0]:
                continue
            outer = poly[0]
            xs = [float(pt[0]) for pt in outer]
            ys = [float(pt[1]) for pt in outer]
            packed.append((min(xs), min(ys), max(xs), max(ys), poly))

        if not packed:
            raise RuntimeError(f"No Polygon/MultiPolygon geometries found in {p}")

        return cls(polygons=packed)
