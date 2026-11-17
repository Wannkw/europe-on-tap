"""Minimal map drawing without GeoPandas: GeoJSON polygons in a Lambert
azimuthal equal-area projection centred on Europe (as in EPSG:3035)."""

from __future__ import annotations

import numpy as np
from matplotlib.collections import PolyCollection

R_KM = 6371.0
LAT0, LON0 = np.radians(52.0), np.radians(10.0)
EXTENT_LONLAT = (-24.0, 45.0, 34.0, 71.0)


def laea(lon, lat):
    """Spherical Lambert azimuthal equal-area projection, output in km."""
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    k = np.sqrt(2 / (1 + np.sin(LAT0) * np.sin(phi) + np.cos(LAT0) * np.cos(phi) * np.cos(lam - LON0)))
    x = R_KM * k * np.cos(phi) * np.sin(lam - LON0)
    y = R_KM * k * (np.cos(LAT0) * np.sin(phi) - np.sin(LAT0) * np.cos(phi) * np.cos(lam - LON0))
    return x, y


def _rings(geometry: dict):
    if geometry["type"] == "Polygon":
        yield geometry["coordinates"][0]
    elif geometry["type"] == "MultiPolygon":
        for poly in geometry["coordinates"]:
            yield poly[0]


def country_shapes(geojson: dict) -> dict[str, list[np.ndarray]]:
    """ISO3 -> list of projected exterior rings. Uses ADM0_A3 because Natural Earth
    sets ISO_A3 to -99 for France, Norway and Kosovo."""
    shapes: dict[str, list[np.ndarray]] = {}
    for feature in geojson["features"]:
        code = feature["properties"]["ADM0_A3"]
        for ring in _rings(feature["geometry"]):
            ring = np.asarray(ring)
            # Skip far-away territories (French Guiana, Svalbard is kept).
            if not (EXTENT_LONLAT[0] - 10 < ring[:, 0].mean() < EXTENT_LONLAT[1] + 40
                    and EXTENT_LONLAT[2] - 5 < ring[:, 1].mean() < 85):
                continue
            x, y = laea(ring[:, 0], ring[:, 1])
            shapes.setdefault(code, []).append(np.column_stack([x, y]))
    return shapes


def set_europe_view(ax) -> None:
    lon = np.linspace(EXTENT_LONLAT[0], EXTENT_LONLAT[1], 50)
    lat = np.linspace(EXTENT_LONLAT[2], EXTENT_LONLAT[3], 50)
    edge_lon = np.concatenate([lon, lon, np.full(50, lon[0]), np.full(50, lon[-1])])
    edge_lat = np.concatenate([np.full(50, lat[0]), np.full(50, lat[-1]), lat, lat])
    x, y = laea(edge_lon, edge_lat)
    ax.set_xlim(np.percentile(x, 3), np.percentile(x, 97))
    ax.set_ylim(y.min(), np.percentile(y, 90))
    ax.set_aspect("equal")
    ax.set_axis_off()  # lets bbox_inches="tight" crop to the map itself


def draw_countries(ax, shapes, values: dict[str, float] | None = None, cmap=None, norm=None,
                   base_color="#e5e7eb", edge="white", linewidth=0.4):
    """Fill countries by ``values`` (ISO3 -> number); others get ``base_color``."""
    polys, colors = [], []
    for code, rings in shapes.items():
        value = None if values is None else values.get(code)
        color = base_color if value is None or np.isnan(value) else cmap(norm(value))
        for ring in rings:
            polys.append(ring)
            colors.append(color)
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=edge, linewidths=linewidth))
