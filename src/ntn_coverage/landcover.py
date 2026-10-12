"""Metro circles for mixed-environment maps, plus a Romania outline for figures."""

from __future__ import annotations

import numpy as np

# lat, lon, radius_deg — metro footprints used when environment="mixed"
ROMANIA_URBAN = (
    (44.4268, 26.1025, 0.28, "Bucharest"),
    (46.7712, 23.6236, 0.16, "Cluj-Napoca"),
    (45.7489, 21.2087, 0.16, "Timisoara"),
    (47.1585, 27.6014, 0.16, "Iasi"),
    (44.1598, 28.6348, 0.14, "Constanta"),
    (44.3302, 23.7949, 0.14, "Craiova"),
    (45.6427, 25.5887, 0.14, "Brasov"),
    (45.4353, 28.0080, 0.12, "Galati"),
    (45.7537, 21.2257, 0.10, "Arad"),
    (47.0465, 21.9189, 0.12, "Oradea"),
    (46.5439, 24.5579, 0.12, "Targu Mures"),
    (45.3000, 24.3000, 0.10, "Pitesti"),
    (44.9250, 26.0360, 0.12, "Ploiesti"),
    (47.6630, 26.2730, 0.12, "Suceava"),
)

# Simplified outer ring (lon, lat), visually recognizable at country scale.
ROMANIA_OUTLINE = np.array(
    [
        [20.26, 45.47],
        [20.74, 46.12],
        [21.33, 46.52],
        [22.11, 47.71],
        [23.01, 47.95],
        [23.54, 47.58],
        [24.45, 47.64],
        [25.19, 47.89],
        [26.12, 48.12],
        [26.62, 48.26],
        [27.23, 47.82],
        [28.11, 46.97],
        [28.21, 46.37],
        [28.96, 46.55],
        [29.63, 46.52],
        [29.84, 46.32],
        [29.69, 45.73],
        [29.15, 45.29],
        [28.68, 45.22],
        [28.55, 44.23],
        [28.21, 43.71],
        [27.52, 44.04],
        [26.63, 43.74],
        [25.99, 43.89],
        [25.45, 43.74],
        [24.38, 43.69],
        [23.90, 43.85],
        [23.23, 44.08],
        [22.71, 44.23],
        [22.41, 44.56],
        [21.67, 44.42],
        [20.66, 44.84],
        [20.26, 45.47],
    ]
)


# lat, lon, radius_deg. Capitals and large metros outside the Romania list,
# so mixed mode is not rural everywhere beyond the Romanian cities.
WORLD_URBAN = (
    (51.51, -0.13, 0.45, "London"),
    (48.86, 2.35, 0.40, "Paris"),
    (52.52, 13.41, 0.35, "Berlin"),
    (40.42, -3.70, 0.35, "Madrid"),
    (41.90, 12.50, 0.30, "Rome"),
    (45.46, 9.19, 0.28, "Milan"),
    (41.39, 2.17, 0.25, "Barcelona"),
    (52.37, 4.90, 0.22, "Amsterdam"),
    (50.85, 4.35, 0.20, "Brussels"),
    (48.21, 16.37, 0.25, "Vienna"),
    (52.23, 21.01, 0.28, "Warsaw"),
    (50.08, 14.44, 0.22, "Prague"),
    (47.50, 19.04, 0.25, "Budapest"),
    (37.98, 23.73, 0.28, "Athens"),
    (59.33, 18.07, 0.25, "Stockholm"),
    (59.91, 10.75, 0.20, "Oslo"),
    (55.68, 12.57, 0.22, "Copenhagen"),
    (60.17, 24.94, 0.22, "Helsinki"),
    (53.35, -6.26, 0.22, "Dublin"),
    (38.72, -9.14, 0.22, "Lisbon"),
    (48.14, 11.58, 0.22, "Munich"),
    (53.55, 10.00, 0.22, "Hamburg"),
    (47.38, 8.54, 0.18, "Zurich"),
    (44.82, 20.46, 0.22, "Belgrade"),
    (42.70, 23.32, 0.22, "Sofia"),
    (50.45, 30.52, 0.30, "Kyiv"),
    (41.01, 28.98, 0.45, "Istanbul"),
    (55.76, 37.62, 0.45, "Moscow"),
    (59.93, 30.34, 0.30, "Saint Petersburg"),
    (35.68, 139.69, 0.50, "Tokyo"),
    (34.69, 135.50, 0.35, "Osaka"),
    (37.57, 126.98, 0.35, "Seoul"),
    (39.90, 116.41, 0.45, "Beijing"),
    (31.23, 121.47, 0.40, "Shanghai"),
    (23.13, 113.26, 0.35, "Guangzhou"),
    (22.54, 114.06, 0.22, "Hong Kong"),
    (25.03, 121.57, 0.25, "Taipei"),
    (1.35, 103.82, 0.18, "Singapore"),
    (13.76, 100.50, 0.35, "Bangkok"),
    (-6.21, 106.85, 0.40, "Jakarta"),
    (14.60, 120.98, 0.35, "Manila"),
    (28.61, 77.21, 0.45, "Delhi"),
    (19.08, 72.88, 0.40, "Mumbai"),
    (22.57, 88.36, 0.35, "Kolkata"),
    (12.97, 77.59, 0.30, "Bengaluru"),
    (24.86, 67.01, 0.35, "Karachi"),
    (23.81, 90.41, 0.30, "Dhaka"),
    (35.69, 51.39, 0.35, "Tehran"),
    (24.71, 46.68, 0.30, "Riyadh"),
    (25.20, 55.27, 0.22, "Dubai"),
    (32.09, 34.78, 0.18, "Tel Aviv"),
    (30.04, 31.24, 0.40, "Cairo"),
    (6.52, 3.38, 0.40, "Lagos"),
    (-1.29, 36.82, 0.25, "Nairobi"),
    (-26.20, 28.05, 0.30, "Johannesburg"),
    (33.57, -7.59, 0.25, "Casablanca"),
    (36.75, 3.06, 0.25, "Algiers"),
    (9.03, 38.74, 0.25, "Addis Ababa"),
    (40.71, -74.01, 0.40, "New York"),
    (34.05, -118.24, 0.45, "Los Angeles"),
    (41.88, -87.63, 0.35, "Chicago"),
    (25.76, -80.19, 0.28, "Miami"),
    (32.78, -96.80, 0.30, "Dallas"),
    (33.75, -84.39, 0.30, "Atlanta"),
    (43.65, -79.38, 0.30, "Toronto"),
    (45.50, -73.57, 0.25, "Montreal"),
    (49.28, -123.12, 0.22, "Vancouver"),
    (19.43, -99.13, 0.50, "Mexico City"),
    (-23.55, -46.63, 0.50, "Sao Paulo"),
    (-22.91, -43.17, 0.35, "Rio de Janeiro"),
    (-34.60, -58.38, 0.40, "Buenos Aires"),
    (-12.05, -77.04, 0.35, "Lima"),
    (4.71, -74.07, 0.35, "Bogota"),
    (-33.45, -70.67, 0.30, "Santiago"),
    (-33.87, 151.21, 0.35, "Sydney"),
    (-37.81, 144.96, 0.30, "Melbourne"),
    (-36.85, 174.76, 0.25, "Auckland"),
)


def urban_mask(lat_deg, lon_deg) -> np.ndarray:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    mask = np.zeros(np.broadcast(lat, lon).shape, dtype=bool)
    # Equirectangular-ish distance: 1° lat = 111 km, lon scaled by cos(lat)
    for city_lat, city_lon, radius, _name in (*ROMANIA_URBAN, *WORLD_URBAN):
        dlat = lat - city_lat
        dlon = (lon - city_lon) * np.cos(np.deg2rad(city_lat))
        mask |= (dlat * dlat + dlon * dlon) <= radius * radius
    return mask
