"""Circular Keplerian Walker and GEO constellations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ntn_coverage.constants import EARTH_RADIUS_M, GEO_ALTITUDE_M, MU_EARTH
from ntn_coverage.geometry import ecef_from_ll, eci_to_ecef


@dataclass(frozen=True)
class ConstellationConfig:
    kind: str  # "walker" or "geo"
    altitude_m: float
    inclination_deg: float = 53.0
    planes: int = 18
    per_plane: int = 20
    phasing: int = 1
    geo_longitudes_deg: tuple[float, ...] = ()
    raan0_deg: float = 0.0
    arglat0_deg: float = 0.0
    tle_source: str = ""

    @property
    def n_satellites(self) -> int:
        if self.kind == "geo":
            return len(self.geo_longitudes_deg)
        if self.kind == "tle":
            return 0
        return self.planes * self.per_plane

    @property
    def semi_major_m(self) -> float:
        return EARTH_RADIUS_M + self.altitude_m

    @property
    def mean_motion_rad_s(self) -> float:
        a = self.semi_major_m
        return float(np.sqrt(MU_EARTH / a**3))

    @property
    def period_s(self) -> float:
        return 2.0 * np.pi / self.mean_motion_rad_s


@dataclass
class ConstellationState:
    config: ConstellationConfig
    t_s: float
    ecef: np.ndarray
    velocity_ecef: np.ndarray
    sat_ids: np.ndarray = field(repr=False)

    @property
    def n_satellites(self) -> int:
        return int(self.ecef.shape[0])


def circular_rv(a: float, inclination_rad, raan_rad, arglat_rad) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position/velocity for a circular Keplerian orbit."""
    inc = np.asarray(inclination_rad, dtype=float)
    raan = np.asarray(raan_rad, dtype=float)
    u = np.asarray(arglat_rad, dtype=float)
    n = np.sqrt(MU_EARTH / a**3)
    cu, su = np.cos(u), np.sin(u)
    ci, si = np.cos(inc), np.sin(inc)
    co, so = np.cos(raan), np.sin(raan)
    x = a * (co * cu - so * su * ci)
    y = a * (so * cu + co * su * ci)
    z = a * (su * si)
    vx = n * a * (-co * su - so * cu * ci)
    vy = n * a * (-so * su + co * cu * ci)
    vz = n * a * (cu * si)
    r = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    v = np.stack(np.broadcast_arrays(vx, vy, vz), axis=-1)
    return r, v


def propagate(config: ConstellationConfig, t_s: float = 0.0) -> ConstellationState:
    """Satellite ECEF positions and velocities at time t_s (seconds from epoch)."""
    if config.kind == "tle":
        from ntn_coverage.tle import propagate_tle

        return propagate_tle(config.tle_source, t_s)

    if config.kind == "geo":
        longs = np.asarray(config.geo_longitudes_deg, dtype=float)
        if longs.size == 0:
            raise ValueError("GEO constellation needs geo_longitudes_deg")
        ecef = ecef_from_ll(np.zeros_like(longs), longs, alt_m=config.altitude_m)
        vel = np.zeros_like(ecef)
        ids = np.array([f"GEO-{lon:.1f}E" for lon in longs])
        return ConstellationState(config=config, t_s=t_s, ecef=ecef, velocity_ecef=vel, sat_ids=ids)

    if config.kind != "walker":
        raise ValueError(f"Unknown constellation kind: {config.kind}")

    planes = int(config.planes)
    per_plane = int(config.per_plane)
    f = int(config.phasing)
    total = planes * per_plane
    a = config.semi_major_m
    n = config.mean_motion_rad_s
    p = np.repeat(np.arange(planes), per_plane)
    k = np.tile(np.arange(per_plane), planes)
    raan = np.deg2rad(config.raan0_deg) + 2.0 * np.pi * p / planes
    u0 = (
        np.deg2rad(config.arglat0_deg)
        + 2.0 * np.pi * k / per_plane
        + 2.0 * np.pi * f * p / total
    )
    u = u0 + n * float(t_s)
    r_eci, v_eci = circular_rv(a, np.deg2rad(config.inclination_deg), raan, u)
    ecef, vel = eci_to_ecef(r_eci, v_eci, t_s)
    ids = np.array([f"P{int(pp):02d}S{int(kk):02d}" for pp, kk in zip(p, k, strict=False)])
    return ConstellationState(config=config, t_s=t_s, ecef=ecef, velocity_ecef=vel, sat_ids=ids)


def leo600_walker(planes: int = 18, per_plane: int = 20, inclination_deg: float = 53.0) -> ConstellationConfig:
    return ConstellationConfig(
        kind="walker",
        altitude_m=600_000.0,
        inclination_deg=inclination_deg,
        planes=planes,
        per_plane=per_plane,
        phasing=1,
    )


def leo1200_walker(planes: int = 18, per_plane: int = 20, inclination_deg: float = 53.0) -> ConstellationConfig:
    return ConstellationConfig(
        kind="walker",
        altitude_m=1_200_000.0,
        inclination_deg=inclination_deg,
        planes=planes,
        per_plane=per_plane,
        phasing=1,
    )


def geo_europe() -> ConstellationConfig:
    return ConstellationConfig(
        kind="geo",
        altitude_m=GEO_ALTITUDE_M,
        inclination_deg=0.0,
        geo_longitudes_deg=(19.2, 25.0, 42.0),
    )


def starlink_shell1() -> ConstellationConfig:
    """Starlink-like Walker geometry (not operator RF). Shell 1: 72 × 22 at 550 km, 53°."""
    return ConstellationConfig(
        kind="walker",
        altitude_m=550_000.0,
        inclination_deg=53.0,
        planes=72,
        per_plane=22,
        phasing=17,
    )
