"""WGS-84 geometry for NTN links, with TR 38.811 spherical slant-range helper."""

from __future__ import annotations

import numpy as np

from ntn_coverage.constants import C_LIGHT, EARTH_RADIUS_M, OMEGA_EARTH, WGS84_A_M, WGS84_E2


def ecef_from_ll(lat_deg, lon_deg, alt_m=0.0) -> np.ndarray:
    """Geodetic WGS-84 lat/lon to ECEF metres. Shape (..., 3)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    slat, clat = np.sin(lat), np.cos(lat)
    slon, clon = np.sin(lon), np.cos(lon)
    n_prime = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * slat * slat)
    x = (n_prime + alt) * clat * clon
    y = (n_prime + alt) * clat * slon
    z = (n_prime * (1.0 - WGS84_E2) + alt) * slat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)


def ll_from_ecef(ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ECEF metres to WGS-84 lat/lon (deg) and altitude (m)."""
    x, y, z = np.moveaxis(np.asarray(ecef, dtype=float), -1, 0)
    lon = np.arctan2(y, x)
    p = np.sqrt(x * x + y * y)
    b = WGS84_A_M * np.sqrt(1.0 - WGS84_E2)
    ep2 = WGS84_E2 / (1.0 - WGS84_E2)
    theta = np.arctan2(z * WGS84_A_M, p * b)
    st, ct = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + ep2 * b * st**3, p - WGS84_E2 * WGS84_A_M * ct**3)
    slat = np.sin(lat)
    n_prime = WGS84_A_M / np.sqrt(1.0 - WGS84_E2 * slat * slat)
    alt = p / np.clip(np.cos(lat), 1e-12, None) - n_prime
    return np.rad2deg(lat), np.rad2deg(lon), alt


def slant_range_from_elevation(elevation_deg, altitude_m: float) -> np.ndarray:
    """TR 38.811 eq. (6.6-3): spherical-Earth slant range."""
    alpha = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    h = float(altitude_m)
    re = EARTH_RADIUS_M
    return np.sqrt((re * np.sin(alpha)) ** 2 + h * h + 2.0 * re * h) - re * np.sin(alpha)


def gmst_rad(jd: float) -> float:
    """Greenwich mean sidereal time (IAU 1982) in radians for a UT1 Julian date.

    Polar motion and the equation of the equinoxes are neglected, which is the
    usual TEME-to-ECEF approximation for TLE states.
    """
    t = (float(jd) - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t
        + 0.093104 * t * t
        - 6.2e-6 * t * t * t
    )
    return (seconds % 86400.0) * (2.0 * np.pi / 86400.0)


def eci_to_ecef(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    t_s: float,
    theta_rad: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate an inertial state into ECEF and express velocity in the rotating frame.

    Walker propagation passes seconds from its own epoch and the angle is
    ``ω t``. TLE propagation passes ``theta_rad`` equal to GMST at the
    absolute element time, because a TLE epoch is not GMST zero.
    """
    theta = OMEGA_EARTH * float(t_s) if theta_rad is None else float(theta_rad)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = np.moveaxis(np.asarray(r_eci, dtype=float), -1, 0)
    vx, vy, vz = np.moveaxis(np.asarray(v_eci, dtype=float), -1, 0)
    r_x = c * x + s * y
    r_y = -s * x + c * y
    r_z = z
    v_x = c * vx + s * vy
    v_y = -s * vx + c * vy
    v_z = vz
    r_ecef = np.stack((r_x, r_y, r_z), axis=-1)
    v_inertial_ecef = np.stack((v_x, v_y, v_z), axis=-1)
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    v_ecef = v_inertial_ecef - np.cross(omega, r_ecef)
    return r_ecef, v_ecef


def intersect_wgs84(origin_m, direction) -> np.ndarray | None:
    """First forward intersection of a ray with the WGS-84 ellipsoid, or None."""
    o = np.asarray(origin_m, dtype=float).reshape(3)
    d = np.asarray(direction, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(d))
    if nrm < 1e-12:
        return None
    d = d / nrm
    a2 = WGS84_A_M * WGS84_A_M
    b2 = a2 * (1.0 - WGS84_E2)
    inv = np.array([1.0 / a2, 1.0 / a2, 1.0 / b2])
    A = float(np.dot(d * d, inv))
    B = float(2.0 * np.dot(o * d, inv))
    C = float(np.dot(o * o, inv) - 1.0)
    disc = B * B - 4.0 * A * C
    if disc < 0.0 or abs(A) < 1e-30:
        return None
    root = np.sqrt(disc)
    hits = [t for t in ((-B - root) / (2.0 * A), (-B + root) / (2.0 * A)) if t > 1.0]
    if not hits:
        return None
    return o + min(hits) * d


def satellite_uv_frame(sat_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal satellite frame: e_z toward Earth, (e_x, e_y) = UV plane."""
    sat = np.asarray(sat_ecef, dtype=float).reshape(3)
    ez = -sat / np.clip(np.linalg.norm(sat), 1.0, None)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(ez, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    ex = np.cross(ref, ez)
    ex = ex / np.clip(np.linalg.norm(ex), 1e-12, None)
    ey = np.cross(ez, ex)
    return ex, ey, ez


def look_uvw(ue_ecef: np.ndarray, sat_ecef: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direction cosines of sat→UE in the satellite UV frame."""
    ex, ey, ez = satellite_uv_frame(sat_ecef)
    sat = np.asarray(sat_ecef, dtype=float).reshape(3)
    ue = np.asarray(ue_ecef, dtype=float)
    d = ue - sat
    d = d / np.clip(np.linalg.norm(d, axis=-1, keepdims=True), 1.0, None)
    return d @ ex, d @ ey, d @ ez


def link_geometry(
    ue_ecef: np.ndarray,
    sat_ecef: np.ndarray,
    sat_v_ecef: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Elevation, slant range, off-nadir angle, and optional Doppler geometry."""
    ue = np.asarray(ue_ecef, dtype=float)
    sat = np.asarray(sat_ecef, dtype=float).reshape(-1, 3)
    ue_b = ue[..., None, :]
    los = sat - ue_b
    dist = np.linalg.norm(los, axis=-1)
    dist_safe = np.clip(dist, 1.0, None)
    ue_norm = np.linalg.norm(ue_b, axis=-1, keepdims=True)
    up = ue_b / np.clip(ue_norm, 1.0, None)
    los_hat = los / dist_safe[..., None]
    sin_el = np.sum(los_hat * up, axis=-1)
    elevation_deg = np.rad2deg(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

    sat_hat = sat / np.clip(np.linalg.norm(sat, axis=-1, keepdims=True), 1.0, None)
    nadir = -sat_hat
    sat_to_ue = -los_hat
    cos_off = np.sum(sat_to_ue * nadir, axis=-1)
    off_nadir_deg = np.rad2deg(np.arccos(np.clip(cos_off, -1.0, 1.0)))

    out: dict[str, np.ndarray] = {
        "slant_range_m": dist,
        "elevation_deg": elevation_deg,
        "off_nadir_deg": off_nadir_deg,
        "los_hat": los_hat,
    }
    if sat_v_ecef is not None:
        # velocity_ecef is already relative to the rotating Earth, so a terminal
        # fixed on the ground has zero velocity in this frame.
        v_sat = np.asarray(sat_v_ecef, dtype=float).reshape(-1, 3)
        range_rate = np.sum(v_sat * los_hat, axis=-1)
        out["range_rate_mps"] = range_rate
        out["doppler_over_f"] = -range_rate / C_LIGHT
    return out
