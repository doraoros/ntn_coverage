"""Ground traces of the serving 19-beam hex cluster."""

from __future__ import annotations

import numpy as np

from ntn_coverage.beams import adjacent_beam_spacing, axial_to_uv, beam_color, hex_disk, hex_round, uv_to_axial_frac
from ntn_coverage.geometry import ecef_from_ll, intersect_wgs84, ll_from_ecef, look_uvw, satellite_uv_frame
from ntn_coverage.orbit import propagate
from ntn_coverage.scenarios import Scenario


def serving_axial(ue_ecef, sat_ecef, abs_spacing: float) -> tuple[int, int]:
    u, v, _w = look_uvw(ue_ecef, sat_ecef)
    q, r = hex_round(*uv_to_axial_frac(float(u), float(v), abs_spacing))
    return int(np.asarray(q)), int(np.asarray(r))


def _hex_ring_uv(u0: float, v0: float, abs_spacing: float) -> tuple[np.ndarray, np.ndarray]:
    # Flat-top hex, centre-to-centre = ABS (TR 38.821 UV lattice).
    radius = abs_spacing / np.sqrt(3.0)
    ang = np.deg2rad(30.0 + 60.0 * np.arange(7))
    return u0 + radius * np.cos(ang), v0 + radius * np.sin(ang)


def _uv_to_ll(sat_ecef, u, v) -> tuple[list, list]:
    ex, ey, ez = satellite_uv_frame(sat_ecef)
    rho2 = np.clip(u * u + v * v, 0.0, 0.999)
    w = np.sqrt(1.0 - rho2)
    lats: list[float | None] = []
    lons: list[float | None] = []
    for i in range(u.size):
        direction = u[i] * ex + v[i] * ey + w[i] * ez
        hit = intersect_wgs84(sat_ecef, direction)
        if hit is None:
            lats.append(None)
            lons.append(None)
            continue
        lat, lon, _alt = ll_from_ecef(hit)
        lats.append(float(np.asarray(lat)))
        lons.append(float(np.asarray(lon)))
    return lats, lons


def serving_beam_cluster(
    lat_deg: float,
    lon_deg: float,
    scenario: Scenario,
    t_s: float = 0.0,
    serving_index: int | None = None,
    serving_qr: tuple[int, int] | None = None,
) -> list[dict]:
    """19 hex footprints centred on the serving beam (rings=2)."""
    state = propagate(scenario.constellation, t_s)
    if state.n_satellites == 0:
        return []
    if serving_index is None or serving_index < 0:
        return []
    sat = state.ecef[int(serving_index)]
    abs_s = adjacent_beam_spacing(scenario.rf.hpbw_deg)
    if serving_qr is not None:
        q0, r0 = (int(v) for v in serving_qr)
    else:
        q0, r0 = serving_axial(ecef_from_ll(lat_deg, lon_deg), sat, abs_s)
    dq, dr, _u, _v = hex_disk(2, abs_s)
    q = dq + q0
    r = dr + r0
    u, v = axial_to_uv(q, r, abs_s)
    colors = beam_color(q, r, scenario.reuse_factor)
    out = []
    for i in range(q.size):
        uu, vv = _hex_ring_uv(float(u[i]), float(v[i]), abs_s)
        lat, lon = _uv_to_ll(sat, uu, vv)
        if any(p is None for p in lat):
            continue
        out.append(
            {
                "q": int(q[i]),
                "r": int(r[i]),
                "color": int(colors[i]),
                "serving": int(q[i]) == q0 and int(r[i]) == r0,
                "lat": lat,
                "lon": lon,
            }
        )
    return out
