"""Hexagonal multi-beam layout and FRF colouring from 3GPP TR 38.821."""

from __future__ import annotations

import numpy as np

from ntn_coverage.antennas import bessel_relative_gain_db


def adjacent_beam_spacing(hpbw_deg: float) -> float:
    """UV-plane ABS = sqrt(3) * sin(HPBW/2), Table 6.1.1.1-4."""
    return float(np.sqrt(3.0) * np.sin(np.deg2rad(float(hpbw_deg)) / 2.0))


def axial_to_uv(q, r, abs_spacing: float) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    u = abs_spacing * (q + 0.5 * r)
    v = abs_spacing * (np.sqrt(3.0) / 2.0 * r)
    return u, v


def uv_to_axial_frac(u, v, abs_spacing: float) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    r = (2.0 / np.sqrt(3.0)) * v / abs_spacing
    q = u / abs_spacing - 0.5 * r
    return q, r


def hex_round(q, r) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    s = -q - r
    qn, rn, sn = np.round(q), np.round(r), np.round(s)
    dq, dr, ds = np.abs(qn - q), np.abs(rn - r), np.abs(sn - s)
    qn = np.where((dq > dr) & (dq > ds), -rn - sn, qn)
    rn = np.where(~((dq > dr) & (dq > ds)) & (dr > ds), -qn - sn, rn)
    return qn.astype(int), rn.astype(int)


def beam_color(q, r, reuse_factor: int) -> np.ndarray:
    q = np.asarray(q, dtype=int)
    r = np.asarray(r, dtype=int)
    if int(reuse_factor) <= 1:
        return np.zeros(np.broadcast(q, r).shape, dtype=int)
    return np.mod(q - r, int(reuse_factor))


def hex_disk(rings: int, abs_spacing: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Axial hex disk. rings=2 → 19 beams (TR 38.821 SLS layout)."""
    qs: list[int] = []
    rs: list[int] = []
    for q in range(-rings, rings + 1):
        for r in range(-rings, rings + 1):
            if max(abs(q), abs(r), abs(-q - r)) <= rings:
                qs.append(q)
                rs.append(r)
    q = np.asarray(qs, dtype=int)
    r = np.asarray(rs, dtype=int)
    u, v = axial_to_uv(q, r, abs_spacing)
    return q, r, u, v


def boresight_uvw(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    rho2 = u * u + v * v
    w = np.sqrt(np.clip(1.0 - rho2, 0.0, None))
    vec = np.stack((u, v, w), axis=-1)
    nrm = np.clip(np.linalg.norm(vec, axis=-1, keepdims=True), 1e-12, None)
    return vec / nrm


def beam_relative_gain_db(
    look_uvw: np.ndarray,
    boresight: np.ndarray,
    diameter_m: float,
    freq_hz: float,
) -> np.ndarray:
    """look (n_ue, 3), boresight (n_beam, 3) or (n_ue, n_beam, 3) → gain dB (n_ue, n_beam)."""
    look = np.asarray(look_uvw, dtype=float).reshape(-1, 3)
    bs = np.asarray(boresight, dtype=float)
    if bs.ndim < 3:
        bs = bs.reshape(1, -1, 3)
    cos_t = np.clip(np.sum(look[:, None, :] * bs, axis=-1), -1.0, 1.0)
    off_deg = np.rad2deg(np.arccos(cos_t))
    return bessel_relative_gain_db(off_deg, diameter_m, freq_hz)
