"""Circular-aperture Bessel antenna pattern from 3GPP TR 38.811 §6.4.1."""

from __future__ import annotations

import numpy as np
from scipy.special import j1

from ntn_coverage.constants import C_LIGHT


def peak_gain_dbi(diameter_m: float, freq_hz: float, efficiency: float = 0.65) -> np.ndarray:
    lam = C_LIGHT / float(freq_hz)
    g_lin = efficiency * (np.pi * float(diameter_m) / lam) ** 2
    return 10.0 * np.log10(np.clip(g_lin, 1e-30, None))


def bessel_relative_gain_db(off_axis_deg, diameter_m: float, freq_hz: float) -> np.ndarray:
    """Relative gain vs boresight, dB. Peak is 0 dB.

    Pattern is [2 J1(u)/u]^2 with u = k a sin(θ), a = D/2, matching the
    3GPP 3 dB beamwidth of 4.4127° for a 2 m S-band aperture.
    """
    theta = np.deg2rad(np.asarray(off_axis_deg, dtype=float))
    lam = C_LIGHT / float(freq_hz)
    radius = 0.5 * float(diameter_m)
    k = 2.0 * np.pi / lam
    u = k * radius * np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        pattern = np.where(np.abs(u) < 1e-9, 1.0, (2.0 * j1(u) / u) ** 2)
    pattern = np.clip(pattern, 1e-12, None)
    return 10.0 * np.log10(pattern)


def eirp_dbw(eirp_peak_dbw, off_axis_deg, diameter_m: float, freq_hz: float, steered: bool) -> np.ndarray:
    """Peak EIRP, optionally reduced by the Bessel pattern for nadir pointing."""
    peak = np.asarray(eirp_peak_dbw, dtype=float)
    if steered:
        return np.broadcast_to(peak, np.asarray(off_axis_deg, dtype=float).shape).copy()
    return peak + bessel_relative_gain_db(off_axis_deg, diameter_m, freq_hz)
