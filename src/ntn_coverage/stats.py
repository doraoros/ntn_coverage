"""Empirical CDFs and 3GPP TR 38.821 link-budget calibration."""

from __future__ import annotations

import numpy as np

from ntn_coverage.channel import path_loss_components
from ntn_coverage.geometry import slant_range_from_elevation
from ntn_coverage.link_budget import VSAT, snr_from_eirp_density


def ecdf(values) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float).ravel())
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1, dtype=float) / x.size
    return x, y


def tr38821_sc6_ka_leo600() -> dict[str, float]:
    """Clear-sky LOS calibration vs TR 38.821 Table 6.1.3.3-1 case SC6 (~8.5 dB)."""
    d = float(slant_range_from_elevation(30.0, 600_000.0))
    losses = path_loss_components(d, 30.0, 20e9, "rural", latitude_deg=45.0)
    pl = float(losses["fspl_db"] + losses["gases_db"] + losses["scintillation_db"])
    snr = float(snr_from_eirp_density(4.0, pl, VSAT.g_over_t_db_k, extra_loss_db=0.0))
    return {
        "elevation_deg": 30.0,
        "altitude_km": 600.0,
        "freq_ghz": 20.0,
        "fspl_db": float(losses["fspl_db"]),
        "fspl_3gpp_db": 179.1,
        "snr_db": snr,
        "snr_3gpp_db": 8.5,
        "fspl_error_db": abs(float(losses["fspl_db"]) - 179.1),
        "snr_error_db": abs(snr - 8.5),
    }
