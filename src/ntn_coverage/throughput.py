"""NR-like MCS mapping from SINR to spectral efficiency / throughput."""

from __future__ import annotations

import numpy as np

# TS 38.214 CQI spectral efficiencies (Table 5.2.2.1-2 up to 64-QAM, top two
# from Table 5.2.2.1-3) at approximate AWGN SINR switching points.
_SINR_DB = np.array(
    [-6.9, -5.1, -3.2, -1.2, 0.7, 2.6, 4.7, 6.5, 8.4, 10.4, 12.2, 14.1, 15.9, 17.8, 19.5, 21.5]
)
_SE_BPS_HZ = np.array(
    [
        0.2344,
        0.3770,
        0.6016,
        0.8770,
        1.1758,
        1.4766,
        1.9141,
        2.4063,
        2.7305,
        3.3223,
        3.9023,
        4.5234,
        5.1152,
        5.5547,
        6.2266,
        6.9141,
    ]
)


def spectral_efficiency(sinr_db, overhead: float = 0.14) -> np.ndarray:
    """b/s/Hz after a simple DMRS/PDCCH overhead factor."""
    x = np.asarray(sinr_db, dtype=float)
    se = np.interp(x, _SINR_DB, _SE_BPS_HZ, left=0.0, right=float(_SE_BPS_HZ[-1]))
    se = np.where(np.isfinite(x), se, 0.0)
    return se * (1.0 - overhead)


def throughput_bps(sinr_db, bandwidth_hz: float, overhead: float = 0.14) -> np.ndarray:
    return spectral_efficiency(sinr_db, overhead=overhead) * float(bandwidth_hz)
