"""Downlink CNR / SNR from 3GPP TR 38.821 link-budget equations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ntn_coverage.constants import K_BOLTZMANN_DBW_PER_K_HZ, T0_K


@dataclass(frozen=True)
class Terminal:
    name: str
    gain_rx_dbi: float
    noise_figure_db: float
    antenna_temp_k: float
    shadowing_margin_db: float
    tx_power_dbm: float = 23.0
    gain_tx_dbi: float = 0.0

    @property
    def g_over_t_db_k(self) -> float:
        t_rx = T0_K * (10.0 ** (self.noise_figure_db / 10.0) - 1.0)
        t_sys = self.antenna_temp_k + t_rx
        return self.gain_rx_dbi - 10.0 * np.log10(t_sys)


HANDHELD = Terminal(
    name="handheld",
    gain_rx_dbi=0.0,
    noise_figure_db=7.0,
    antenna_temp_k=290.0,
    shadowing_margin_db=3.0,
    tx_power_dbm=23.0,
    gain_tx_dbi=0.0,
)

VSAT = Terminal(
    name="vsat",
    gain_rx_dbi=39.7,
    noise_figure_db=1.2,
    antenna_temp_k=150.0,
    shadowing_margin_db=0.0,
    tx_power_dbm=33.0,
    gain_tx_dbi=43.2,
)


def snr_from_eirp_density(
    eirp_density_dbw_per_mhz,
    path_loss_db,
    g_over_t_db_k,
    extra_loss_db=0.0,
) -> np.ndarray:
    """Downlink SNR (dB) for a white EIRP density specified in dBW/MHz.

    With constant PSD, signal and noise both scale with bandwidth, so SNR is
    independent of B. Equivalent to TR 38.821 (6.1.3.1-2) after substituting
    EIRP = density + 10 log10(B / 1 MHz).
    """
    return (
        np.asarray(eirp_density_dbw_per_mhz, dtype=float)
        - 60.0
        - np.asarray(path_loss_db, dtype=float)
        - np.asarray(extra_loss_db, dtype=float)
        + float(g_over_t_db_k)
        - K_BOLTZMANN_DBW_PER_K_HZ
    )


def cn0_dbhz(snr, bandwidth_hz) -> np.ndarray:
    return np.asarray(snr, dtype=float) + 10.0 * np.log10(np.clip(float(bandwidth_hz), 1.0, None))


def shannon_rate_bps(snr_db_value, bandwidth_hz, overhead: float = 0.2) -> np.ndarray:
    snr_lin = 10.0 ** (np.asarray(snr_db_value, dtype=float) / 10.0)
    se = np.minimum(np.log2(1.0 + np.clip(snr_lin, 0.0, None)), 8.0)
    return se * float(bandwidth_hz) * (1.0 - overhead)
