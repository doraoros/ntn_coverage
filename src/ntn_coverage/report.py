"""Text link-budget dump and short pass timeline at a terminal."""

from __future__ import annotations

import numpy as np

from ntn_coverage.coverage import PointPrediction, predict_point
from ntn_coverage.scenarios import Scenario


def format_link_budget(pred: PointPrediction) -> str:
    flag = "OK" if pred.covered else "OUT"
    return (
        f"{pred.serving_sat:<8} {flag:<3}  "
        f"elev {pred.elevation_deg:5.1f} deg  "
        f"FSPL {pred.fspl_db:6.1f}  "
        f"C/N {pred.snr_db:6.2f}  "
        f"C/I {pred.cir_db:5.1f}  "
        f"SINR {pred.sinr_db:6.2f} dB  "
        f"{pred.rate_mbps:5.1f} Mbps"
    )


def format_pass_table(rows: list[dict]) -> str:
    lines = ["# t_s    sat      elev   fdop_Hz   SINR    Mbps"]
    for row in rows:
        lines.append(
            f"{row['t_s']:7.0f}  {str(row['serving_sat']):<8} "
            f"{row['elevation_deg']:6.1f}  {row['doppler_hz']:8.0f}  "
            f"{row['sinr_db']:6.2f}  {row['rate_mbps']:6.1f}"
        )
    return "\n".join(lines)


def pass_timeline(
    lat_deg: float,
    lon_deg: float,
    scenario: Scenario,
    t0_s: float = 0.0,
    duration_s: float = 1200.0,
    dt_s: float = 30.0,
    rain_rate_mm_h: float = 0.0,
    indoor: bool = False,
) -> list[dict]:
    times = np.arange(float(t0_s), float(t0_s) + float(duration_s) + 0.5 * dt_s, float(dt_s))
    rows = []
    for t in times:
        pred = predict_point(
            lat_deg, lon_deg, scenario, t_s=float(t), rain_rate_mm_h=rain_rate_mm_h, indoor=indoor
        )
        rows.append(
            {
                "t_s": float(t),
                "serving_sat": pred.serving_sat,
                "elevation_deg": pred.elevation_deg,
                "doppler_hz": pred.doppler_hz,
                "sinr_db": pred.sinr_db,
                "snr_db": pred.snr_db,
                "rate_mbps": pred.rate_mbps,
                "n_visible": pred.n_visible,
                "covered": pred.covered,
            }
        )
    return rows
