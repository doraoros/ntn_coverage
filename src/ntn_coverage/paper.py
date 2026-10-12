"""Export SINR maps, CDFs, availability, and 3GPP calibration figures."""

from __future__ import annotations

import json
from pathlib import Path

from ntn_coverage.coverage import predict_coverage
from ntn_coverage.figures import plot_calibration_png, plot_hero_png, plot_sinr_cdf
from ntn_coverage.plot import plot_compare_png, plot_coverage_png, plot_dashboard_png
from ntn_coverage.scenarios import get_scenario


def run_paper(
    region: str = "romania",
    resolution_deg: float = 0.2,
    samples: int = 8,
    out_dir: str | Path = "docs/figures",
    scenario_name: str = "leo600-s-handheld",
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = get_scenario(scenario_name)
    grids = {}
    summaries = {}
    for frf in (1, 3):
        scenario = base.with_reuse(frf)
        grid = predict_coverage(scenario, region=region, resolution_deg=resolution_deg)
        grids[frf] = grid
        summaries[f"frf{frf}"] = grid.summary()
        plot_coverage_png(grid, out / f"sinr_frf{frf}.png")
        plot_dashboard_png(grid, out / f"dashboard_frf{frf}.png")

    plot_compare_png(grids[1], grids[3], out / "sinr_frf1_vs_frf3.png")
    plot_sinr_cdf(
        {"FRF=1": grids[1], "FRF=3": grids[3]},
        out / "sinr_cdf.png",
    )
    compare = plot_hero_png(grids[1], grids[3], out / "reuse_compare.png")
    cal_png, cal = plot_calibration_png(out / "calibration_tr38821_sc6.png")

    avail = predict_coverage(
        base.with_reuse(3),
        region=region,
        resolution_deg=max(resolution_deg, 0.25),
        mode="availability",
        n_times=int(samples),
    )
    plot_coverage_png(avail, out / "availability_frf3.png", field="availability")
    plot_dashboard_png(avail, out / "availability_dashboard.png")
    summaries["availability"] = avail.summary()

    payload = {
        "region": region,
        "scenario": scenario_name,
        "resolution_deg": resolution_deg,
        "reuse_compare": Path(compare).as_posix(),
        "calibration_png": Path(cal_png).as_posix(),
        "calibration": cal,
        **summaries,
    }
    (out / "figures_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
