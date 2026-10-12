"""Command-line interface for the NTN 6G coverage predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ntn_coverage.coverage import predict_coverage, predict_point
from ntn_coverage.export import write_grid_csv, write_grid_npz
from ntn_coverage.paper import run_paper
from ntn_coverage.plot import plot_compare_png, plot_coverage_html, plot_coverage_png, plot_dashboard_png
from ntn_coverage.report import format_link_budget, format_pass_table, pass_timeline
from ntn_coverage.scenarios import REGIONS, SCENARIOS, get_scenario


def _apply_overrides(scenario, args):
    if getattr(args, "environment", None):
        scenario = scenario.with_environment(args.environment)
    if getattr(args, "antenna", None):
        scenario = scenario.with_antenna(args.antenna)
    if getattr(args, "reuse", None) is not None:
        scenario = scenario.with_reuse(args.reuse)
    if getattr(args, "coordination", None):
        scenario = scenario.with_coordination(args.coordination)
    if getattr(args, "threshold", None) is not None:
        from dataclasses import replace

        scenario = replace(scenario, snr_threshold_db=args.threshold)
    if getattr(args, "tle", None):
        from dataclasses import replace as _replace

        from ntn_coverage.orbit import ConstellationConfig

        scenario = _replace(
            scenario,
            constellation=ConstellationConfig(kind="tle", altitude_m=550_000.0, tle_source=args.tle),
            title=f"{scenario.title} · TLE",
        )
    return scenario


def _cmd_point(args) -> int:
    scenario = _apply_overrides(get_scenario(args.scenario), args)
    pred = predict_point(
        args.lat,
        args.lon,
        scenario,
        t_s=args.t,
        rain_rate_mm_h=args.rain,
        indoor=args.indoor,
    )
    if args.json:
        payload = {
            "scenario": scenario.name,
            "lat": pred.lat_deg,
            "lon": pred.lon_deg,
            "covered": pred.covered,
            "snr_db": pred.snr_db,
            "sinr_db": pred.sinr_db,
            "cir_db": pred.cir_db,
            "p_cover": pred.p_cover,
            "cn0_dbhz": pred.cn0_dbhz,
            "elevation_deg": pred.elevation_deg,
            "slant_range_km": pred.slant_range_km,
            "path_loss_db": pred.path_loss_db,
            "fspl_db": pred.fspl_db,
            "doppler_hz": pred.doppler_hz,
            "n_visible": pred.n_visible,
            "serving_sat": pred.serving_sat,
            "rate_mbps": pred.rate_mbps,
            **pred.extras,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(format_link_budget(pred))
    return 0


def _write_maps(grid, out: Path, stem: str) -> dict:
    png = plot_coverage_png(grid, out / f"{stem}.png", field="sinr_db")
    dash = plot_dashboard_png(grid, out / f"{stem}_dashboard.png")
    html = plot_coverage_html(grid, out / f"{stem}.html")
    npz = write_grid_npz(grid, out / f"{stem}.npz")
    csv = write_grid_csv(grid, out / f"{stem}.csv")
    summary = grid.summary()
    (out / f"{stem}_summary.json").write_text(
        json.dumps({"stem": stem, **summary}, indent=2),
        encoding="utf-8",
    )
    return {
        "png": str(png),
        "dashboard": str(dash),
        "html": str(html),
        "npz": str(npz),
        "csv": str(csv),
        **summary,
    }


def _cmd_map(args) -> int:
    scenario = _apply_overrides(get_scenario(args.scenario), args)
    grid = predict_coverage(
        scenario,
        region=args.region,
        resolution_deg=args.res,
        t_s=args.t,
        mode=args.mode,
        n_times=args.samples,
        rain_rate_mm_h=args.rain,
        indoor=args.indoor,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result = _write_maps(grid, out, f"{scenario.name}_{args.region}_frf{scenario.reuse_factor}")
    print(json.dumps(result, indent=2))
    return 0


def _cmd_compare(args) -> int:
    left = _apply_overrides(get_scenario(args.left), args)
    right = _apply_overrides(get_scenario(args.right), args)
    g0 = predict_coverage(left, region=args.region, resolution_deg=args.res, t_s=args.t)
    g1 = predict_coverage(right, region=args.region, resolution_deg=args.res, t_s=args.t)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    png = plot_compare_png(g0, g1, out / f"compare_{left.name}_vs_{right.name}_{args.region}.png")
    print(
        json.dumps(
            {
                "png": str(png),
                "left": {"scenario": left.name, **g0.summary()},
                "right": {"scenario": right.name, **g1.summary()},
            },
            indent=2,
        )
    )
    return 0


def _cmd_study(args) -> int:
    base = _apply_overrides(get_scenario(args.scenario), args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    grids = {}
    for frf in (1, 3):
        scenario = base.with_reuse(frf)
        grid = predict_coverage(
            scenario,
            region=args.region,
            resolution_deg=args.res,
            t_s=args.t,
            rain_rate_mm_h=args.rain,
            indoor=args.indoor,
        )
        grids[frf] = grid
        results[f"frf{frf}"] = _write_maps(grid, out, f"study_{scenario.name}_{args.region}_frf{frf}")
    png = plot_compare_png(
        grids[1],
        grids[3],
        out / f"study_{base.name}_{args.region}_frf1_vs_frf3.png",
    )
    payload = {
        "region": args.region,
        "scenario": base.name,
        "compare_png": str(png),
        **results,
    }
    (out / f"study_{base.name}_{args.region}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_paper(args) -> int:
    payload = run_paper(
        region=args.region,
        resolution_deg=args.res,
        samples=args.samples,
        out_dir=args.out,
        scenario_name=args.scenario,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_pass(args) -> int:
    scenario = _apply_overrides(get_scenario(args.scenario), args)
    rows = pass_timeline(
        args.lat,
        args.lon,
        scenario,
        t0_s=args.t,
        duration_s=args.minutes * 60.0,
        dt_s=args.dt,
        rain_rate_mm_h=args.rain,
        indoor=args.indoor,
    )
    print(format_pass_table(rows))
    return 0


def _cmd_workbench(args) -> int:
    from ntn_coverage.workbench import serve_workbench

    serve_workbench(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ntn-coverage",
        description="Physics-based NTN/6G coverage predictor (3GPP TR 38.811 / 38.821).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp, *, region=False):
        sp.add_argument("--scenario", default="leo600-s-handheld", choices=sorted(SCENARIOS))
        sp.add_argument(
            "--environment",
            choices=["rural", "suburban", "urban", "dense_urban", "mixed"],
        )
        sp.add_argument("--antenna", choices=["cellular", "steered", "nadir"])
        sp.add_argument("--reuse", type=int, choices=[1, 3], help="Frequency reuse factor")
        sp.add_argument(
            "--coordination",
            choices=["exclusive", "uncoordinated"],
            help="exclusive: only the serving sat tiles the user; others leak a 19-beam nadir cluster",
        )
        sp.add_argument("--threshold", type=float, default=None, help="SINR coverage threshold in dB")
        sp.add_argument("--t", type=float, default=0.0, help="Snapshot time in seconds from epoch")
        sp.add_argument(
            "--rain",
            type=float,
            default=0.0,
            help="Snapshot rain rate in mm/h (0 = clear sky), not an annual 0.01%% statistic",
        )
        sp.add_argument("--indoor", action="store_true")
        sp.add_argument("--tle", help="TLE file or text; overrides the Walker/GEO constellation")
        if region:
            sp.add_argument("--region", default="romania", choices=sorted(REGIONS))
            sp.add_argument("--res", type=float, default=0.15, help="Grid step in degrees")
            sp.add_argument("--out", default="outputs")

    pt = sub.add_parser("point", help="Link budget at a lat/lon")
    add_common(pt)
    pt.add_argument("--lat", type=float, required=True)
    pt.add_argument("--lon", type=float, required=True)
    pt.add_argument("--json", action="store_true", help="Print JSON instead of the text dump")
    pt.set_defaults(func=_cmd_point)

    mp = sub.add_parser("map", help="Build SINR / throughput coverage maps")
    add_common(mp, region=True)
    mp.add_argument("--mode", choices=["snapshot", "availability"], default="snapshot")
    mp.add_argument("--samples", type=int, default=12, help="Time samples for availability mode")
    mp.set_defaults(func=_cmd_map)

    cp = sub.add_parser("compare", help="Compare two scenarios on the same grid")
    add_common(cp, region=True)
    cp.add_argument("--left", default="leo600-s-handheld")
    cp.add_argument("--right", default="leo1200-s-handheld")
    cp.set_defaults(func=_cmd_compare)

    st = sub.add_parser("study", help="FRF=1 vs FRF=3 SINR study on one region")
    add_common(st, region=True)
    st.set_defaults(func=_cmd_study)

    fg = sub.add_parser("figures", help="Export SINR maps, CDF, calibration, and reuse comparison")
    add_common(fg, region=True)
    fg.set_defaults(out="docs/figures", res=0.2)
    fg.add_argument("--samples", type=int, default=8, help="Orbit samples for availability")
    fg.set_defaults(func=_cmd_paper)

    ps = sub.add_parser("pass", help="Elevation / Doppler / serving-sat vs time at a terminal")
    add_common(ps)
    ps.add_argument("--lat", type=float, required=True)
    ps.add_argument("--lon", type=float, required=True)
    ps.add_argument("--minutes", type=float, default=20.0)
    ps.add_argument("--dt", type=float, default=30.0, help="Sample step in seconds")
    ps.set_defaults(func=_cmd_pass)

    wb = sub.add_parser("workbench", help="Open the local coverage GUI")
    wb.add_argument("--host", default="127.0.0.1")
    wb.add_argument("--port", type=int, default=8765)
    wb.add_argument("--no-browser", action="store_true")
    wb.set_defaults(func=_cmd_workbench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
