"""Grid and point coverage prediction, including cellular SINR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
from scipy.stats import norm

from ntn_coverage.antennas import eirp_dbw
from ntn_coverage.beams import (
    adjacent_beam_spacing,
    axial_to_uv,
    beam_color,
    beam_relative_gain_db,
    boresight_uvw,
    hex_disk,
    hex_round,
    uv_to_axial_frac,
)
from ntn_coverage.channel import Environment, path_loss_components
from ntn_coverage.geometry import ecef_from_ll, link_geometry, ll_from_ecef, look_uvw
from ntn_coverage.landcover import urban_mask
from ntn_coverage.link_budget import cn0_dbhz, snr_from_eirp_density
from ntn_coverage.orbit import ConstellationState, propagate
from ntn_coverage.scenarios import REGIONS, Scenario
from ntn_coverage.throughput import throughput_bps

# 19-beam cluster per satellite (TR 38.821 SLS layout).
CLUSTER_RINGS = 2
# Upper bound on UE x satellite pairs held in memory at once.
MAX_UE_SAT_PAIRS = 500_000
MIN_RESOLUTION_DEG = 0.02
MAX_GRID_POINTS = 150_000
# Availability: average these only while a satellite is serving.
_SERVED_MEAN = {
    "snr_db",
    "sinr_db",
    "cir_db",
    "cn0_dbhz",
    "elevation_deg",
    "slant_range_m",
    "path_loss_db",
    "fspl_db",
    "p_los",
    "doppler_hz",
    "off_nadir_deg",
    "sinr_los_db",
    "sinr_nlos_db",
}
# Availability: a satellite index has no meaningful mean. Keep the middle sample.
_DISCRETE = {"serving_index", "serving_q", "serving_r"}
# Beams beyond this UV radius would point past the Earth limb.
_MAX_UV_RHO2 = 0.98


@dataclass
class PointPrediction:
    lat_deg: float
    lon_deg: float
    covered: bool
    snr_db: float
    sinr_db: float
    cir_db: float
    cn0_dbhz: float
    elevation_deg: float
    slant_range_km: float
    path_loss_db: float
    fspl_db: float
    doppler_hz: float
    n_visible: int
    serving_sat: str
    rate_mbps: float
    p_cover: float
    extras: dict = field(default_factory=dict)


@dataclass
class CoverageGrid:
    """Gridded prediction.

    In availability mode SINR, C/N, C/I and elevation are means over samples
    that have a serving link. Throughput and P(cover) are means over every
    sample, and an outage counts as zero. ``coverage_fraction`` is the
    fraction of time above the SINR threshold; ``ever_covered_fraction`` is
    the share of pixels that clear it at least once.
    """

    scenario: Scenario
    lats: np.ndarray
    lons: np.ndarray
    snr_db: np.ndarray
    sinr_db: np.ndarray
    cir_db: np.ndarray
    elevation_deg: np.ndarray
    path_loss_db: np.ndarray
    n_visible: np.ndarray
    serving_index: np.ndarray
    rate_mbps: np.ndarray
    p_cover: np.ndarray
    availability: np.ndarray | None = None
    t_s: float = 0.0

    @property
    def covered(self) -> np.ndarray:
        metric = self.sinr_db if np.isfinite(self.sinr_db).any() else self.snr_db
        return np.isfinite(metric) & (metric >= self.scenario.snr_threshold_db)

    def summary(self) -> dict[str, float]:
        sinr = self.sinr_db[np.isfinite(self.sinr_db)]
        snr = self.snr_db[np.isfinite(self.snr_db)]
        rate = self.rate_mbps[np.isfinite(self.rate_mbps)]
        out = {
            "coverage_fraction": float(np.mean(self.covered)),
            "mean_snr_db": float(np.mean(snr)) if snr.size else float("nan"),
            "mean_sinr_db": float(np.mean(sinr)) if sinr.size else float("nan"),
            "median_sinr_db": float(np.median(sinr)) if sinr.size else float("nan"),
            "p5_sinr_db": float(np.percentile(sinr, 5)) if sinr.size else float("nan"),
            "p50_sinr_db": float(np.percentile(sinr, 50)) if sinr.size else float("nan"),
            "p95_sinr_db": float(np.percentile(sinr, 95)) if sinr.size else float("nan"),
            "mean_rate_mbps": float(np.mean(rate)) if rate.size else float("nan"),
            "mean_p_cover": float(np.nanmean(self.p_cover)),
            "mean_visible_sats": float(np.mean(self.n_visible)),
            "mean_serving_elevation_deg": float(
                np.nanmean(np.where(self.covered, self.elevation_deg, np.nan))
            ),
            "reuse_factor": float(self.scenario.reuse_factor),
        }
        if self.availability is not None:
            out["ever_covered_fraction"] = float(np.mean(self.availability > 0.0))
            out["coverage_fraction"] = float(np.mean(self.availability))
            out["mean_availability"] = out["coverage_fraction"]
        return out


def _blend_mixed_losses(ue_lat, ue_lon, rural, urban) -> dict[str, np.ndarray]:
    mask = urban_mask(ue_lat, ue_lon)
    if rural["path_loss_mean_db"].ndim == mask.ndim + 1:
        m = mask[..., None]
    else:
        m = mask
    return {key: np.where(m, urban[key], rural[key]) for key in rural}


def _path_losses(scenario: Scenario, dist, elev, ue_lat, ue_lon, rain_rate_mm_h, indoor):
    env = scenario.environment
    if env == "mixed":
        rural = path_loss_components(
            dist, np.clip(elev, 0.1, 90.0), scenario.rf.freq_hz, "rural",
            ue_lat[..., None], rain_rate_mm_h, indoor,
        )
        urban = path_loss_components(
            dist, np.clip(elev, 0.1, 90.0), scenario.rf.freq_hz, "urban",
            ue_lat[..., None], rain_rate_mm_h, indoor,
        )
        return _blend_mixed_losses(ue_lat, ue_lon, rural, urban)
    return path_loss_components(
        dist,
        np.clip(elev, 0.1, 90.0),
        scenario.rf.freq_hz,
        cast(Environment, env),
        ue_lat[..., None],
        rain_rate_mm_h,
        indoor,
    )


def take_safe(arr, serving):
    idx = np.where(np.asarray(serving) < 0, 0, serving)
    return np.take_along_axis(np.asarray(arr), idx[..., None], axis=-1)[..., 0]


def _coverage_probability(sinr_los, sinr_nlos, p_los, sig_los, sig_nlos, threshold_db):
    sig_l = np.clip(np.asarray(sig_los, dtype=float), 0.25, None)
    sig_n = np.clip(np.asarray(sig_nlos, dtype=float), 0.25, None)
    p_los = np.asarray(p_los, dtype=float)
    p = p_los * norm.sf((threshold_db - sinr_los) / sig_l)
    p = p + (1.0 - p_los) * norm.sf((threshold_db - sinr_nlos) / sig_n)
    return np.clip(p, 0.0, 1.0)


def _noise_limited(scenario, geo, losses, visible):
    steered = scenario.antenna_mode != "nadir"
    eirp_dens = eirp_dbw(
        scenario.rf.eirp_density_dbw_per_mhz,
        geo["off_nadir_deg"],
        scenario.rf.aperture_m,
        scenario.rf.freq_hz,
        steered=steered,
    )
    extra = scenario.terminal.shadowing_margin_db
    snr = snr_from_eirp_density(
        eirp_dens, losses["path_loss_mean_db"], scenario.terminal.g_over_t_db_k, extra
    )
    return np.where(visible, snr, -np.inf)


def _cluster_cn(scenario: Scenario, look, q, r, abs_s, path_loss_db) -> np.ndarray:
    """Linear C/N per (UE, beam) for hex beams (q, r) of one satellite."""
    u, v = axial_to_uv(q, r, abs_s)
    gain = beam_relative_gain_db(look, boresight_uvw(u, v), scenario.rf.aperture_m, scenario.rf.freq_hz)
    snr = snr_from_eirp_density(
        scenario.rf.eirp_density_dbw_per_mhz + gain,
        path_loss_db[:, None],
        scenario.terminal.g_over_t_db_k,
        scenario.terminal.shadowing_margin_db,
    )
    return np.where(u * u + v * v < _MAX_UV_RHO2, 10.0 ** (snr / 10.0), 0.0)


def _cellular_sinr(scenario: Scenario, ue_ecef, sat_ecef, losses, visible):
    """Best-beam attachment with intra/inter-satellite co-channel interference.

    Every satellite lights a fixed CLUSTER_RINGS hex cluster. For the serving
    satellite (and, when uncoordinated, for every other satellite) the cluster
    is centred on the beam cell containing the UE, which is the wrap-around
    equivalent of the TR 38.821 SLS. With exclusive coordination the other
    satellites keep their cluster at nadir. Results therefore depend only on
    the UE location, not on the size of the evaluated grid.
    """
    n_ue, n_sat = visible.shape
    pl = losses["path_loss_mean_db"]
    pl_los = losses["path_loss_los_db"]
    pl_nlos = losses["path_loss_nlos_db"]
    rows = np.arange(n_ue)

    abs_s = adjacent_beam_spacing(scenario.rf.hpbw_deg)
    reuse = max(int(scenario.reuse_factor), 1)
    exclusive = scenario.coordination != "uncoordinated"
    dq, dr, _du, _dv = hex_disk(CLUSTER_RINGS, abs_s)
    nadir_colors = beam_color(dq, dr, reuse)

    best_cn = np.full(n_ue, -np.inf)
    serve_sat = np.full(n_ue, -1, dtype=int)
    serve_color = np.zeros(n_ue, dtype=int)
    serve_q = np.zeros(n_ue, dtype=int)
    serve_r = np.zeros(n_ue, dtype=int)

    clusters: dict[int, tuple] = {}
    for s in np.flatnonzero(visible.any(axis=0)):
        vis_s = visible[:, s]
        look = np.stack(look_uvw(ue_ecef, sat_ecef[s]), axis=-1)
        q0, r0 = hex_round(*uv_to_axial_frac(look[:, 0], look[:, 1], abs_s))
        q = q0[:, None] + dq[None, :]
        r = r0[:, None] + dr[None, :]
        colors = beam_color(q, r, reuse)
        cn = np.where(vis_s[:, None], _cluster_cn(scenario, look, q, r, abs_s, pl[:, s]), 0.0)
        local = np.argmax(cn, axis=1)
        peak = cn[rows, local]
        better = vis_s & (peak > 0.0) & (peak > best_cn)
        best_cn = np.where(better, peak, best_cn)
        serve_sat = np.where(better, s, serve_sat)
        serve_color = np.where(better, colors[rows, local], serve_color)
        serve_q = np.where(better, q[rows, local], serve_q)
        serve_r = np.where(better, r[rows, local], serve_r)
        clusters[int(s)] = (look, q, r, colors, cn)

    i_cn = np.zeros(n_ue)
    for s, (look, q, r, colors, cn) in clusters.items():
        own = serve_sat == s
        same = colors == serve_color[:, None]
        on_serving = (q == serve_q[:, None]) & (r == serve_r[:, None])
        i_cn += np.where(own, np.sum(np.where(same & ~on_serving, cn, 0.0), axis=1), 0.0)

        other = visible[:, s] & ~own
        if not np.any(other):
            continue
        if exclusive:
            cn_o = _cluster_cn(scenario, look, dq[None, :], dr[None, :], abs_s, pl[:, s])
            same_o = nadir_colors[None, :] == serve_color[:, None]
        else:
            cn_o, same_o = cn, same
        i_cn += np.where(other, np.sum(np.where(same_o, cn_o, 0.0), axis=1), 0.0)

    best_snr = 10.0 * np.log10(np.clip(best_cn, 1e-30, None))
    sinr_lin = best_cn / np.clip(i_cn + 1.0, 1e-30, None)
    sinr = 10.0 * np.log10(np.clip(sinr_lin, 1e-30, None))
    cir = 10.0 * np.log10(np.clip(best_cn / np.clip(i_cn, 1e-30, None), 1e-30, None))
    none = serve_sat < 0
    sinr = np.where(none, np.nan, sinr)
    cir = np.where(none, np.nan, cir)
    best_snr = np.where(none, np.nan, best_snr)

    # LOS/NLOS SINR for coverage probability: scale serving C, keep I+N.
    idx = np.clip(serve_sat, 0, None)
    pl_s = pl[rows, idx]
    d_los = pl_s - pl_los[rows, idx]
    d_nlos = pl_s - pl_nlos[rows, idx]
    c_los = best_cn * 10.0 ** (d_los / 10.0)
    c_nlos = best_cn * 10.0 ** (d_nlos / 10.0)
    sinr_los = 10.0 * np.log10(np.clip(c_los / (i_cn + 1.0), 1e-30, None))
    sinr_nlos = 10.0 * np.log10(np.clip(c_nlos / (i_cn + 1.0), 1e-30, None))
    sinr_los = np.where(none, np.nan, sinr_los)
    sinr_nlos = np.where(none, np.nan, sinr_nlos)

    return {
        "snr_db": best_snr,
        "sinr_db": sinr,
        "cir_db": cir,
        "serving_index": np.where(none, -1, serve_sat),
        "serving_q": np.where(none, -1, serve_q),
        "serving_r": np.where(none, -1, serve_r),
        "sinr_los_db": sinr_los,
        "sinr_nlos_db": sinr_nlos,
        "i_over_n": i_cn,
    }


def _serving_metrics(
    scenario: Scenario,
    ue_lat: np.ndarray,
    ue_lon: np.ndarray,
    state: ConstellationState,
    rain_rate_mm_h: float = 0.0,
    indoor: bool = False,
) -> dict[str, np.ndarray]:
    """Serving-link metrics for UEs of any shape, processed in memory-bounded chunks."""
    ue_lat = np.asarray(ue_lat, dtype=float)
    ue_lon = np.broadcast_to(np.asarray(ue_lon, dtype=float), ue_lat.shape)
    shape = ue_lat.shape
    flat_lat, flat_lon = ue_lat.ravel(), ue_lon.ravel()
    chunk = max(1, MAX_UE_SAT_PAIRS // max(state.n_satellites, 1))
    parts = [
        _serving_metrics_flat(
            scenario, flat_lat[i : i + chunk], flat_lon[i : i + chunk], state, rain_rate_mm_h, indoor
        )
        for i in range(0, flat_lat.size, chunk)
    ]
    return {key: np.concatenate([p[key] for p in parts]).reshape(shape) for key in parts[0]}


def _serving_metrics_flat(
    scenario: Scenario,
    ue_lat: np.ndarray,
    ue_lon: np.ndarray,
    state: ConstellationState,
    rain_rate_mm_h: float,
    indoor: bool,
) -> dict[str, np.ndarray]:
    ue_ecef = ecef_from_ll(ue_lat, ue_lon)
    geo = link_geometry(ue_ecef, state.ecef, state.velocity_ecef)
    visible = geo["elevation_deg"] >= scenario.min_elevation_deg
    keep = np.flatnonzero(visible.any(axis=0))
    if keep.size == 0:
        keep = np.array([0])
    geo = {key: val[:, keep] for key, val in geo.items()}
    visible = visible[:, keep]
    elev = geo["elevation_deg"]
    dist = geo["slant_range_m"]
    losses = _path_losses(scenario, dist, elev, ue_lat, ue_lon, rain_rate_mm_h, indoor)

    if scenario.antenna_mode == "cellular":
        cell = _cellular_sinr(scenario, ue_ecef, state.ecef[keep], losses, visible)
        serving = cell["serving_index"]
    else:
        snr = _noise_limited(scenario, geo, losses, visible)
        serving = np.argmax(snr, axis=-1)
        cell = None

    n_visible = np.sum(visible, axis=-1)

    none = n_visible == 0
    if cell is None:
        serving_out = np.where(none, -1, serving)
        best_snr = np.where(none, np.nan, take_safe(snr, serving_out))
        sinr = best_snr
        cir = np.full(best_snr.shape, np.nan)
        steered = scenario.antenna_mode != "nadir"
        eirp_nl = eirp_dbw(
            scenario.rf.eirp_density_dbw_per_mhz,
            geo["off_nadir_deg"],
            scenario.rf.aperture_m,
            scenario.rf.freq_hz,
            steered=steered,
        )
        extra = scenario.terminal.shadowing_margin_db
        gt = scenario.terminal.g_over_t_db_k
        sinr_los = np.where(
            none,
            np.nan,
            take_safe(
                snr_from_eirp_density(eirp_nl, losses["path_loss_los_db"], gt, extra),
                serving_out,
            ),
        )
        sinr_nlos = np.where(
            none,
            np.nan,
            take_safe(
                snr_from_eirp_density(eirp_nl, losses["path_loss_nlos_db"], gt, extra),
                serving_out,
            ),
        )
        serving_q = serving_r = np.full(serving_out.shape, -1)
    else:
        best_snr = cell["snr_db"]
        sinr = cell["sinr_db"]
        cir = cell["cir_db"]
        sinr_los = cell["sinr_los_db"]
        sinr_nlos = cell["sinr_nlos_db"]
        serving_out = cell["serving_index"]
        serving_q, serving_r = cell["serving_q"], cell["serving_r"]

    p_los = take_safe(losses["p_los"], serving_out)
    sig_l = take_safe(losses["los_sf_sigma_db"], serving_out)
    sig_n = take_safe(losses["nlos_sf_sigma_db"], serving_out)
    p_cov = _coverage_probability(
        sinr_los, sinr_nlos, p_los, sig_l, sig_n, scenario.snr_threshold_db
    )
    rate = throughput_bps(sinr, scenario.rf.bandwidth_hz) / 1e6
    doppler = take_safe(geo["doppler_over_f"], serving_out) * scenario.rf.freq_hz
    cn0 = cn0_dbhz(best_snr, scenario.rf.bandwidth_hz)

    return {
        "snr_db": best_snr,
        "sinr_db": sinr,
        "cir_db": cir,
        "cn0_dbhz": np.where(none, np.nan, cn0),
        "elevation_deg": np.where(none, np.nan, take_safe(elev, serving_out)),
        "slant_range_m": np.where(none, np.nan, take_safe(dist, serving_out)),
        "path_loss_db": np.where(none, np.nan, take_safe(losses["path_loss_mean_db"], serving_out)),
        "fspl_db": np.where(none, np.nan, take_safe(losses["fspl_db"], serving_out)),
        "p_los": np.where(none, np.nan, p_los),
        "doppler_hz": np.where(none, np.nan, doppler),
        "n_visible": n_visible,
        "serving_index": np.where(serving_out >= 0, keep[np.clip(serving_out, 0, None)], -1),
        "serving_q": serving_q,
        "serving_r": serving_r,
        "off_nadir_deg": np.where(none, np.nan, take_safe(geo["off_nadir_deg"], serving_out)),
        "rate_mbps": np.where(none, np.nan, rate),
        "p_cover": np.where(none, np.nan, p_cov),
        "sinr_los_db": sinr_los,
        "sinr_nlos_db": sinr_nlos,
    }


def predict_point(
    lat_deg: float,
    lon_deg: float,
    scenario: Scenario,
    t_s: float = 0.0,
    rain_rate_mm_h: float = 0.0,
    indoor: bool = False,
) -> PointPrediction:
    state = propagate(scenario.constellation, t_s)
    metrics = _serving_metrics(
        scenario,
        np.array(lat_deg),
        np.array(lon_deg),
        state,
        rain_rate_mm_h=rain_rate_mm_h,
        indoor=indoor,
    )
    idx = int(metrics["serving_index"])
    sat_name = str(state.sat_ids[idx]) if idx >= 0 else "none"
    sinr = float(metrics["sinr_db"])
    snr = float(metrics["snr_db"])
    covered = bool(np.isfinite(sinr) and sinr >= scenario.snr_threshold_db)
    serving_q, serving_r = int(metrics["serving_q"]), int(metrics["serving_r"])
    if idx >= 0 and scenario.antenna_mode != "cellular":
        from ntn_coverage.footprints import serving_axial

        serving_q, serving_r = serving_axial(
            ecef_from_ll(lat_deg, lon_deg),
            state.ecef[idx],
            adjacent_beam_spacing(scenario.rf.hpbw_deg),
        )
    return PointPrediction(
        lat_deg=float(lat_deg),
        lon_deg=float(lon_deg),
        covered=covered,
        snr_db=snr,
        sinr_db=sinr,
        cir_db=float(metrics["cir_db"]),
        cn0_dbhz=float(metrics["cn0_dbhz"]),
        elevation_deg=float(metrics["elevation_deg"]),
        slant_range_km=float(metrics["slant_range_m"]) / 1000.0,
        path_loss_db=float(metrics["path_loss_db"]),
        fspl_db=float(metrics["fspl_db"]),
        doppler_hz=float(metrics["doppler_hz"]),
        n_visible=int(metrics["n_visible"]),
        serving_sat=sat_name,
        rate_mbps=float(metrics["rate_mbps"]),
        p_cover=float(metrics["p_cover"]),
        extras={
            "p_los": float(metrics["p_los"]),
            "off_nadir_deg": float(metrics["off_nadir_deg"]),
            "g_over_t_db_k": float(scenario.terminal.g_over_t_db_k),
            "t_s": float(t_s),
            "reuse_factor": int(scenario.reuse_factor),
            "coordination": scenario.coordination,
            "antenna_mode": scenario.antenna_mode,
            "sinr_los_db": float(metrics["sinr_los_db"]),
            "sinr_nlos_db": float(metrics["sinr_nlos_db"]),
            "serving_index": idx,
            "serving_q": serving_q,
            "serving_r": serving_r,
        },
    )


def _grid_axes(region: dict[str, float], resolution_deg: float) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(resolution_deg) or resolution_deg < MIN_RESOLUTION_DEG:
        raise ValueError(f"resolution_deg must be >= {MIN_RESOLUTION_DEG}")
    lats = np.arange(region["lat_min"], region["lat_max"] + 1e-9, resolution_deg)
    lons = np.arange(region["lon_min"], region["lon_max"] + 1e-9, resolution_deg)
    if lats.size < 2 or lons.size < 2:
        raise ValueError("Region/resolution produced a degenerate grid")
    n_points = int(lats.size * lons.size)
    if n_points > MAX_GRID_POINTS:
        raise ValueError(f"Grid is {n_points} points (max {MAX_GRID_POINTS}); use a coarser grid")
    return lats, lons


def _grid_from_metrics(scenario, lats, lons, m, t_s, availability=None) -> CoverageGrid:
    return CoverageGrid(
        scenario=scenario,
        lats=lats,
        lons=lons,
        snr_db=m["snr_db"],
        sinr_db=m["sinr_db"],
        cir_db=m["cir_db"],
        elevation_deg=m["elevation_deg"],
        path_loss_db=m["path_loss_db"],
        n_visible=m["n_visible"],
        serving_index=m["serving_index"],
        rate_mbps=m["rate_mbps"],
        p_cover=m["p_cover"],
        availability=availability,
        t_s=t_s,
    )


def predict_coverage(
    scenario: Scenario,
    region: str | dict[str, float] = "romania",
    resolution_deg: float = 0.15,
    t_s: float = 0.0,
    mode: str = "snapshot",
    n_times: int = 12,
    rain_rate_mm_h: float = 0.0,
    indoor: bool = False,
) -> CoverageGrid:
    box = REGIONS[region] if isinstance(region, str) else region
    lats, lons = _grid_axes(box, resolution_deg)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    if mode == "snapshot":
        state = propagate(scenario.constellation, t_s)
        m = _serving_metrics(scenario, lat_grid, lon_grid, state, rain_rate_mm_h, indoor)
        return _grid_from_metrics(scenario, lats, lons, m, t_s)

    if mode != "availability":
        raise ValueError("mode must be 'snapshot' or 'availability'")

    n_times = int(n_times)
    if n_times < 1:
        raise ValueError("n_times must be >= 1")
    period = scenario.constellation.period_s
    times = np.linspace(0.0, period, n_times, endpoint=False)
    sums: dict[str, np.ndarray] = {}
    served: np.ndarray | None = None
    hits: np.ndarray | None = None
    discrete: dict[str, np.ndarray] = {}
    for index, t in enumerate(times):
        state = propagate(scenario.constellation, float(t))
        sample = _serving_metrics(scenario, lat_grid, lon_grid, state, rain_rate_mm_h, indoor)
        finite = np.isfinite(sample["sinr_db"])
        if served is None:
            served = np.zeros(finite.shape, dtype=float)
            hits = np.zeros(finite.shape, dtype=float)
        served += finite
        hits += finite & (sample["sinr_db"] >= scenario.snr_threshold_db)
        if index == n_times // 2:
            discrete = {
                "serving_index": sample["serving_index"],
                "serving_q": sample["serving_q"],
                "serving_r": sample["serving_r"],
            }
        for key, values in sample.items():
            if key in _DISCRETE:
                continue
            if key not in sums:
                sums[key] = np.zeros(np.shape(values), dtype=float)
            usable = finite if key in _SERVED_MEAN else np.isfinite(values)
            sums[key] += np.where(usable, values, 0.0)
    assert served is not None and hits is not None
    averaged: dict[str, np.ndarray] = {}
    for key, total in sums.items():
        if key in _SERVED_MEAN:
            averaged[key] = np.where(served > 0.0, total / np.maximum(served, 1.0), np.nan)
        else:
            averaged[key] = total / n_times
    averaged.update(discrete)
    return _grid_from_metrics(scenario, lats, lons, averaged, 0.0, availability=hits / n_times)


def subsatellite_points(state: ConstellationState) -> tuple[np.ndarray, np.ndarray]:
    lat, lon, _ = ll_from_ecef(state.ecef)
    return lat, lon
