"""Static PNG and interactive HTML coverage maps."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ntn_coverage.coverage import CoverageGrid, subsatellite_points
from ntn_coverage.landcover import ROMANIA_OUTLINE, ROMANIA_URBAN
from ntn_coverage.orbit import propagate


def _mesh(grid: CoverageGrid) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = np.meshgrid(grid.lons, grid.lats)
    return lon, lat


def _field_spec(grid: CoverageGrid, field: str) -> tuple[np.ndarray, str, str]:
    if field == "availability" and grid.availability is not None:
        return grid.availability * 100.0, "Availability (%)", "RdYlGn"
    if field == "elevation":
        return grid.elevation_deg, "Serving elevation (°)", "plasma"
    if field == "p_cover":
        return grid.p_cover * 100.0, "P(SINR ≥ threshold) (%)", "RdYlGn"
    if field == "rate_mbps":
        return grid.rate_mbps, "Throughput (Mbps)", "magma"
    if field == "cir_db":
        return grid.cir_db, "C/I (dB)", "cividis"
    if field == "snr_db":
        return grid.snr_db, "Downlink C/N (dB)", "viridis"
    return grid.sinr_db, "Downlink SINR (dB)", "viridis"


def _limits(z: np.ndarray, field: str) -> tuple[float, float]:
    if field in {"availability", "p_cover"}:
        return 0.0, 100.0
    finite = z[np.isfinite(z)]
    if not finite.size:
        return -10.0, 20.0
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if vmax - vmin < 1.0:
        vmin -= 1.0
        vmax += 1.0
    return vmin, vmax


def _decorate_geo(ax, grid: CoverageGrid, legend: bool = True) -> None:
    if grid.lats.min() < 49 and grid.lats.max() > 43 and grid.lons.min() < 30 and grid.lons.max() > 20:
        ax.plot(ROMANIA_OUTLINE[:, 0], ROMANIA_OUTLINE[:, 1], color="white", lw=1.1, alpha=0.9)
        ax.plot(ROMANIA_OUTLINE[:, 0], ROMANIA_OUTLINE[:, 1], color="black", lw=0.4, alpha=0.8)
        for lat, lon, _r, _name in ROMANIA_URBAN[:8]:
            if grid.lats.min() <= lat <= grid.lats.max() and grid.lons.min() <= lon <= grid.lons.max():
                ax.plot(lon, lat, "o", ms=3.0, color="white", markeredgecolor="black", markeredgewidth=0.4)
    state = propagate(grid.scenario.constellation, grid.t_s)
    slat, slon = subsatellite_points(state)
    pad = 6.0
    nearby = (
        (slat >= grid.lats.min() - pad)
        & (slat <= grid.lats.max() + pad)
        & (slon >= grid.lons.min() - pad)
        & (slon <= grid.lons.max() + pad)
    )
    ax.scatter(
        slon[nearby],
        slat[nearby],
        s=28,
        c="crimson",
        edgecolors="white",
        linewidths=0.5,
        zorder=3,
        label="Sub-satellite" if legend else None,
        clip_on=True,
    )
    ax.set_xlim(float(grid.lons.min()), float(grid.lons.max()))
    ax.set_ylim(float(grid.lats.min()), float(grid.lats.max()))
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(float(np.mean(grid.lats)))))


def plot_coverage_png(grid: CoverageGrid, path: str | Path, field: str = "sinr_db") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lon, lat = _mesh(grid)
    z, label, cmap = _field_spec(grid, field)
    vmin, vmax = _limits(z, field)
    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=140)
    if field == "sinr_db":
        from matplotlib.colors import TwoSlopeNorm

        mesh = ax.pcolormesh(
            lon,
            lat,
            z,
            shading="auto",
            cmap="RdYlGn",
            norm=TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=8.0),
        )
    else:
        mesh = ax.pcolormesh(lon, lat, z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, pad=0.02).set_label(label)
    _decorate_geo(ax, grid)
    rf = grid.scenario.reuse_factor
    ax.set_title(f"{grid.scenario.title}  ·  FRF={rf}  ·  {grid.scenario.antenna_mode}")
    ax.legend(loc="lower left", framealpha=0.85, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_dashboard_png(grid: CoverageGrid, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lon, lat = _mesh(grid)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6), dpi=140)
    averaged = grid.availability is not None
    panels = [
        ("availability" if averaged else "sinr_db", axes[0, 0]),
        ("p_cover", axes[0, 1]),
        ("rate_mbps", axes[1, 0]),
        ("elevation", axes[1, 1]),
    ]
    for field, ax in panels:
        z, label, cmap = _field_spec(grid, field)
        vmin, vmax = _limits(z, field)
        mesh = ax.pcolormesh(lon, lat, z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04).set_label(label, fontsize=8)
        _decorate_geo(ax, grid, legend=False)
        ax.set_title(label, fontsize=10)
    note = "time average" if averaged else grid.scenario.antenna_mode
    fig.suptitle(
        f"{grid.scenario.title}  ·  FRF={grid.scenario.reuse_factor}  ·  {note}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_coverage_html(grid: CoverageGrid, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    averaged = grid.availability is not None
    titles = (
        ("Availability (%)", "P(cover) (%)", "Mean throughput (Mbps)", "Mean elevation (°)")
        if averaged
        else ("SINR (dB)", "P(cover) (%)", "Throughput (Mbps)", "Elevation (°)")
    )
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )
    first = grid.availability * 100.0 if averaged else grid.sinr_db
    traces = [
        (first, "RdYlGn" if averaged else "Viridis", "Availability %" if averaged else "SINR dB", 1, 1),
        (grid.p_cover * 100.0, "RdYlGn", "P(cover) %", 1, 2),
        (grid.rate_mbps, "Magma", "Mbps", 2, 1),
        (grid.elevation_deg, "Plasma", "elev °", 2, 2),
    ]
    for z, scale, title, r, c in traces:
        fig.add_trace(
            go.Heatmap(
                x=grid.lons,
                y=grid.lats,
                z=z,
                colorscale=scale,
                colorbar=dict(title=title, len=0.35, y=0.80 if r == 1 else 0.20, x=0.46 if c == 1 else 1.0),
                hovertemplate="lat=%{y:.2f}° lon=%{x:.2f}°<br>%{z:.2f}<extra></extra>",
            ),
            row=r,
            col=c,
        )
    fig.update_xaxes(title_text="Longitude (°)")
    fig.update_yaxes(title_text="Latitude (°)")
    fig.update_layout(
        title=f"{grid.scenario.title} · FRF={grid.scenario.reuse_factor}",
        height=860,
        margin=dict(l=60, r=40, t=70, b=50),
    )
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def plot_compare_png(left: CoverageGrid, right: CoverageGrid, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.colors import TwoSlopeNorm

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=140)
    lon, lat = _mesh(left)
    shared = TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=8.0)
    pairs = [
        (left.sinr_db, f"{left.scenario.name}  FRF={left.scenario.reuse_factor}", "RdYlGn", shared),
        (right.sinr_db, f"{right.scenario.name}  FRF={right.scenario.reuse_factor}", "RdYlGn", shared),
        (right.sinr_db - left.sinr_db, "Δ SINR (right − left)", "coolwarm", None),
    ]
    for ax, (z, title, cmap, norm) in zip(axes, pairs, strict=False):
        if cmap == "coolwarm":
            peak = float(np.nanpercentile(np.abs(z), 98)) if np.isfinite(z).any() else 1.0
            mesh = ax.pcolormesh(lon, lat, z, shading="auto", cmap=cmap, vmin=-peak, vmax=peak)
        else:
            mesh = ax.pcolormesh(lon, lat, z, shading="auto", cmap=cmap, norm=norm)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        _decorate_geo(ax, left, legend=False)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
