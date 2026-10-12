"""Analysis figures: SINR CDFs, 3GPP calibration, reuse comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from ntn_coverage.coverage import CoverageGrid
from ntn_coverage.landcover import ROMANIA_URBAN
from ntn_coverage.plot import _decorate_geo, _mesh
from ntn_coverage.stats import ecdf, tr38821_sc6_ka_leo600

FRF1_COLOR = "#0072B2"
FRF3_COLOR = "#D55E00"
SINR_VMIN = -5.0
SINR_VMAX = 8.0
SINR_CMAP = "RdYlGn"


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sinr_norm() -> TwoSlopeNorm:
    return TwoSlopeNorm(vmin=SINR_VMIN, vcenter=0.0, vmax=SINR_VMAX)


def _city_labels(
    ax, grid: CoverageGrid, names: tuple[str, ...] = ("Bucharest", "Cluj-Napoca", "Iasi", "Timisoara")
) -> None:
    for lat, lon, _r, name in ROMANIA_URBAN:
        if name not in names:
            continue
        if not (grid.lats.min() <= lat <= grid.lats.max() and grid.lons.min() <= lon <= grid.lons.max()):
            continue
        ax.annotate(
            name,
            (lon, lat),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
            color="0.08",
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")],
        )


def _outage_at(x: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> float:
    if x.size == 0:
        return float("nan")
    below = x < threshold
    if not np.any(below):
        return 0.0
    return float(y[below][-1])


def plot_sinr_cdf(grids: dict[str, CoverageGrid], path: str | Path) -> Path:
    apply_paper_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"FRF=1": FRF1_COLOR, "FRF=3": FRF3_COLOR}
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    for label, grid in grids.items():
        x, y = ecdf(grid.sinr_db)
        ax.plot(x, y, lw=2.2, color=colors.get(label), label=label)
        p_out = _outage_at(x, y, 0.0)
        med = float(np.nanmedian(grid.sinr_db))
        ax.axvline(med, color=colors.get(label, "0.4"), ls=":", lw=1.0, alpha=0.85)
        if p_out >= 0.01:
            ax.text(
                0.22,
                p_out + 0.06,
                f"{label} outage {100 * p_out:.1f}%",
                fontsize=8,
                color=colors.get(label, "0.2"),
            )
    ax.axvline(0.0, color="0.35", ls="--", lw=1.0, label="Coverage threshold 0 dB")
    ax.set_xlabel("Downlink SINR (dB)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("SINR distribution over Romania (LEO-600, S-band handheld)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_calibration_png(path: str | Path) -> tuple[Path, dict[str, float]]:
    apply_paper_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cal = tr38821_sc6_ka_leo600()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    panels = [
        (axes[0], "fspl_3gpp_db", "fspl_db", "fspl_error_db", "FSPL (dB)", "Free-space path loss", (178.85, 179.42)),
        (axes[1], "snr_3gpp_db", "snr_db", "snr_error_db", "CNR (dB)", "Downlink CNR", (0.0, 10.5)),
    ]
    for ax, ref_k, eng_k, err_k, ylab, title, ylim in panels:
        ref, eng = cal[ref_k], cal[eng_k]
        bars = ax.bar([0, 1], [ref, eng], 0.55, color=[FRF1_COLOR, FRF3_COLOR])
        ax.set_xticks([0, 1], ["TR 38.821", "Engine"])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_ylim(*ylim)
        for bar, val in zip(bars, (ref, eng), strict=False):
            pad = 0.012 if "FSPL" in ylab else 0.12
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + pad,
                f"{val:.3f}" if "FSPL" in ylab else f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax.text(
            0.5,
            0.08,
            f"|Δ| = {cal[err_k]:.3f} dB",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="0.25",
        )
        ax.grid(axis="x", visible=False)
    fig.suptitle("Calibration · LEO-600 Ka VSAT · 30° elevation · clear sky (SC6)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path, cal


def plot_hero_png(frf1: CoverageGrid, frf3: CoverageGrid, path: str | Path) -> Path:
    """Shared-scale FRF=1 vs FRF=3 maps, CDF, and snapshot KPIs."""
    apply_paper_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.6, 7.8), dpi=180)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.95], height_ratios=[1.12, 1.0], wspace=0.28, hspace=0.34)
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax_cdf = fig.add_subplot(gs[1, :2])
    ax_kpi = fig.add_subplot(gs[:, 2])

    lon, lat = _mesh(frf3)
    norm = sinr_norm()
    mesh = None
    for ax, grid, letter in ((ax1, frf1, "a"), (ax3, frf3, "b")):
        mesh = ax.pcolormesh(lon, lat, grid.sinr_db, shading="auto", cmap=SINR_CMAP, norm=norm)
        _decorate_geo(ax, grid, legend=False)
        _city_labels(ax, grid)
        cov = 100.0 * float(grid.summary()["coverage_fraction"])
        ax.set_title(f"({letter})  FRF = {grid.scenario.reuse_factor}   ·   {cov:.1f}% covered")
    cbar = fig.colorbar(mesh, ax=[ax1, ax3], fraction=0.046, pad=0.03)
    cbar.set_label("Downlink SINR (dB)")

    x1, y1 = ecdf(frf1.sinr_db)
    x3, y3 = ecdf(frf3.sinr_db)
    ax_cdf.plot(x1, y1, lw=2.4, color=FRF1_COLOR, label="FRF = 1")
    ax_cdf.plot(x3, y3, lw=2.4, color=FRF3_COLOR, label="FRF = 3")
    ax_cdf.axvline(0.0, color="0.35", ls="--", lw=1.0)
    ax_cdf.set_xlabel("Downlink SINR (dB)")
    ax_cdf.set_ylabel("Empirical CDF")
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_title("(c)  Same C/N on both maps — coverage is interference-limited")
    ax_cdf.legend(frameon=False, loc="lower right")
    p1 = _outage_at(x1, y1)
    p3 = _outage_at(x3, y3)
    ax_cdf.annotate(
        f"P(SINR < 0) = {100 * p1:.1f}%",
        xy=(0.0, p1),
        xytext=(-14, 10),
        textcoords="offset points",
        ha="right",
        color=FRF1_COLOR,
        fontsize=8,
    )
    if p3 >= 0.01:
        ax_cdf.annotate(
            f"P(SINR < 0) = {100 * p3:.1f}%",
            xy=(0.0, p3),
            xytext=(-12, 12),
            textcoords="offset points",
            ha="right",
            color=FRF3_COLOR,
            fontsize=8,
        )

    s1, s3 = frf1.summary(), frf3.summary()
    ax_kpi.set_axis_off()
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    rows = [
        ("NTN 6G coverage predictor", "", True),
        ("3GPP TR 38.811 / 38.821", "", False),
        ("", "", False),
        ("Coverage  FRF=1", f"{100 * s1['coverage_fraction']:.1f}%", False),
        ("Coverage  FRF=3", f"{100 * s3['coverage_fraction']:.1f}%", False),
        ("Median SINR  FRF=1", f"{s1['median_sinr_db']:.2f} dB", False),
        ("Median SINR  FRF=3", f"{s3['median_sinr_db']:.2f} dB", False),
        ("Reuse gain (median)", f"+{s3['median_sinr_db'] - s1['median_sinr_db']:.2f} dB", False),
        ("Mean rate  FRF=3", f"{s3['mean_rate_mbps']:.1f} Mbps", False),
        ("P(cover)  FRF=3", f"{100 * s3['mean_p_cover']:.1f}%", False),
        ("C/N (both)", f"{s3['mean_snr_db']:.2f} dB", False),
    ]
    y = 0.96
    for name, val, hero in rows:
        if not name:
            y -= 0.035
            continue
        ax_kpi.text(
            0.0,
            y,
            name,
            fontsize=13 if hero else 10,
            fontweight="bold" if (hero or not val) else "normal",
            transform=ax_kpi.transAxes,
            color="0.08",
        )
        if val:
            ax_kpi.text(
                1.0,
                y,
                val,
                fontsize=12,
                ha="right",
                fontweight="bold",
                transform=ax_kpi.transAxes,
                color=FRF3_COLOR if "FRF=3" in name or "gain" in name else "0.12",
            )
        y -= 0.072
    ax_kpi.text(
        0.0,
        0.03,
        "Romania · 0.2° grid · LEO 600 km\nS-band handheld · exclusive coordination\nWalker 360/18/1 · threshold 0 dB",
        fontsize=8,
        color="0.38",
        transform=ax_kpi.transAxes,
        va="bottom",
    )
    fig.suptitle("Non-terrestrial 6G coverage: frequency reuse on a LEO downlink", fontsize=14, y=0.995)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_architecture_png(path: str | Path) -> Path:
    apply_paper_style()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 3.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 3.4)
    ax.axis("off")
    boxes = [
        (0.25, 1.15, "Constellation\nWalker / GEO / TLE"),
        (2.45, 1.15, "WGS-84 geometry\nslant, elev, Doppler"),
        (4.65, 1.15, "TR 38.811 channel\nLOS, clutter, gases"),
        (6.85, 1.15, "Hex beams + FRF\nBessel aperture"),
        (9.05, 1.15, "SINR → MCS\nP(cover), Mbps"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                1.95,
                1.35,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor="#F4F7FB",
                edgecolor="#1F3A5F",
                linewidth=1.2,
            )
        )
        ax.text(x + 0.975, y + 0.68, text, ha="center", va="center", fontsize=9)
    for x in (2.20, 4.40, 6.60, 8.80):
        ax.annotate(
            "", xy=(x + 0.22, 1.82), xytext=(x, 1.82), arrowprops=dict(arrowstyle="->", color="#1F3A5F", lw=1.4)
        )
    ax.set_title("Engine pipeline (downlink, snapshot or availability campaign)", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
