"""Write coverage grids as NPZ / CSV for QGIS or numpy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ntn_coverage.coverage import CoverageGrid


def write_grid_npz(grid: CoverageGrid, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lats=grid.lats,
        lons=grid.lons,
        sinr_db=grid.sinr_db,
        snr_db=grid.snr_db,
        cir_db=grid.cir_db,
        rate_mbps=grid.rate_mbps,
        p_cover=grid.p_cover,
        elevation_deg=grid.elevation_deg,
        n_visible=grid.n_visible,
    )
    return path


def write_grid_csv(grid: CoverageGrid, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lat, lon = np.meshgrid(grid.lats, grid.lons, indexing="ij")
    cols = np.column_stack(
        [
            lat.ravel(),
            lon.ravel(),
            grid.sinr_db.ravel(),
            grid.snr_db.ravel(),
            grid.cir_db.ravel(),
            grid.rate_mbps.ravel(),
            grid.p_cover.ravel(),
            grid.elevation_deg.ravel(),
            np.asarray(grid.n_visible, dtype=float).ravel(),
        ]
    )
    header = "lat,lon,sinr_db,snr_db,cir_db,rate_mbps,p_cover,elevation_deg,n_visible"
    np.savetxt(path, cols, delimiter=",", header=header, comments="", encoding="utf-8")
    return path
