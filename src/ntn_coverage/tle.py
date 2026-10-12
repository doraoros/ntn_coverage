"""SGP4 / TLE ingest. TEME is rotated with the same GMST helper as Walker states."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ntn_coverage.geometry import eci_to_ecef, gmst_rad
from ntn_coverage.orbit import ConstellationConfig, ConstellationState


def _read_tle_pairs(source: str) -> list[tuple[str, str, str]]:
    text = Path(source).read_text(encoding="utf-8") if Path(source).exists() else source
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    pairs: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = lines[i - 1] if i and not lines[i - 1].startswith(("1 ", "2 ")) else f"SAT-{len(pairs)+1}"
            pairs.append((name.strip(), lines[i], lines[i + 1]))
            i += 2
            continue
        i += 1
    if not pairs:
        raise ValueError("No TLE pairs found")
    return pairs


def propagate_tle(source: str, t_s: float = 0.0) -> ConstellationState:
    """Propagate a TLE file or raw TLE text to ECEF at epoch + t_s seconds.

    SGP4 returns TEME at the absolute element time. That vector is rotated by
    GMST at the same instant; ``t_s`` alone is not an Earth angle.
    """
    try:
        from sgp4.api import SGP4_ERRORS, Satrec
    except ImportError as exc:
        raise ImportError("TLE support needs the sgp4 package: pip install sgp4") from exc

    pairs = _read_tle_pairs(source)
    ecef = []
    vel = []
    ids = []
    failures: list[str] = []
    for name, l1, l2 in pairs:
        try:
            sat = Satrec.twoline2rv(l1, l2)
        except ValueError as exc:
            failures.append(f"{name} ({exc})")
            continue
        err, r, v = sat.sgp4_tsince(float(t_s) / 60.0)
        if err != 0:
            failures.append(f"{name} ({SGP4_ERRORS.get(err, err)})")
            continue
        jd = float(sat.jdsatepoch) + float(sat.jdsatepochF) + float(t_s) / 86400.0
        r_m = np.asarray(r, dtype=float) * 1000.0
        v_ms = np.asarray(v, dtype=float) * 1000.0
        r_ecef, v_ecef = eci_to_ecef(r_m, v_ms, t_s, theta_rad=gmst_rad(jd))
        ecef.append(r_ecef)
        vel.append(v_ecef)
        ids.append(name[:16])
    if failures:
        raise RuntimeError("SGP4 failed for " + "; ".join(failures))
    if not ecef:
        raise RuntimeError("SGP4 failed for every TLE")
    mean_r = float(np.mean(np.linalg.norm(np.stack(ecef), axis=1)))
    cfg = ConstellationConfig(kind="walker", altitude_m=mean_r - 6_371_000.0, planes=1, per_plane=len(ecef))
    return ConstellationState(
        config=cfg,
        t_s=t_s,
        ecef=np.stack(ecef),
        velocity_ecef=np.stack(vel),
        sat_ids=np.asarray(ids),
    )
