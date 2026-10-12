"""Local HTTP GUI for coverage maps, beam traces, and pass timelines."""

from __future__ import annotations

import json
import math
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import plotly

from ntn_coverage.coverage import (
    MAX_GRID_POINTS,
    MIN_RESOLUTION_DEG,
    predict_coverage,
    predict_point,
    subsatellite_points,
)
from ntn_coverage.footprints import serving_beam_cluster
from ntn_coverage.landcover import ROMANIA_OUTLINE, ROMANIA_URBAN
from ntn_coverage.orbit import propagate
from ntn_coverage.report import format_link_budget, pass_timeline
from ntn_coverage.scenarios import REGIONS, SCENARIOS, get_scenario
from ntn_coverage.stats import ecdf, tr38821_sc6_ka_leo600

WEB_DIR = Path(__file__).resolve().parent / "web"
PLOTLY_JS = Path(plotly.__file__).resolve().parent / "package_data" / "plotly.min.js"


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    return value


def _dumps(payload) -> bytes:
    return json.dumps(_clean(payload), allow_nan=False).encode("utf-8")


def _scenario_from_body(body: dict):
    scenario = get_scenario(str(body.get("scenario") or "leo600-s-handheld"))
    if body.get("environment"):
        scenario = scenario.with_environment(str(body["environment"]))
    if body.get("antenna"):
        scenario = scenario.with_antenna(str(body["antenna"]))
    if body.get("reuse") is not None:
        scenario = scenario.with_reuse(int(body["reuse"]))
    if body.get("coordination"):
        scenario = scenario.with_coordination(str(body["coordination"]))
    return scenario


MAX_PASS_SAMPLES = 2_000


def _run_params(body: dict) -> dict:
    region = str(body.get("region") or "romania")
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Choose one of: {', '.join(sorted(REGIONS))}")
    resolution = float(body.get("resolution_deg") or 0.25)
    if not math.isfinite(resolution) or resolution < MIN_RESOLUTION_DEG:
        raise ValueError(f"resolution_deg must be >= {MIN_RESOLUTION_DEG}")
    box = REGIONS[region]
    n_points = (int((box["lat_max"] - box["lat_min"]) / resolution) + 1) * (
        int((box["lon_max"] - box["lon_min"]) / resolution) + 1
    )
    if n_points > MAX_GRID_POINTS:
        raise ValueError(
            f"{region} at {resolution} deg is {n_points} points (max {MAX_GRID_POINTS}); use a coarser grid"
        )
    rain = float(body.get("rain_rate_mm_h") or 0.0)
    if not math.isfinite(rain) or rain < 0.0:
        raise ValueError("rain_rate_mm_h must be >= 0")
    return {
        "region": region,
        "resolution_deg": resolution,
        "t_s": float(body.get("t_s") or 0.0),
        "rain_rate_mm_h": rain,
        "indoor": bool(body.get("indoor") or False),
    }


def _grid_payload(grid) -> dict:
    x, y = ecdf(grid.sinr_db)
    return {
        "lats": grid.lats,
        "lons": grid.lons,
        "sinr_db": grid.sinr_db,
        "snr_db": grid.snr_db,
        "cir_db": grid.cir_db,
        "rate_mbps": grid.rate_mbps,
        "p_cover": grid.p_cover * 100.0,
        "elevation_deg": grid.elevation_deg,
        "n_visible": grid.n_visible,
        "summary": grid.summary(),
        "ecdf": {"x": x, "y": y},
        "title": grid.scenario.title,
        "reuse_factor": grid.scenario.reuse_factor,
    }


def _probe_payload(body: dict) -> dict:
    scenario = _scenario_from_body(body)
    params = _run_params(body)
    lat = float(body["lat"])
    lon = float(body["lon"])
    pred = predict_point(
        lat,
        lon,
        scenario,
        t_s=params["t_s"],
        rain_rate_mm_h=params["rain_rate_mm_h"],
        indoor=params["indoor"],
    )
    idx = int(pred.extras.get("serving_index", -1))
    beams = serving_beam_cluster(
        lat,
        lon,
        scenario,
        t_s=params["t_s"],
        serving_index=idx,
        serving_qr=(pred.extras["serving_q"], pred.extras["serving_r"]),
    )
    return {
        "lat": pred.lat_deg,
        "lon": pred.lon_deg,
        "covered": pred.covered,
        "snr_db": pred.snr_db,
        "sinr_db": pred.sinr_db,
        "cir_db": pred.cir_db,
        "cn0_dbhz": pred.cn0_dbhz,
        "elevation_deg": pred.elevation_deg,
        "slant_range_km": pred.slant_range_km,
        "path_loss_db": pred.path_loss_db,
        "fspl_db": pred.fspl_db,
        "doppler_hz": pred.doppler_hz,
        "n_visible": pred.n_visible,
        "serving_sat": pred.serving_sat,
        "rate_mbps": pred.rate_mbps,
        "p_cover": pred.p_cover,
        "extras": pred.extras,
        "budget": format_link_budget(pred),
        "beams": beams,
    }


def _pass_payload(body: dict) -> dict:
    scenario = _scenario_from_body(body)
    params = _run_params(body)
    duration_s = float(body.get("duration_s") or 1200.0)
    dt_s = float(body.get("dt_s") or 30.0)
    if not (dt_s > 0.0 and 0.0 <= duration_s / dt_s <= MAX_PASS_SAMPLES):
        raise ValueError(f"need dt_s > 0 and at most {MAX_PASS_SAMPLES} samples")
    rows = pass_timeline(
        float(body["lat"]),
        float(body["lon"]),
        scenario,
        t0_s=params["t_s"],
        duration_s=duration_s,
        dt_s=dt_s,
        rain_rate_mm_h=params["rain_rate_mm_h"],
        indoor=params["indoor"],
    )
    return {"rows": rows}


def _meta() -> dict:
    return {
        "scenarios": [
            {"name": name, "title": sc.title, "n_satellites": sc.constellation.n_satellites}
            for name, sc in SCENARIOS.items()
        ],
        "regions": {name: box for name, box in REGIONS.items()},
        "outline": ROMANIA_OUTLINE.tolist(),
        "cities": [{"lat": lat, "lon": lon, "name": name} for lat, lon, _r, name in ROMANIA_URBAN[:8]],
        "calibration": tr38821_sc6_ka_leo600(),
    }


_CACHE: dict[tuple, dict] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(body: dict, reuse: int) -> tuple:
    params = _run_params(body)
    return (
        str(body.get("scenario") or "leo600-s-handheld"),
        str(body.get("environment") or ""),
        str(body.get("antenna") or ""),
        int(reuse),
        str(body.get("coordination") or ""),
        params["region"],
        round(params["resolution_deg"], 5),
        round(params["t_s"], 3),
        round(params["rain_rate_mm_h"], 4),
        bool(params["indoor"]),
    )


def _run_one(body: dict, reuse: int) -> dict:
    key = _cache_key(body, reuse)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    scenario = _scenario_from_body({**body, "reuse": reuse})
    params = _run_params(body)
    t0 = time.perf_counter()
    grid = predict_coverage(scenario, **params)
    payload = _grid_payload(grid)
    payload["elapsed_s"] = time.perf_counter() - t0
    sat_state = propagate(scenario.constellation, params["t_s"])
    slat, slon = subsatellite_points(sat_state)
    nearby = (
        (slat >= grid.lats.min() - 6)
        & (slat <= grid.lats.max() + 6)
        & (slon >= grid.lons.min() - 6)
        & (slon <= grid.lons.max() + 6)
    )
    payload["sats"] = {
        "lat": slat[nearby],
        "lon": slon[nearby],
        "id": [str(v) for v in sat_state.sat_ids[nearby]],
    }
    with _CACHE_LOCK:
        _CACHE[key] = payload
        if len(_CACHE) > 12:
            _CACHE.pop(next(iter(_CACHE)))
    return payload


def handle_run(body: dict) -> dict:
    compare = bool(body.get("compare"))
    reuses = (1, 3) if compare else (int(body.get("reuse") or 3),)
    layers = {f"frf{rf}": _run_one(body, rf) for rf in reuses}
    params = _run_params(body)
    return {
        "compare": compare,
        "layers": layers,
        "region": params["region"],
        "resolution_deg": params["resolution_deg"],
        "t_s": params["t_s"],
    }


class WorkbenchHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        print("[workbench]", fmt % args, file=sys.stderr)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/plotly.min.js":
                self._send(200, PLOTLY_JS.read_bytes(), "application/javascript")
                return
            if path == "/api/meta":
                self._send(200, _dumps(_meta()), "application/json")
                return
            if path == "/api/health":
                self._send(200, _dumps({"ok": True}), "application/json")
                return
            self._send(404, b'{"error":"not found"}', "application/json")
        except Exception as exc:  # noqa: BLE001
            self._send(500, _dumps({"error": str(exc)}), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            body = self._read_json()
            if path == "/api/run":
                self._send(200, _dumps(handle_run(body)), "application/json")
                return
            if path == "/api/probe":
                self._send(200, _dumps(_probe_payload(body)), "application/json")
                return
            if path == "/api/pass":
                self._send(200, _dumps(_pass_payload(body)), "application/json")
                return
            self._send(404, b'{"error":"not found"}', "application/json")
        except (ValueError, KeyError, TypeError) as exc:
            self._send(400, _dumps({"error": str(exc)}), "application/json")
        except Exception as exc:  # noqa: BLE001
            self._send(500, _dumps({"error": str(exc)}), "application/json")


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class _Server(ThreadingHTTPServer):
        allow_reuse_address = True
        daemon_threads = True

    return _Server((host, int(port)), WorkbenchHandler)


def serve_workbench(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    httpd = make_server(host, port)
    url = f"http://{host}:{httpd.server_port}/"
    print(f"ntn-coverage GUI -> {url}", flush=True)
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping workbench.")
    finally:
        httpd.server_close()
