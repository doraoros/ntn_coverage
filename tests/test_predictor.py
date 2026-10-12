"""3GPP TR 38.811 / 38.821 unit checks."""

import numpy as np
import pytest

from ntn_coverage.antennas import bessel_relative_gain_db
from ntn_coverage.channel import SFCL, fspl_db, los_probability, path_loss_components
from ntn_coverage.constants import EARTH_RADIUS_M
from ntn_coverage.coverage import predict_coverage, predict_point
from ntn_coverage.geometry import ecef_from_ll, link_geometry, ll_from_ecef, slant_range_from_elevation
from ntn_coverage.link_budget import HANDHELD, VSAT, snr_from_eirp_density
from ntn_coverage.orbit import geo_europe, leo600_walker, propagate
from ntn_coverage.scenarios import get_scenario


def test_nadir_geometry():
    ue = ecef_from_ll(0.0, 0.0)
    sat = ecef_from_ll(0.0, 0.0, alt_m=600_000.0)
    geo = link_geometry(ue, sat.reshape(1, 3))
    assert geo["elevation_deg"][0] == np.clip(geo["elevation_deg"][0], 89.9, 90.0)
    assert abs(float(geo["slant_range_m"][0]) - 600_000.0) < 1.0
    assert float(geo["off_nadir_deg"][0]) < 0.05


def test_slant_range_formula_matches_ecef():
    elev = 30.0
    h = 600_000.0
    d = float(slant_range_from_elevation(elev, h))
    # Place the satellite on the equator at the Earth-central angle implied by d.
    ue = ecef_from_ll(0.0, 0.0)
    re, rs = EARTH_RADIUS_M, EARTH_RADIUS_M + h
    cos_g = (re * re + rs * rs - d * d) / (2.0 * re * rs)
    gamma = np.rad2deg(np.arccos(np.clip(cos_g, -1, 1)))
    sat = ecef_from_ll(0.0, gamma, alt_m=h)
    geo = link_geometry(ue, sat.reshape(1, 3))
    assert abs(float(geo["slant_range_m"][0]) - d) < 2000.0
    assert abs(float(geo["elevation_deg"][0]) - elev) < 0.4
    assert float(geo["off_nadir_deg"][0]) < 90.0


def test_fspl_tr38811():
    # d in metres, fc in GHz: 32.45 + 20log10(fc) + 20log10(d)
    got = float(fspl_db(600_000.0, 2e9))
    expected = 32.45 + 20 * np.log10(2.0) + 20 * np.log10(600_000.0)
    assert abs(got - expected) < 1e-6
    assert abs(got - 154.034) < 0.01


def test_los_and_clutter_tables():
    assert los_probability(30.0, "rural") == 0.919
    assert los_probability(10.0, "dense_urban") == 0.282
    row = SFCL["rural"][2]  # 30°
    assert row[2] == 18.42
    assert row[5] == 21.9


def test_bessel_hpbw_s_band_2m():
    angles = np.linspace(0.0, 5.0, 501)
    g = bessel_relative_gain_db(angles, diameter_m=2.0, freq_hz=2e9)
    hpbw_half = angles[np.where(g <= -3.0)[0][0]]
    assert abs(2.0 * hpbw_half - 4.4127) < 0.08


def test_terminal_g_over_t():
    assert abs(VSAT.g_over_t_db_k - 15.9) < 0.1
    assert abs(HANDHELD.g_over_t_db_k - (-31.62)) < 0.05


def test_leo_ka_vsat_snr_near_3gpp_link_budget():
    """TR 38.821 Table 6.1.3.3-1 case SC6: LEO-600 Ka VSAT, ~8.5 dB at 30°."""
    d = float(slant_range_from_elevation(30.0, 600_000.0))
    losses = path_loss_components(
        d, 30.0, 20e9, "rural", latitude_deg=45.0, rain_rate_mm_h=0.0
    )
    # Calibration link budget is clear-sky LOS: drop mean clutter, keep gases+scint.
    pl = float(losses["fspl_db"] + losses["gases_db"] + losses["scintillation_db"])
    snr = float(snr_from_eirp_density(4.0, pl, VSAT.g_over_t_db_k, extra_loss_db=0.0))
    assert abs(float(losses["fspl_db"]) - 179.1) < 0.2
    assert abs(snr - 8.5) < 1.5


def test_walker_and_geo_counts():
    leo = propagate(leo600_walker(), 0.0)
    assert leo.n_satellites == 360
    geo = propagate(geo_europe(), 0.0)
    assert geo.n_satellites == 3
    _lat, _lon, alt = ll_from_ecef(geo.ecef)
    assert np.allclose(alt, 35_786_000.0, atol=50.0)


def test_bucharest_point_is_served():
    scenario = get_scenario("leo600-s-handheld")
    pred = predict_point(44.4268, 26.1025, scenario, t_s=0.0)
    assert pred.n_visible >= 1
    assert pred.covered
    assert np.isfinite(pred.sinr_db)
    assert pred.sinr_db <= pred.snr_db + 0.25
    assert 0.0 <= pred.p_cover <= 1.0
    assert pred.elevation_deg >= scenario.min_elevation_deg
    assert pred.rate_mbps > 0


def test_romania_grid_shape():
    scenario = get_scenario("leo600-s-handheld")
    grid = predict_coverage(scenario, region="romania", resolution_deg=0.5, t_s=0.0)
    assert grid.snr_db.ndim == 2
    assert grid.snr_db.shape[0] == grid.lats.size
    assert grid.snr_db.shape[1] == grid.lons.size
    assert grid.sinr_db.shape == grid.snr_db.shape
    assert grid.summary()["coverage_fraction"] > 0.5


def test_wgs84_roundtrip():
    lat0, lon0, alt0 = 44.43, 26.10, 120.0
    ecef = ecef_from_ll(lat0, lon0, alt0)
    lat, lon, alt = ll_from_ecef(ecef)
    assert abs(float(lat) - lat0) < 1e-6
    assert abs(float(lon) - lon0) < 1e-6
    assert abs(float(alt) - alt0) < 0.5


def test_hex_disk_has_19_beams():
    from ntn_coverage.beams import hex_disk

    q, r, u, v = hex_disk(2, 0.0668)
    assert q.size == 19
    assert (0, 0) in list(zip(q.tolist(), r.tolist(), strict=False))


def test_hex_reuse_colors_differ_on_neighbors():
    from ntn_coverage.beams import beam_color

    origin = int(beam_color(0, 0, 3))
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, 1), (1, -1)]
    for q, r in neighbors:
        assert int(beam_color(q, r, 3)) != origin


def test_reuse3_not_worse_than_reuse1_in_bucharest():
    base = get_scenario("leo600-s-handheld")
    p1 = predict_point(44.4268, 26.1025, base.with_reuse(1))
    p3 = predict_point(44.4268, 26.1025, base.with_reuse(3))
    assert np.isfinite(p1.sinr_db) and np.isfinite(p3.sinr_db)
    assert p3.sinr_db >= p1.sinr_db - 0.1


@pytest.mark.parametrize("reuse", [1, 3])
def test_point_sinr_independent_of_grid_region(reuse):
    scenario = get_scenario("leo600-s-handheld").with_reuse(reuse)
    lat, lon = 44.5, 26.0
    point = predict_point(lat, lon, scenario).sinr_db
    small = {"lat_min": lat, "lat_max": lat + 0.2, "lon_min": lon, "lon_max": lon + 0.2}
    wide = {"lat_min": lat - 10.0, "lat_max": lat + 10.0, "lon_min": lon - 20.0, "lon_max": lon + 20.0}
    for region, res in ((small, 0.2), ("romania", 0.2), (wide, 1.0)):
        grid = predict_coverage(scenario, region=region, resolution_deg=res)
        i = int(np.argmin(np.abs(grid.lats - lat)))
        j = int(np.argmin(np.abs(grid.lons - lon)))
        assert abs(grid.lats[i] - lat) < 1e-6 and abs(grid.lons[j] - lon) < 1e-6
        assert float(grid.sinr_db[i, j]) == pytest.approx(point, abs=1e-6), region


def test_mixed_environment_runs_and_is_worse_in_cities():
    scenario = get_scenario("leo600-s-mixed")
    city = predict_point(44.4268, 26.1025, scenario)
    rural = predict_point(44.4268, 26.1025, scenario.with_environment("rural"))
    assert np.isfinite(city.sinr_db)
    assert city.path_loss_db > rural.path_loss_db
    grid = predict_coverage(scenario, region="bucharest", resolution_deg=0.25)
    assert np.isfinite(grid.sinr_db).all()


def test_scenario_rejects_unknown_options():
    base = get_scenario("leo600-s-handheld")
    for bad in (
        lambda: base.with_environment("forest"),
        lambda: base.with_antenna("phased"),
        lambda: base.with_coordination("magic"),
        lambda: base.with_reuse(4),
    ):
        with pytest.raises(ValueError):
            bad()


def test_rain_matches_itu_p838_and_p618():
    from ntn_coverage.channel import rain_attenuation_db, rain_specific_attenuation_db_km

    # P.838-3 at 20 GHz, circular pol.: k ~ 0.0939, alpha ~ 1.020.
    assert rain_specific_attenuation_db_km(20e9, 25.0) == pytest.approx(2.50, abs=0.05)
    assert rain_specific_attenuation_db_km(2e9, 25.0) < 0.01
    # Uncapped vertical factor is 1.14 here, which P.618 forbids; the fade is ~13 dB.
    ka = float(rain_attenuation_db(30.0, 20e9, 25.0, latitude_deg=45.0))
    assert ka == pytest.approx(13.0, abs=0.3)
    assert float(rain_attenuation_db(30.0, 2e9, 25.0, latitude_deg=45.0)) < 0.2
    assert float(rain_attenuation_db(30.0, 20e9, 0.0)) == 0.0
    assert float(rain_attenuation_db(60.0, 20e9, 25.0)) < ka
    # A light rate must not grow a path longer than the rain layer.
    gamma = rain_specific_attenuation_db_km(20e9, 1.0)
    from ntn_coverage.channel import rain_height_km

    geometric_km = float(rain_height_km(45.0)) / np.sin(np.deg2rad(30.0))
    light = float(rain_attenuation_db(30.0, 20e9, 1.0, latitude_deg=45.0))
    assert light <= gamma * geometric_km + 1e-9


def test_availability_summary_is_time_averaged():
    scenario = get_scenario("leo600-s-handheld").with_reuse(1)
    grid = predict_coverage(scenario, region="romania", resolution_deg=1.0, mode="availability", n_times=4)
    s = grid.summary()
    assert s["coverage_fraction"] == pytest.approx(float(np.mean(grid.availability)))
    assert s["ever_covered_fraction"] == pytest.approx(float(np.mean(grid.availability > 0)))
    assert s["coverage_fraction"] <= s["ever_covered_fraction"]


def test_availability_fields_match_sample_means_not_peaks():
    scenario = get_scenario("leo600-s-handheld").with_reuse(3)
    box = {"lat_min": 44.4, "lat_max": 44.6, "lon_min": 26.0, "lon_max": 26.2}
    n_times = 3
    grid = predict_coverage(scenario, region=box, resolution_deg=0.2, mode="availability", n_times=n_times)
    times = np.linspace(0.0, scenario.constellation.period_s, n_times, endpoint=False)
    sinr, rate = [], []
    for t in times:
        sample = predict_coverage(scenario, region=box, resolution_deg=0.2, t_s=float(t))
        sinr.append(sample.sinr_db)
        rate.append(np.where(np.isfinite(sample.rate_mbps), sample.rate_mbps, 0.0))
    sinr = np.stack(sinr)
    np.testing.assert_allclose(grid.sinr_db, np.nanmean(sinr, axis=0), equal_nan=True)
    np.testing.assert_allclose(grid.rate_mbps, np.mean(np.stack(rate), axis=0))
    assert float(np.nanmax(sinr)) > float(np.nanmax(grid.sinr_db)) + 0.1


def test_fixed_terminal_doppler_has_no_earth_rate():
    ue = ecef_from_ll(0.0, 0.0)
    sat = ecef_from_ll(0.0, 10.0, alt_m=600_000.0).reshape(1, 3)
    stopped = link_geometry(ue, sat, np.zeros((1, 3)))
    assert abs(float(stopped["range_rate_mps"][0])) < 1e-6
    closing = link_geometry(ue, sat, np.array([[0.0, -1000.0, 0.0]]))
    assert float(closing["range_rate_mps"][0]) < 0.0
    assert float(closing["doppler_over_f"][0]) > 0.0
    geo = predict_point(44.43, 26.10, get_scenario("geo-s-handheld"))
    assert abs(geo.doppler_hz) < 1e-3


def test_grid_rejects_oversized_request():
    scenario = get_scenario("leo600-s-handheld")
    with pytest.raises(ValueError, match="coarser"):
        predict_coverage(scenario, region="world", resolution_deg=0.5)


def test_grid_chunking_matches_single_pass(monkeypatch):
    import ntn_coverage.coverage as cov

    scenario = get_scenario("leo600-s-handheld").with_reuse(1)
    full = predict_coverage(scenario, region="bucharest", resolution_deg=0.1)
    monkeypatch.setattr(cov, "MAX_UE_SAT_PAIRS", 1)
    chunked = predict_coverage(scenario, region="bucharest", resolution_deg=0.1)
    np.testing.assert_allclose(chunked.sinr_db, full.sinr_db)
    np.testing.assert_array_equal(chunked.serving_index, full.serving_index)


def test_mcs_is_monotonic():
    from ntn_coverage.throughput import spectral_efficiency

    se = spectral_efficiency(np.array([-8.0, 0.0, 10.0, 20.0]))
    assert se[0] <= se[1] <= se[2] <= se[3]
    assert se[0] == 0.0


def test_mixed_landcover_flags_bucharest():
    from ntn_coverage.landcover import urban_mask

    assert bool(urban_mask(44.43, 26.10))
    assert not bool(urban_mask(45.15, 25.0))
    assert bool(urban_mask(51.51, -0.12))
    assert bool(urban_mask(40.71, -74.00))
    assert not bool(urban_mask(0.0, -30.0))


def test_ecdf_and_calibration():
    from ntn_coverage.stats import ecdf, tr38821_sc6_ka_leo600

    x, y = ecdf([3.0, 1.0, 2.0])
    assert list(x) == [1.0, 2.0, 3.0]
    x2, y2 = ecdf(np.array([[3.0, 1.0], [np.nan, 2.0]]))
    assert list(x2) == [1.0, 2.0, 3.0]
    assert abs(float(y[-1]) - 1.0) < 1e-12
    cal = tr38821_sc6_ka_leo600()
    assert cal["fspl_error_db"] < 0.3
    assert cal["snr_error_db"] < 1.5


def test_starlink_like_count():
    from ntn_coverage.orbit import starlink_shell1

    assert starlink_shell1().n_satellites == 72 * 22


def test_tle_iss_optional():
    pytest.importorskip("sgp4")
    from sgp4.api import Satrec

    from ntn_coverage.geometry import gmst_rad, ll_from_ecef
    from ntn_coverage.tle import propagate_tle

    tle = """
ISS (ZARYA)
1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991
2 25544  51.6431 211.2001 0007417  17.6667  85.6398 15.50103472202482
"""
    state = propagate_tle(tle, 0.0)
    assert state.n_satellites == 1
    assert state.ecef.shape == (1, 3)
    lines = [ln.strip() for ln in tle.splitlines() if ln.strip()]
    sat = Satrec.twoline2rv(lines[1], lines[2])
    _err, r, _v = sat.sgp4_tsince(0.0)
    _lat_teme, lon_teme, _alt = ll_from_ecef(np.asarray(r) * 1000.0)
    lat, lon, _alt_ecef = ll_from_ecef(state.ecef[0])
    assert abs(float(lat) - float(_lat_teme)) < 1e-6
    shift = (float(lon) - float(lon_teme) + 180.0) % 360.0 - 180.0
    expected = -np.rad2deg(gmst_rad(sat.jdsatepoch + sat.jdsatepochF))
    expected = (expected + 180.0) % 360.0 - 180.0
    assert shift == pytest.approx(expected, abs=1e-4)


def test_tle_bad_element_is_reported():
    pytest.importorskip("sgp4")
    from ntn_coverage.tle import propagate_tle

    with pytest.raises(RuntimeError, match="SGP4 failed"):
        propagate_tle("BAD SAT\n1 not-a-tle\n2 not-a-tle\n")


def test_gmst_at_j2000():
    from ntn_coverage.geometry import gmst_rad

    assert np.rad2deg(gmst_rad(2451545.0)) == pytest.approx(280.460618375, abs=1e-6)


def test_link_budget_text_and_pass():
    from ntn_coverage.report import format_link_budget, pass_timeline
    from ntn_coverage.scenarios import get_scenario

    scenario = get_scenario("leo600-s-handheld").with_reuse(3)
    pred = predict_point(44.4268, 26.1025, scenario)
    line = format_link_budget(pred)
    assert pred.serving_sat in line
    assert "SINR" in line
    rows = pass_timeline(44.4268, 26.1025, scenario, t0_s=0.0, duration_s=60.0, dt_s=60.0)
    assert len(rows) == 2
    assert "serving_sat" in rows[0]


def test_beam_cluster_bucharest():
    from ntn_coverage.footprints import serving_beam_cluster

    scenario = get_scenario("leo600-s-handheld").with_reuse(3)
    pred = predict_point(44.4268, 26.1025, scenario)
    beams = serving_beam_cluster(
        44.4268, 26.1025, scenario, serving_index=pred.extras["serving_index"]
    )
    assert len(beams) >= 7
    assert sum(1 for b in beams if b["serving"]) == 1
    ring = beams[0]
    assert ring["lat"][0] == ring["lat"][-1]
    assert all(abs(lat - 44.43) < 8 for lat in ring["lat"])


def test_wgs84_ray_nadir():
    from ntn_coverage.geometry import ecef_from_ll, intersect_wgs84, ll_from_ecef

    sat = ecef_from_ll(0.0, 0.0, alt_m=600_000.0)
    hit = intersect_wgs84(sat, -sat)
    assert hit is not None
    lat, lon, alt = ll_from_ecef(hit)
    assert abs(float(lat)) < 0.05
    assert abs(float(lon)) < 0.05
    assert abs(float(alt)) < 200.0


def test_grid_csv_npz(tmp_path):
    from ntn_coverage.export import write_grid_csv, write_grid_npz

    scenario = get_scenario("leo600-s-handheld")
    grid = predict_coverage(scenario, region="bucharest", resolution_deg=0.25)
    npz = write_grid_npz(grid, tmp_path / "g.npz")
    csv = write_grid_csv(grid, tmp_path / "g.csv")
    data = np.load(npz)
    assert "sinr_db" in data
    text = csv.read_text(encoding="utf-8").splitlines()
    assert text[0].startswith("lat,lon,sinr_db")


def test_workbench_http():
    import json
    import threading
    import urllib.request

    from ntn_coverage.workbench import make_server

    httpd = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_port}"
        html = urllib.request.urlopen(base + "/", timeout=10).read()
        assert b"ntn-coverage" in html
        assert b"NTN COVERAGE WORKBENCH" not in html
        meta = json.loads(urllib.request.urlopen(base + "/api/meta", timeout=10).read())
        names = [row["name"] for row in meta["scenarios"]]
        assert "leo600-s-handheld" in names
        req = urllib.request.Request(
            base + "/api/probe",
            data=json.dumps({"lat": 44.4268, "lon": 26.1025, "reuse": 3}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        probe = json.loads(urllib.request.urlopen(req, timeout=30).read())
        assert probe["n_visible"] >= 1
        assert "SINR" in probe["budget"]
        assert len(probe["beams"]) >= 1
        assert any(beam["serving"] for beam in probe["beams"])

        import urllib.error

        too_big = urllib.request.Request(
            base + "/api/run",
            data=json.dumps({"region": "world", "resolution_deg": 0.05}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(too_big, timeout=10)
        assert err.value.code == 400
        assert "coarser" in json.loads(err.value.read())["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()


