"""Named 3GPP NTN / 6G scenarios (TR 38.821 Set-1 RF + Walker LEO)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from ntn_coverage.channel import SCENE_ENVIRONMENTS, SceneEnvironment
from ntn_coverage.constants import (
    KA_BAND_BW_HZ,
    KA_DL_HZ,
    LEO600_ALTITUDE_M,
    LEO1200_ALTITUDE_M,
    S_BAND_BW_HZ,
    S_BAND_HZ,
)
from ntn_coverage.link_budget import HANDHELD, VSAT, Terminal
from ntn_coverage.orbit import ConstellationConfig, geo_europe, leo600_walker, leo1200_walker, starlink_shell1


@dataclass(frozen=True)
class RFConfig:
    band: str
    freq_hz: float
    bandwidth_hz: float
    eirp_density_dbw_per_mhz: float
    tx_gain_dbi: float
    aperture_m: float
    beam_diameter_km: float
    hpbw_deg: float


def set1_rf(orbit: str, band: str) -> RFConfig:
    """TR 38.821 Table 6.1.1.1-1 Set-1 downlink parameters."""
    key = (orbit, band)
    table = {
        ("leo600", "s"): RFConfig("s", S_BAND_HZ, S_BAND_BW_HZ, 34.0, 30.0, 2.0, 50.0, 4.4127),
        ("leo600", "ka"): RFConfig("ka", KA_DL_HZ, KA_BAND_BW_HZ, 4.0, 38.5, 0.5, 20.0, 1.7647),
        ("leo1200", "s"): RFConfig("s", S_BAND_HZ, S_BAND_BW_HZ, 40.0, 30.0, 2.0, 90.0, 4.4127),
        ("leo1200", "ka"): RFConfig("ka", KA_DL_HZ, KA_BAND_BW_HZ, 10.0, 38.5, 0.5, 40.0, 1.7647),
        ("geo", "s"): RFConfig("s", S_BAND_HZ, S_BAND_BW_HZ, 59.0, 51.0, 22.0, 250.0, 0.4011),
        ("geo", "ka"): RFConfig("ka", KA_DL_HZ, KA_BAND_BW_HZ, 40.0, 58.5, 5.0, 110.0, 0.1765),
    }
    if key not in table:
        raise KeyError(f"No Set-1 RF for {orbit}/{band}")
    return table[key]


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    constellation: ConstellationConfig
    rf: RFConfig
    terminal: Terminal
    environment: SceneEnvironment = "rural"
    min_elevation_deg: float = 10.0
    antenna_mode: str = "cellular"
    snr_threshold_db: float = 0.0
    reuse_factor: int = 3
    coordination: str = "exclusive"

    def __post_init__(self) -> None:
        _check("environment", self.environment, SCENE_ENVIRONMENTS)
        _check("antenna_mode", self.antenna_mode, ANTENNA_MODES)
        _check("coordination", self.coordination, COORDINATION_MODES)
        if int(self.reuse_factor) not in REUSE_FACTORS:
            raise ValueError(f"reuse_factor must be one of {REUSE_FACTORS}, got {self.reuse_factor!r}")

    def with_environment(self, environment: str) -> Scenario:
        return replace(self, environment=cast(SceneEnvironment, environment))

    def with_antenna(self, antenna_mode: str) -> Scenario:
        return replace(self, antenna_mode=antenna_mode)

    def with_reuse(self, reuse_factor: int) -> Scenario:
        return replace(self, reuse_factor=int(reuse_factor))

    def with_coordination(self, coordination: str) -> Scenario:
        return replace(self, coordination=coordination)


ANTENNA_MODES = ("cellular", "steered", "nadir")
COORDINATION_MODES = ("exclusive", "uncoordinated")
REUSE_FACTORS = (1, 3)


def _check(field_name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")


def _leo(
    name: str, title: str, altitude_m: float, band: str, terminal: Terminal, env: SceneEnvironment
) -> Scenario:
    if altitude_m == LEO600_ALTITUDE_M:
        orbit_key = "leo600"
        cst = leo600_walker()
    elif altitude_m == LEO1200_ALTITUDE_M:
        orbit_key = "leo1200"
        cst = leo1200_walker()
    else:
        raise ValueError(altitude_m)
    return Scenario(name, title, cst, set1_rf(orbit_key, band), terminal, env)


SCENARIOS: dict[str, Scenario] = {
    "leo600-s-handheld": _leo(
        "leo600-s-handheld",
        "LEO 600 km · S-band handheld (3GPP Set-1)",
        LEO600_ALTITUDE_M,
        "s",
        HANDHELD,
        "rural",
    ),
    "leo600-s-urban": _leo(
        "leo600-s-urban",
        "LEO 600 km · S-band handheld · urban",
        LEO600_ALTITUDE_M,
        "s",
        HANDHELD,
        "urban",
    ),
    "leo600-s-mixed": _leo(
        "leo600-s-mixed",
        "LEO 600 km · S-band handheld · mixed land cover",
        LEO600_ALTITUDE_M,
        "s",
        HANDHELD,
        "mixed",
    ),
    "leo600-ka-vsat": _leo(
        "leo600-ka-vsat",
        "LEO 600 km · Ka-band VSAT (3GPP Set-1)",
        LEO600_ALTITUDE_M,
        "ka",
        VSAT,
        "rural",
    ),
    "leo1200-s-handheld": _leo(
        "leo1200-s-handheld",
        "LEO 1200 km · S-band handheld (3GPP Set-1)",
        LEO1200_ALTITUDE_M,
        "s",
        HANDHELD,
        "rural",
    ),
    "geo-ka-vsat": Scenario(
        name="geo-ka-vsat",
        title="GEO · Ka-band VSAT (3GPP Set-1, Europe slots)",
        constellation=geo_europe(),
        rf=set1_rf("geo", "ka"),
        terminal=VSAT,
        environment="rural",
        min_elevation_deg=10.0,
        antenna_mode="cellular",
        snr_threshold_db=0.0,
        reuse_factor=3,
    ),
    "geo-s-handheld": Scenario(
        name="geo-s-handheld",
        title="GEO · S-band handheld (3GPP Set-1, Europe slots)",
        constellation=geo_europe(),
        rf=set1_rf("geo", "s"),
        terminal=HANDHELD,
        environment="rural",
        min_elevation_deg=10.0,
        antenna_mode="cellular",
        snr_threshold_db=0.0,
        reuse_factor=3,
    ),
    "starlink-s-handheld": Scenario(
        name="starlink-s-handheld",
        title="Starlink-like 72×22 @ 550 km · 3GPP S-band RF",
        constellation=starlink_shell1(),
        rf=set1_rf("leo600", "s"),
        terminal=HANDHELD,
        environment="rural",
        min_elevation_deg=10.0,
        antenna_mode="cellular",
        snr_threshold_db=0.0,
        reuse_factor=3,
    ),
}


REGIONS: dict[str, dict[str, float]] = {
    "romania": {"lat_min": 43.5, "lat_max": 48.3, "lon_min": 20.2, "lon_max": 29.8},
    "europe": {"lat_min": 35.0, "lat_max": 72.0, "lon_min": -12.0, "lon_max": 40.0},
    "bucharest": {"lat_min": 44.2, "lat_max": 44.7, "lon_min": 25.8, "lon_max": 26.3},
    "world": {"lat_min": -70.0, "lat_max": 70.0, "lon_min": -180.0, "lon_max": 180.0},
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario '{name}'. Choose one of: {known}")
    return SCENARIOS[name]
