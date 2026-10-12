"""3GPP TR 38.811 large-scale NTN channel (path loss, LOS, clutter, gases)."""

from __future__ import annotations

from typing import Literal, get_args

import numpy as np

Environment = Literal["dense_urban", "urban", "suburban", "rural"]
# "mixed" blends rural and urban per pixel from the land-cover mask.
SceneEnvironment = Literal["dense_urban", "urban", "suburban", "rural", "mixed"]
SCENE_ENVIRONMENTS: tuple[str, ...] = get_args(SceneEnvironment)

ELEV_REF_DEG = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])

# TR 38.811 Table 6.6.1-1
P_LOS = {
    "dense_urban": np.array([0.282, 0.331, 0.398, 0.468, 0.537, 0.612, 0.738, 0.820, 0.981]),
    "urban": np.array([0.246, 0.386, 0.493, 0.613, 0.726, 0.805, 0.919, 0.968, 0.992]),
    "suburban": np.array([0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998]),
    "rural": np.array([0.782, 0.869, 0.919, 0.929, 0.935, 0.940, 0.949, 0.952, 0.998]),
}

# TR 38.811 Tables 6.6.2-1..3: σ_SF LOS, σ_SF NLOS, CL NLOS for S then Ka.
# Columns: S_LOS_sf, S_NLOS_sf, S_CL, Ka_LOS_sf, Ka_NLOS_sf, Ka_CL
SFCL = {
    "dense_urban": np.array(
        [
            [3.5, 15.5, 34.3, 2.9, 17.1, 44.3],
            [3.4, 13.9, 30.9, 2.4, 17.1, 39.9],
            [2.9, 12.4, 29.0, 2.7, 15.6, 37.5],
            [3.0, 11.7, 27.7, 2.4, 14.6, 35.8],
            [3.1, 10.6, 26.8, 2.4, 14.2, 34.6],
            [2.7, 10.5, 26.2, 2.7, 12.6, 33.8],
            [2.5, 10.1, 25.8, 2.6, 12.1, 33.3],
            [2.3, 9.2, 25.5, 2.8, 12.3, 33.0],
            [1.2, 9.2, 25.5, 0.6, 12.3, 32.9],
        ]
    ),
    "urban": np.array(
        [
            [4.0, 6.0, 34.3, 4.0, 6.0, 44.3],
            [4.0, 6.0, 30.9, 4.0, 6.0, 39.9],
            [4.0, 6.0, 29.0, 4.0, 6.0, 37.5],
            [4.0, 6.0, 27.7, 4.0, 6.0, 35.8],
            [4.0, 6.0, 26.8, 4.0, 6.0, 34.6],
            [4.0, 6.0, 26.2, 4.0, 6.0, 33.8],
            [4.0, 6.0, 25.8, 4.0, 6.0, 33.3],
            [4.0, 6.0, 25.5, 4.0, 6.0, 33.0],
            [4.0, 6.0, 25.5, 4.0, 6.0, 32.9],
        ]
    ),
    "suburban": np.array(
        [
            [1.79, 8.93, 19.52, 1.9, 10.7, 29.5],
            [1.14, 9.08, 18.17, 1.6, 10.0, 24.6],
            [1.14, 8.78, 18.42, 1.9, 11.2, 21.9],
            [0.92, 10.25, 18.28, 2.3, 11.6, 20.0],
            [1.42, 10.56, 18.63, 2.7, 11.8, 18.7],
            [1.56, 10.74, 17.68, 3.1, 10.8, 17.8],
            [0.85, 10.17, 16.50, 3.0, 10.8, 17.2],
            [0.72, 11.52, 16.30, 3.6, 10.8, 16.9],
            [0.72, 11.52, 16.30, 0.4, 10.8, 16.8],
        ]
    ),
}
SFCL["rural"] = SFCL["suburban"]

# Approximate ITU-R P.676 zenith gaseous attenuation (standard atmosphere), dB.
_GAS_F_GHZ = np.array(
    [1.0, 2.0, 4.0, 6.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 35.0, 40.0, 45.0, 50.0]
)
_GAS_ZENITH_DB = np.array(
    [0.030, 0.035, 0.040, 0.050, 0.080, 0.10, 0.15, 0.28, 0.40, 1.05, 0.45, 0.28, 0.24, 0.22, 0.28, 0.40, 0.55, 1.20]
)


# ITU-R P.838-3 Table 5: k_H, α_H, k_V, α_V.
_RAIN_F_GHZ = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])
_RAIN_KH = np.array(
    [2.59e-5, 8.47e-5, 1.071e-4, 7.056e-4, 4.115e-3, 1.217e-2, 2.386e-2, 4.481e-2,
     9.164e-2, 0.1571, 0.2403, 0.3374, 0.4431, 0.5521, 0.6600]
)
_RAIN_AH = np.array(
    [0.9691, 1.0664, 1.6009, 1.5900, 1.3905, 1.2571, 1.1825, 1.1233,
     1.0568, 0.9991, 0.9485, 0.9047, 0.8673, 0.8355, 0.8084]
)
_RAIN_KV = np.array(
    [3.08e-5, 9.98e-5, 2.461e-4, 4.878e-4, 3.450e-3, 1.129e-2, 2.455e-2, 5.008e-2,
     9.611e-2, 0.1533, 0.2291, 0.3224, 0.4274, 0.5375, 0.6472]
)
_RAIN_AV = np.array(
    [0.8592, 0.9490, 1.2476, 1.5728, 1.3797, 1.2156, 1.1216, 1.0440,
     0.9847, 0.9491, 0.9129, 0.8761, 0.8421, 0.8123, 0.7871]
)


def fspl_db(distance_m, freq_hz) -> np.ndarray:
    """TR 38.811 eq. (6.6-2): d in metres, fc in GHz."""
    d = np.clip(np.asarray(distance_m, dtype=float), 1.0, None)
    fc_ghz = np.asarray(freq_hz, dtype=float) / 1e9
    return 32.45 + 20.0 * np.log10(fc_ghz) + 20.0 * np.log10(d)


def _nearest_elev_index(elevation_deg) -> np.ndarray:
    elev = np.asarray(elevation_deg, dtype=float)
    elev = np.clip(elev, 10.0, 90.0)
    idx = np.argmin(np.abs(ELEV_REF_DEG[:, None] - elev.ravel()[None, :]), axis=0)
    return idx.reshape(elev.shape)


def los_probability(elevation_deg, environment: Environment) -> np.ndarray:
    table = P_LOS[environment]
    return table[_nearest_elev_index(elevation_deg)]


def _band_weights(freq_hz: float) -> tuple[float, float]:
    """Linear blend in log-frequency between 2 GHz (S) and 20 GHz (Ka)."""
    f = float(freq_hz)
    if f <= 2e9:
        return 1.0, 0.0
    if f >= 20e9:
        return 0.0, 1.0
    t = (np.log10(f) - np.log10(2e9)) / (np.log10(20e9) - np.log10(2e9))
    return 1.0 - t, t


def clutter_and_sf(elevation_deg, freq_hz: float, environment: Environment) -> dict[str, np.ndarray]:
    table = SFCL[environment]
    rows = table[_nearest_elev_index(elevation_deg)]
    w_s, w_ka = _band_weights(freq_hz)
    los_sf = w_s * rows[..., 0] + w_ka * rows[..., 3]
    nlos_sf = w_s * rows[..., 1] + w_ka * rows[..., 4]
    nlos_cl = w_s * rows[..., 2] + w_ka * rows[..., 5]
    return {
        "los_sf_sigma_db": los_sf,
        "nlos_sf_sigma_db": nlos_sf,
        "nlos_clutter_db": nlos_cl,
    }


def gaseous_attenuation_db(elevation_deg, freq_hz: float) -> np.ndarray:
    """TR 38.811 §6.6.4: A_zenith(f) / sin(α)."""
    elev = np.clip(np.asarray(elevation_deg, dtype=float), 0.1, 90.0)
    a_zenith = float(np.interp(freq_hz / 1e9, _GAS_F_GHZ, _GAS_ZENITH_DB))
    return a_zenith / np.sin(np.deg2rad(elev))


def scintillation_loss_db(elevation_deg, freq_hz: float, latitude_deg) -> np.ndarray:
    """Ionospheric (low/high latitude, f<6 GHz) + tropospheric scintillation.

    Calibration in TR 38.821 sets ionospheric loss to 0 between 20° and 60°
    latitude. Outside that belt, 2.2 dB at 4 GHz scaled as (4 GHz / f)^1.5.
    """
    elev = np.clip(np.asarray(elevation_deg, dtype=float), 0.1, 90.0)
    lat = np.abs(np.asarray(latitude_deg, dtype=float))
    f = float(freq_hz)
    iono = np.zeros(np.broadcast(elev, lat).shape, dtype=float)
    if f < 6e9:
        scale = (4e9 / f) ** 1.5
        mask = (lat < 20.0) | (lat > 60.0)
        iono = np.where(mask, 2.2 * scale, 0.0)
        iono = np.broadcast_to(iono, np.broadcast(elev, lat).shape).copy()
    tropo = 0.02 * (f / 4e9) ** 1.3 / np.sin(np.deg2rad(elev))
    return iono + tropo


def rain_specific_attenuation_db_km(freq_hz: float, rain_rate_mm_h: float) -> float:
    """ITU-R P.838-3 specific attenuation γ_R (dB/km), circular polarisation."""
    f_ghz = float(np.clip(float(freq_hz) / 1e9, _RAIN_F_GHZ[0], _RAIN_F_GHZ[-1]))
    log_f = np.log10(_RAIN_F_GHZ)
    lf = np.log10(f_ghz)
    k_h = 10.0 ** np.interp(lf, log_f, np.log10(_RAIN_KH))
    k_v = 10.0 ** np.interp(lf, log_f, np.log10(_RAIN_KV))
    a_h = float(np.interp(lf, log_f, _RAIN_AH))
    a_v = float(np.interp(lf, log_f, _RAIN_AV))
    k = 0.5 * (k_h + k_v)
    alpha = (k_h * a_h + k_v * a_v) / (2.0 * k)
    return float(k * max(float(rain_rate_mm_h), 0.0) ** alpha)


def rain_height_km(latitude_deg) -> np.ndarray:
    """ITU-R P.839-2 rain height from latitude."""
    lat = np.asarray(latitude_deg, dtype=float)
    return np.select(
        [lat >= 23.0, lat >= -21.0, lat >= -71.0],
        [5.0 - 0.075 * (lat - 23.0), np.full(lat.shape, 5.0), 5.0 + 0.1 * (lat + 21.0)],
        default=0.0,
    )


def rain_attenuation_db(
    elevation_deg, freq_hz: float, rain_rate_mm_h: float, latitude_deg=45.0
) -> np.ndarray:
    """Slant-path fade while rain is falling at ``rain_rate_mm_h``.

    γ_R is ITU-R P.838-3. The rainy path is the geometric path through the
    P.839 rain layer, shortened by the ITU-R P.618 horizontal and vertical
    reduction factors. Both factors are capped at 1: this is the fade at the
    supplied snapshot rate, not the annual attenuation exceeded 0.01% of the
    time, so the path is never longer than the rain layer.
    """
    elev = np.clip(np.asarray(elevation_deg, dtype=float), 5.0, 90.0)
    lat = np.asarray(latitude_deg, dtype=float)
    shape = np.broadcast(elev, lat).shape
    if rain_rate_mm_h <= 0.0:
        return np.zeros(shape)
    elev, lat = np.broadcast_to(elev, shape), np.broadcast_to(lat, shape)
    f_ghz = float(freq_hz) / 1e9
    gamma = rain_specific_attenuation_db_km(freq_hz, rain_rate_mm_h)
    h_r = np.clip(rain_height_km(lat), 0.0, None)
    sin_e = np.sin(np.deg2rad(elev))
    l_s = h_r / sin_e
    l_g = l_s * np.cos(np.deg2rad(elev))
    r01 = 1.0 / (1.0 + 0.78 * np.sqrt(l_g * gamma / f_ghz) - 0.38 * (1.0 - np.exp(-2.0 * l_g)))
    r01 = np.minimum(r01, 1.0)
    zeta = np.rad2deg(np.arctan2(h_r, l_g * r01))
    l_r = np.where(zeta > elev, l_g * r01 / np.cos(np.deg2rad(elev)), l_s)
    chi = np.where(np.abs(lat) < 36.0, 36.0 - np.abs(lat), 0.0)
    denom = 1.0 + np.sqrt(sin_e) * (
        31.0 * (1.0 - np.exp(-elev / (1.0 + chi))) * np.sqrt(l_r * gamma) / f_ghz**2 - 0.45
    )
    v01 = np.where(denom > 0.0, 1.0 / denom, 1.0)
    v01 = np.clip(v01, 0.0, 1.0)
    return gamma * l_r * v01


def path_loss_components(
    slant_range_m,
    elevation_deg,
    freq_hz: float,
    environment: Environment,
    latitude_deg,
    rain_rate_mm_h: float = 0.0,
    indoor: bool = False,
) -> dict[str, np.ndarray]:
    """Mean large-scale losses. Shadow fading mean is 0 dB."""
    fspl = fspl_db(slant_range_m, freq_hz)
    p_los = los_probability(elevation_deg, environment)
    sfcl = clutter_and_sf(elevation_deg, freq_hz, environment)
    clutter_mean = (1.0 - p_los) * sfcl["nlos_clutter_db"]
    gases = gaseous_attenuation_db(elevation_deg, freq_hz)
    scint = scintillation_loss_db(elevation_deg, freq_hz, latitude_deg)
    rain = rain_attenuation_db(elevation_deg, freq_hz, rain_rate_mm_h, latitude_deg)
    o2i = np.full_like(fspl, 20.0 if indoor else 0.0)
    total = fspl + clutter_mean + gases + scint + rain + o2i
    return {
        "fspl_db": fspl,
        "p_los": p_los,
        "clutter_mean_db": clutter_mean,
        "nlos_clutter_db": sfcl["nlos_clutter_db"],
        "los_sf_sigma_db": sfcl["los_sf_sigma_db"],
        "nlos_sf_sigma_db": sfcl["nlos_sf_sigma_db"],
        "gases_db": gases,
        "scintillation_db": scint,
        "rain_db": rain,
        "o2i_db": o2i,
        "path_loss_los_db": fspl + gases + scint + rain + o2i,
        "path_loss_nlos_db": fspl + gases + scint + rain + o2i + sfcl["nlos_clutter_db"],
        "path_loss_mean_db": total,
    }
