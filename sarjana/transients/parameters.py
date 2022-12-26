import numpy as np
import scipy
import pandas as pd

from sarjana.transients.constants import (
    Hubble_distance,
    Omega_m,
    Omega_Lambda,
    Mpc_to_cm,
)


# TODO Understand this
def comoving_distance_at_z(z):  # Mpc
    zp1 = 1.0 + z
    h0_up = np.sqrt(1 + Omega_m / Omega_Lambda) * scipy.special.hyp2f1(
        1 / 3, 1 / 2, 4 / 3, -Omega_m / Omega_Lambda
    )
    hz_up = (
        zp1
        * np.sqrt(1 + Omega_m * zp1**3 / Omega_Lambda)
        * scipy.special.hyp2f1(1 / 3, 1 / 2, 4 / 3, -Omega_m * zp1**3 / Omega_Lambda)
    )
    h0_down = np.sqrt(Omega_Lambda + Omega_m)
    hz_down = np.sqrt(Omega_Lambda + Omega_m * zp1**3)
    return Hubble_distance * (hz_up / hz_down - h0_up / h0_down)


def luminosity_distance_at_z(z):  # Mpc
    return (1.0 + z) * comoving_distance_at_z(z)


def burst_energy(data: pd.DataFrame) -> pd.DataFrame:
    return (
        1e-23
        * data["frequency"]
        * 1e6
        * data["fluence"]
        / 1000
        * (4 * np.pi * (luminosity_distance_at_z(data["redshift"]) * Mpc_to_cm) ** 2)
        / (1 + data["redshift"])
    )


def rest_frequency(data: pd.DataFrame) -> pd.DataFrame:
    return data["frequency"] * (1 + data["redshift"])


def frequency_diff_by_dm(data: pd.DataFrame) -> pd.DataFrame:
    pass


def rest_frequency_bandwidth(data: pd.DataFrame) -> pd.DataFrame:
    return (data["high_freq"] - data["low_freq"])(1 + data["redshift"])


def rest_time_width(data: pd.DataFrame) -> pd.DataFrame:
    return data["width"] / (1 + data["redshift"])


def brightness_temperature(data: pd.DataFrame) -> pd.DataFrame:
    return (
        1.1e35
        * data["flux"]
        * (data["width"] * 1000) ** (-2)
        * (data["frequency"] / 1000) ** (-2)
        * (luminosity_distance_at_z(data["redshift"]) / 1000) ** 2
        / (1 + data["redshift"])
    )
