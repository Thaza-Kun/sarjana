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
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
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
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        z (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (1.0 + z) * comoving_distance_at_z(z)


def rest_luminosity(
    z: float,
    fluence: float,
    frequency: float,
    rest_frequency: float,
    spectral_idx: float,
) -> float:
    """Adopted from Hashimoto 2019 https://doi.org/10.1093/mnras/stz1715

    Args:
        z (float): _description_
        fluence (float): _description_
        frequency (float): _description_
        rest_frequency (float): _description_
        spectral_idx (float): _description_

    Returns:
        float: _description_
    """
    numerator = 4 * np.pi * (luminosity_distance_at_z(z) ** 2)
    denominator = (1 + z) ** (2 + spectral_idx)
    frequency_ratio = (rest_frequency / frequency) ** spectral_idx
    return (numerator / denominator) * frequency_ratio * fluence


def burst_energy(center_frequency, fluence, redshift) -> float:
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        center_frequency (_type_): _description_
        fluence (_type_): _description_
        redshift (_type_): _description_

    Returns:
        float: _description_
    """
    return (
        1e-23
        * center_frequency
        * 1e6
        * fluence
        / 1000
        * (4 * np.pi * (luminosity_distance_at_z(redshift) * Mpc_to_cm) ** 2)
        / (1 + redshift)
    )


def rest_frequency(center_frequency, redshift) -> float:
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        center_frequency (_type_): _description_
        redshift (_type_): _description_

    Returns:
        float: _description_
    """
    return center_frequency * (1 + redshift)


def rest_frequency_bandwidth(low_freq, high_freq, redshift) -> float:
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        low_freq (_type_): _description_
        high_freq (_type_): _description_
        redshift (_type_): _description_

    Returns:
        float: _description_
    """
    return (high_freq - low_freq)(1 + redshift)


def rest_time_width(time_width, redshift) -> float:
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        time_width (_type_): _description_
        redshift (_type_): _description_

    Returns:
        float: _description_
    """
    return time_width / (1 + redshift)


def brightness_temperature(flux, width, center_frequency, redshift) -> float:
    """Adopted from Luo Jia-Wei 2022 https://doi.org/10.1093/mnras/stac3206

    Args:
        flux (_type_): _description_
        width (_type_): _description_
        center_frequency (_type_): _description_
        redshift (_type_): _description_

    Returns:
        float: _description_
    """
    return (
        1.1e35
        * flux
        * (width * 1000) ** (-2)
        * (center_frequency / 1000) ** (-2)
        * (luminosity_distance_at_z(redshift) / 1000) ** 2
        / (1 + redshift)
    )
