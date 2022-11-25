import numpy as np
import scipy

from sarjana.transients.constants import Hubble_distance, Omega_m, Omega_Lambda


def comoving_distance_at_z(z):  # Mpc
    zp1 = 1.0 + z
    h0_up = np.sqrt(1 + Omega_m / Omega_Lambda) * scipy.special.hyp2f1(
        1 / 3, 1 / 2, 4 / 3, -Omega_m / Omega_Lambda
    )
    hz_up = (
        zp1
        * np.sqrt(1 + Omega_m * zp1 ** 3 / Omega_Lambda)
        * scipy.special.hyp2f1(1 / 3, 1 / 2, 4 / 3, -Omega_m * zp1 ** 3 / Omega_Lambda)
    )
    h0_down = np.sqrt(Omega_Lambda + Omega_m)
    hz_down = np.sqrt(Omega_Lambda + Omega_m * zp1 ** 3)
    return Hubble_distance * (hz_up / hz_down - h0_up / h0_down)


def luminosity_distance_at_z(z):  # Mpc
    return (1.0 + z) * comoving_distance_at_z(z)
