import numpy as np

H_0 = 67.4 * 1000 * 100  # cm s^-1 Mpc^-1
c = 29979245800  # cm s^-1
Mpc_to_cm = 3.085677581491367e24
Gyr_to_s = 3.15576e16
Hubble_time = 1 / (H_0 / Mpc_to_cm * Gyr_to_s)  # Gyr
Hubble_distance = c / H_0  # Mpc
Omega_b = 0.0224 / ((H_0) / 1000 / 100 / 100) ** 2
Omega_m = 0.315
Omega_Lambda = 0.685
f_IGM = 0.83
chi = 7 / 8
G = 6.6743e-8  # cm^3 g^-1 s^-2
m_p = 1.67262192e-24  # g
dm_factor = (
    3 * c * H_0 / (Mpc_to_cm) ** 2 * 1e6 * Omega_b * f_IGM * chi / (8 * np.pi * G * m_p)
)
DM_host_lab = 70.0  # pc cm^-3
DM_halo = 30.0
