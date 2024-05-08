#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-03-13
# Python Version: 3.7

"""Harmonic sum
"""

import argparse
import pathlib
from typing import Tuple

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", "-f", help="Frequency", type=float, required=True)
    parser.add_argument("--harmonics", "-H", help="Harmonics", type=int, default=16)
    parser.add_argument("--noise", "-n", help="Noise", type=float, default=5e-1)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    t = np.linspace(0, 3, 500)  # s
    print(f"{arguments.freq} osc/s")  # /s
    signal = np.sin(2 * np.pi * arguments.freq * t) + np.random.default_rng(42).normal(
        scale=arguments.noise, size=t.shape
    )
    plt.figure()
    plt.title(f"Test signal freq={arguments.freq} Hz noise={arguments.noise:.2f}")
    plt.plot(t, signal)
    plt.plot(t, np.sin(2 * np.pi * arguments.freq * t))
    plt.savefig(f"test-greedy-{arguments.freq}Hz-noise={arguments.noise}.png")

    frequency = np.linspace(0.5, 5 * arguments.freq, 1000)
    power = LombScargle(t, signal).power(frequency)
    plt.figure()
    plt.title(
        f"Power of test signal freq={arguments.freq} Hz noise={arguments.noise:.2f}"
    )
    plt.axvline(arguments.freq, color="orange")
    plt.text(
        arguments.freq + 0.5,
        power.max(),
        f"Power at {arguments.freq} Hz = {power[np.digitize(arguments.freq, frequency)]:.3f}",
    )
    plt.plot(frequency, power)
    plt.savefig(f"test-greedy-power-{arguments.freq}Hz-noise={arguments.noise}.png")

    def get_snr(sums: np.ndarray, mean: float, std: float) -> float:
        return np.max((sums - mean) / std)

    def compare_snr(
        snr: float, h: int, snr_max: float, h_m: float
    ) -> Tuple[float, float]:
        return np.nanmax([snr, snr_max]), np.nanmax([h, h_m])

    SNR = np.empty(len(frequency))
    for i, f in enumerate(frequency):
        let_sum = 0
        d = 0
        snr_max = 0
        h_m = 0
        # fundamental bin
        x_d = power[i]
        x_d_plus_1 = power[min(i + 1, len(frequency) - 1)]
        if x_d_plus_1 > x_d:
            d += 1
            let_sum += x_d_plus_1
        else:
            let_sum += x_d
        snr_max = get_snr(let_sum, power.mean(), power.std())
        # higher harmonics
        for h in range(arguments.harmonics):
            x_d = power[min(int(h * i + d), len(frequency) - 1)]
            x_d_plus_1 = power[min(int(h * i + d + 1), len(frequency) - 1)]
            if x_d_plus_1 > x_d:
                d += 1
                let_sum += x_d_plus_1
            else:
                let_sum += x_d
            snr = get_snr(let_sum, power.mean(), power.std())
            snr_max, h_m = compare_snr(snr_max, h_m, snr, h)
        SNR[i] = snr_max
        h_0 = h_m

    plt.figure()
    plt.title(
        f"SNR of test signal freq={arguments.freq} Hz noise={arguments.noise:.2f}"
    )
    plt.axvline(arguments.freq, color="orange")
    plt.text(
        arguments.freq + 0.5,
        SNR.max(),
        f"SNR at {arguments.freq} Hz = {SNR[np.digitize(arguments.freq, frequency)]:.3f}",
    )
    plt.plot(frequency, SNR)
    plt.savefig(f"test-greedy-snr-{arguments.freq}Hz-noise={arguments.noise}.png")


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
