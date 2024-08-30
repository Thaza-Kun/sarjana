#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-07-09
# Python Version: 3.12

"""Plots the power-power distribution
"""

import argparse
import pathlib
import warnings

from scipy import stats, spatial
from scipy.stats import chisquare
from scipy.signal import find_peaks
from userust import greedy_harmonic_sum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import astropy.units as u
from astropy.time import Time
from typing import Tuple
from astropy.timeseries import LombScargle

from pdmpy import pdm

# from ndtest import ks2d2s

warnings.filterwarnings("ignore")


def LombScargle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    LS = LombScargle(time, obs)
    power = LS.power(freq_grid.value)
    return power, freq_grid.value


def pdm_periodogram(
    time: np.ndarray, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    if type(time) is Time:
        time = time.value
    f, theta = pdm(
        time,
        obs,
        f_min=freq_grid.min().value,
        f_max=freq_grid.max().value,
        delf=np.mean(freq_grid.diff().value),
    )
    return -theta, f


def main():
    period = 3
    rng = np.random.default_rng(seed=42)
    time = np.linspace(0, 10 * np.pi, 150)

    true_density = 0.8
    filter = rng.choice([True, False], len(time), p=[true_density, 1 - true_density])
    signal = (
        3 + np.sin(2 * np.pi * time / period) + rng.normal(scale=0.5, size=time.shape)
    )
    # signal = 3 + rng.normal(scale=2, size=time.shape)

    print(time.shape)
    print(signal.shape)
    time, signal = np.compress(filter, [time, signal], axis=1)
    print(time.shape)
    print(signal.shape)
    plt.plot(time, signal, "+")
    plt.show()

    freq_grid = np.linspace(0.1, 1, 1000) / u.s

    n_bins = 6
    stat = list()
    for p in 1 / freq_grid.value:
        phases = (time / p) % 1
        sort_idx = np.lexsort([signal, phases])

        counts_ = signal[sort_idx]
        phases = phases[sort_idx]

        _, bins = np.histogram(phases, n_bins)
        bin_idx = np.digitize(bins, phases)
        phase_count = np.array([sum(i) for i in np.split(counts_, bin_idx)[:-1]])

        chi2 = chisquare(phase_count)
        stat.append(chi2.statistic)
    stat = np.array(stat)
    stat_power_period = np.digitize(period, 1 / freq_grid.value)

    peaks_stat_idx, _ = find_peaks(stat, height=stat.mean() + stat.std())

    plt.plot(1 / freq_grid, stat)
    plt.plot(
        1 / freq_grid[stat_power_period], stat[stat_power_period], "x", color="orange"
    )
    plt.plot(1 / freq_grid[peaks_stat_idx], stat[peaks_stat_idx], "+")
    plt.axvline(period)
    plt.show()

    ls_power, ls_freq = LombScargle_periodogram(time, signal, freq_grid)
    ls_power_period = np.digitize(period, 1 / ls_freq)

    peaks_ls_idx, _ = find_peaks(ls_power)

    peaks_idx = np.array([*peaks_stat_idx, *peaks_ls_idx])

    plt.plot(1 / ls_freq, ls_power)
    plt.plot(
        1 / ls_freq[ls_power_period], ls_power[ls_power_period], "x", color="orange"
    )
    plt.plot(1 / freq_grid[peaks_idx], ls_power[peaks_idx], "+")
    plt.axvline(period)
    plt.show()

    pythagoras = np.sqrt(
        (ls_power[ls_power > ls_power.mean()] - ls_power.min()) ** 2
        + (stat[ls_power > ls_power.mean()] - stat.min()) ** 2
    )
    max_idx = pythagoras.argmax()

    pythagoras_ = np.sqrt(
        (ls_power - ls_power[ls_power > ls_power.mean()][max_idx]) ** 2
        + (stat - stat[ls_power > ls_power.mean()][max_idx]) ** 2
    )
    print(len(pythagoras_[pythagoras_ <= 0 + 2 * pythagoras_.std()]) / len(pythagoras_))

    plt.scatter(ls_power[peaks_idx], stat[peaks_idx])
    plt.plot(ls_power[ls_power_period], stat[stat_power_period], "x", color="orange")
    print(2 * pythagoras_.std())

    # stats.linregress(ls_power, stat)
    slope, intercept, r, p, se = stats.linregress(ls_power[peaks_idx], stat[peaks_idx])
    print(slope, intercept, r, p, se)
    plt.plot(ls_power, slope * ls_power + intercept, color="green")

    # t = np.linspace(0, 2*np.pi, 1000)
    # plt.plot(ls_power[ls_power > ls_power.mean()][max_idx]+(2*pythagoras_.std()*np.cos(t)) , stat[ls_power > ls_power.mean()][max_idx]+(2*pythagoras_.std()*np.sin(t)), color='orange')
    # plt.scatter(ls_power[ls_power > ls_power.mean()][max_idx], stat[ls_power > ls_power.mean()][max_idx], s=100, edgecolors='red', facecolors=None)
    # plt.xlim(ls_power.min()-0.1, ls_power.max()+0.1)
    # plt.ylim(stat.min()-0.1, stat.max()+0.1)
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    main()
