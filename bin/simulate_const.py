#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-04-30
# Python Version: 3.12

"""Simulate a constant signal
"""

from userust import parse_arguments, iterate_periodogram, Ensemble

import argparse
import pathlib
from typing import Tuple
from dataclasses import dataclass, field
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, peak_prominences

from astropy.time import Time
from astropy.timeseries import LombScargle
import astropy.units as u


def LombScargle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> np.ndarray:
    LS = LombScargle(time, obs)
    power = LS.power(freq_grid)
    return power


def snr_greedy_harmonic_sum(power: np.ndarray, grid: np.ndarray) -> np.ndarray:
    def get_snr(sums: np.ndarray, mean: float, std: float) -> float:
        return np.max((sums - mean) / std)

    def compare_snr(
        snr: float, h: int, snr_max: float, h_m: float
    ) -> Tuple[float, float]:
        return np.nanmax([snr, snr_max]), np.nanmax([h, h_m])

    SNR = np.empty(len(grid))
    for i, f in enumerate(grid):
        let_sum = 0
        d = 0
        snr_max = 0
        h_m = 0
        # fundamental bin
        x_d = power[i]
        x_d_plus_1 = power[min(i + 1, len(grid) - 1)]
        if x_d_plus_1 > x_d:
            d += 1
            let_sum += x_d_plus_1
        else:
            let_sum += x_d
        snr_max = get_snr(let_sum, np.mean(power), np.std(power))
        # higher harmonics
        for h in range(arguments.harmonics):
            x_d = power[min(int(h * i + d), len(grid) - 1)]
            x_d_plus_1 = power[min(int(h * i + d + 1), len(grid) - 1)]
            if x_d_plus_1 > x_d:
                d += 1
                let_sum += x_d_plus_1
            else:
                let_sum += x_d
            snr = get_snr(let_sum, np.mean(power), np.std(power))
            snr_max, h_m = compare_snr(snr_max, h_m, snr, h)
        SNR[i] = snr_max
        h_0 = h_m
    return SNR


@dataclass
class Transient:
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    signal_arrivals: np.ndarray = field(default_factory=lambda: np.array([]))
    telescope_observations: np.ndarray = field(default_factory=lambda: np.array([]))
    telescope_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    time: Time = field(default_factory=lambda: np.array([]))
    noise: float = 0.0

    def __repr__(self) -> str:
        return (
            "+++++++++\n"
            + "Transient\n"
            + f"signal: {len(self.signal)}\n"
            + f"obs: {len(self.telescope_observations)}\n"
            + f"noise: {self.noise:5>0.2f}\n"
            + "+++++++++\n"
        )


def read_frb_data(name: str, arguments: argparse.Namespace) -> Transient:
    print(f"Reading FRB: {name}")

    def get_column(columnname: str) -> np.ndarray:
        return np.load(pathlib.Path(arguments.parent, name, f"{columnname}.npy"))

    data = Transient(
        signal=get_column("fluence"),
        signal_arrivals=get_column("mjd_400_barycentered"),
        telescope_observations=get_column("mjd_400_observations_barycentered"),
    )

    data.noise = max(get_column("fluence_err") / data.signal)
    data.telescope_observations.sort()

    Y = np.zeros_like(data.telescope_observations)
    for i, fluence in zip(
        np.digitize(data.signal_arrivals, data.telescope_observations), data.signal
    ):
        Y[i - 1] = fluence

    data.telescope_signal = Y
    data.time = Time(data.telescope_observations, format="mjd")
    print(data)
    return data


def create_hnull_data(noise: float, rng: np.random.Generator, arguments: argparse.Namespace) -> Transient:
    print("Creating data...")
    data = Transient(
        telescope_observations=np.load(
            pathlib.Path(arguments.parent, "observations.npy")
        ),
    )
    data.telescope_observations.sort()
    data.signal = np.ones(shape=data.telescope_observations.shape) + rng.normal(
        scale=noise, size=data.telescope_observations.shape
    )
    data.time = Time(data.telescope_observations, format="mjd")
    print(data)
    return data


def main(arguments: argparse.Namespace):
    rng = np.random.default_rng(seed=arguments.seed)

    name = arguments.name
    outdir = arguments.outdir

    T = arguments.period

    cutoff_SNR = arguments.min_SNR
    cutoff_power = arguments.min_power
    runs = arguments.runs

    thisfrb = read_frb_data(name, arguments)
    frb_ensemble = Ensemble()

    sim_frb = create_hnull_data(noise=thisfrb.noise, rng=rng, arguments=arguments)
    sim_ensemble = Ensemble()


    begin = sim_frb.telescope_observations.min()
    end = sim_frb.telescope_observations.max()
    window_filter = (sim_frb.telescope_observations >= begin) & (
        sim_frb.telescope_observations <= end
    )

    view = sim_frb.signal[window_filter]
    detection_rate = len(thisfrb.signal) / len(view)

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)

    if arguments.freq_grid is not None:
        freq_grid = np.load(arguments.freq_grid) * (1 / u.hour)

    view_index = len(sim_frb.signal[(sim_frb.telescope_observations < begin)])
    view_len = len(view)
    # Random detections between event window
    sim_ensemble, frb_ensemble = iterate_periodogram(
        sim_frb.signal,
        sim_frb.time,
        thisfrb.telescope_signal,
        thisfrb.time,
        view_index,
        view_len,
        detection_rate,
        arguments,
        rng,
        LombScargle_periodogram,
        snr_greedy_harmonic_sum,
        find_peaks,
        sim_ensemble,
        frb_ensemble,
        freq_grid,
    )

    # with open(pathlib.Path(arguments.parent, name, "ensemble-simfrb.pkl"), "w+b") as f:
    #     print("Saving ensemble-simfrb.pkl data")
    #     pickle.dump(sim_ensemble, f)
    # with open(pathlib.Path(arguments.parent, name, "ensemble-frb.pkl"), "w+b") as f:
    #     print("Saving ensemble-frb.pkl data")
    #     pickle.dump(frb_ensemble, f)

    print(f"Drawing figure to {outdir}")
    sim_ensemble.filter(snr=cutoff_SNR, power=cutoff_power)
    frb_ensemble.filter(snr=cutoff_SNR, power=cutoff_power)
    plotted = {
        "freq": np.concatenate((1 / np.array(sim_ensemble.freq), 1 / np.array(frb_ensemble.freq))),
        "power": np.concatenate((sim_ensemble.power, frb_ensemble.power)),
        "snr": np.concatenate((sim_ensemble.snr, frb_ensemble.snr)),
        "label": np.concatenate(
            (
                ["simulated"] * len(sim_ensemble.freq),
                ["observed"] * len(frb_ensemble.freq),
            )
        ),
    }
    print(plotted)

    fig, axs = plt.subplots(2, 2)

    g = sns.histplot(
        data=plotted,
        x="freq",
        y="power",
        hue="label",
        ax=axs[0, 0],
        log_scale=(True, False),
        common_norm=False,
        legend=False,
        alpha=0.5,
    )
    g.set_xlabel("Period")
    g.set_ylabel("Power")

    g = sns.histplot(
        data=plotted,
        x="snr",
        y="freq",
        hue="label",
        ax=axs[1, 1],
        log_scale=(False, True),
        common_norm=False,
        legend=False,
        alpha=0.5,
    )
    g.set_xlabel("SNR")
    g.set_ylabel("Period")

    g = sns.histplot(
        data=plotted,
        x="snr",
        y="power",
        hue="label",
        ax=axs[0, 1],
        common_norm=False,
        legend=False,
        alpha=0.5,
    )
    g.set_xlabel("SNR")
    g.set_ylabel("Power")
    print(plotted)

    g = sns.histplot(
        data=plotted,
        x="freq",
        hue="label",
        ax=axs[1, 0],
        log_scale=True,
        common_norm=False,
        element="step",
    )
    g.set_xlabel("Period")

    plt.suptitle(f"{name}")
    plt.tight_layout()
    plt.savefig(
        pathlib.Path(simdir, f"min_SNR={cutoff_SNR}-runs={runs:>0}-period-{T}.png")
    )


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
