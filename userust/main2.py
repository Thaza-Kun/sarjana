#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-04-30
# Python Version: 3.12

"""Simulate a constant signal
"""

from userust import parse_arguments, generate_periodogram_ensembles, Ensemble

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


# def snr_greedy_harmonic_sum(power: np.ndarray, grid: np.ndarray) -> np.ndarray:
#     def get_snr(sums: np.ndarray, mean: float, std: float) -> float:
#         return np.max((sums - mean) / std)

#     def compare_snr(
#         snr: float, h: int, snr_max: float, h_m: float
#     ) -> Tuple[float, float]:
#         return np.nanmax([snr, snr_max]), np.nanmax([h, h_m])

#     SNR = np.empty(len(grid))
#     for i, f in enumerate(grid):
#         let_sum = 0
#         d = 0
#         snr_max = 0
#         h_m = 0
#         # fundamental bin
#         x_d = power[i]
#         x_d_plus_1 = power[min(i + 1, len(grid) - 1)]
#         if x_d_plus_1 > x_d:
#             d += 1
#             let_sum += x_d_plus_1
#         else:
#             let_sum += x_d
#         snr_max = get_snr(let_sum, np.mean(power), np.std(power))
#         # higher harmonics
#         for h in range(arguments.harmonics):
#             x_d = power[min(int(h * i + d), len(grid) - 1)]
#             x_d_plus_1 = power[min(int(h * i + d + 1), len(grid) - 1)]
#             if x_d_plus_1 > x_d:
#                 d += 1
#                 let_sum += x_d_plus_1
#             else:
#                 let_sum += x_d
#             snr = get_snr(let_sum, np.mean(power), np.std(power))
#             snr_max, h_m = compare_snr(snr_max, h_m, snr, h)
#         SNR[i] = snr_max
#         h_0 = h_m
#     return SNR


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


def create_hnull_data(
    noise: float, rng: np.random.Generator, arguments: argparse.Namespace
) -> Transient:
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

    cutoff_SNR = arguments.min_SNR
    cutoff_power = arguments.min_power
    runs = arguments.runs

    thisfrb = read_frb_data(name, arguments)
    sim_frb = create_hnull_data(noise=thisfrb.noise, rng=rng, arguments=arguments)

    begin = sim_frb.telescope_observations.min()
    end = sim_frb.telescope_observations.max()
    window_filter = (sim_frb.telescope_observations >= begin) & (
        sim_frb.telescope_observations <= end
    )

    view = sim_frb.signal[window_filter]
    detection_rate = len(thisfrb.signal) / len(view)

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)

    observations = thisfrb.telescope_observations
    if arguments.freq_grid is not None:
        freq_grid = np.load(arguments.freq_grid) * (1 / u.hour)
    elif arguments.rate is not None:
        n_eval = arguments.grid
        freq_min = 2 * arguments.rate * (1 / u.hour)
        freq_max = 3 / ((observations.max() - observations.min()) * u.day)

        freq_grid = np.linspace(freq_max, freq_min, n_eval)
    else:
        n_eval = arguments.grid
        freq_min = 0.1 * (1 / u.hour)
        freq_max = 3 / ((observations.max() - observations.min()) * u.day)

        freq_grid = np.linspace(freq_max, freq_min, n_eval)

    view_index = len(sim_frb.signal[(sim_frb.telescope_observations < begin)])
    view_len = len(view)
    # Random detections between event window
    sim_ensemble, frb_ensemble = generate_periodogram_ensembles(
        sim_frb.signal,
        sim_frb.time,
        thisfrb.telescope_signal,
        thisfrb.time,
        view_index,
        view_len,
        detection_rate,
        arguments,
        arguments.seed,
        LombScargle_periodogram,
        find_peaks,
        freq_grid,
    )

    print(f"Saving ensemble data to {datadir}")
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-sim-power.npy"),
        sim_ensemble.power,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-sim-snr.npy"),
        sim_ensemble.snr,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-sim-freq.npy"),
        sim_ensemble.freq,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-sim-group.npy"),
        frb_ensemble.group,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-frb-power.npy"),
        frb_ensemble.power,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-frb-snr.npy"),
        frb_ensemble.snr,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-frb-freq.npy"),
        frb_ensemble.freq,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(datadir, f"n{runs:0>6}-frb-group.npy"),
        frb_ensemble.group,
        allow_pickle=False,
    )


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
