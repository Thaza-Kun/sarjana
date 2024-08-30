#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-05-10
# Python Version: 3.12

"""Generate an ensemble of periodogram peaks
"""

from copy import deepcopy
from typing import Tuple
from userust import (
    parse_arguments,
    Generator,
    generate_timeseries_subsample,
)

import argparse
import pathlib
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
import pandas as pd

import matplotlib.pyplot as plt

from astropy.time import Time
from astropy.timeseries import LombScargle, TimeSeries
import astropy.units as u

from pdmpy import pdm

# Suppress FutureWarning messages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def LombScargle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    LS = LombScargle(Time(time, format="mjd"), obs)
    power = LS.power(freq_grid)
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


# It seems that this is not useful
# because the simulation has no inactive period
# making it null by definition
# Maybe we can fix this by generating a function with inactive period
# Or maybe this can be a strong indication that
# the active period is indeed periodic and not random
def duty_cycle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    def calc_inactive_frac(time: Time, data: np.ndarray, trial_periods: np.ndarray):
        def check_cumsum(cumsum: np.ndarray) -> float:
            inactive = 0
            state = 0
            prev = 0
            for current in cumsum:
                if current == prev:
                    state += 1
                else:
                    prev = current
                    inactive = state if state > inactive else inactive
                    state = 0
            return inactive

        fracs = []
        timeseries = TimeSeries(
            time=Time(time, format="mjd"), data={"detections": data}
        )
        for period in trial_periods:
            folded_ = timeseries.fold(period=period, wrap_phase=1, normalize_phase=True)
            # print(folded_)
            phases = np.array(folded_["time"])
            counts = np.array(folded_["detections"]).flatten()
            phases = (
                pd.DataFrame({"phase": phases, "detections": counts})
                .groupby(pd.cut(phases, np.arange(0, 1, 0.005)))["detections"]
                .sum()
            )
            phases = phases.reset_index().rename(columns={"index": "phase_bin"})
            phases["cumsum"] = phases["detections"].cumsum()
            inactive = check_cumsum(phases["cumsum"].to_numpy())
            fracs.append(inactive / len(phases))
        return np.array(fracs)

    return calc_inactive_frac(time, np.floor(obs), 1 / freq_grid), freq_grid.value


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
    runs = arguments.runs

    thisfrb = read_frb_data(name, arguments)
    sim_frb = create_hnull_data(noise=thisfrb.noise, rng=rng, arguments=arguments)

    pdgram_name = arguments.periodogram.upper()
    PDGRAM = {
        "LS": LombScargle_periodogram,
        "PDM": pdm_periodogram,
        "DC": duty_cycle_periodogram,
    }

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)

    observations = thisfrb.telescope_observations

    if arguments.rate is not None:
        n_eval = arguments.grid
        max_period = 200 / (arguments.rate * (1 / u.hour))
        min_period = 1 / (arguments.rate * (1 / u.hour))
        # freq_min = 2 * arguments.rate * (1 / u.hour)
        # freq_max = 3 / ((observations.max() - observations.min()) * u.day)

        freq_grid = np.linspace(1 / max_period, 1 / min_period, num=n_eval)
    else:
        n_eval = arguments.grid
        max_period = 3 / ((observations.max() - observations.min()) * u.day)
        min_period = 0.1 * np.diff(observations).max() * u.day

        freq_grid = np.linspace(1 / max_period, 1 / min_period, num=n_eval)

    gen = Generator(1)
    noise_sample = generate_timeseries_subsample(
        sim_frb.time.value, sim_frb.signal, len(thisfrb.signal), gen
    )
    print(len(thisfrb.time))
    print(len(thisfrb.signal_arrivals))
    print(len(thisfrb.signal))
    print(len(noise_sample.magnitude))
    frb_pdgram = PDGRAM[pdgram_name](thisfrb.signal_arrivals, thisfrb.signal, freq_grid)

    noise_pdgram = PDGRAM[pdgram_name](
        noise_sample.time, noise_sample.magnitude, freq_grid
    )
    residue = frb_pdgram[0] - noise_pdgram[0]
    plt.plot(1 / frb_pdgram[1], residue)
    plt.plot(1 / frb_pdgram[1], frb_pdgram[0])
    plt.plot(1 / frb_pdgram[1], noise_pdgram[0])
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
