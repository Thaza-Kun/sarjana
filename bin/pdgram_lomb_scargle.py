#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-03-27
# Python Version: 3.12

"""Lomb Scargle Periodogram
"""

import pathlib
import argparse
from typing import Callable

from datetime import datetime

from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        "-f",
        help="Folder containing {repeater_name}/{column}.npy",
        type=pathlib.Path,
    )
    parser.add_argument("--name", help="Repeater name", type=str)
    parser.add_argument("--begin", help="Begin date", type=datetime)
    parser.add_argument("--end", help="End date", type=datetime)
    parser.add_argument("--n", help="Coefficient", type=int, default=5)
    parser.add_argument("--rate", help="Burst rate (hr-1)", type=float)
    parser.add_argument("--outdir", help="Output dir", type=pathlib.Path)
    return parser.parse_args()


def LombScargle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> np.ndarray:
    LS = LombScargle(time, obs)
    power = LS.power(freq_grid)
    return power


def leave_one_out(array: np.ndarray) -> np.ndarray:
    N = len(array)
    return array[1:] - np.tri(N, N - 1, k=-1, dtype=bool)


def estimate_error_leave_one_out(
    Ys: np.ndarray,
    Ts: np.ndarray,
    freq_grid: u.Quantity,
    periodogram: Callable,
    optimize: Callable,
) -> float:
    best_periods = []
    for y, t in tqdm(zip(leave_one_out(Ys), leave_one_out(Ts))):
        power_ = periodogram(t, y, freq_grid)
        best_periods.append(1 / freq_grid[optimize(power_)])
    return np.array([p.to(u.day).value for p in best_periods]).std()


def main(arguments: argparse.Namespace):
    chosen_name: str = arguments.name

    eventdir = pathlib.Path(arguments.folder, chosen_name)

    def get_column(columnname: str) -> np.ndarray:
        return np.load(pathlib.Path(arguments.folder, chosen_name, f"{columnname}.npy"))

    fluences = get_column("fluence")
    fluences_time = get_column("mjd_400_barycentered")
    observations = get_column("mjd_400_observations_barycentered")
    observations.sort()
    # print(observations)

    Y = np.zeros_like(observations)
    for i, fluence in zip(np.digitize(fluences_time, observations), fluences):
        Y[i] = fluence

    if arguments.begin:
        Y = Y[Y > Time(arguments.begin).to_datetime()]
    if arguments.end:
        Y = Y[Y < Time(arguments.end).to_datetime()]

    time = Time(observations, format="mjd")

    freq_min = 2 * arguments.rate * (1 / u.hour)
    freq_max = 3 / ((observations.max() - observations.min()) * u.day)

    print(freq_min)
    print(freq_max)

    n_eval = max(
        min(
            int(
                arguments.n * freq_min.value * (observations.max() - observations.min())
            ),
            5_000,
        ),
        500,
    )

    print(f"{n_eval=}")
    print(f"{(observations.max()-observations.min())=}")
    freq_grid = np.linspace(freq_max, freq_min, n_eval)

    # Lomb-Scargle
    print(arguments.name)
    print("Periodogram")
    power = LombScargle_periodogram(time, Y, freq_grid)
    ls_period = 1 / freq_grid[np.nanargmax(power)]
    print("FAP")
    # ls_fap = LombScargle(time, Y).false_alarm_probability(
    #     np.nanmax(power),
    #     method="bootstrap",
    #     minimum_frequency=freq_grid.min(),
    #     maximum_frequency=freq_grid.max(),
    # )
    print(f"Period = {ls_period.value} d")

    # Leave-One-Out (Lomb-Scargle)
    print("Leave one out")
    ls_stdev = estimate_error_leave_one_out(
        Ys=Y,
        Ts=time,
        freq_grid=freq_grid,
        periodogram=LombScargle_periodogram,
        optimize=np.nanargmax,
    )
    low_ = ls_period.value - ls_stdev * 2
    high_ = ls_period.value + ls_stdev * 2

    g = sns.lineplot(x=1 / freq_grid, y=power)
    g.axvline(ls_period.value, color="red", alpha=1)
    g.axvspan(min(low_, 1 / freq_grid.min().value), high_, alpha=0.3)
    g.set_xscale("log")
    g.set_xlabel("period")
    g.set_ylabel("LS Power")
    g.set_title(
        f"Lomb-Scargle Periodogram of {chosen_name} ({ls_period.to(u.day):.2f}) ±{2*ls_stdev:.2f} d)"
    )

    outdir = pathlib.Path(f"{arguments.outdir}/{chosen_name}")
    outdir.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(f"{outdir}/pdgram-LombScargle.png")
    plt.figure()

    ## SAVE
    np.save(pathlib.Path(eventdir, "pdgram-freq-grid.npy"), freq_grid.value)
    np.save(pathlib.Path(eventdir, "pdgram-LombScargle.npy"), power.value)
    result = {
        "period": ls_period.value,
        "stdev": 2 * ls_stdev,
        # "false_alarm_probability": ls_fap,
    }
    pd.DataFrame(result, index=[0]).to_csv(
        pathlib.Path(eventdir, "pdgram-LombScargle.csv"), index=False
    )


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
