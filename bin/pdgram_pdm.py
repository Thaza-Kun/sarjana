import argparse
from collections import defaultdict
import os
import pathlib
from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u

from datetime import datetime

from tqdm import tqdm

from pdmpy import pdm


"""Periodogram by Phase Dispersion Minimization
"""


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


def pdm_periodogram(
    time: np.ndarray, obs: np.ndarray, freq_grid: u.Quantity
) -> np.ndarray:
    if type(time) is Time:
        time = time.value
    f, theta = pdm(
        time,
        obs,
        f_min=freq_grid.min().value,
        f_max=freq_grid.max().value,
        delf=freq_grid.diff()[0].value,
    )
    return f, theta


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
        f_, power_ = periodogram(t, y, freq_grid)
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

    # Phase Folding
    print("Periodogram...")
    freq, theta = pdm_periodogram(time, Y, freq_grid)
    PDM_period = 1 / freq_grid[np.nanargmin(theta)]

    ## Leave One Out (PDM)
    print("Estimate LOO...")
    PDM_stdev = estimate_error_leave_one_out(
        Ys=Y,
        Ts=time,
        freq_grid=freq_grid,
        periodogram=pdm_periodogram,
        optimize=np.nanargmin,
    )
    low = PDM_period.value - PDM_stdev * 2
    high = PDM_period.value + PDM_stdev * 2

    print(theta.shape)
    print(freq.shape)
    g = sns.lineplot(x=1 / freq, y=theta)
    g.axvline(PDM_period.value, color="red", alpha=1)
    g.axvspan(low, high, alpha=0.3)
    g.set_xscale("log")
    g.set_xlabel("Period (days)")
    g.set_ylabel("Theta")
    g.set_title(
        f"PDM Periodogram of {chosen_name} ({PDM_period.to(u.day):.2f}±{max(2*PDM_stdev, 0.01):.2f} d)"
    )

    outdir = pathlib.Path(f"{arguments.outdir}/{chosen_name}")
    outdir.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(f"{outdir}/pdgram-PhaseDispMin.png")

    ## SAVE
    np.save(pathlib.Path(eventdir, "pdgram-PhaseDispMin.npy"), theta)
    np.save(pathlib.Path(eventdir, "pdgram-PhaseDispMin-freq-grid.npy"), freq)
    result = {
        "period": PDM_period.value,
        "stdev": 2 * PDM_stdev,
    }
    pd.DataFrame(result, index=[0]).to_csv(
        pathlib.Path(eventdir, "pdgram-PhaseDispMin.csv"), index=False
    )


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
