"""Periodogram by Duty Cucle
"""

import argparse
from collections import defaultdict
import os
import pathlib
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from numba import jit, float64

# from numba.experimental import jitclass

from astropy.timeseries import LombScargle
from astropy.time import Time
import astropy.units as u
from astropy.timeseries import TimeSeries

from datetime import datetime

from tqdm import tqdm
from multiprocessing import Pool, Queue, Process, Manager

# Suppress FutureWarning messages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# @jitclass([('detections', float64[:]), ('time', float64[:])])
# class CTimeSeries:
#     def __init__(self, time, data):
#         self._timeseries = TimeSeries(time=time, data={'detections':data})
#         self.detections = self.timeseries['detections']
#         self.time = self.timeseries.time
#         self.fold = self.timeseries.fold

# @jitclass([('values', float64[:])])
# class CTime:
#     def __init__(self, data):
#         self.values = data


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


@jit
def check_cumsum(cumsum: float64[:]) -> float:
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


# @jit(parallel=True, nopython=False)
def calc_inactive_frac(
    time: Time, data: np.ndarray, trial_periods: np.ndarray, quiet: bool = False
):
    def iterator(iterable: Iterable, quiet: bool) -> Iterable:
        if quiet:
            return iterable
        else:
            return tqdm(iterable)

    fracs = []
    processes = []
    timeseries = TimeSeries(time=time, data={"detections": data.reshape(-1, 1)})
    for period in iterator(trial_periods, quiet=quiet):
        folded_ = timeseries.fold(period=period, wrap_phase=1, normalize_phase=True)
        phases = np.array(folded_["time"])
        counts = np.array(folded_["detections"]).flatten()
        phases = (
            pd.DataFrame({"phase": phases, "detections": counts})
            .groupby(pd.cut(phases, np.arange(0, 1, 0.05)))["detections"]
            .sum()
        )
        phases = phases.reset_index().rename(columns={"index": "phase_bin"})
        phases["cumsum"] = phases["detections"].cumsum()
        inactive = check_cumsum(phases["cumsum"].to_numpy())
        fracs.append(inactive / len(phases))
    return fracs


def duty_cycle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity, quiet: bool = False
) -> np.ndarray:
    return calc_inactive_frac(time, obs, 1 / freq_grid, quiet)


def leave_one_out(array: np.ndarray) -> np.ndarray:
    N = len(array)
    return array[1:] - np.tri(N, N - 1, k=-1, dtype=bool)


def loo_subroutine(y, t, periodogram, grid, optimize: Callable, periods: list) -> float:
    power_ = periodogram(Time(t, format="mjd"), y, grid, quiet=True)
    periods.append(1 / grid[optimize(power_)])


# @jit(parallel=True, nopython=False)
def estimate_error_leave_one_out(
    Ys: np.ndarray,
    Ts: np.ndarray,
    freq_grid: u.Quantity,
    periodogram: Callable,
    optimize: Callable,
) -> float:
    processes = []
    P = []
    with Manager() as m:
        best_periods = m.list()
        for y, t in tqdm(
            zip(leave_one_out(Ys), leave_one_out(Ts)), total=len(leave_one_out(Ys))
        ):
            p = Process(
                target=loo_subroutine,
                args=(y, t, periodogram, freq_grid, optimize, best_periods),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        P = list([p.to(u.day).value for p in best_periods])
    return np.array(P).std()


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
    frac = duty_cycle_periodogram(time, Y, freq_grid)
    max_inactive_idx = np.nanargmax(frac)
    PF_period = 1 / freq_grid[max_inactive_idx]

    ## Leave-One-Out (Phase-Folding)
    print("Estimating LOO...")
    PF_stdev = estimate_error_leave_one_out(
        Ys=Y,
        Ts=time,
        freq_grid=freq_grid,
        periodogram=duty_cycle_periodogram,
        optimize=np.nanargmax,
    )

    print("Graphing...")
    low_ = PF_period.value - (PF_stdev * 2)
    high_ = PF_period.value + (PF_stdev * 2)

    g = sns.lineplot(x=1 / freq_grid, y=frac)
    g.axvline(PF_period.value, color="red", alpha=1)
    g.axvspan(low_, high_, alpha=0.3)
    g.set_xscale("log")
    g.set_xlabel("Period (days)")
    g.set_ylabel("Inactive Fraction (%)")
    g.set_title(
        f"Phase Folding Periodogram of {chosen_name} ({PF_period.to(u.day):.2f}±{2*PF_stdev:.2f} d)"
    )

    outdir = pathlib.Path(f"{arguments.outdir}/{chosen_name}")
    outdir.mkdir(parents=True, exist_ok=True)
    g.figure.savefig(f"{outdir}/pdgram-DutyCycle.png")
    plt.figure()

    ## SAVE
    print("Saving...")
    np.save(pathlib.Path(eventdir, "pdgram-DutyCycle.npy"), frac)
    result = {
        "period": PF_period.value,
        "stdev": 2 * PF_stdev,
        "duty_cycle": 1 - frac[max_inactive_idx],
    }
    pd.DataFrame(result, index=[0]).to_csv(
        pathlib.Path(eventdir, "pdgram-DutyCycle.csv"), index=False
    )


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
