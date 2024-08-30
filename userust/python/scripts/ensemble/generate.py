#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-05-10
# Python Version: 3.12

"""Generate an ensemble of periodogram peaks
"""

from typing import Tuple
from userust import (
    parse_arguments,
    generate_periodogram_ensembles,
    # Generator,
    # generate_signal_filter,
    greedy_harmonic_sum,
)

import argparse
import pathlib
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import find_peaks
import pandas as pd

from astropy.time import Time
from astropy.timeseries import LombScargle, TimeSeries
import astropy.units as u

# from pdmpy import pdm


def LombScargle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    LS = LombScargle(time, obs)
    power = LS.power(freq_grid.value)
    return power, freq_grid.value


def pdm_periodogram(
    time: np.ndarray, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    # if type(time) is Time:
    #     time = time.value
    # f, theta = pdm(
    #     time,
    #     obs,
    #     f_min=freq_grid.min().value,
    #     f_max=freq_grid.max().value,
    #     delf=np.mean(freq_grid.diff().value),
    # )
    # def fold_time(array: np.ndarray, period: float) -> np.ndarray:
    #     return (array / period) % 1
    
    variance = np.sum((obs - np.mean(obs))**2) / (len(obs) - 1)

    result = list()
    for period in (1 / freq_grid).to(u.day).value:
        phases = fold_time(time, period)
        ordered_index = np.lexsort([obs, phases])
        obs = np.array(obs)[ordered_index]
        phases = np.array(phases)[ordered_index]
        _ , bins = np.histogram(phases, bins=10)
        idx_separator = np.digitize(bins, phases)
        bin_items = np.split(obs, idx_separator)
        bin_means = [np.mean(i) for i in bin_items[:-1]]
        bin_counts = [len(i) for i in bin_items[:-1]]
        bin_variance = [np.sum((i - bin_means[n])**2) / (bin_counts[n] - 1) for n, i in enumerate(bin_items[:-1])]
        s_2 = np.nansum([(len(obs) - 1)*i for i in bin_variance])/np.nansum([len(obs)-i for i in bin_counts])
        result.append(s_2/variance)
    return -np.array(result), freq_grid.value


def duty_cycle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:
    time, obs = np.compress([o > 0.0 for o in obs], [time, obs], axis=1)

    # def fold_time(array: np.ndarray, period: float) -> np.ndarray:
    #     return (array / period) % 1

    result = list()
    for period in (1 / freq_grid).to(u.day).value:
        phases = fold_time(time, period)
        ordered_index = np.lexsort([obs, phases])
        obs = obs[ordered_index]
        phases = phases[ordered_index]
        head = phases[0] - 0
        tail = 1 - phases[-1]
        disconnected = head + tail
        longest = max(np.diff(phases).max(), disconnected)
        result.append(longest)
    return (result, freq_grid.value)

def fold_time(array: np.ndarray, period: float) -> np.ndarray:
    return (array / period) % 1

from scipy.stats import chisquare

def h_test_count_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:

    result = list()
    for period in (1 / freq_grid).to(u.day).value:
        phases = fold_time(time, period)
        counts, bins = np.histogram(phases, 20)
        stat = chisquare(counts)
        result.append(stat.statistic)
    return (np.array(result), freq_grid.value)

def h_test_mag_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> Tuple[np.ndarray, np.ndarray]:

    result = list()
    for period in (1 / freq_grid).to(u.day).value:
        phases = fold_time(time, period)
        phases.sort()
        _, bins = np.histogram(phases, 20)
        idx_separator = np.digitize(bins, phases)
        bin_items = np.split(obs, idx_separator)
        bin_means = [0 if len(i) == 0 else np.mean(i) for i in bin_items[:-1]]
        stat = chisquare(bin_means)
        result.append(stat.statistic)
    return (np.array(result), freq_grid.value)


@dataclass
class Transient:
    signal: np.ndarray = field(default_factory=lambda: np.array([]))
    signal_arrivals: np.ndarray = field(default_factory=lambda: np.array([]))
    # telescope_observations: np.ndarray = field(default_factory=lambda: np.array([]))
    telescope_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    time: Time = field(default_factory=lambda: np.array([]))
    noise: float = 0.0

    def __repr__(self) -> str:
        return (
            "+++++++++\n"
            + "Transient\n"
            + f"signal: {len(self.signal)}\n"
            + f"obs: {len(self.signal_arrivals)}\n"
            + f"noise: {self.noise:5>0.2f}\n"
            + "+++++++++\n"
        )


def read_frb_data(name: str, arguments: argparse.Namespace) -> Transient:
    print(f"Reading FRB: {name}")

    def get_column(columnname: str) -> np.ndarray:
        return np.load(pathlib.Path(arguments.parent, name, f"{columnname}.npy"))

    s = get_column("fluence")
    t = get_column("mjd_400_barycentered")
    e = get_column("fluence_err")
    t.sort()
    _, hourly_t = np.histogram(t, bins=int(t.max() - t.min())*24)
    # print(t)
    # print(len(t))
    # print(hourly_t)
    # print(len(hourly_t))
    idx_t = np.unique(np.digitize(hourly_t, t))
    s = np.array([np.mean(i) for i in np.split(s, idx_t)[:-1]])
    e = np.array([np.mean(i) for i in np.split(e, idx_t)[:-1]])
    t = np.array([np.mean(i) for i in np.split(t, idx_t)[:-1]])
    
    # idx_ht = np.digitize(t, hourly_t)
    # ht_s = np.zeros_like(hourly_t)
    # ht_e = np.zeros_like(hourly_t)

    # ht_s[idx_ht] = s
    # ht_e[idx_ht] = e
    # s, t, e = np.unique([s, t, e], axis=1)
    data = Transient(
        signal=s,
        signal_arrivals=t,
        noise=np.mean(e)
        # telescope_observations=get_column("mjd_400_observations_barycentered"),
    )

    # data.telescope_observations.sort()

    # Y = np.zeros_like(data.telescope_observations)
    # for i, fluence in zip(
    #     np.digitize(data.signal_arrivals, data.telescope_observations), data.signal
    # ):
    #     Y[i - 1] = fluence

    # data.telescope_signal = Y
    data.time = Time(data.signal_arrivals, format="mjd")
    print(data)
    return data


def create_hnull_data(
    magnitude: float, noise: float, time: np.ndarray, rng: np.random.Generator, arguments: argparse.Namespace
) -> Transient:
    print("Creating data...")
    data = Transient(
        signal_arrivals=time
    )
    data.signal = rng.normal(
        loc=magnitude, scale=np.sqrt(noise), size=data.signal_arrivals.shape
    )
    data.time = Time(data.signal_arrivals, format="mjd")
    print(data)
    return data


def main(arguments: argparse.Namespace):
    rng = np.random.default_rng(seed=arguments.seed)

    name = arguments.name
    outdir = arguments.outdir
    runs = arguments.runs

    global thisfrb
    thisfrb = read_frb_data(name, arguments)
    sim_frb = create_hnull_data(magnitude=np.mean(thisfrb.signal[thisfrb.signal != 0]),noise=thisfrb.noise, time=thisfrb.signal_arrivals, rng=rng, arguments=arguments)

    global PDGRAM, pdgram_name
    pdgram_name = arguments.periodogram.upper()
    PDGRAM = {
        "LS": LombScargle_periodogram,
        "PDM": pdm_periodogram,
        "DC": duty_cycle_periodogram,
        "HTC": h_test_count_periodogram,
        "HTM": h_test_mag_periodogram,
    }

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)
    inspectdir = pathlib.Path(simdir, "inspect")
    inspectdir.mkdir(parents=True, exist_ok=True)

    observations = thisfrb.signal_arrivals

    if arguments.rate is not None:
        n_eval = arguments.grid
        max_period = 200 / (arguments.rate * (1 / u.hour))
        min_period = 1 / (arguments.rate * (1 / u.hour))

        freq_grid = np.geomspace(1 / max_period, 1 / min_period, num=n_eval)
    else:
        n_eval = arguments.grid
        max_period = 3 / ((observations.max() - observations.min()) * u.day)
        min_period = 0.1 * np.diff(observations).max() * u.day

        freq_grid = np.geomspace(1 / max_period, 1 / min_period, num=n_eval)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    def inspect_evaluation(
        magnitude: np.ndarray, time: np.ndarray, power: np.ndarray, snr: np.ndarray, freq_grid: np.ndarray, iternum: int
    ) -> None:
        data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "flux": thisfrb.signal,
                        "time": thisfrb.signal_arrivals,
                        "label": "observed",
                    }
                ),
                pd.DataFrame(
                    {
                        "flux": np.array(magnitude),
                        "time": time,
                        "label": "simulation",
                    }
                ),
            ]
        )
        fig, axs = plt.subplots(2, 2)
        sns.scatterplot(data, x="time", y="flux", hue="label", ax=axs[0,0])
        axs[0,0].set_title(f"n_obs = {len(thisfrb.signal)}, n_sim = {len(magnitude)}")
        axs[0,1].plot(1/np.array(freq_grid), power)
        axs[0,1].set_title("Power")
        axs[1,1].plot(1/np.array(freq_grid), snr)
        axs[1,1].set_title("SNR")
        c2 = chisquare(power)
        axs[1,0].text(0.2,0.2, f"{c2.statistic:.2f} | {c2.pvalue:.2f}")
        plt.tight_layout()
        plt.savefig(
            pathlib.Path(
                inspectdir, f"n{iternum:0>6}-g{n_eval:0>6}-pgram={pdgram_name}.png"
            )
        )
        plt.close()

    sim_ensemble = generate_periodogram_ensembles(
        sim_frb.signal,
        sim_frb.time.value,
        len(thisfrb.signal),
        arguments.runs,
        arguments.seed,
        PDGRAM[pdgram_name],
        find_peaks,
        freq_grid,
        arguments.harmonics,
        arguments.snr_scale,
        inspect_evaluation,
    )

    frb_power, frb_freq = PDGRAM[pdgram_name](
        thisfrb.signal_arrivals, thisfrb.signal, freq_grid
    )
    frb_peaks, _ = find_peaks(frb_power)
    frb_snr = greedy_harmonic_sum(
        frb_power, frb_freq, arguments.harmonics
    )
    inspect_evaluation(thisfrb.signal, thisfrb.signal_arrivals, frb_power, frb_snr, freq_grid, arguments.runs)

    print(f"Saving ensemble data to {datadir}")
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-sim-power.npy"
        ),
        sim_ensemble.power,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-sim-snr.npy"
        ),
        sim_ensemble.snr,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-sim-freq.npy"
        ),
        sim_ensemble.freq,
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-sim-group.npy"
        ),
        sim_ensemble.group,
        allow_pickle=False,
    )
    print(f"Saving frb data to {datadir}")
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-frb-power.npy"
        ),
        np.array(frb_power)[frb_peaks],
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-frb-snr.npy"
        ),
        np.array(frb_snr)[frb_peaks],
        allow_pickle=False,
    )
    np.save(
        pathlib.Path(
            datadir, f"n{runs:0>6}-g{n_eval:0>6}-pgram={pdgram_name}-frb-freq.npy"
        ),
        np.array(frb_freq)[frb_peaks],
        allow_pickle=False,
    )


def run():
    # Suppress FutureWarning messages
    import warnings

    warnings.simplefilter(action="ignore", category=FutureWarning)

    print(__doc__)
    arguments = parse_arguments()
    main(arguments)


if __name__ == "__main__":
    run()
