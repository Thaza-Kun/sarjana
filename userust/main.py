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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, peak_prominences

from astropy.time import Time
from astropy.timeseries import LombScargle
import astropy.units as u


def main(arguments: argparse.Namespace):
    name = arguments.name
    outdir = arguments.outdir

    period = arguments.period

    cutoff_SNR = arguments.min_SNR
    cutoff_power = arguments.min_power
    runs = arguments.runs

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ensemble data from {datadir}")
    frb_ensemble = Ensemble(
        power=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-power.npy")),
        snr=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-snr.npy")),
        freq=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-freq.npy")),
        group=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-group.npy")),
    )
    # print(len(frb_ensemble.power))
    # print(len(frb_ensemble.snr))
    # print(len(frb_ensemble.freq))
    # print(len(frb_ensemble.group))

    sim_ensemble = Ensemble(
        power=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-power.npy")),
        snr=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-snr.npy")),
        freq=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-freq.npy")),
        group=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-group.npy")),
    )
    frb_ensemble.filter(snr=0, power=0)
    sim_ensemble.filter(snr=0, power=0)
    # print(len(sim_ensemble.power))
    # print(len(sim_ensemble.snr))
    # print(len(sim_ensemble.freq))
    # print(len(sim_ensemble.group))

    plotted = pd.DataFrame(
        {
            "Period": np.concatenate(
                (1 / np.array(sim_ensemble.freq), 1 / np.array(frb_ensemble.freq))
            ),
            "Power": np.concatenate((sim_ensemble.power, frb_ensemble.power)),
            "SNR": np.concatenate((sim_ensemble.snr, frb_ensemble.snr)),
            "group": np.concatenate((sim_ensemble.group, frb_ensemble.group)),
            "label": np.concatenate(
                (
                    ["simulated"] * len(sim_ensemble.freq),
                    ["observed"] * len(frb_ensemble.freq),
                )
            ),
        }
    )
    plotted["ΔPOWER/ΔSNR"] = (
        plotted.groupby(["group", "label"])
        .apply(lambda x: np.gradient(x["Power"], x["SNR"]))
        .explode()
        .to_numpy()
    )
    print(plotted["ΔPOWER/ΔSNR"].describe())
    print(plotted["ΔPOWER/ΔSNR"].max())
    print(plotted["ΔPOWER/ΔSNR"].min())
    print(plotted["ΔPOWER/ΔSNR"].mean())
    print(plotted["ΔPOWER/ΔSNR"].std())

    for i in np.linspace(0, cutoff_SNR, 10):
        # if (len(sim_ensemble.power) != 0) or (len(frb_ensemble.power) != 0):
        print(f"Drawing figure to {outdir}")
        fig, axs = plt.subplots(2, 3, sharex="col", figsize=(15, 7))

        for col, colname in enumerate(["Period", "Power", "SNR"]):
            if colname == "Period":
                log_scale = True
            else:
                log_scale = False
            try:
                g = sns.histplot(
                    data=plotted[plotted["SNR"] >= i],
                    x=colname,
                    hue="label",
                    ax=axs[0, col],
                    log_scale=log_scale,
                    common_norm=False,
                    cumulative=True,
                    legend=False,
                    element="step",
                    fill=False,
                    stat="density",
                )
                g.set_xlabel(colname)
                g.set_ylabel("Cumulative Density")
                g = sns.histplot(
                    data=plotted[plotted["SNR"] >= i],
                    x=colname,
                    hue="label",
                    ax=axs[1, col],
                    log_scale=log_scale,
                    common_norm=False,
                    element="step",
                )
                g.set_xlabel(colname)
                if colname == "ΔPOWER/ΔSNR":
                    axs[0, col].set_xlim(plotted[plotted["SNR"] >= i]["ΔPOWER/ΔSNR"].quantile([0.05, 0.95]))
                    # axs[1, col].set_xlim(plotted["ΔPOWER/ΔSNR"].quantile([0.1, 0.9]))
                if period and (colname == "Period"):
                    axs[0, col].axvline(period)
                    axs[1, col].axvline(period)
            except TypeError:
                ...
        plt.suptitle(f"{name}")
        plt.tight_layout()
        plt.savefig(pathlib.Path(simdir, f"n{runs:0>6}-min_SNR={i:0>2.2f}.png"))


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
