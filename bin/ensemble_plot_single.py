#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-05-10
# Python Version: 3.12

"""Plots the distribution of peak properties from ensemble
"""

from userust import parse_arguments, Ensemble

import argparse
import pathlib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, wilcoxon, chisquare, mannwhitneyu
# from ndtest import ks2d2s


def read_frb_data(name: str, arguments: argparse.Namespace) -> np.ndarray:
    print(f"Reading FRB data: {name}")
    return np.load(pathlib.Path(arguments.parent, name, f"fluence.npy"))


def main(arguments: argparse.Namespace):
    name = arguments.name
    outdir = arguments.outdir

    period = arguments.period
    rate = arguments.rate

    min_SNR = arguments.min_SNR
    min_power = arguments.min_power
    runs = arguments.runs
    # TEMPORARILY uses the seed argument
    grid_num = arguments.seed if arguments.seed != 16 else 5  # Overriding the default
    pdgram_name = arguments.periodogram.upper()

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)
    plotdir = pathlib.Path(simdir, "single", f"n{runs:0>6}-g{arguments.grid:0>6}")
    plotdir.mkdir(parents=True, exist_ok=True)

    n_events = len(read_frb_data(name, arguments))

    print(f"Loading ensemble data from {datadir}")
    frb_ensemble = Ensemble(
        power=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-frb-power.npy")
        ),
        snr=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-frb-snr.npy")
        ),
        freq=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-frb-freq.npy")
        ),
        group=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-frb-group.npy")
        ),
    )
    sim_ensemble = Ensemble(
        power=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-sim-power.npy")
        ),
        snr=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-sim-snr.npy")
        ),
        freq=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-sim-freq.npy")
        ),
        group=np.load(
            pathlib.Path(datadir, f"n{runs:0>6}-g{arguments.grid:0>6}-pdgram={pdgram_name}-sim-group.npy")
        ),
    )
    # Required to filter first to unify length (for some reason the length isnt uniform)
    # frb_ensemble.filter(snr=0, power=0)
    # sim_ensemble.filter(snr=0, power=0)

    plotted = pd.DataFrame(
        {
            "Period": np.concatenate(
                (1 / np.array(sim_ensemble.freq), 1 / np.array(frb_ensemble.freq))
            ),
            "Power": np.concatenate((sim_ensemble.power, frb_ensemble.power)),
            "SNR": np.concatenate((sim_ensemble.snr, frb_ensemble.snr)),
            "label": np.concatenate(
                (
                    ["simulated"] * len(sim_ensemble.freq),
                    ["observed"] * len(frb_ensemble.freq),
                )
            ),
        }
    )

    print(f"Drawing figure {name} [{min_SNR:.2f}, {min_power:.8f}] to {outdir}")
    data = plotted[(plotted["SNR"] >= min_SNR) & (plotted["Power"] >= min_power)]
    if len(data) == 0:
        print("Data is empty. Skipping...")
        exit()
    _fig, axs = plt.subplots(3, 3, sharex="col", figsize=(15, 7))
    for col, colname in enumerate(["Period", "Power", "SNR"]):
        warnings.filterwarnings('ignore')
        if colname == "Period":
            log_scale = True
        else:
            log_scale = False
        num_bins=100
        if log_scale:
            bins = np.geomspace(data[colname].min(), data[colname].max(), num_bins)
            # bins = np.linspace(data[colname].min(), data[colname].max(), 100)
        else:
            bins = np.linspace(data[colname].min(), data[colname].max(), num_bins)
        # print(data[colname].min())
        # print(data[colname].max())
        # print(log_scale)
        g = sns.histplot(
            data,
            x=colname,
            hue="label",
            ax=axs[0, col],
            # log_scale=log_scale,
            common_norm=False,
            cumulative=True,
            legend=False,
            element="step",
            fill=False,
            stat="density",
            bins=bins
        )
        g.set_ylabel("Cumulative Density")
        # print(bins)
        # print(np.histogram(data[data["label"] == "observed"][colname].to_numpy(), density=True, bins=bins))
        # print(np.histogram(data[data["label"] == "simulated"][colname].to_numpy(), density=True, bins=bins))
        hist_obs, b = np.histogram(data[data["label"] == "observed"][colname].to_numpy(),  density=True, bins=bins)
        hist_sim, _ = np.histogram(data[data["label"] == "simulated"][colname].to_numpy(), density=True,  bins=bins)
        lessthan = mannwhitneyu(hist_obs, hist_sim, alternative="less")
        equal = mannwhitneyu(hist_obs, hist_sim, alternative='two-sided')
        greaterthan = mannwhitneyu(hist_obs, hist_sim, alternative="greater")
        if lessthan != np.nan:
            axs[0, col].text(
                data[colname].max(),
                0.25,
                (
                    f"(Obs < Sim) p-value {f"= {lessthan.pvalue:.3f}" if lessthan.pvalue >= 0.001 else "< 0.001"}"
                ),
                ha="right",
                va="center",
            )
        if equal != np.nan:
            axs[0, col].text(
                data[colname].max(),
                0.15,
                (
                    f"(Obs = Sim) p-value {f"= {equal.pvalue:.3f}" if equal.pvalue >= 0.001 else "< 0.001"}"
                ),
                ha="right",
                va="center",
            )
        if greaterthan != np.nan:
            axs[0, col].text(
                data[colname].max(),
                0.05,
                (
                    f"(Obs > Sim) p-value {f"= {greaterthan.pvalue:.3f}" if greaterthan.pvalue >= 0.001 else "< 0.001"}"
                ),
                ha="right",
                va="center",
            )
        axs[0, col].set_ylim(0, 1)
        g = sns.histplot(
            data,
            x=colname,
            hue="label",
            ax=axs[1, col],
            # log_scale=log_scale,
            common_norm=False,
            element="step",
            legend=col == 2,
            bins=bins
        )
        g.set_ylabel("Counts")
        if colname == "Period":
            Y = "Power"
        elif colname == "Power":
            Y = "SNR"
        elif colname == "SNR":
            Y = "Period"
        # stat_test2d = ks2d2s(
        #     data[data["label"] == "observed"][colname].to_numpy(),
        #     data[data["label"] == "observed"][Y].to_numpy(),
        #     data[data["label"] == "simulated"][colname].to_numpy(),
        #     data[data["label"] == "simulated"][Y].to_numpy(),
        #     nboot=10
        # )
        g = sns.histplot(
            data,
            x=colname,
            y=Y,
            hue="label",
            ax=axs[2, col],
            log_scale=(log_scale, False),
            common_norm=False,
            legend=False,
            bins=num_bins
        )
        # axs[2, col].text(
        #     data[colname].max(),
        #     data[Y].max(),
        #     (
        #         f"p-value = {stat_test2d:.3e}"
        #     ),
        #     ha="right",
        #     va="top",
        # )
        g.set_ylabel(Y)
        g.set_xlabel(colname)
        if period and (colname == "Period"):
            axs[0, col].axvline(period)
            axs[1, col].axvline(period)
            axs[2, col].axvline(period)
    title = f"{name} | events = {n_events}"
    if rate:
        title += f" | burst rate = {rate:0>2.2e} 1/h"
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(
        pathlib.Path(
            plotdir, f"n{runs:0>6}-min_SNR={min_SNR:0>2.1f}-min_power={min_power:0>8.5f}.png"
        )
    )
    plt.close()


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
