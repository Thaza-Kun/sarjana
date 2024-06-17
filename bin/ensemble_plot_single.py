#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-05-10
# Python Version: 3.12

"""Plots the distribution of peak properties from ensemble
"""

from matplotlib.ticker import LogLocator, MaxNLocator
from userust import parse_arguments, Ensemble

import argparse
import pathlib
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu

# from ndtest import ks2d2s

warnings.filterwarnings("ignore")


def read_frb_data(name: str, arguments: argparse.Namespace) -> np.ndarray:
    print(f"Reading FRB data: {name}")
    return np.load(pathlib.Path(arguments.parent, name, f"fluence.npy"))


def main(arguments: argparse.Namespace):
    name = arguments.name
    outdir = arguments.outdir

    def get_column(columnname: str) -> np.ndarray:
        return np.load(pathlib.Path(arguments.parent, name, f"{columnname}.npy"))

    observations = get_column("mjd_400_barycentered")
    obs_window = observations.max() - observations.min()

    period = arguments.period
    rate = arguments.rate

    min_SNR = arguments.min_SNR
    min_power = arguments.min_power
    runs = arguments.runs
    # TEMPORARILY uses the seed argument
    grid_num = arguments.seed if arguments.seed != 16 else 5  # Overriding the default
    pdgram_name = arguments.periodogram.upper()

    PDGRAM = {"LS": "Lomb-Scargle", "PDM": "Phase Dispersion Minimization"}

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)
    plotdir = pathlib.Path(simdir, "single", f"n{runs:0>6}-g{arguments.grid:0>6}")
    plotdir.mkdir(parents=True, exist_ok=True)
    pdgramdir = pathlib.Path(simdir, "periodogram")
    pdgramdir.mkdir(parents=True, exist_ok=True)

    n_events = len(read_frb_data(name, arguments))

    # print(f"Loading example data from {datadir}")
    # eg_frb_power = np.load(
    #     pathlib.Path(datadir, f"pgram={pdgram_name}-eg-frb-power.npy")
    # )
    # eg_frb_freq = np.load(pathlib.Path(datadir, f"pgram={pdgram_name}-eg-frb-freq.npy"))
    # eg_sim_power = np.load(
    #     pathlib.Path(datadir, f"pgram={pdgram_name}-eg-sim-power.npy")
    # )
    # eg_sim_freq = np.load(pathlib.Path(datadir, f"pgram={pdgram_name}-eg-sim-freq.npy"))

    figscale = 0.5
    palette = sns.color_palette()

    # _fig, axs = plt.subplots(2, 1, figsize=(15*figscale, 9*figscale), sharex=True)
    # axs[0].semilogx(1 / eg_frb_freq, eg_frb_power, color=palette[1])
    # axs[0].set_title(f"Observed | events = {n_events}")
    # axs[0].set_xlabel("Period, T (d)")
    # axs[0].set_ylabel("Power, P")
    # limscale = max(
    #     max(eg_frb_power) - min(eg_frb_power), max(eg_sim_power) - min(eg_sim_power)
    # )
    # axs[0].set_ylim(
    #     min(min(eg_frb_power), min(eg_sim_power)) - 0.05 * limscale,
    #     max(max(eg_frb_power), max(eg_sim_power)) + 0.05 * limscale,
    # )
    # axs[1].semilogx(1 / eg_sim_freq, eg_sim_power, color=palette[0])
    # axs[1].set_title("Simulated Constant Noise")
    # axs[1].set_xlabel("Period, T (d)")
    # axs[1].set_ylabel("Power, P")
    # axs[1].set_ylim(
    #     min(min(eg_frb_power), min(eg_sim_power)) - 0.05 * limscale,
    #     max(max(eg_frb_power), max(eg_sim_power)) + 0.05 * limscale,
    # )
    # plt.suptitle(f"{PDGRAM[pdgram_name]} Periodogram for {name}")
    # plt.tight_layout()
    # plt.savefig(pathlib.Path(pdgramdir, f"pgram={pdgram_name}.png"))
    # plt.close()

    print(f"Loading ensemble data from {datadir}")
    # frb_ensemble = Ensemble(
    #     power=np.load(
    #         pathlib.Path(
    #             datadir,
    #             f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-frb-power.npy",
    #         )
    #     ),
    #     snr=np.load(
    #         pathlib.Path(
    #             datadir,
    #             f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-frb-snr.npy",
    #         )
    #     ),
    #     freq=np.load(
    #         pathlib.Path(
    #             datadir,
    #             f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-frb-freq.npy",
    #         )
    #     ),
    #     group=np.load(
    #         pathlib.Path(
    #             datadir,
    #             f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-frb-group.npy",
    #         )
    #     ),
    # )
    sim_ensemble = Ensemble(
        power=np.load(
            pathlib.Path(
                datadir,
                f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-sim-power.npy",
            )
        ),
        snr=np.load(
            pathlib.Path(
                datadir,
                f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-sim-snr.npy",
            )
        ),
        freq=np.load(
            pathlib.Path(
                datadir,
                f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-sim-freq.npy",
            )
        ),
        group=np.array([1])
        # group=np.load(
        #     pathlib.Path(
        #         datadir,
        #         f"n{runs:0>6}-g{arguments.grid:0>6}-pgram={pdgram_name}-sim-group.npy",
        #     )
        # ),
    )
    # Required to filter first to unify length (for some reason the length isnt uniform)
    # frb_ensemble.filter(snr=0, power=-1)
    # sim_ensemble.filter(snr=0, power=-1)

    # frb_df = pd.DataFrame(
    #     {
    #         "Period": 1/np.array(frb_ensemble.freq),
    #         "Power": frb_ensemble.power,
    #         "SNR": frb_ensemble.snr,
    #         "group": frb_ensemble.group,
    #         "label": "observed"
    #     }
    # )
    sim_df = pd.DataFrame(
        {
            "Period": 1/np.array(sim_ensemble.freq),
            "Power": sim_ensemble.power,
            "SNR": sim_ensemble.snr,
            "label": "simulated"
        }
    )

    # plotted = pd.concat([sim_df, frb_df[frb_df["group"] == 1]])
    plotted = sim_df

    print(f"Drawing figure {name} [{min_SNR:.2f}, {min_power:.8f}] to {outdir}")
    data = plotted[(plotted["SNR"] >= min_SNR) & (plotted["Power"] >= min_power)]
    if len(data) == 0:
        print("Data is empty. Skipping...")
        exit()

    _fig, axs = plt.subplots(2, 2, sharex="col", figsize=(15*figscale, 7*figscale))

    num_bins_period = 50
    num_bins_power = 20

    bins_period = np.geomspace(data["Period"].min(), data["Period"].max(), num_bins_period)
    bins_power = np.linspace(data["Power"].min(), data["Power"].max(), num_bins_power)
    # hist_obs_period, _ = np.histogram(
    #     data[data["label"] == "observed"]["Period"].to_numpy(),
    #     density=True,
    #     bins=bins_period,
    # )
    # hist_sim_period, _ = np.histogram(
    #     data[data["label"] == "simulated"]["Period"].to_numpy(),
    #     density=True,
    #     bins=bins_period,
    # )
    # hist_obs_power, _ = np.histogram(
    #     data[data["label"] == "observed"]["Power"].to_numpy(),
    #     density=True,
    #     bins=bins_power,
    # )
    # hist_sim_power, _ = np.histogram(
    #     data[data["label"] == "simulated"]["Power"].to_numpy(),
    #     density=True,
    #     bins=bins_power,
    # )
    # mwu_period = ks_2samp(
    #     data[(data["label"] == "observed")]["Period"].to_numpy(),
    #     data[data["label"] == "simulated"]["Period"].to_numpy(),
    #     alternative="two-sided",
    # )
    # mwu_power = ks_2samp(
    #     data[(data["label"] == "observed")]["Power"].to_numpy(),
    #     data[data["label"] == "simulated"]["Power"].to_numpy(),
    #     alternative="two-sided",
    # )
    # mwu_power_gr = ks_2samp(
    #     data[(data["label"] == "observed")]["Power"].to_numpy(),
    #     data[data["label"] == "simulated"]["Power"].to_numpy(),
    #     alternative="less",
    # )
    axs[0, 1].axis("off")
    # label_size="large"
    # scale = 0.14
    # shift = 2.7*scale
    # axs[0, 1].text(
    #     0,
    #     4*scale + shift,
    #     f"Event count    = {n_events}",
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    # axs[0, 1].text(
    #     0,
    #     3*scale + shift,
    #     f"Event window = {obs_window:.2f} days",
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    # axs[0, 1].text(
    #     0,
    #     2*scale + shift,
    #     f"Burst rate       = {rate:.3e} 1/h",
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    # axs[0, 1].text(
    #     0,
    #     3*scale,
    #     f"Minimum SNR = {min_SNR:.2f}",
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    # axs[0, 1].text(
    #     0,
    #     2*scale,
    #     r"$P(F_\text{obs}(T) = F_\text{noise}(T))$"
    #     + (
    #         f" = {mwu_period.pvalue:.5f}"
    #         if mwu_period.pvalue > 1e-5
    #         else r"$\leq$ 0.00001"
    #     ),
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    # axs[0, 1].text(
    #     0,
    #     1*scale,
    #     r"$P(F_\text{obs}(\text{P}) = F_\text{noise}(\text{P}))$"
    #     + (
    #         f" = {mwu_power.pvalue:.5f}"
    #         if mwu_power.pvalue > 1e-5
    #         else r"$\leq$ 0.00001"
    #     ),
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    # axs[0, 1].text(
    #     0,
    #     0.00,
    #     r"$P(F_\text{obs}(\text{P}) \ngtr F_\text{noise}(\text{P}))$"
    #     + (
    #         f" = {mwu_power_gr.pvalue:.5f}"
    #         if mwu_power_gr.pvalue > 1e-5
    #         else r"$\leq$ 0.00001"
    #     ),
    #     transform=axs[0, 1].transAxes,
    #     fontsize=label_size
    # )
    g = sns.histplot(
        data,
        x="Period",
        hue="label",
        ax=axs[0, 0],
        common_norm=False,
        element="step",
        legend=True,
        bins=bins_period,
        stat='density'
    )
    g.set_yticks([])
    g.set_ylabel("Density")
    g.set_xlabel("Period, T (d)")
    g = sns.histplot(
        data,
        x="Power",
        hue="label",
        ax=axs[1, 1],
        common_norm=False,
        element="step",
        legend=False,
        bins=bins_power,
        stat='density'
    )
    g.set_yticks([])
    axs[1,1].xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 5]))
    g.set_ylabel("Density")
    g.set_xlabel("Power, P (d)")
    g = sns.histplot(
        data,
        x="Period",
        y="Power",
        hue="label",
        ax=axs[1, 0],
        log_scale=(True, False),
        common_norm=False,
        legend=False,
        bins=(num_bins_period, num_bins_power),
        stat='count'
    )
    axs[1,0].xaxis.set_major_locator(LogLocator(numticks=3))
    g.set_yticks([])
    g.set_ylabel("Power, P")
    g.set_xlabel("Period, T")
    plt.suptitle(f"{PDGRAM[pdgram_name]} Periodogram Profile for {name}", fontsize='x-large')
    plt.tight_layout()
    plt.savefig(
        pathlib.Path(
            plotdir,
            f"pgram={pdgram_name}-n{runs:0>6}-min_SNR={min_SNR:0>2.1f}-min_power={min_power:0>8.5f}.png",
        ),
    )
    plt.close()


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
