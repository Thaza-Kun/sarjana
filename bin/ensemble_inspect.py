#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-05-10
# Python Version: 3.12

"""Plots the distribution of peak properties from ensemble
"""

from userust import parse_arguments, Ensemble

import argparse
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_frb_data(name: str, arguments: argparse.Namespace) -> np.ndarray:
    print(f"Reading FRB data: {name}")
    return np.load(pathlib.Path(arguments.parent, name, f"fluence.npy"))


def main(arguments: argparse.Namespace):
    name = arguments.name
    outdir = arguments.outdir

    period = arguments.period
    rate = arguments.rate

    cutoff_SNR = arguments.min_SNR
    runs = arguments.runs

    simdir = pathlib.Path(outdir, name)
    simdir.mkdir(parents=True, exist_ok=True)
    datadir = pathlib.Path(simdir, "ensembles")
    datadir.mkdir(parents=True, exist_ok=True)

    n_events = len(read_frb_data(name, arguments))

    print(f"Loading ensemble data from {datadir}")
    frb_ensemble = Ensemble(
        power=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-power.npy")),
        snr=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-snr.npy")),
        freq=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-freq.npy")),
        group=np.load(pathlib.Path(datadir, f"n{runs:0>6}-frb-group.npy")),
    )
    sim_ensemble = Ensemble(
        power=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-power.npy")),
        snr=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-snr.npy")),
        freq=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-freq.npy")),
        group=np.load(pathlib.Path(datadir, f"n{runs:0>6}-sim-group.npy")),
    )
    print(f"Before filter")
    print(f"SIM-ENSEMBLE")
    print(len(sim_ensemble.power))
    print(len(sim_ensemble.snr))
    print(len(sim_ensemble.freq))
    print(len(sim_ensemble.group))

    print(f"FRB-ENSEMBLE")
    print(len(frb_ensemble.power))
    print(len(frb_ensemble.snr))
    print(len(frb_ensemble.freq))
    print(len(frb_ensemble.group))
    # Required to filter first to unify length (for some reason the length isnt uniform)
    frb_ensemble.filter(snr=0, power=0)
    sim_ensemble.filter(snr=0, power=0)

    print(f"After filter with snr=0 power=0")
    print(f"SIM-ENSEMBLE")
    print(len(sim_ensemble.power))
    print(len(sim_ensemble.snr))
    print(len(sim_ensemble.freq))
    print(len(sim_ensemble.group))

    print(f"FRB-ENSEMBLE")
    print(len(frb_ensemble.power))
    print(len(frb_ensemble.snr))
    print(len(frb_ensemble.freq))
    print(len(frb_ensemble.group))


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
