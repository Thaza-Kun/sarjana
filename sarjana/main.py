# TUI
from copy import deepcopy
import rich
import typer

from pathlib import Path
import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sarjana import commands
from sarjana.collections import generate_scattered_gaussian_model
from sarjana.signal.optimize import fit_tilted_2d_gaussian, fit_time_series
from sarjana.signal.analyze import (
    find_frequency_drift,
)
from sarjana.handlers import ParquetWaterfall
from sarjana.signal.properties import full_width_nth_maximum, find_peaks
from sarjana.signal.template import scattered_gaussian_signal

app = typer.Typer()

# TODO Pleunis categories:
# 1. Broadband simple bursts composed of one peak that can be reasonably well described
#   by one Gaussian function in time, convolved with an exponential scattering tail if
#   scattering is not negligible. Their spectra can be well described by a power-law
#   function.
# 2. Narrowband simple bursts—with spectra that are more like Gaussians.
# 3. Complex bursts composed of multiple peaks with similar frequency extent, with one
#   of the peaks sometimes being a much dimmer precursor or post-cursor—they can be
#   broadband or narrowband.
# 4. Complex bursts composed of multiple subbursts that drift downward in frequency
#    as time progresses.

# TODO So apparently these are not even properly measured in Janskys?
# So I need to find the proper values from the calibrated wfall?
# Rujuk: https://chime-frb-open-data.github.io/waterfall/#plotting-calibrated-data

# TODO Get all multipeak programmatically and run the algorithm
# TODO Refine dfdt algorithm
# TODO Plot the dfdt (for visual confirmation)

import seaborn as sns


def plot_mean(data, **kwargs):
    axes = plt.gca()
    _column: np.ndarray = data.to_numpy()
    # n25, median, n75 = np.quantile(_column, [0.1, 0.5, 0.9])
    midpoint = np.median(_column)
    stdev = _column.std()
    lower = midpoint - 1.5 * stdev
    upper = midpoint + 1.5 * stdev
    axes.axvspan(
        max(0, lower),
        min(upper, _column.max()),
        facecolor="red",
        edgecolor=None,
        alpha=0.1,
    )
    # axes.axvline(midpoint)


@app.command()
def debug(file: str = typer.Argument(...)):
    """Misc functions to be debug"""
    wfall = np.load(Path(os.getenv("DATAPATH"), "23891929_DM348.8_waterfall.npy"))
    print(find_frequency_drift(wfall))


@app.command()
def drift(
    n: int = typer.Option(10, help="Number of monte carlo run"),
    m: int = typer.Option(10, help="Number of DM trial run"),
    events: int = typer.Option(0, help="How many events to calculate. (0 = all)"),
    path: str = typer.Option('.', help="Path of results"),
):
    """Find frequency drift"""
    commands.find_frequency_drift_acmc_all(n_mc=n,n_dm=m, event_total=events, path=Path(path))


@app.command()
def plot(
    profile: str = typer.Argument(
        ..., help="Path to flux profile data (in `.parquet`)"
    ),
    embedding: str = typer.Argument(..., help="Path to embedding data (in `.csv`)"),
    savefile: str = typer.Argument(
        ..., help="The name of the saved plot file. No extension in name."
    ),
    path: str = typer.Option(".", help="The path to save figure."),
    size: int = typer.Option(30, help="The number of FRB in each plot."),
    burst: bool = typer.Option(False, help="Whether to show peaks"),
):
    """Plots a FacetGrid of flux profiles of each FRB based on categories defined in embedding file."""
    commands.plot_many_flux_profile_by_clustering_groups(
        profile, embedding, savefile, size, highlight_burst=burst
    )


@app.command()
def download(
    eventnames: typer.FileText = typer.Argument(
        ..., help="A newline delimited `.txt` file listing eventnames."
    ),
    tofile: str = typer.Option(
        None, help="Filename in `.parquet` to collect downloaded data into."
    ),
    path: str = typer.Argument(".", help="Download file to this path"),
    limit: int = typer.Option(None, help="How many to download"),
):
    """Download waterfall data from CHIME/FRB database"""
    commands.download_waterfall_data_from_chimefrb_database(
        eventnames=eventnames, tofile=tofile, path=path, limit=limit
    )


@app.command()
def combine(
    eventnames: typer.FileText = typer.Argument(
        ..., help="A newline delimited `.txt` file listing eventnames."
    ),
    collectionfile: str = typer.Argument(
        ..., help="Filename in `.parquet` to collect downloaded data into."
    ),
    filepattern: str = typer.Option(
        "{}_waterfall.h5.parquet", help="File pattern to search for parquet."
    ),
):
    """Combine files into a single parquet file"""
    commands.combine_multifile_into_single_parquet_file(
        eventnames=eventnames, collectionfile=collectionfile, filepattern=filepattern
    )


if __name__ == "__main__":
    app()
