# TUI
from copy import deepcopy
import rich
import typer

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sarjana import commands
from sarjana.collections import generate_scattered_gaussian_model
from sarjana.optimize import fit_time_series
from sarjana.signal import (
    find_full_width_nth_maximum,
    scattered_gaussian_signal,
    find_peaks,
)
from sarjana.handlers import ParquetWaterfall

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
# TODO Investigate wfall by strip

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
def debug(
    frb: str = typer.Argument(...),
):
    """Misc functions to be debug"""
    import dfdt
    from pathlib import Path
    import os
    from sarjana.signal.transform import autocorrelate_waterfall

    burstpath = Path(os.getenv("DATAPATH"), "23891929_DM348.8_waterfall.npy")
    burst_ = np.load(burstpath)
    burst = ParquetWaterfall(Path(os.getenv("DATAPATH"), "raw", "wfall", frb))

    # burst parameters
    dm_uncertainty = 0.2  # pc cm-3
    source = "R3"
    eventid = "23891929"

    # instrument parameters
    dt_s = 0.00098304
    df_mhz = 0.0244140625
    nchan = 16384
    freq_bottom_mhz = 400.1953125
    freq_top_mhz = 800.1953125

    timeseries = np.nansum(burst.wfall, axis=0)
    timeseries_var = np.nanstd(np.diff(timeseries))
    peaks, _ = find_peaks(timeseries, prominence=timeseries_var)
    widths = find_full_width_nth_maximum(timeseries, peaks, n=10)
    width = np.array([*widths]).max()

    peak = peaks[0]

    ds = dfdt.DynamicSpectrum(dt_s, df_mhz, nchan, freq_bottom_mhz, freq_top_mhz)
    DS = dfdt.DynamicSpectrum(
        burst.dt,
        np.diff(burst.plot_freq)[0],
        burst.wfall.shape[1],
        burst.plot_freq.min(),
        burst.plot_freq.max(),
    )
    data = dfdt.ac_mc_drift(
        burst.wfall,
        dm_uncertainty,
        burst.eventname,
        burst.eventname,
        DS,
        dm_trials=100,
        mc_trials=100,
        peak=peak,
        width=width,
    )

    plt.imshow(data, aspect="auto")
    plt.title(burst.eventname)
    plt.savefig(f"dfdt_{burst.eventname}_autocorrelate2d.png")
    data = autocorrelate_waterfall(
        burst.wfall,
        burst.dt,
        np.diff(burst.plot_freq).min(),
        nchannels=len(burst.plot_freq),
        bottom_freq=burst.plot_freq.min(),
        top_freq=burst.plot_freq.max(),
    )
    plt.imshow(data, aspect="auto")
    plt.title(burst.eventname)
    plt.savefig(f"custom_{burst.eventname}_autocorrelate2d.png")
    # plt.show()


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
