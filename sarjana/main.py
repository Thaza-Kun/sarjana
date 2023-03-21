# TUI
from copy import deepcopy
import rich
import typer

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sarjana import commands
from sarjana.optimize import fit_time_series
from sarjana.signal import (
    scattered_gaussian_signal,
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

# TODO Betulkan find_burst untuk multiburst
# 👉 Tengok ni https://www.chime-frb.ca/catalog/FRB20190109A
# TODO Cari taburan sigma/tau dan sigma
# TODO Gunakan taburan sigma/tau dan sigma untuk tentukan broadband vs shortband
# TODO FIX PEAKS (Some peaks are not aligned)


@app.command()
def debug(
    frb: str = typer.Argument(...),
):
    """Misc functions to be debug"""
    # rich.print([*deepcopy(scattered_gaussian_signal.__annotations__).keys()])
    matplotlib.use("TkAgg")
    data = ParquetWaterfall(frb).remove_rfi()
    func = scattered_gaussian_signal
    plt.plot(data.plot_time, data.model_ts, label="data", drawstyle="steps-post")
    params = fit_time_series(
        func,
        data.plot_time,
        data.model_ts,
        params=[1, 1, data.model_ts.max(), data.plot_time[data.model_ts.argmax()]],
    )
    rich.print([i[0] for i in params.values()])
    # plt.plot(
    #     data.plot_time,
    #     func(data.plot_time, *params),
    #     label=f"sigma/tau={params[0]/params[1]:.3e}",
    #     drawstyle='steps-post'
    # )
    # # if tail:
    # #     plt.plot(data.plot_time, reciprocal(data.plot_time, params[-1], center))
    # #     plt.plot(data.plot_time, gauss(data.plot_time, amplitude, *params[1:-1]))
    # rich.print(np.sqrt(np.diag(pcovar)))
    # plt.title(data.eventname)
    # plt.legend()
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
    peaks: bool = typer.Option(False, help="Whether to show peaks"),
):
    """Plots a FacetGrid of flux profiles of each FRB based on categories defined in embedding file."""
    commands.plot_many_flux_profile_by_clustering_groups(
        profile, embedding, savefile, size, draw_peaks=peaks
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
