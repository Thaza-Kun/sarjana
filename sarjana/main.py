from pathlib import Path
from typing import Any, Optional, Union

# Data manipulation
import numpy as np
import pandas as pd

# Data vis
import matplotlib.pyplot as plt
import seaborn as sns

# Signal processing
import scipy

# TUI
import typer
import rich
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)

from sarjana.handlers import H5Waterfall, ParquetWaterfall
from sarjana.download import (
    compress_to_parquet,
    manage_download_waterfall_data_task,
)


def merge_embedding_into_profile(
    profiles: pd.DataFrame, embedding: pd.DataFrame
) -> pd.DataFrame:
    """Injects the embedding data from machine learning into a dataframe of flux profiles.

    Args:
        profiles (pd.DataFrame): A `.parquet` file containing a list of flux profiles.
        embedding (pd.DataFrame): A `.csv` file containing groups, classification, and embedding coordinates.

    Returns:
        pd.DataFrame: A merge of the two.
    """
    return profiles.merge(embedding, left_on="eventname", right_on="tns_name")


def boxcar_kernel(width):
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_burst(ts, min_width=1, max_width=128):
    min_width = int(min_width)
    max_width = int(max_width)
    # do not search widths bigger than timeseries
    widths = list(range(min_width, min(max_width + 1, len(ts) - 2)))
    # envelope finding
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)
    for i in range(len(widths)):
        convolved = scipy.signal.convolve(ts, boxcar_kernel(widths[i]), mode="same")
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]
    try:
        best_idx = np.nanargmax(snrs)
    except ValueError:
        return peaks, widths, snrs
    return peaks[best_idx], widths[best_idx], snrs[best_idx]


Vector = Union[np.ndarray, pd.Series]


def plot_flux_profile(
    flux: pd.Series,
    model_flux: pd.Series,
    time: pd.Series,
    timedelta: pd.Series,
    axes: Optional[plt.Axes] = None,
    **kwargs,
) -> Any:
    """A single flux plot.

    Args:
        flux (pd.Series): Flux data as a time series
        model_flux (pd.Series): Denoised flux data as a time series
        time (pd.Series): Time axis
        timedelta (pd.Series): Differences between two units of time
        eventname (pd.Series): The name of the FRB
        axes (Optional[plt.Axes], optional): The axes to draw to. If None, it queries `plt.gca()`. Defaults to None.

    Returns:
        Any: A Seaborn plot.
    """
    axes = plt.gca() if axes is None else axes
    # Reshape data
    _time: np.ndarray = time.to_numpy()[0]
    _model_flux: np.ndarray = model_flux.to_numpy()[0]
    _flux: np.ndarray = flux.to_numpy()[0]
    _timedelta: float = timedelta.to_numpy()[0]

    # Find burst
    peak, width, _ = find_burst(_flux)

    # Resize time
    _time = _time - _time[np.argmax(_flux)]
    _time = _time - (_timedelta / 2)

    # TODO Remove RFI from flux data as described here https://chime-frb-open-data.github.io/waterfall/#removing-the-radio-frequency-interference
    # REQUIRES Waterfall data

    # Add one more step after final point
    _time = np.append(_time, _time[-1] + timedelta)
    _flux = np.append(_flux, _flux[-1])
    _model_flux = np.append(_model_flux, _model_flux[-1])

    g = sns.lineplot(
        x=_time,
        y=_flux,
        drawstyle="steps-post",
        ax=axes,
        color="orange",
    )
    sns.lineplot(x=_time, y=_model_flux, drawstyle="steps-post", ax=g)
    g.set(xlim=[_time[0], _time[-1]])

    # Color the burst event with a greyish box
    g.axvspan(
        max(
            _time.min(),
            _time[peak] + 0.5 * _timedelta - (0.5 * width) * _timedelta,
        ),
        min(
            _time.max(),
            _time[peak] + 0.5 * _timedelta + (0.5 * width) * _timedelta,
        ),
        facecolor="tab:blue",
        edgecolor=None,
        alpha=0.1,
    )
    return g


app = typer.Typer()


@app.command()
def plot(
    profile: str = typer.Argument(
        ..., help="Path to flux profile data (in `.parquet`)"
    ),
    embedding: str = typer.Argument(..., help="Path to embedding data (in `.csv`)"),
    savefile: str = typer.Argument(
        ..., help="The name of the saved plot file. No extension in name."
    ),
    size: int = typer.Option(30, help="The number of FRB in each plot."),
):
    """Plots a FacetGrid of flux profiles of each FRB based on categories defined in embedding file."""
    prof = pd.read_parquet(profile)
    emb = pd.read_csv(embedding)
    data = merge_embedding_into_profile(prof, emb)
    categories = data["hdbscan_group"].unique()
    for cat in categories:
        to_plot = data[data["hdbscan_group"] == cat].drop_duplicates(subset="eventname")
        for pos in range(0, len(to_plot), size):
            loop_num = (pos / size) + 1
            g = sns.FacetGrid(
                to_plot[pos : pos + size],
                col="eventname",
                col_wrap=5,
                sharex=False,
                sharey=False,
            )
            g.map(
                plot_flux_profile,
                "ts",
                "model_ts",
                "plot_time",
                "dt",
                "eventname",
            )
            g.fig.suptitle(cat + " " + str(loop_num))
            g.set_ylabels("flux (Jy)")
            g.set_xlabels("time (ms)")
            g.savefig(f"{savefile}-{cat}-{loop_num}.png")


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
    basepath = Path(path)
    baseurl = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data"
    try:
        if tofile is None:
            raise FileNotFoundError
        currently_available_names = pd.read_parquet(
            tofile, engine="pyarrow", columns=["eventname"]
        )["eventname"].tolist()
    except FileNotFoundError:
        currently_available_names = []

    names_to_download = [
        *{i.strip("\n") for i in eventnames if i not in currently_available_names}
    ]
    if limit:
        names_to_download = names_to_download[:limit]
    progress = Progress(
        TextColumn("{task.id:>3d}/"),
        TextColumn(f"{len(names_to_download)-1:>3d} "),
        TextColumn(
            "[bold blue]{task.fields[filename]}",
            justify="right",
        ),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeElapsedColumn(),
        "/",
        TimeRemainingColumn(),
        transient=True,
    )
    manage_download_waterfall_data_task(
        names_to_download,
        progress_manager=progress,
        basepath=basepath,
        baseurl=baseurl,
        collect_to=tofile,
        DataHandler=H5Waterfall,
    )


@app.command()
def debug(frb: str = typer.Argument(..., help="FRB file")):
    rich.print(ParquetWaterfall(frb).plot_freq)
    # compress_to_parquet(fromfile=frb, tofile=None, data_handler=H5Waterfall)
