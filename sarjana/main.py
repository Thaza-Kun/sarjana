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
    eventname: pd.Series,
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
    savefile: str = typer.Argument(..., help="The name of the saved plot file."),
):
    """Plots a FacetGrid of flux profiles of each FRB based on categories defined in embedding file."""
    prof = pd.read_parquet(profile)
    emb = pd.read_csv(embedding)
    data = merge_embedding_into_profile(prof, emb)
    g = sns.FacetGrid(data.loc[:14, :], col="eventname", col_wrap=5, sharex=False)
    g.map(
        plot_flux_profile,
        "ts",
        "model_ts",
        "plot_time",
        "dt",
        "eventname",
    )
    g.set_ylabels("flux (Jy)")
    g.set_xlabels("time (ms)")
    g.savefig(savefile)


@app.command()
def debug():
    """Use this space to input any operation for debugging purposes"""
    rich.print(np.append(np.array([1, 2, 3, 4, 5]), 1))
