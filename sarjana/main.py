from typing import Any
import numpy as np
import pandas as pd
import seaborn as sns

import scipy

import typer


def merge_embedding_in_profile(
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
    best_idx = np.nanargmax(snrs)
    return peaks[best_idx], widths[best_idx], snrs[best_idx]


def plot_flux_profile(
    flux: np.ndarray,
    model_flux: np.ndarray,
    time: np.ndarray,
    timedelta: float,
    eventname: str,
) -> Any:
    peak, width, _ = find_burst(flux)
    time = time - time[np.argmax(flux)]
    time = time - timedelta / 2
    time = np.append(time, time[-1] + timedelta)
    flux = np.append(flux, flux[-1])
    model_flux = np.append(model_flux, model_flux[-1])
    g = sns.lineplot(x=time, y=flux, drawstyle="steps-post")
    sns.lineplot(x=time, y=model_flux, drawstyle="steps-post", ax=g)
    g.set_title(eventname)
    g.axvspan(
        max(
            time.min(),
            time[peak] + 0.5 * timedelta - (0.5 * width) * timedelta,
        ),
        min(
            time.max(),
            time[peak] + 0.5 * timedelta + (0.5 * width) * timedelta,
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
    prof = pd.read_parquet(profile)
    emb = pd.read_csv(embedding)
    data = merge_embedding_in_profile(prof, emb)
    item = data.loc[0, :]
    g = plot_flux_profile(
        item["ts"], item["model_ts"], item["plot_time"], item["dt"], item["eventname"]
    )
    g.get_figure().savefig(savefile)
