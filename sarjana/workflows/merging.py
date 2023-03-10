from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns

import scipy

from sarjana.data.collections import load_2d_embedding, load_catalog, load_profiles

datasource = Path("/mnt/d/home/datasets/sarjana/")

cfod_path = Path(datasource, "./raw/cfod/")
external_data_path = Path(datasource, "./raw/external/")

chimefrb_profile = Path(cfod_path, "chimefrb_profile.parquet")
embedding_data = Path(external_data_path, "2d-embeddings_chen2022.csv")

profiles = load_profiles(chimefrb_profile)

embedding = load_2d_embedding(embedding_data)

catalog = load_catalog(Path(external_data_path, "catalog_Hashimoto2022.csv"))


def merge() -> pd.DataFrame:
    return profiles.merge(embedding, left_on="eventname", right_on="tns_name")


def boxcar_kernel(width: int) -> np.ndarray:
    """
    Returns the boxcar kernel of given width normalized by
    sqrt(width) for S/N reasons.

    Parameters
    ----------
    width : int
        Width of the boxcar.
    Returns
    -------
    boxcar : np.ndarray
        Boxcar of width `width` normalized by sqrt(width).
    """
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_burst(ts, min_width=1, max_width=128) -> Tuple[int, int, float]:
    """
    Find burst peak and width using boxcar convolution.

    Parameters
    ----------
    ts : array_like
        Time-series.
    min_width : int, optional
        Minimum width to search from, in number of time samples.
        1 by default.
    max_width : int, optional
        Maximum width to search up to, in number of time samples.
        128 by default.
    plot : bool, optional
        If True, show figure to summarize burst finding results.
    Returns
    -------
    peak : int
        Index of the peak of the burst in the time-series.
    width : int
        Width of the burst in number of samples.
    snr : float
        S/N of the burst.

    """
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


def plot(data: pd.DataFrame) -> None:
    item = data.loc[0, :]
    peak, width, snr = find_burst(item["ts"])
    item["plot_time"] = item["plot_time"] - item["plot_time"][np.argmax(item["ts"])]
    item["plot_time"] = item["plot_time"] - item["dt"] / 2
    item["plot_time"] = np.append(item["plot_time"], item["plot_time"][-1] + item["dt"])
    item["ts"] = np.append(item["ts"], item["ts"][-1])
    item["model_ts"] = np.append(item["model_ts"], item["model_ts"][-1])
    g = sns.lineplot(item, x="plot_time", y="ts", drawstyle="steps-post")
    sns.lineplot(item, x="plot_time", y="model_ts", drawstyle="steps-post", ax=g)
    g.set_title(item["eventname"])
    g.axvspan(
        max(
            item["plot_time"].min(),
            item["plot_time"][peak] + 0.5 * item["dt"] - (0.5 * width) * item["dt"],
        ),
        min(
            item["plot_time"].max(),
            item["plot_time"][peak] + 0.5 * item["dt"] + (0.5 * width) * item["dt"],
        ),
        facecolor="tab:blue",
        edgecolor=None,
        alpha=0.1,
    )
    g.get_figure().savefig(f"{item['eventname']}.png")
