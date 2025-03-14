# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "polars",
#     "pyarrow",
#     "scipy",
# ]
# ///
"""Plot many waterfall for repeater from parquet file"""
import argparse
import pathlib
from typing import Tuple
import numpy as np
import pyarrow.parquet as pq
import polars as pl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec
import scipy


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dir", type=pathlib.Path, required=True)
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path)
    return parser.parse_args()


def boxcar_kernel(width) -> np.ndarray:
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_burst_snr(ts, min_width=1, max_width=128) -> np.ndarray:
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
    return snrs[best_idx]


def main(arguments: argparse.Namespace) -> None:
    name: str = arguments.name
    dir: pathlib.Path = arguments.dir
    catalog: pathlib.Path = arguments.catalog

    catalog = pl.read_csv(catalog, null_values="-9999")
    f = catalog.filter(pl.col("repeater_name") == name)
    events = f.select(pl.col("tns_name")).unique().to_numpy().flatten()
    mjd_400 = (
        f.group_by(pl.col("tns_name"))
        .agg(pl.col("mjd_400").max())
        .select(pl.col("mjd_400"))
        .to_numpy()
        .flatten()
    )

    rows = int(len(events) / 3)
    fig = plt.figure(figsize=(10, 3.5 * rows))
    gs_main = gspec.GridSpec(nrows=rows, ncols=3, figure=fig)

    for i, e in enumerate(events):
        try:
            file = dir / f"{e}_waterfall.h5.parquet"
            arr = pq.read_table(file)
        except FileNotFoundError:
            file = dir / f"{e}_waterfall.parquet"
            arr = pq.read_table(file)

        ### plot dynamic spectrum
        wfall = arr["wfall"].to_numpy()[0].reshape(arr["wfall_shape"].to_numpy()[0])
        ts = arr["ts"].to_numpy()[0]
        time = arr["plot_time"].to_numpy()[0]
        extent = arr["extent"].to_numpy()[0]
        dt = arr["dt"].to_numpy()[0]
        DM = arr["dm"].to_numpy()[0]

        snr = find_burst_snr(ts)

        # time stamps relative to peaks
        idx_peak = np.argmax(ts)
        extent = np.array([*extent[0:2] - time[idx_peak], *extent[2:]])
        time = time - (time[idx_peak] + (dt / 2))
        time = np.array([*time, time[-1] + dt])

        ts = [*ts, ts[-1]]

        wfall[np.isnan(wfall)] = np.nanmedian(
            wfall
        )  # replace nans in the data with the data median
        # use standard deviation of residuals to set color scale
        vmin = np.nanpercentile(wfall, 1)
        vmax = np.nanpercentile(wfall, 90)

        gs = gspec.GridSpecFromSubplotSpec(
            ncols=1, nrows=2, subplot_spec=gs_main[i], height_ratios=[1, 3], hspace=0.0
        )

        plot_img = plt.subplot(gs[1])
        plot_ts = plt.subplot(gs[0], sharex=plot_img)

        cmap = plt.cm.Greys
        plot_img.imshow(
            wfall,
            extent=extent,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="none",
        )
        # plot_img.set_xlabel("Time [ms]", fontsize=9)
        # plot_img.set_ylabel("Frequency [MHz]", fontsize=9)
        plot_img.set_yticks([400, 500, 600, 700, 800])

        plot_ts.step(time, ts, where="post")
        plot_ts.set_xlim(time[0], time[-1])
        plt.setp(plot_ts.get_xticklabels(), visible=False)
        plot_ts.set_yticklabels([], visible=True)
        plot_ts.set_yticks([])
        plot_ts.set_xlim(extent[0], extent[1])
        plt.setp(plot_img.get_xticklabels(), fontsize=9)
        plt.setp(plot_img.get_yticklabels(), fontsize=9)

        plot_ts.annotate(
            f"DM = {DM:.2f}\nSNR = {snr:.2f}",
            (time[-2], 0.9),
            xycoords=("data", "axes fraction"),
            ha="right",
            va="top",
            fontsize=9,
        )

        plt.title(f"MJD {mjd_400[i]:.3f}")

    if arguments.out is None:
        plt.show()
        exit()

    fig.suptitle(name)
    fig.supxlabel("Time [ms]", fontsize=9)
    fig.supylabel("Frequency [MHz]", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{arguments.out}/{name}_waterfall_many.pdf", dpi=300)
    print(f"Saved to {arguments.out}/{name}_waterfall_many.pdf")


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
