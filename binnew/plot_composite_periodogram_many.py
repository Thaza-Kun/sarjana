import pathlib
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.rcParams["savefig.dpi"] = 500
FIGSCALE = 2.5
BASE_SIZE = 6
plt.rcParams["font.size"] = FIGSCALE*12/1.5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--simname", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--size", type=float, default=1.0)
    return parser.parse_args()


arguments = parse_arguments()

names = arguments.names
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence
size = arguments.size

outdir = pathlib.Path(f"{arguments.output}/{simname}")
print(names)
cols = 4
row_lim = 5
rows = (len(names) // cols) + 1
pages = 1 + ((rows - 1) // row_lim)
table = defaultdict()
colors = {
    "filtered": "tab:blue",
    "antifiltered": "orange",
    "filtered_p": "tab:red",
    "antifiltered_p": "mediumseagreen",
}
labels = {
    "filtered": r"$1.5<P\leq\tau$",
    "antifiltered": r"$\tau< P\leq 2\tau$",
    "filtered_p": r"$P_{cc} \geq 0.050$ ($P\leq\tau$)",
    "antifiltered_p": r"$P_{cc} \geq 0.050$ ($\tau< P$)",
}
sizes = {
    "filtered": 1.5 * BASE_SIZE,
    "antifiltered": 1 * BASE_SIZE,
    "filtered_p": BASE_SIZE,
    "antifiltered_p": BASE_SIZE,
}
markers = {
    "filtered": "+",
    "antifiltered": "x",
    "filtered_p": "D",
    "antifiltered_p": "s",
}
PEAK_COLOR = "black"
PEAK_SIZE = sizes["filtered_p"]*25
for p in range(pages):
    fig, axe = plt.subplots(
        row_lim,
        cols,
        figsize=(FIGSCALE*8.72, FIGSCALE*8.30),
        sharey=True,
        sharex=False,
    )
    axes = np.array(axe).reshape(-1)
    N = cols * row_lim
    for i, name in enumerate(tqdm(names[(p) * N : N + (p) * N])):
        if i >= axes.shape[0]:
            i -= 1
            break
        if i + 1 == len(names):
            axes[i].set_axis_off()
            i += 1
        ax = axes[i]
        outdirsize = pathlib.Path(outdir, name, f"{size:.2f}")
        for d in outdirsize.iterdir():
            if d.is_dir():
                vname, val = d.name.split("=")
                if vname == "value_number_of_events":
                    n_event = int(val)
                elif vname == "value_total_exposure":
                    exposure_hr = float(val)
                elif vname == "value_event_window":
                    event_window = int(float(val))
                # elif vname == "value_burst_rate":
                #     burst_rate = float(val)
        x_mean = np.load(pathlib.Path(outdirsize, f"kde-{runs}.mean.x.npy"))
        y_mean = np.load(pathlib.Path(outdirsize, f"kde-{runs}.mean.y.npy"))
        covar = np.load(pathlib.Path(outdirsize, f"kde-{runs}.covar.npy"))

        CUTOFF = 1
        filtered = lambda x: x <= CUTOFF * event_window
        antifiltered = lambda x: x > CUTOFF * event_window
        periods = np.load(pathlib.Path(outdirsize, f"transient-{runs}.grid.period.npy"))
        chisquare_stat = np.load(
            pathlib.Path(outdirsize, f"transient-{runs}.power.chisquare.npy")
        )
        inactive_stat = np.load(
            pathlib.Path(outdirsize, f"transient-{runs}.power.inactive.npy")
        )

        rv = multivariate_normal([x_mean, y_mean], covar)
        x, y = np.mgrid[
            chisquare_stat.min() : chisquare_stat.max() : 0.1,
            0:1.6:0.1,
        ]
        pos = np.dstack((x, y))
        pdf = rv.pdf(pos)

        cdf1 = rv.cdf(
            np.dstack(
                (chisquare_stat[filtered(periods)], inactive_stat[filtered(periods)])
            )
        )
        cdf2 = rv.cdf(
            np.dstack(
                (
                    chisquare_stat[antifiltered(periods)],
                    inactive_stat[antifiltered(periods)],
                )
            )
        )

        prob = 1 - cdf1
        point5_peak = np.argwhere(cdf1 > 1 - confidence)
        prob2 = 1 - cdf2
        point5_peak2 = np.argwhere(cdf2 > 1 - confidence)
        peak = np.argmin(prob)
        # accept_peak = prob[peak] <= confidence

        x_peak = np.argmax(chisquare_stat[filtered(periods)])
        y_peak = np.argmax(inactive_stat[filtered(periods)])
        # Sometimes the peaks do not align
        # if prob[peak] == prob[x_peak]:
        #     peak = x_peak
        # if prob[peak] == prob[y_peak]:
        #     peak = y_peak
        ax.plot(
            chisquare_stat[antifiltered(periods)],
            inactive_stat[antifiltered(periods)],
            linestyle="",
            rasterized=True,
            marker=markers["antifiltered"],
            c=colors["antifiltered"],
            label=labels["antifiltered"],
            markersize=sizes["antifiltered"]
        )
        ax.plot(
            chisquare_stat[filtered(periods)][0],
            inactive_stat[filtered(periods)][0],
            linestyle="",
            rasterized=True,
            marker=markers["filtered"],
            label=labels["filtered"],
            markersize=sizes["filtered"],
            c=colors["filtered"],
        )
        ax.plot(
            chisquare_stat[antifiltered(periods)][point5_peak2],
            inactive_stat[antifiltered(periods)][point5_peak2],
            linestyle="",
            # rasterized=True,
            marker=markers["antifiltered_p"],
            c=colors["antifiltered_p"],
            label=labels["antifiltered_p"],
            markersize=sizes["antifiltered_p"],
            zorder=2.1
        )
        ax.plot(
            chisquare_stat[filtered(periods)],
            inactive_stat[filtered(periods)],
            linestyle="",
            rasterized=True,
            marker=markers["filtered"],
            c=colors["filtered"],
            markersize=sizes["filtered"],
        )
        ax.plot(
            chisquare_stat[filtered(periods)][point5_peak],
            inactive_stat[filtered(periods)][point5_peak],
            linestyle="",
            # rasterized=True,
            marker=markers["filtered_p"],
            c=colors["filtered_p"],
            label=labels["filtered_p"],
            markersize=sizes["filtered_p"],
            zorder=2.2
        )
        ax.scatter(
            chisquare_stat[peak],
            inactive_stat[peak],
            PEAK_SIZE,
            facecolor="None",
            edgecolors=PEAK_COLOR,
            # rasterized=True,
            zorder=3
        )
        ax.text(
            0.95,
            0.05,
            f"n = {n_event}\nE = {exposure_hr:.2f} hr\n"
            + r"$\tau$ = "
            + f"{event_window} d",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=FIGSCALE*12/2
        )
        ax.set(ylim=(-0.1, 1.1), title=name)
        table[name] = {
            "period": periods[peak].round(2),
            "p_cc": prob[peak].round(3),
            "chisquare": chisquare_stat[peak].round(2),
            "inactive_frac": inactive_stat[peak].round(2),
            "tau": int(event_window),
            "avg_sep": (int(event_window) / (n_event - 1)) / periods[peak],
            "n": n_event,
            "exposure": exposure_hr,
        }

    # fig.suptitle("Composite periodograms")
    fig.supxlabel(r"$\chi^2$ / ($\Phi$ - 1)")
    fig.supylabel(r"$F_0$")
    if axes[i] != axes[-1]:
        handles, labels = ax.get_legend_handles_labels()
        axes[i + 1].legend(handles, labels, loc="upper left", fontsize=FIGSCALE*12/2)
        [ax.set_axis_off() for ax in axes[i + 1 :]]
    plt.tight_layout()
    plt.savefig(
        pathlib.Path(outdir, f"Composite-periodogram-many-{p+1}_of_{pages}.pdf")
    )

import pandas as pd

pd.DataFrame(table).transpose().round(3).to_csv(pathlib.Path(outdir, f"Summary.csv"))
