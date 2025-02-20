from collections import defaultdict
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

plt.rcParams["mathtext.fontset"] = "dejavuserif"
FIGSCALE = 2.5
BASE_SIZE = 6
plt.rcParams["font.size"] = FIGSCALE*12/1.5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="FRB20180916B")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--simname", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--vertical", action="store_true")
    return parser.parse_args()


arguments = parse_arguments()
name = arguments.name
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence

outdir = pathlib.Path(f"{arguments.output}/{simname}/{name}/")

Tsizes = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.0,
]
Tsizes = sorted(Tsizes, reverse=True)
fig, axe = (
    plt.subplots(5, 2, figsize=(FIGSCALE*4.15, FIGSCALE*8.72), sharey=True, sharex=True)
    if arguments.vertical
    else plt.subplots(2, 5, figsize=(FIGSCALE*8.72, FIGSCALE*4.15), sharey=True, sharex=True)
)
axes = np.array(axe).reshape(-1)
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
    "antifiltered": BASE_SIZE,
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
NBINS = 36
for i, size in enumerate(tqdm(Tsizes)):
    ax = axes[i]
    outdirsize = pathlib.Path(outdir, f"{size:.2f}")
    DELTA_F0 = 1 / NBINS
    for i in outdirsize.iterdir():
        if i.is_dir():
            vname, val = i.name.split("=")
            if vname == "value_number_of_events":
                n_event = int(val)
            elif vname == "value_total_exposure":
                exposure_hr = float(val)
            elif vname == "value_event_window":
                event_window = int(float(val))
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
        np.dstack((chisquare_stat[filtered(periods)], inactive_stat[filtered(periods)]))
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
    accept_peak = prob[peak] <= confidence

    x_peak = np.argmax(chisquare_stat[filtered(periods)])
    y_peak = np.argmax(inactive_stat[filtered(periods)])
    # Sometimes the peaks do not align
    if prob[peak] == prob[x_peak]:
        peak = x_peak
    if prob[peak] == prob[y_peak]:
        peak = y_peak

    inactive_stat -= DELTA_F0
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
        zorder=2.3
    )
    ax.text(
        0.95,
        0.95,
        f"n = {n_event}\nE = {exposure_hr:.2f} hr\n"
        + r"$\tau$ = "
        + f"{event_window} d",
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=FIGSCALE*12/2
    )

    ax.set(ylim=(-0.1, 1.1), title=f"{size*100:.0f}% T")
    table[f"{size:2f}"] = {
        "period": periods[peak].round(2),
        "p_cc": prob[peak].round(3),
        "chisquare": chisquare_stat[peak].round(2),
        "inactive_frac": inactive_stat[peak].round(2),
        "tau": int(event_window),
        "avg_sep": (int(event_window) / (n_event - 1)) / periods[peak],
        "n": n_event,
        "exposure": exposure_hr,
    }

fig.suptitle(name)
fig.supxlabel(r"$\chi^2$ / ($\Phi$ - 1)")
fig.supylabel(r"$F_0$")
axes[-1].legend(loc="lower right", fontsize=FIGSCALE*12/2)
plt.xlim(-1, 9)
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, "Composite-periodogram-subwindows.pdf"))

import pandas as pd

pd.DataFrame(table).transpose().round(3).to_csv(pathlib.Path(outdir, f"Subwindows.csv"))
