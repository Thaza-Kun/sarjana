# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "tqdm",
# ]
# ///
import pathlib
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 15


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--simname", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--size", type=float, default=1.0)
    return parser.parse_args()


arguments = parse_arguments()

name = arguments.name
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence
size = arguments.size

outdir = pathlib.Path(f"{arguments.output}/{simname}/{name}/")

outdirsize = pathlib.Path(outdir, f"{size:.2f}")
for i in outdirsize.iterdir():
    if i.is_dir():
        fname, val = i.name.split("=")
        if fname == "value_number_of_events":
            n_event = int(val)
        elif fname == "value_total_exposure":
            exposure_hr = float(val)
        elif fname == "value_event_window":
            event_window = int(float(val))
        elif fname == "value_burst_rate":
            burst_rate = float(val)

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

## STEP [7] Use CDF from [5] to calculate P(X^2, Inactive_frac)
cdf1 = rv.cdf(
    np.dstack((chisquare_stat[filtered(periods)], inactive_stat[filtered(periods)]))
)
cdf2 = rv.cdf(
    np.dstack(
        (chisquare_stat[antifiltered(periods)], inactive_stat[antifiltered(periods)])
    )
)

prob = 1 - cdf1
point5_peak = np.argwhere(cdf1 > 1 - confidence)
prob2 = 1 - cdf2
point5_peak2 = np.argwhere(cdf2 > 1 - confidence)
peak = np.argmin(prob)
accept_peak = prob[peak] <= confidence

# # Single peak
x_peak = np.argmax(chisquare_stat[filtered(periods)])
y_peak = np.argmax(inactive_stat[filtered(periods)])
# Sometimes the peaks do not align
if prob[peak] == prob[x_peak]:
    peak = x_peak
if prob[peak] == prob[y_peak]:
    peak = y_peak

colors = {
    "filtered": "tab:blue",
    "antifiltered": "orange",
    "filtered_p": "tab:red",
    "antifiltered_p": "pink",
}

fig, axs = plt.subplots(
    2,
    2,
    figsize=(9, 8),
    gridspec_kw={
        "hspace": 0,
        "wspace": 0,
        "width_ratios": [5, 1],
        "height_ratios": [1, 5],
    },
)
axs[0, 0].axis("off")
axs[0, 1].axis("off")
axs[1, 1].axis("off")
axs[1, 0].plot(
    chisquare_stat[antifiltered(periods)],
    inactive_stat[antifiltered(periods)],
    "x",
    c=colors["antifiltered"],
    label=r"$\tau\leq P\leq 2\tau$",
    markersize=4,
)
axs[1, 0].plot(
    chisquare_stat[filtered(periods)],
    inactive_stat[filtered(periods)],
    "+",
    label=r"$1.5<P<\tau$",
    c=colors["filtered"],
)
axs[1, 0].plot(
    chisquare_stat[antifiltered(periods)][point5_peak2],
    inactive_stat[antifiltered(periods)][point5_peak2],
    "x",
    c=colors["antifiltered_p"],
    label=r"$p \geq 0.05 (\tau\leq P)$",
    markersize=4,
)
axs[1, 0].plot(
    chisquare_stat[filtered(periods)][point5_peak],
    inactive_stat[filtered(periods)][point5_peak],
    "+",
    c=colors["filtered_p"],
    label=r"$p\geq 0.05 (P<\tau)$",
)
axs[1, 0].scatter(
    chisquare_stat[peak],
    inactive_stat[peak],
    50,
    facecolor="None",
    edgecolors="green",
)

axs[0, 1].text(
    0,
    0,
    f"n = {n_event}\nE = {exposure_hr:.2f} hr\n"
    + r"$\tau$"
    + f" = {int(event_window)} days",
    ha="left",
    va="bottom",
    transform=axs[0, 1].transAxes,
)

_, ch_bins = np.histogram(chisquare_stat)
_, in_bins = np.histogram(inactive_stat)

histcollection = [
    chisquare_stat[filtered(periods)],
    chisquare_stat[antifiltered(periods)],
    # chisquare_stat[filtered(periods)][point5_peak],
    # chisquare_stat[antifiltered(periods)][point5_peak2]
]
ihistcollection = [
    inactive_stat[filtered(periods)],
    inactive_stat[antifiltered(periods)],
    # inactive_stat[filtered(periods)][point5_peak],
    # inactive_stat[antifiltered(periods)][point5_peak2]
]
axs[0, 0].hist(
    histcollection,
    orientation="vertical",
    color=[*colors.values()][:2],
    histtype="step",
    density=True,
    bins=ch_bins,
)
axs[0, 0].plot(
    histcollection[1][point5_peak2],
    [0.01] * len(histcollection[1][point5_peak2]),
    "|",
    color=[*colors.values()][3],
)
axs[0, 0].plot(
    histcollection[0][point5_peak],
    [0.01] * len(histcollection[0][point5_peak]),
    "|",
    color=[*colors.values()][2],
)
axs[1, 1].hist(
    ihistcollection,
    orientation="horizontal",
    color=[*colors.values()][:2],
    histtype="step",
    bins=in_bins,
    density=True,
)
axs[1, 1].plot(
    [0.5] * len(ihistcollection[1][point5_peak2]),
    ihistcollection[1][point5_peak2],
    "|",
    color=[*colors.values()][3],
)
axs[1, 1].plot(
    [0.5] * len(ihistcollection[0][point5_peak]),
    ihistcollection[0][point5_peak],
    "|",
    color=[*colors.values()][2],
)

plt.suptitle(name)
axs[1, 0].set(xlabel=r"$\chi^2$ / ($\Phi$ - 1)", ylabel=r"$F_0$")
axs[1, 0].legend(loc="lower right")
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, f"Composite-periodogram-{name}.pdf"))
