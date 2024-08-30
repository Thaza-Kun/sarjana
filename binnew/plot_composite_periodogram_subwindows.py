import pathlib
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.size'] = 15

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

sizes = [
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
sizes = sorted(sizes, reverse=True)
fig, axe = plt.subplots(5, 2, figsize=(10, 21), sharey=True, sharex=True) if arguments.vertical else plt.subplots(2, 5, figsize=(21, 10), sharey=True, sharex=True)
axes = np.array(axe).reshape(-1)

for i, size in enumerate(tqdm(sizes)):
    ax = axes[i]
    outdirsize = pathlib.Path(outdir, f"{size:.2f}")
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

    colors = {"filtered": "tab:blue", "antifiltered": "orange", "filtered_p": "tab:red", "antifiltered_p": "pink"}

    ax.plot(
        chisquare_stat[antifiltered(periods)],
        inactive_stat[antifiltered(periods)],
        "x",
        c=colors["antifiltered"],
        label=r"$\tau\leq P\leq 2\tau$",
    )
    ax.plot(
        chisquare_stat[filtered(periods)][0],
        inactive_stat[filtered(periods)][0],
        "+",
        label=r"$1.5<P<\tau$",
        c=colors["filtered"]
    )
    ax.plot(
        chisquare_stat[antifiltered(periods)][point5_peak2],
        inactive_stat[antifiltered(periods)][point5_peak2],
        "x",
        c=colors["antifiltered_p"],
        label=r"$p \geq 0.05 (\tau\leq P)$",
    )
    ax.plot(
        chisquare_stat[filtered(periods)],
        inactive_stat[filtered(periods)],
        "+",
        # label=r"$1.5<P<\tau$",
        c=colors["filtered"]
    )
    ax.plot(
        chisquare_stat[filtered(periods)][point5_peak],
        inactive_stat[filtered(periods)][point5_peak],
        "+",
        c=colors["filtered_p"],
        label=r"$p\geq 0.05 (P<\tau)$",
    )
    ax.scatter(
        chisquare_stat[peak],
        inactive_stat[peak],
        50,
        facecolor="None",
        edgecolors="green",
    )

    ax.text(
        0.95,
        0.05,
        f"n = {n_event}\nE = {exposure_hr:.2f} hr",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set(ylim=(-0.1, 1.1), title=f"{size*100:.0f}% T ($\\tau$ = {event_window} day)")

fig.suptitle(name)
fig.supxlabel(r"$\chi^2$ / ($\Phi$ - 1)")
fig.supylabel(r"$F_0$")
plt.legend(loc='right')
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, "Composite-periodogram-subwindows.png"))
