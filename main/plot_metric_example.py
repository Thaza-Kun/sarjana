# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astropy",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
# ]
# ///
from copy import deepcopy
import itertools
import pathlib
import argparse
from dataclasses import dataclass

from matplotlib import gridspec
import pandas as pd
import numpy as np
from scipy.stats import chisquare

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--nbins", type=int, action="extend", nargs=3)
    parser.add_argument("--shifts", type=float, action="extend", nargs=3)
    parser.add_argument("--datadir", type=str, default="../data/repeaters/")
    parser.add_argument("--other", type=float, default=None)
    return parser.parse_args()


### STEPS
# [1] Get:
#       - arrival_time:
#       - exposure_time:
#       - exposure_hour:
#       - period_grid:
# [2] Calculate burst rate for use as probability, where P(x) = burst_rate * exposure_hour
# [3] Generate mock data ensemble
# [4] For each mock data: Generate (X^2, Inactive_frac) pair
# [5] Build pdf using mock (X^2, Inactive_frac) pair
# [6] Generate (X^2, Inactive_frac) pair for real data
# [7] Use CDF from [5] to calculate P(X^2, Inactive_frac)


## STEP [1] Get data
#       - arrival_time:
#       - exposure_time:
#       - exposure_hour:
#       - period_grid:

arguments = parse_arguments()

name = arguments.name
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence
size = arguments.size
NBINS = arguments.nbins
shifts = arguments.shifts
predict_period = arguments.period

FRACTION = 1 / 2
DATADIR = arguments.datadir
OUTDIR = arguments.output

size_other = arguments.other

exposure_csv = f"{DATADIR}/{name}/exposure-UL.csv"
arrivals_csv = f"{DATADIR}/{name}/arrivals.txt"

ra = pd.read_csv(f"{DATADIR}/{name}/ra.txt", header=None).to_numpy()[0][0]
dec = pd.read_csv(f"{DATADIR}/{name}/dec.txt", header=None).to_numpy()[0][0]


outdir = pathlib.Path(f"{OUTDIR}/{simname}/{name}/{size:.2f}")
if size_other != None:
    outdirother = pathlib.Path(f"{OUTDIR}/{simname}/{name}/{size_other:.2f}")
    for i in outdirother.iterdir():
        if i.is_dir():
            fname, val = i.name.split("=")
            if fname == "value_number_of_events":
                n_event_other = int(val)

coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
chime = EarthLocation.of_site("CHIME")

exposures = pd.read_csv(exposure_csv, parse_dates=["date"]).dropna()
exposures = exposures[: int(len(exposures) * size)]
exposure_date = Time(exposures["date"].to_list(), format="datetime", location=chime)
exposure_date = np.array(
    (exposure_date + exposure_date.light_travel_time(coord)).to_value("mjd")
)
exposures_hr = exposures["exposure_U"] + exposures["exposures_L"]

arrivals = pd.read_csv(arrivals_csv, parse_dates=[0], header=None)
try:
    arrivals = arrivals[::-1][:n_event_other]
except NameError:
    arrivals = arrivals[::-1]

arrivals = Time(arrivals[0].to_list(), format="datetime", location=chime)
arrivals = np.array((arrivals + arrivals.light_travel_time(coord)).to_value("mjd"))


def phasefold(time, period, values: np.ndarray | None = None, nbins=36, shift=0):
    phases = ((time - shift) / period) % 1
    if values is None:
        return np.histogram(phases, nbins, range=(0, 1))
    sort_idx = np.lexsort([values, phases])
    try:
        curve = values[sort_idx]
        phases = phases[sort_idx]
    except KeyError:
        curve = values.to_numpy()[sort_idx]
        phases = phases[sort_idx]
    _, bins = np.histogram(phases, nbins, range=(0, 1))
    bin_idx = np.digitize(bins, phases)
    hrs = np.array([sum(i) for i in np.split(curve, bin_idx[:-1])][1:])
    return hrs, bins


@dataclass
class PhaseSpan:
    start: float
    end: float
    length: float
    active: bool


fig = plt.figure(figsize=(16, 12))
outer = gridspec.GridSpec(3, 3, figure=fig)
# inner = gridspec.GridSpec()

# f, ax = plt.subplots(5,1, height_ratios=[5,1,1,5,1], sharex='col', gridspec_kw={'hspace': 0}, figsize=(7, 10))
fig.set_tight_layout(True)
# ax[2].set_axis_off()

for i, (s, b) in enumerate(itertools.product(shifts, NBINS)):
    old = arrivals[arrivals <= exposure_date.max()]
    curve, bins = phasefold(old, predict_period, shift=exposure_date.min() + s, nbins=b)
    rate = sum(curve) / sum(exposures_hr)
    exp_curve, exp_bins = phasefold(
        exposure_date,
        predict_period,
        values=exposures_hr,
        shift=exposure_date.min() + s,
        nbins=b,
    )
    exp_curve = exp_curve * rate

    active_phase = curve > 0

    def split_by_mask(a, mask):
        return np.split(a, np.flatnonzero(np.diff(mask)) + 1)

    tfalse = np.array(
        list(map(lambda x: x[0], split_by_mask(active_phase, active_phase)))
    )
    active = np.array(
        list(
            map(
                lambda x: (x[0], x[-1] + 1 / b, x[-1] - x[0] + 1 / b),
                split_by_mask(bins[:-1], active_phase),
            )
        )
    )
    spans = list(
        PhaseSpan(start=x[0], end=x[1], length=x[2], active=y)
        for x, y in zip(active, tfalse)
    )
    headspan = deepcopy(spans[0])
    tailspan = deepcopy(spans[-1])
    if not headspan.active and not tailspan.active:
        headtail = headspan.length + tailspan.length
    else:
        headtail = 0
    spans = sorted(spans, key=lambda x: x.length, reverse=True)

    inner = outer[i].subgridspec(2, 1, height_ratios=[6, 1], wspace=0.2, hspace=0)

    if len(fig.axes) == 0:
        ax0 = fig.add_subplot(inner[0])
        ax1 = fig.add_subplot(inner[1], sharex=ax0)
    else:
        ax0 = fig.add_subplot(inner[0], sharey=fig.axes[0])
        ax1 = fig.add_subplot(inner[1], sharex=ax0, sharey=fig.axes[1])

    ax0.step(bins, [*curve, 0], color="tab:blue", where="post", label=r"$N_\phi$")
    ax0.step(
        exp_bins,
        [*exp_curve, 0],
        color="tab:green",
        alpha=0.3,
        linestyle="--",
        where="post",
        label=r"$r \cdot E_\phi$",
    )
    ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax0.set_title(f"bins = {b}, shift = {s} d")
    ax0.set_xlim(0, 1)

    for s in spans:
        if not s.active:
            if headtail > s.length and not s.end + 1 / b >= 1:
                continue
            # get 'textheight' from prev iteration or set it now
            textheight = max(curve) / 2
            text2height = max(curve - exp_curve)

            ax0.axvspan(s.start, s.end, alpha=0.1, color="black")
            if s.end + 1 / b >= 1 and not headspan.active:
                ax0.axvspan(headspan.start, headspan.end, alpha=0.1, color="black")
                span_length = s.length + headspan.length
            else:
                span_length = s.length
            ax0.text(
                s.start + 0.5 * (s.end - s.start),
                textheight,
                r"$F_0$ = " + f"{span_length:.2f}",
                ha="center",
                va="center",
                size=11,
            )
            break

    if i % 2 == 0:
        ax0.set_ylabel("Count")
        ax1.set_ylabel(r"$N_\phi - r\cdot E_\phi$")

    ax1.step(bins, [*(curve - exp_curve), 0], where="post")
    ax1.set_yticks([])
    ax1.fill_between(bins, [*(curve - exp_curve), 0], step="post", alpha=0.3)
    ax1.axhline(0)
    exp_curve[exp_curve == 0] = 1e-20
    ax1.text(
        1,
        1,
        r"$\chi^2 / (\Phi - 1)$ = "
        + f"{chisquare(curve, exp_curve).statistic / (b - 1):.2f}",
        ha="right",
        va="top",
        size=11,
        transform=ax1.transAxes,
    )

ax0.legend()
fig.supxlabel(r"$\phi$")
fig.align_ylabels()
for ax in fig.get_axes():
    ax.label_outer()
fig.suptitle(f"Example phase-folds using {name} at {predict_period:.2f} d")
fig.savefig(
    pathlib.Path(
        outdir,
        f"Example-phasefold-metric-{name}~{predict_period:.2f}x{'+'.join(str(b) for b in NBINS)}x{'+'.join(str(s) for s in shifts)}.pdf",
    )
)
print(
    pathlib.Path(
        outdir,
        f"Example-phasefold-metric-{name}~{predict_period:.2f}x{'+'.join(str(b) for b in NBINS)}x{'+'.join(str(s) for s in shifts)}.pdf",
    )
)
plt.close()
