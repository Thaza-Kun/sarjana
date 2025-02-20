# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astropy",
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///
import pathlib
import argparse
from itertools import zip_longest, chain
from dataclasses import dataclass
from copy import deepcopy
from typing import List

import pandas as pd
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rcParams["mathtext.fontset"] = "dejavuserif"
# plt.rcParams["savefig.dpi"] = 500
FIGSCALE = 1
BASE_SIZE = 6
plt.rcParams["font.size"] = FIGSCALE*12/1.5


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", action='extend', nargs="+", type=str, required=True)
    parser.add_argument("--selects", action='extend', nargs="+", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--periods", type=float, action='extend', nargs="+", required=True)
    parser.add_argument("--nbin", type=int, default=36)
    parser.add_argument("--simname", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    parser.add_argument("--datadir", type=str, default="../data/repeaters/")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--size", type=float, default=1.0)
    return parser.parse_args()

arguments = parse_arguments()

names = arguments.names
selects = arguments.selects
fold_periods = arguments.periods
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence
size = arguments.size
NBINS = arguments.nbin
FRACTION = 1 / 2
OUTDIR = arguments.output
DATADIR = arguments.datadir

assert_same_length = '\n'.join(["Arguments for `--names` and `--periods` are not the same length. Current values:",
                      ', '.join(
                          [f"(name: {n}, period: {p})" for n, p in zip_longest(names, fold_periods)]
                          )
                        ])

assert len(names) == len(fold_periods), assert_same_length

@dataclass
class PhaseSpan:
    start: float
    end: float
    length: float
    active: bool

def phasefold(time, period, values: np.ndarray | None = None, nbins=36, shift=0):
    phases = ((time - shift) / period) % 1
    if values is None:
        return np.histogram(phases, nbins, range=(0, 1))
    sort_idx = np.lexsort([values, phases])
    try:
        curve = values.to_numpy()[sort_idx]
        phases = phases[sort_idx]
    except KeyError:
        print(values)
        print(phases)
        print(sort_idx)
        exit()
    _, bins = np.histogram(phases, nbins, range=(0, 1))
    bin_idx = np.digitize(bins, phases)
    hrs = np.array(
        [sum(i) for i in np.split(curve, bin_idx[:-1])][1:]
    )
    return hrs, bins

NROW = 5
NCOL = 4
fig, axes = plt.subplots(NROW, NCOL, figsize=(FIGSCALE*8.72, FIGSCALE*8.30), sharex=True, sharey=True)
outer = list(chain(*axes))
# outer = gridspec.GridSpec(4, 5, wspace=0.2, hspace=0.2, figure=fig)
fig.set_tight_layout(True)

lastrow = range(NROW * NCOL)[:-NCOL]
residuerow = int(NROW*NCOL) - len(names)

def mapindex(x: int) -> int:
    if residuerow == 3 and NCOL == 5:
        if x == int((NROW - 1)*NCOL) + 0:
            return int((NROW - 1)*NCOL) + 1
        elif x == int((NROW - 1)*NCOL) + 1:
            return int((NROW - 1)*NCOL) + 3
    if residuerow == 3 and NCOL == 4:
        if x == int((NROW - 1)*NCOL) + 0:
            return int((NROW - 1)*NCOL) + 1
    else:
        raise ValueError(f"Value {residuerow} and col={NCOL}Not yet implemented")
def blankindex() -> List[int]:
    if residuerow == 3 and NCOL == 5:
        return [int((NROW - 1)*NCOL) + n for n in [0,2,4]]
    if residuerow == 3 and NCOL == 4:
        return [int((NROW - 1)*NCOL) + n for n in [0,2,3]]
    else:
        raise ValueError(f"Value {residuerow} and col={NCOL} Not yet implemented")

for i, (name, period) in enumerate(zip(names, fold_periods)):
    if i == (NROW - 1) * NCOL:
        for n in blankindex():
            outer[n].set_axis_off()
    if i >= (NROW - 1) * NCOL:
        i = mapindex(i)
    exposure_csv = f"{DATADIR}/{name}/exposure-UL.csv"
    arrivals_csv = f"{DATADIR}/{name}/arrivals.txt"

    ra = pd.read_csv(f"{DATADIR}/{name}/ra.txt", header=None).to_numpy()[0][0]
    dec = pd.read_csv(f"{DATADIR}/{name}/dec.txt", header=None).to_numpy()[0][0]

    coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
    chime = EarthLocation.of_site("CHIME")

    exposures = pd.read_csv(exposure_csv, parse_dates=["date"]).dropna()
    exposures = exposures[: int(len(exposures) * size)]
    exposure_date = Time(exposures["date"].to_list(), format="datetime", location=chime)
    exposure_date = np.array(
        (exposure_date + exposure_date.light_travel_time(coord)).to_value("mjd")
    )
    exposures_hr = exposures['exposure_U'] + exposures["exposures_L"]

    arrivals = pd.read_csv(arrivals_csv, parse_dates=[0], header=None)
    arrivals = arrivals[::-1]
    arrivals = Time(arrivals[0].to_list(), format="datetime", location=chime)
    arrivals = np.array((arrivals + arrivals.light_travel_time(coord)).to_value("mjd"))

    folded = outer[i]

    old = arrivals[arrivals <= exposure_date.max()]
    curve, bins = phasefold(old, period, shift=exposure_date.min())
    rate = sum(curve)/sum(exposures_hr)
    exp_curve, exp_bins = phasefold(exposure_date, period, values=exposures_hr, shift=exposure_date.min())
    exp_curve = np.round(exp_curve) * rate

    folded.step(bins, [*curve, 0], color="tab:blue", where="post", label=r"$N_\phi$")
    folded.step(exp_bins, [*exp_curve, 0], color="tab:green", alpha=0.3, linestyle="--", where="post", label=r"$r \cdot E_\phi$")
    folded.yaxis.set_major_locator(MaxNLocator(integer=True))
    title = name if name not in selects else f'{name}*'
    folded.set_title(title)
    folded.set_xlim(0,1)

    active_phase = curve > 0
    def split_by_mask(a, mask):
        return np.split(a, np.flatnonzero(np.diff(mask)) + 1)
    tfalse = np.array(list(map(lambda x: x[0] ,split_by_mask(active_phase, active_phase))))
    active = np.array(list(map(lambda x: (x[0], x[-1] + 1/NBINS, x[-1]-x[0] + 1/NBINS), split_by_mask(bins[:-1], active_phase))))
    spans = list(PhaseSpan(start=x[0], end=x[1], length=x[2], active=y) for x, y in zip(active, tfalse))
    headspan = deepcopy(spans[0])
    tailspan = deepcopy(spans[-1])
    if not headspan.active and not tailspan.active:
        headtail = headspan.length + tailspan.length
    else:
        headtail = 0
    spans = sorted(spans, key=lambda x: x.length, reverse=True)
    for s in spans:
        if not s.active:
            if headtail > s.length and not (s.end == tailspan.end or s.end == headspan.end):
                continue
            textheight = max(curve)/2
            text2height = max(curve-exp_curve)

            if s.end == tailspan.end and not headspan.active:
                folded.axvspan(s.start, s.end, alpha=0.1, color="black")
                folded.axvspan(headspan.start, headspan.end, alpha=0.1, color="black")
            elif s.end == headspan.end and not tailspan.active:
                folded.axvspan(s.start, s.end, alpha=0.1, color="black")
                folded.axvspan(tailspan.start, tailspan.end, alpha=0.1, color="black")
            else:
                folded.axvspan(s.start, s.end, alpha=0.1, color="black")
            break
    folded.set_xlabel(f'P = {period:.2f} days')

# We use the leaked `i` value to use the last value
if outer[i] != outer[-1]:
    handles, labels = outer[i].get_legend_handles_labels()
    outer[i + 1].legend(handles, labels, loc="upper left")
    [ax.set_axis_off() for ax in outer[i + 1 :]]
fig.supylabel("Event counts")
fig.supxlabel(r"$\phi$")
fig.suptitle(f'Resulting phase fold from their respective evaluated periods')
plt.savefig(pathlib.Path(OUTDIR, simname, f"Result-phasefold-metric.pdf"))
print(pathlib.Path(OUTDIR, simname, f"Result-phasefold-metric.pdf"))
plt.show()
plt.close()