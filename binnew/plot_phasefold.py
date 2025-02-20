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

import pandas as pd
import numpy as np

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
    parser.add_argument("--nbin", type=int, default=36)
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

arguments = parse_arguments()

name = arguments.name
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence
size = arguments.size
NBINS = arguments.nbin
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

arrivals = pd.read_csv(arrivals_csv, parse_dates=[0], header=None)
try:
    arrivals = arrivals[::-1][:n_event_other]
except NameError:
    arrivals = arrivals[::-1]
length_all_arrival = len(arrivals[0])

arrivals = Time(arrivals[0].to_list(), format="datetime", location=chime)

arrivals = np.array((arrivals + arrivals.light_travel_time(coord)).to_value("mjd"))


def phasefold(time, period, nbins=36, shift=0):
    phases = ((time - shift) / period) % 1
    return np.histogram(phases, nbins, range=(0, 1))


old = arrivals[arrivals <= exposure_date.max()]
curve, bins = phasefold(old, predict_period, shift=exposure_date.min())
tau_old = old.max() - old.min()
tau_new = arrivals.max() - arrivals.min()

curve_, bins_ = phasefold(arrivals, predict_period, shift=exposure_date.min())

window = np.array(np.cumsum(curve)).astype(bool)
window_anti = np.array(np.cumsum(curve[::-1])).astype(bool)[::-1]
window = [w and w_a for (w, w_a) in zip(window, window_anti)]
within = curve_[window]

# label = "T" if size == 1.0 else f"{int(size*100)} % T"
# label_other = "T + new" if size_other == None else f"{int(size_other*100)} % T"
plt.step(bins, [*curve, 0], label=None, color="tab:blue", where="post")
# plt.step(bins_, [*curve_, 0], label=label_other, color="tab:orange", where="post")
plt.step(bins, [*curve, 0], color="tab:blue", where="post")
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title(name)
plt.ylabel("Count")
plt.xlabel(r"$\Phi$" + f" (Period = {predict_period:.2f} days)")
plt.tight_layout()
plt.xlim(0,1)
plt.savefig(pathlib.Path(outdir, f"Phasefold-extended-{name}-{predict_period:.2f}.png"))
print(pathlib.Path(outdir, f"Phasefold-extended-{name}-{predict_period:.2f}.png"))
plt.show()
plt.close()