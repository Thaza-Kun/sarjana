# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "astropy",
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "scipy",
#     "tqdm",
# ]
# ///
import datetime
import argparse
from typing import Tuple
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import numpy as np

import pathlib

from scipy.signal import find_peaks, peak_widths
from scipy.stats import chisquare, multivariate_normal

import matplotlib.pyplot as plt

import tqdm

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--simname", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=120)
    parser.add_argument("--nbins", type=int, default=36)
    parser.add_argument(
        "--ngrid",
        type=float,
        default=1.0,
        help="Modifies grid spacing to be 1/n*(original spacing)",
    )
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--datadir", type=str, default="../data/repeaters")
    return parser.parse_args()


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
SNR = arguments.snr
seed = arguments.seed
NBINS = arguments.nbins
ngrid = arguments.ngrid

FRACTION = 1 / 2
DATADIR = arguments.datadir
OUTDIR = arguments.output

exposure_csv = f"{DATADIR}/{name}/exposure-UL.csv"
arrivals_csv = f"{DATADIR}/{name}/arrivals.txt"
snr_csv = f"{DATADIR}/{name}/SNR.txt"

ra = pd.read_csv(f"{DATADIR}/{name}/ra.txt", header=None).to_numpy()[0][0]
dec = pd.read_csv(f"{DATADIR}/{name}/dec.txt", header=None).to_numpy()[0][0]

outdir = pathlib.Path(f"{OUTDIR}/{simname}/{name}/{size:.2f}")
outdir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(seed=seed)
coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
chime = EarthLocation.of_site("CHIME")

exposures = pd.read_csv(exposure_csv, parse_dates=["date"]).dropna()
exposures = exposures[: int(len(exposures) * size)]
exposure_date = Time(exposures["date"].to_list(), format="datetime", location=chime)
exposure_date = np.array(
    (exposure_date + exposure_date.light_travel_time(coord)).to_value("mjd")
)
exposure_window = exposure_date.max() - exposure_date.min()

exposure_hr_U = exposures["exposure_U"].to_numpy()
exposure_hr_L = exposures["exposures_L"].to_numpy()
U_exposure_total = np.sum(exposure_hr_U)
L_exposure_total = np.sum(exposure_hr_L)
exposure_hr = exposure_hr_U + exposure_hr_L

snrs = pd.read_csv(snr_csv, header=None)
arrivals = pd.read_csv(arrivals_csv, parse_dates=[0], header=None)
length_all_arrival = len(arrivals[0])

within_exposure = (
    arrivals[0] < (exposures["date"].max() + datetime.timedelta(days=1))
) & (arrivals[0] > (exposures["date"].min() - datetime.timedelta(days=1)))


arrivals = arrivals[snrs.to_numpy() >= SNR][within_exposure]
snrs = snrs[snrs.to_numpy() >= SNR][within_exposure]
event_window = arrivals[0].max() - arrivals[0].min()
arrivals = Time(arrivals[0].to_list(), format="datetime", location=chime)

table = {
    "arrivals": arrivals.to_value("mjd").round(3),
    "snrs": snrs.to_numpy().flatten().round(1),
}

arrivals = np.array((arrivals + arrivals.light_travel_time(coord)).to_value("mjd"))
table["arrivals_barycenter"] = arrivals.round(3)
# SAVE
pd.DataFrame(table).sort_values(by="arrivals").to_csv(
    pathlib.Path(OUTDIR, simname, name, f"{size:.2f}", "Arrivals.csv"), index=False
)
arrivals.sort()
length_some_arrival = len(arrivals)

split_idx = np.digitize(exposure_date, arrivals)
arrivals = np.array([len(i) for i in np.split(arrivals, split_idx)][1:])

contains_more = length_all_arrival > length_some_arrival

assert np.sum(arrivals) > 1, "Events must be more than one"

plt.plot(exposure_date, exposure_hr)
plt.vlines(
    exposure_date[arrivals != 0], ymin=0, ymax=1.5 * exposure_hr.max(), color="r"
)
plt.savefig(pathlib.Path(outdir, "Timeline.png"))


## STEP [2] Calculate burst rate
assert np.nansum(exposure_hr) > 0, "Exposure cannot be zero"
burst_rate = sum(arrivals) / np.nansum(exposure_hr)

burst_rate_day = burst_rate * 24  # (1/hr) * (hr/day)

print("rate (1/d): ", burst_rate_day)
print("tau (d): ", (event_window.days * 1 / FRACTION))
print("spacing (d): ", 1 / (ngrid * burst_rate_day))

periods = np.arange(
    # np.sum(arrivals) / burst_rate_day,
    1.5,
    (event_window.days * 1 / FRACTION),
    1 / (ngrid * burst_rate_day),
)
print("Trial periods:", periods.min(), periods.max(), np.diff(periods)[0])
print("Trial periods len:", periods.shape)

## STEP [3] Generate mock data ensemble
mask_rng = lambda: [
    rng.choice([True, False], size=1, p=[burst_rate * i, 1 - (burst_rate * i)])
    for i in exposure_hr
]


def chisquare_inactive_pdgram(
    time: np.ndarray,
    detections: np.ndarray,
    exposure: np.ndarray,
    periods: np.ndarray,
    rate: float,
    nbins: int = NBINS,
) -> Tuple[np.ndarray, np.ndarray]:
    stat = list()
    inactive_stat = list()

    for p in periods:
        phases = (time / p) % 1

        sort_idx = np.lexsort([detections, phases])
        curve = detections[sort_idx]
        curve_exposure = exposure[sort_idx]
        phases = phases[sort_idx]

        _, bins = np.histogram(phases, nbins, range=(0, 1))
        bin_idx = np.digitize(bins, phases)
        curve_count = np.array([sum(i) for i in np.split(curve, bin_idx[:-1])])
        expos_count = np.array(
            [sum(i) * rate for i in np.split(curve_exposure, bin_idx[:-1])]
        )

        # Avoiding divide by zero
        curve_count = curve_count[expos_count != 0]
        bins = bins[expos_count != 0]
        expos_count = expos_count[expos_count != 0]

        # If no expos_count, there is no use continuing
        if len(expos_count) == 0:
            continue
        # shift residue so that sum(curve) == sum(exposure)
        expos_count = expos_count + (
            (sum(curve_count) - sum(expos_count)) / len(expos_count)
        )

        active_phase = bins[curve_count > 0]
        wait_head = active_phase[0] - (1 / nbins)
        wait_tail = 1 - active_phase[-1]
        if len(np.diff(active_phase)) == 0:
            max_inactive = wait_tail + wait_head
        else:
            max_inactive = max(np.diff(active_phase).max(), wait_head + wait_tail)

        chi2 = chisquare(curve_count, expos_count)
        stat.append(
            chi2.statistic / (len(curve_count) - 1 if len(curve_count) >= 2 else 1)
        )
        inactive_stat.append(max_inactive)
    return np.array(stat), np.array(inactive_stat)


## STEP [4] For each mock data: Generate (X^2, Inactive_frac) pair
mock_chisquare = list()
mock_inactive = list()
for _ in tqdm.tqdm(range(runs)):
    mock_arrivals = np.array(mask_rng()).flatten().astype(int)
    mock_burst_rate = sum(mock_arrivals) / np.sum(exposure_hr)
    chisquare_stat, inactive_stat = chisquare_inactive_pdgram(
        time=exposure_date,
        detections=mock_arrivals,
        exposure=exposure_hr,
        periods=periods,
        rate=mock_burst_rate,
    )
    mock_chisquare.extend(chisquare_stat)
    mock_inactive.extend(inactive_stat)

    # Manually delete to avoid namespace leak
    del chisquare_stat, inactive_stat
mock_chisquare = np.array(mock_chisquare)
mock_inactive = np.array(mock_inactive)

## STEP [5] Build pdf using mock (X^2, Inactive_frac) pair
x_mean = mock_chisquare.mean()
y_mean = mock_inactive.mean()
covar = np.cov([mock_chisquare, mock_inactive])

rv = multivariate_normal([x_mean, y_mean], covar)


# [6] Generate (X^2, Inactive_frac) pair for real data
chisquare_stat, inactive_stat = chisquare_inactive_pdgram(
    time=exposure_date,
    detections=arrivals,
    exposure=exposure_hr,
    periods=periods,
    rate=burst_rate,
)
x, y = np.mgrid[
    chisquare_stat.min() : chisquare_stat.max() : 0.1,
    0:1.6:0.1,
]
pos = np.dstack((x, y))
pdf = rv.pdf(pos)

## STEP [7] Use CDF from [5] to calculate P(X^2, Inactive_frac)
cdf = rv.cdf(np.dstack((chisquare_stat, inactive_stat)))

prob = 1 - cdf
point5_peak = np.argwhere(cdf > 1 - confidence)
peak = np.argmin(prob)
accept_peak = prob[peak] <= confidence

###############
## OUTPUT: DATA
pathlib.Path(outdir, f"value_contains_more_detections={contains_more}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_number_of_events={np.sum(arrivals)}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_total_exposure={np.sum(exposure_hr):.3f}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_burst_rate={burst_rate:.3f}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_total_exposure_U={U_exposure_total:.3f}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_total_exposure_L={L_exposure_total:.3f}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_event_window={event_window.days:.3f}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_exposure_window={exposure_window:.3f}").mkdir(
    parents=True, exist_ok=True
)
pathlib.Path(outdir, f"value_peak_index={peak}").mkdir(parents=True, exist_ok=True)
np.save(pathlib.Path(outdir, f"kde-{runs}.mean.x.npy"), x_mean)
np.save(pathlib.Path(outdir, f"kde-{runs}.mean.y.npy"), y_mean)
np.save(pathlib.Path(outdir, f"kde-{runs}.covar.npy"), covar)
np.save(pathlib.Path(outdir, f"transient-{runs}.power.chisquare.npy"), chisquare_stat)
np.save(pathlib.Path(outdir, f"transient-{runs}.power.inactive.npy"), inactive_stat)
np.save(pathlib.Path(outdir, f"transient-{runs}.grid.period.npy"), periods)


###############
## OUTPUT: PLOT
def remap(x, idx: float) -> np.ndarray:
    remainder = idx % 1
    index = np.floor(idx).astype(int)
    before = x[index]
    after = x[index + 1]
    return before + ((after - before) * remainder)


subfigs = plt.figure(figsize=(15, 7)).subfigures(1, 2)
ax = subfigs[0].subplots(1, 1)
ax.plot(chisquare_stat, inactive_stat, "+")
ax.plot(chisquare_stat[point5_peak], inactive_stat[point5_peak], "r+")


# Single peak
x_peak = np.argmax(chisquare_stat)
y_peak = np.argmax(inactive_stat)
# Sometimes the peaks do not align
if prob[peak] == prob[x_peak]:
    peak = x_peak
if prob[peak] == prob[y_peak]:
    peak = y_peak

if accept_peak:
    ax.scatter(
        chisquare_stat[peak],
        inactive_stat[peak],
        50,
        facecolor="None",
        edgecolors="green",
    )


std = np.std(pdf - pdf.max())
CS = plt.contour(x, y, (pdf.max() - pdf) / std, levels=[1, 2, 3, 4, 5], colors="red")
plt.clabel(CS, fmt=lambda x: f"{int(x)}σ")
plt.xlabel(r"Uniformity measure ($\chi^2$)")
plt.ylabel("Inactivity fraction")

ax = subfigs[1].subplots(3, 1, sharex=True)
ax[0].plot(periods, chisquare_stat)
ax[0].set_ylabel(r"Uniformity measure ($\chi^2$)")
ax[0].plot(periods[point5_peak], chisquare_stat[point5_peak], "r+")

if accept_peak:
    _, _, l, r = peak_widths(cdf - 1, [peak], rel_height=confidence)
    period_neg = periods[peak] - remap(periods, l)[0]
    period_pos = remap(periods, r)[0] - periods[peak]

    for axis in ax:
        [
            axis.axvline(
                periods[peak] * n,
                color="green",
                alpha=0.1,
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
            )
            for n in range(5)
        ]
        [
            axis.axvline(
                periods[peak] * n,
                color="green",
                alpha=0.1,
                ymin=axis.get_ylim()[0],
                ymax=axis.get_ylim()[1],
                linestyle=":",
            )
            for n in np.arange(-4.5, 4.5, 1)
        ]
        axis.axvspan(period_neg, period_pos, alpha=0.1, color="black")
    ax[0].annotate(
        f"{periods[peak]:.2f} (-{period_neg:.2f},+{period_pos:.2f}) [{inactive_stat[peak]:.2f}% inactive]",
        xy=(periods[peak] + 0.5, chisquare_stat.max() - 0.5),
    )

ax[1].plot(periods, inactive_stat)
ax[1].plot(periods[point5_peak], inactive_stat[point5_peak], "r+")
ax[1].set_ylabel("Inactivity fraction")

ax[2].semilogy(periods, 1 - cdf)
ax[2].plot(periods[point5_peak], 1 - cdf[point5_peak], "r+")
ax[2].axhline(confidence, linestyle="--", alpha=0.5, color="red")
ax[2].set_ylabel("Probability")

plt.xlim(periods[0], periods[-1])
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, "Composite-with-stack.png"))
