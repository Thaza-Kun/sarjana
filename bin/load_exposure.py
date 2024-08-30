import datetime
from typing import Tuple
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import numpy as np

import pathlib

from scipy.signal import find_peaks, peak_widths
from scipy.stats import chisquare, multivariate_normal

# from scipy.misc import derivative as deriv

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


## STEP [1] Get data
#       - arrival_time:
#       - exposure_time:
#       - exposure_hour:
#       - period_grid:
# name = "FRB20190113A"

# name = "FRB20190804E"
name = "FRB20191106C"
# name = "FRB20200202A"
# name = "FRB20200223B"
# name = "FRB20200926A"
# name = "FRB20200929C"
# name = "FRB20180916B"

# EXTRA
# name = "FRB20201130A"

# NEW SAMPLE
# name = FRB20180910A
# name = FRB20180916B
# name = FRB20181226F
# name = FRB20190110C
# name = FRB20190113A
# name = FRB20190226B
# name = FRB20190430C
# name = FRB20190609C
# name = FRB20190804E
# name = FRB20190915D
# name = FRB20191106C
# name = FRB20191114A
# name = FRB20200202A
# name = FRB20200223B
# name = FRB20200619A
# name = FRB20200809E
# name = FRB20200926A
# name = FRB20200929C
# name = FRB20201130A
# name = FRB20201221B


SNR = 5

seed = 120
runs = 100
NBINS = 36
size = 1.0
FRACTION = 1 / 2
confidence = 0.05
simname = "2024-08-29"

exposure_csv = f"../data/repeaters/{name}/exposure-UL.csv"
arrivals_csv = f"../data/repeaters/{name}/arrivals.txt"
snr_csv = f"../data/repeaters/{name}/SNR.txt"

ra = pd.read_csv(f"../data/repeaters/{name}/ra.txt", header=None).to_numpy()[0][0]
dec = pd.read_csv(f"../data/repeaters/{name}/dec.txt", header=None).to_numpy()[0][0]

outdir = pathlib.Path(f"../output/{simname}/{name}/{size:.2f}")
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
# print(exposure_date.shape)

exposure_hr_U = exposures["exposure_U"].to_numpy()
exposure_hr_L = exposures["exposures_L"].to_numpy()
U_exposure_total = np.sum(exposure_hr_U)
L_exposure_total = np.sum(exposure_hr_L)
exposure_hr = exposure_hr_U + exposure_hr_L

snrs = pd.read_csv(snr_csv, header=None)
arrivals = pd.read_csv(arrivals_csv, parse_dates=[0], header=None)[snrs.to_numpy() > SNR]
length_all_arrival = len(arrivals[0])

arrivals = arrivals[
    (arrivals[0] < (exposures["date"].max() + datetime.timedelta(days=1)))
    & (arrivals[0] > (exposures["date"].min() - datetime.timedelta(days=1)))
].sort_values(by=[0])
event_window = arrivals[0].max() - arrivals[0].min()
arrivals = Time(arrivals[0].to_list(), format="datetime", location=chime)

# mask_arrivals_rng = np.array([rng.choice([True, False], size=1, p=[size, 1-size]) for _ in arrivals])
arrivals = np.array(
    (arrivals + arrivals.light_travel_time(coord)).to_value("mjd")
)  # [mask_arrivals_rng.flatten()]
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

periods = np.arange(
    np.sum(arrivals) / (burst_rate * 24),
    (event_window.days * 1 / FRACTION),
    1 / (burst_rate * 24),
)


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
        wait_head = active_phase[0] - 0
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
# ax[1].plot(periods[peak], inactive_stat[peak], 'x')
ax[1].plot(periods[point5_peak], inactive_stat[point5_peak], "r+")
ax[1].set_ylabel("Inactivity fraction")

ax[2].semilogy(periods, 1 - cdf)
ax[2].plot(periods[point5_peak], 1 - cdf[point5_peak], "r+")
# ax[2].semilogy(periods[peak], 1-cdf[peak], 'x')
ax[2].axhline(confidence, linestyle="--", alpha=0.5, color="red")
ax[2].set_ylabel("Probability")

plt.xlim(periods[0], periods[-1])
# plt.tight_layout()
plt.savefig(pathlib.Path(outdir, "Figure_1.png"))

exit()
# idx = np.digitize(exposure_date, arrivals)
# # counts = np.array([0 if len(i)==0 else 1 for i in np.split(arrival_date, idx)[:-1]])
# # print(counts)
# # print(exposure_date[coincidence])

# # print(exposure_date)
# # print(exposure_hr)
# # print(arrival_date)

# plt.plot(exposure_date, exposure_hr, '+')
# plt.vlines(arrival_date, ymin=0, ymax=exposure_hr.max(), color='orange')
# plt.vlines(exposure_date[coincidence], ymin=0, ymax=exposure_hr.max(), color='red')
# plt.show()
# # exit()

# rate_1 = len(count)/np.sum(exposure_hr)
# print(rate_1)


# stat = list()
# exp_stat = list()
# inactive_stat = list()

# n_bins = 16
# for p in tqdm.tqdm(periods):
# # with plt.figure():
#     phases = (exposure_date / p) % 1
#     sort_idx = np.lexsort([counts, phases])

#     counts_ = counts[sort_idx]
#     exposure_hr_ = exposure_hr[sort_idx]
#     phases = phases[sort_idx]


#     _, bins = np.histogram(phases, n_bins)
#     bin_idx = np.digitize(bins, phases)
#     phase_count = np.array([sum(i) for i in np.split(counts_, bin_idx)[:-1]])
#     expos_count = np.array([sum(i) * rate_1 for i in np.split(exposure_hr_, bin_idx)[:-1]])

#     # Avoiding divide by zero
#     phase_count = phase_count[expos_count!=0]
#     bins = bins[expos_count!=0]

#     expos_count = expos_count[expos_count!=0]

#     expos_count = expos_count + ((sum(phase_count) - sum(expos_count))/len(expos_count))

#     # plt.step(bins, phase_count)
#     # plt.step(bins, expos_count, color='black', alpha=0.5)
#     # plt.xlim(0, 1)
#     # plt.savefig(f'./plots/{p}.png')
#     # # plt.show()
#     # plt.close()

#     active_phase = bins[phase_count > 0]
#     wait_head = active_phase[0] - 0
#     wait_tail = 1 - active_phase[-1]
#     if len(np.diff(active_phase)) == 0:
#         max_inactive = wait_tail + wait_head
#     else:
#         max_inactive = max(np.diff(active_phase).max(), wait_head + wait_tail)
#     inactive_stat.append(max_inactive)

#     chi2 = chisquare(phase_count, expos_count)
#     stat.append(np.nan_to_num(chi2.statistic))
#     exp_chi2 = chisquare(expos_count)
#     exp_stat.append(exp_chi2.statistic)

# def moving_average(x, w):
#     return np.convolve(np.nan_to_num(x), np.ones(w), 'same') / w

# print(phase_count)
# print(expos_count)
# print(chi2)

# # DO NOT WINDOW
# window = 1
# coeff_std = 2.5

# stat = np.array(moving_average(stat, window))
# exp_stat = np.array(moving_average(exp_stat, window))
# inactive_stat = np.array(moving_average(inactive_stat, window))

# # PROB
# # left = left.round(0).astype(int)
# # right = right.round(0).astype(int)
# dx = np.diff(np.sort(stat)).max()
# print(dx)
# dy = np.diff(np.sort(inactive_stat)).max()
# print(dy)
# x, y = np.mgrid[stat.min():stat.max():dx, inactive_stat.min():inactive_stat.max():dy]
# pos = np.dstack((x, y))

# x_mean = stat.mean()
# y_mean = inactive_stat.mean()
# covar = np.cov([stat, inactive_stat])

# rv = multivariate_normal([x_mean, y_mean], covar)
# cdf = rv.cdf(np.dstack((stat, inactive_stat)))
# prob = rv.pdf(pos) * dx * dy

# prob_period = rv.pdf(np.dstack((stat,inactive_stat)))

# composed_stat = stat * inactive_stat

# min_prominence = stat.mean() + (coeff_std*stat.std())
# peaks, props = find_peaks(stat, height=min_prominence, distance=window)
# _widths, heights, left, right = peak_widths(stat, peaks, rel_height=0.5)

# def remap(x, idx: float) -> np.ndarray:
#     remainder = idx % 1
#     index = np.floor(idx).astype(int)
#     before = x[index]
#     after = x[index + 1]
#     return before + ((after - before)*remainder)


# left = remap(periods, left)
# right = remap(periods, right)

# x_selected = periods[peaks]
# x_selected_delta_l = periods[peaks] - left
# x_selected_delta_r = right - periods[peaks]

# min_prominence = inactive_stat.mean() + (coeff_std*inactive_stat.std())
# inactive_peaks, _ = find_peaks(inactive_stat, height=min_prominence, distance=window)
# _inactive_widths, inactive_heights, inactive_left, inactive_right = peak_widths(inactive_stat, inactive_peaks, rel_height=0.5)

# inactive_left = remap(periods, inactive_left)
# inactive_right = remap(periods, inactive_right)

# x_inactive_selected = periods[inactive_peaks]
# x_inactive_selected_delta_l = periods[inactive_peaks] - inactive_left
# x_inactive_selected_delta_r = inactive_right - periods[inactive_peaks]


# min_prominence = composed_stat.mean() + (coeff_std*composed_stat.std())
# composed_peaks, _ = find_peaks(composed_stat, height=min_prominence, distance=window)
# composed_widths, composed_heights, composed_left, composed_right = peak_widths(composed_stat, composed_peaks, rel_height=0.5)

# composed_left = remap(periods, composed_left)
# composed_right = remap(periods, composed_right)

# x_composed_selected = periods[composed_peaks]
# x_composed_selected_delta_l = periods[composed_peaks] - composed_left
# x_composed_selected_delta_r = composed_right - periods[composed_peaks]

# plt.plot(stat, inactive_stat, 'x')

# std = np.std(prob-prob.max())
# CS = plt.contour(x, y, (prob.max()-prob)/std, levels=[1,2,3,4,5])
# plt.clabel(CS, fmt=lambda x: f"{int(x)}σ")
# plt.show()
# # breakpoint()


# f, ax = plt.subplots(4,1, sharex=True)
# ax[0].set_title(f'Chi-square n_bins={n_bins}')
# ax[0].plot(periods, stat)
# ax[0].axhspan(stat.mean(), stat.mean() + coeff_std*stat.std(), alpha=0.1, color='black')
# # ax[0].text(0, 0.1, pdgram_stat)
# for i, p in enumerate(peaks):
#     ax[0].annotate(f"{x_selected[i]:.2f} _ {x_selected_delta_l[i]:.2f} + {x_selected_delta_r[i]:.2f}", (periods[p], stat[p]))
# ax[1].set_title('Inactive fraction')
# ax[1].plot(periods, inactive_stat)
# ax[1].axhspan(inactive_stat.mean(), inactive_stat.mean() + coeff_std*inactive_stat.std(), alpha=0.1, color='black')
# # ax[1].text(0, 0.1, pdgram_inactive)
# for i, p in enumerate(inactive_peaks):
#     ax[1].annotate(f"{x_inactive_selected[i]:.2f} _ {x_inactive_selected_delta_l[i]:.2f} + {x_inactive_selected_delta_r[i]:.2f}", (periods[p], inactive_stat[p]))
# ax[2].set_title('Composed')
# ax[2].plot(periods, composed_stat)
# ax[2].axhspan(composed_stat.mean(), composed_stat.mean() + coeff_std*composed_stat.std(), alpha=0.1, color='black')
# # ax[2].text(0, 0.1, pdgram_composed)
# for i, p in enumerate(composed_peaks):
#     ax[2].annotate(f"{x_composed_selected[i]:.2f} _ {x_composed_selected_delta_l[i]:.2f} + {x_composed_selected_delta_r[i]:.2f} (p={1-cdf[p]:.2e})", (periods[p], composed_stat[p]))


# cdf_peaks, _ = find_peaks(cdf, height=0.95, distance=50)


# ax[3].set_title('Probability')
# ax[3].semilogy(periods, 1-cdf)
# ax[3].axhline(0.05, linestyle="--", color='orange')
# for i, p in enumerate(cdf_peaks):
#     ax[3].annotate(f"{periods[p]:.2f} (p={1-cdf[p]:.2e})", (periods[p], 1-cdf[p]))
# plt.tight_layout()
# plt.show()
