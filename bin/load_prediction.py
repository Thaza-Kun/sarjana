import pathlib

import pandas as pd
import numpy as np

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.size'] = 15


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
# name = "FRB20191106C"
# name = "FRB20200202A"
# name = "FRB20200926A"
# name = "FRB20200929C"


name = "FRB20190804E"
# predict_period = 171.08+0.87
predict_period = 163.48
# predict_period = 171.08-2.11
# predict_period = 171.08 * 0.6

name = "FRB20200223B"
# predict_period = 25.14+0.02
# predict_period = 25.14
predict_period = 65.73
# predict_period = 25.14-0.16
# predict_period = 25.14 * 3#.26
# predict_period = 25.14 * 7.28

# name = "FRB20180916B"
# predict_period = 16.32+0.10
# predict_period = 16.32
# predict_period = 16.32-0.07
# predict_period = 15.91 #size = 0.1
# predict_period = 16.36 #size = 0.5
# predict_period = 16.30 #size = 0.2

seed = 120
runs = 100
NBINS = 36
size = 1.0
size_other = None
FRACTION = 1 / 2
confidence = 0.05
simname = "2024-08-28"
# inactive_frac = 0.64
# active_period = (1-.83)*predict_period

exposure_csv = f"../data/repeaters/{name}/exposure-UL.csv"
arrivals_csv = f"../data/repeaters/{name}/arrivals.txt"

ra = pd.read_csv(f"../data/repeaters/{name}/ra.txt", header=None).to_numpy()[0][0]
dec = pd.read_csv(f"../data/repeaters/{name}/dec.txt", header=None).to_numpy()[0][0]


outdir = pathlib.Path(f"../output/{simname}/{name}/{size:.2f}")
if size_other != None:
    outdirother = pathlib.Path(f"../output/{simname}/{name}/{size_other:.2f}")
    for i in outdirother.iterdir():
        if i.is_dir():
            fname, val = i.name.split("=")
            if fname == "value_number_of_events":
                n_event_other = int(val)
# grid = np.load(pathlib.Path(outdir, f"transient-{runs}.grid.period.npy"))
# inactive_power = np.load(pathlib.Path(outdir, f"transient-{runs}.power.inactive.npy"))
# chisquar_power = np.load(pathlib.Path(outdir, f"transient-{runs}.power.chisquare.npy"))

rng = np.random.default_rng(seed=seed)
coord = SkyCoord(ra=ra, dec=dec, unit=u.deg)
chime = EarthLocation.of_site("CHIME")

exposures = pd.read_csv(exposure_csv, parse_dates=["date"]).dropna()
exposures = exposures[: int(len(exposures) * size)]
exposure_date = Time(exposures["date"].to_list(), format="datetime", location=chime)
exposure_date = np.array(
    (exposure_date + exposure_date.light_travel_time(coord)).to_value("mjd")
)
# print(exposure_date.shape)

# exposure_hr_U = exposures["exposure_U"].to_numpy()
# exposure_hr_L = exposures["exposures_L"].to_numpy()
# U_exposure_total = np.sum(exposure_hr_U)
# L_exposure_total = np.sum(exposure_hr_L)
# exposure_hr = exposure_hr_U + exposure_hr_L

arrivals = pd.read_csv(arrivals_csv, parse_dates=[0], header=None)
try:
    arrivals = arrivals[::-1][:n_event_other]
except NameError:
    arrivals = arrivals[::-1]
length_all_arrival = len(arrivals[0])

# arrivals = arrivals[
#     (arrivals[0] < (exposures["date"].max() + datetime.timedelta(days=1)))
#     & (arrivals[0] > (exposures["date"].min() - datetime.timedelta(days=1)))
# ].sort_values(by=[0])
# event_window = arrivals[0].max() - arrivals[0].min()
arrivals = Time(arrivals[0].to_list(), format="datetime", location=chime)

# mask_arrivals_rng = np.array([rng.choice([True, False], size=1, p=[size, 1-size]) for _ in arrivals])
arrivals = np.array(
    (arrivals + arrivals.light_travel_time(coord)).to_value("mjd")
)  # [mask_arrivals_rng.flatten()]

def phasefold(time, period, nbins=36, shift=0):
    phases = (((time - shift) / period) % 1)
    return np.histogram(phases, nbins, range=(0, 1))

old = arrivals[arrivals<=exposure_date.max()]
curve, bins = phasefold(old, predict_period, shift=exposure_date.min())
tau_old = old.max() - old.min()
print(tau_old)
print((tau_old/(len(old) - 1))/predict_period)
tau_new = arrivals.max() - arrivals.min()
print(tau_new)
print((tau_new/(len(arrivals) - 1))/predict_period)

print(curve)
curve_, bins_ = phasefold(arrivals, predict_period, shift=exposure_date.min())
print(curve_)

window = np.array(np.cumsum(curve)).astype(bool)
window_anti = np.array(np.cumsum(curve[::-1])).astype(bool)[::-1]
window = [w and w_a for (w, w_a) in zip(window, window_anti)]
within = curve_[window]
print(curve[window])
print(within)
print(sum(within), "/", sum(curve_))

label = 'T' if size == 1.0 else f'{int(size*100)} % T'
label_other = 'T + new' if size_other == None else f'{int(size_other*100)} % T'
plt.step(bins, [*curve, 0], label=label, color="tab:blue", where='post')
plt.step(bins_, [*curve_, 0], label=label_other, color='tab:orange', where='post')
plt.step(bins, [*curve, 0], color="tab:blue", where='post')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.title(name)
plt.ylabel('Count')
plt.xlabel(r'$\Phi$' + f' (Period = {predict_period:.2f} days)')
plt.legend()
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, f'Phasefold-extended-{name}-{predict_period:.2f}.png'))
print(pathlib.Path(outdir, f'Phasefold-extended-{name}-{predict_period:.2f}.png'))
plt.close()

# curve, bins = phasefold(arrivals[arrivals<=exposure_date.max()], predict_period, shift=arrivals.min()-1)
max_p = int(np.ceil((arrivals.max() - arrivals.min()) / predict_period))
curve = np.tile([*curve,0], max_p)
bins = np.tile(bins*predict_period, max_p) + np.repeat([i*predict_period for i in range(max_p)], bins.shape[0]) + arrivals.min()
plt.vlines(arrivals, ymin=0, ymax=curve.max() + 0.5, color="r")
plt.step(bins, curve)
# plt.plot(exposure_date, exposure_hr)
plt.axvline(exposure_date.max(), linestyle="dashed", color="black")
plt.show()
# for i in range(60):
#     plt.axvspan((arrivals.min()-1 + i*predict_period), (arrivals.min()-1 + i*predict_period) + active_period, alpha=0.1)
# plt.savefig(pathlib.Path(outdir, "Timeline-extended.png"))