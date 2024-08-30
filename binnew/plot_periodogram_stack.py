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
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--simname", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--period", type=float, default=None)
    parser.add_argument("--periodpos", type=float, default=0.0)
    parser.add_argument("--periodmin", type=float, default=0.0)
    return parser.parse_args()

arguments = parse_arguments()

name = arguments.name
runs = arguments.runs
simname = arguments.simname
confidence = arguments.confidence
size = arguments.size
p = arguments.period
p_p = arguments.periodpos
p_m = arguments.periodmin

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
        # elif name == "value_burst_rate":
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

## STEP [7] Use CDF from [5] to calculate P(X^2, Inactive_frac)
cdf1 = rv.cdf(np.dstack((chisquare_stat, inactive_stat)))

prob = 1 - cdf1
point5_peak = np.argwhere(cdf1 > 1 - confidence)
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

fig, ax = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
ax[0].plot(periods, chisquare_stat)
ax[0].set(ylabel=r"$\chi^2$ / ($\Phi$ - 1)")
ax[1].plot(periods, inactive_stat)
ax[1].set(ylabel=r"$F_0$")
ax[2].semilogy(periods, prob)
ax[2].axhline(confidence, c="r", linestyle="-.")
ax[2].set(ylabel=r"$p$")
ax[2].invert_yaxis()

fig.suptitle(name + r" ($\tau$ = " + f"{event_window} d)")
fig.supxlabel("Period (d)", y=0.05)

if p is not None:
    for a in ax:
        a.axvline(p, c="g", alpha=0.5)
        a.axvspan((p + p_p), (p - p_m), color="g", alpha=0.3)
    plt.annotate(r'$'+f'{p:.2f}'+r'_{-'+f'{p_m:.2f}'+r'}^{+' + f'{p_p:.2f}' +r'}$ ' + f'(p = {prob[peak]:.2f})', xy=(p, 0), xytext=(p, -0.35), ha='center', xycoords=('data', 'axes fraction'), arrowprops={'arrowstyle': '->'})

fig.align_ylabels()
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, f"{name}-Periodogram-stack-{size:.2f}.png"))