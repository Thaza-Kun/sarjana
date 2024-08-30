import pathlib
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams['font.size'] = 15

name = "FRB20180916B"
# events = 9
# exposure = 28.558
runs = 100
simname = "2024-08-04"

outdir = pathlib.Path(f"../output/{simname}/{name}/")

confidence = 0.05

# sizes = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]
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
]  # , 0.16, 0.17, 0.18, 0.19, 0.20]
sizes = sorted(sizes, reverse=True)
fig, axe = plt.subplots(5, 2, figsize=(10, 21), sharey=True, sharex=True)
axes = np.array(axe).reshape(-1)

for i, size in enumerate(tqdm(sizes)):
    ax = axes[i]
    outdirsize = pathlib.Path(outdir, f"{size:.2f}")
    for i in outdirsize.iterdir():
        if i.is_dir():
            name, val = i.name.split("=")
            if name == "value_number_of_events":
                n_event = int(val)
            elif name == "value_total_exposure":
                exposure_hr = float(val)
            elif name == "value_event_window":
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
    # ax.plot((x_mean, x_mean + vec[0][1]), (y_mean, y_mean+vec[0][0]))
    # ax.plot((x_mean, x_mean + vec[1][1]), (y_mean, y_mean+vec[1][0]))
    ax.scatter(
        chisquare_stat[peak],
        inactive_stat[peak],
        50,
        facecolor="None",
        edgecolors="green",
    )

    # std = np.std(pdf - pdf.max())
    # CS = ax.contour(x, y, (pdf.max() - pdf) / std, levels=[1, 2, 3, 4, 5], colors="black", alpha=0.5)
    # ax.clabel(CS, fmt=lambda x: f"{int(x)}σ")
    # CS = ax.contour(x, y, prob, levels=[confidence], colors="red", linestyle="dashed")
    # ax.clabel(CS, fmt=lambda x: f"{x:.2f} %")
    # ax.xlabel(r"Uniformity measure ($\chi^2$)")
    # ax.ylabel("Inactivity fraction")
    ax.text(
        0.95,
        0.05,
        f"n = {n_event}\nE = {exposure_hr:.2f} hr",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
    )

    ax.set(ylim=(-0.1, 1.1), title=f"{size*100:.0f}% T ($\\tau$ = {event_window} day)")

# ax = subfigs[1].subplots(3, 1, sharex=True)
# ax[0].plot(periods, chisquare_stat)
# ax[0].scatter(periods[peak], chisquare_stat[peak], 50, facecolor="None", edgecolor="green")
# ax[0].set_ylabel(r"Uniformity measure ($\chi^2$)")

# if accept_peak:
#     _, _, l, r = peak_widths(cdf - 1, [peak], rel_height=confidence)
#     period_neg = periods[peak] - remap(periods, l)[0]
#     period_pos = remap(periods, r)[0] - periods[peak]

#     for axis in ax:
#         [
#             axis.axvline(
#                 periods[peak] * n,
#                 color="green",
#                 alpha=0.1,
#                 ymin=axis.get_ylim()[0],
#                 ymax=axis.get_ylim()[1],
#             )
#             for n in range(5)
#         ]
#         [
#             axis.axvline(
#                 periods[peak] * n,
#                 color="green",
#                 alpha=0.1,
#                 ymin=axis.get_ylim()[0],
#                 ymax=axis.get_ylim()[1],
#                 linestyle=":",
#             )
#             for n in np.arange(-4.5, 4.5, 1)
#         ]
#         axis.axvspan(period_neg, period_pos, alpha=0.1, color="black")
#     ax[0].annotate(
#         f"{periods[peak]:.2f} (-{period_neg:.2f},+{period_pos:.2f}) [{inactive_stat[peak]:.2f}% inactive]",
#         xy=(periods[peak] + 0.5, chisquare_stat.max() - 0.5),
#     )

# ax[1].plot(periods, inactive_stat)
# ax[1].scatter(periods[peak], inactive_stat[peak], 50, facecolor="None", edgecolor="green")
# # ax[1].plot(periods[peak], inactive_stat[peak], 'x')
# ax[1].set_ylabel("Inactivity fraction")

# ax[2].semilogy(periods, 1 - cdf)
# ax[2].semilogy(periods[point5_peak], 1 - cdf[point5_peak], 'r+')
# ax[2].semilogy(periods[chisquare_peak], 1-cdf[chisquare_peak], 'kx')
# ax[2].scatter(periods[peak], 1-cdf[peak], 50, facecolor="None", edgecolor="green")
# # ax[2].semilogy(periods[peak], 1-cdf[peak], 'x')
# ax[2].axhline(confidence, linestyle="--", alpha=0.5, color="red")
# ax[2].set_ylabel("Probability")

# plt.xlim(periods[0], periods[-1])
fig.suptitle("FRB20180916B")
fig.supxlabel(r"$\chi^2$ / ($\Phi$ - 1)")
fig.supylabel(r"$F_0$")
plt.legend(loc='right')
plt.tight_layout()
plt.savefig(pathlib.Path(outdir, "Composite-periodogram-5-x-5.png"))
