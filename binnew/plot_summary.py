import pathlib
import argparse
from collections import defaultdict
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from tqdm import tqdm

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 12
plt.rcParams["savefig.dpi"] = 300


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--root", type=str, default="../output/")
    parser.add_argument("--select", action="append", type=str)
    return parser.parse_args()


arguments = parse_arguments()
simname = arguments.folder
root = arguments.root
selections = [] if arguments.select == None else arguments.select

data = pd.read_csv(pathlib.Path(root, simname, "Summary.csv"), index_col=0)
datasub = pd.read_csv(pathlib.Path(root, simname, "Subwindows.csv"), index_col=0)

print(data.columns)
SELECTED = data[data.index.isin(selections)]
data = data[data.index.isin(selections) == False]

print(selections)
print(SELECTED)


def figure(
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    xlim: Tuple[float, float] | None = None,
    ylim: Tuple[float, float] | None = None,
    xlog: bool = True,
    ylog: bool = True,
    hspan: Tuple[float, float] | None = None,
    vline: float | None = None,
    linelabel: Tuple[str, str] | None = None,
    savefig: str | None = None,
    overlay: Callable | None = None,
) -> None:
    plt.figure(figsize=(6.92, 4.15))
    plt.plot(data[xcol], data[ycol], "s", label=r"Sample with $P_{cc} > 0.050$", markersize=4)
    plt.plot(datasub[xcol], datasub[ycol], "D", label="FRB20180916B", markersize=4)
    for name in SELECTED.iterrows():
        plt.scatter(
            name[1][xcol],
            name[1][ycol],
            marker="*",
            label=name[0],
            # markeredgecolor="red",
            # markeredgewidth=0.5,
        )
    if hspan != None:
        plt.axhspan(hspan[0], hspan[1], facecolor='None', edgecolor="green", alpha=0.5, hatch="/")
        plt.axhspan(hspan[0], hspan[1], color="green", alpha=0.2)
    if vline != None:
        plt.axvline(vline, color="grey")
        plt.text(
            vline - 0.1,
            0.04,
            r"$\downarrow$ " + linelabel[0],
            ha="right",
            va="top",
            rotation=-90,
        )
        plt.text(
            vline + 0.1,
            0.04,
            r"$\uparrow$ " + linelabel[1],
            ha="left",
            va="top",
            rotation=-90,
        )
    if overlay != None:
        overlay(plt.gca())
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    if xlim != None:
        plt.xlim(xlim[0], xlim[1])
    if ylog:
        plt.yscale("log")
    if xlog:
        plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=10)
    plt.tight_layout()
    if savefig != None:
        plt.savefig(pathlib.Path(root, simname, savefig))
    else:
        plt.show()


figure(
    xcol="avg_sep",
    ycol="p_cc",
    xlabel="Average days between events relative to period",
    ylabel=r"$P_{cc}$",
    ylim=(1e-4, 1),
    hspan=(0.0, 0.05),
    vline=1.0,
    linelabel=("clustered", "sparse"),
    savefig="Summary-pcc-vs-D.pdf",
)

def overlay_alpha(axis) -> None:
    for alpha in range(2, 6):
        k = np.log2(alpha)
        X = np.arange(2.0, 100.0, 1)
        funclin = lambda x: alpha * (x ** (-k))
        PccLimLin = lambda x: funclin(x)
        axis.step(
            X,
            PccLimLin(X),
            "--",
            alpha=0.7 if alpha != 3 else 1,
            where="pre",
            label=r"$\alpha$ = " + f"{alpha}, k = {k:.2f}",
        )


def overlay_k(axis) -> None:
    for k in np.linspace(1, 3, 5):
        alpha = 2**k
        X = np.arange(2.0, 100.0, 1)
        funclin = lambda x: alpha * (x ** (-k))
        PccLimLin = lambda x: funclin(x)
        axis.step(
            X,
            PccLimLin(X),
            ":",
            where="pre",
            label=f"k = {k:.2f}" + r", $\alpha$ = " + f"{alpha:.2f}",
        )


def overlay(axis) -> None:
    # overlay_k(axis)
    overlay_alpha(axis)


figure(
    xcol="n",
    ycol="p_cc",
    xlabel="Event counts",
    ylabel=r"$P_{cc}$",
    xlim=(2, 100),
    ylim=(1e-4, 1),
    hspan=(0.0, 0.05),
    savefig="Summary-pcc-vs-N.pdf",
    overlay=overlay,
)


data["n_exposure"] = data["n"] / data["exposure"]
datasub["n_exposure"] = datasub["n"] / datasub["exposure"]
SELECTED["n_exposure"] = SELECTED["n"] / SELECTED["exposure"]

figure(
    xcol="exposure",
    ycol="p_cc",
    xlabel="Exposure (hr)",
    ylabel=r"$P_{cc}$",
    ylim=(1e-4, 1),
    hspan=(0.0, 0.05),
    savefig="Summary-exposure-vs-p_cc.pdf",
)

data["exposure_tau"] = data["exposure"] / (data["tau"] * 24)
datasub["exposure_tau"] = datasub["exposure"] / (datasub["tau"] * 24)
SELECTED["exposure_tau"] = SELECTED["exposure"] / (SELECTED["tau"] * 24)

figure(
    xcol="exposure_tau",
    ycol="p_cc",
    xlabel=r"Exposure / $\tau$",
    ylabel=r"$P_{cc}$",
    ylim=(1e-4, 1),
    hspan=(0.0, 0.05),
    savefig="Summary-pcc-vs-exposure_tau.pdf",
)

data["rate_period"] = (data["n_exposure"] / data["tau"]) * 24
datasub["rate_period"] = (datasub["n_exposure"] / datasub["tau"]) * 24
SELECTED["rate_period"] = (SELECTED["n_exposure"] / SELECTED["tau"]) * 24

data["tau_rate"] = (data["n_exposure"] * data["period"])
datasub["tau_rate"] = (datasub["n_exposure"] * datasub["period"])
SELECTED["tau_rate"] = (SELECTED["n_exposure"] * SELECTED["period"])

figure(
    xcol="rate_period",
    ycol="p_cc",
    xlabel=r"Burst rate / $\tau$ ($d^{-2}$)",
    ylabel=r"$P_{cc}$",
    ylim=(1e-4, 1),
    hspan=(0.0, 0.05),
    savefig="Summary-rate_period-vs-p_cc.pdf",
)

figure(
    xcol="tau_rate",
    ycol="period",
    xlabel=r"Period (d)",
    ylabel=r"$\tau \times$ Burst rate (d/hr)",
    # ylim=(1e-4, 1),
    # hspan=(0.0, 0.05),
    savefig="Summary-tau_rate-vs-p.pdf",
)

figure(
    xcol="rate_period",
    ycol="n",
    xlabel=r"Burst rate / Period ($d^{-2}$)",
    ylabel="Event count",
    ylog=False,
    # xlog=False,
    # ylim=(1e-4, 1),
    # hspan=(0.0, 0.05),
    savefig="Summary-rate_period-vs-N.pdf",
)

figure(
    xcol="n_exposure",
    ycol="period",
    xlabel=r"Burst rate ($\text{hr}^{-1}$)",
    ylabel="Period (d)",
    # ylog=False,
    # xlog=False,
    # ylim=(1e-4, 1),
    # hspan=(0.0, 0.05),
    savefig="Summary-rate-vs-period.pdf",
)

data["tau_period"] = (data["tau"] - data["period"]) / data["tau"]
datasub["tau_period"] = (datasub["tau"] - datasub["period"]) / datasub["tau"]
SELECTED["tau_period"] = (SELECTED["tau"] - SELECTED["period"]) / SELECTED["tau"]

figure(
    xcol="tau_period",
    ycol="p_cc",
    xlabel=r"$(P - \tau) / \tau$",
    ylabel=r"$P_{cc}$",
    # ylabel="Burst rate (1/hr)",
    ylim=(1e-4, 1),
    xlim=(0, 1),
    xlog=False,
    hspan=(0.0, 0.05),
    savefig="Summary-pcc-vs-tau_period.pdf",
)

# figure(
#     xcol="n",
#     ycol="dec",
#     xlabel="Event count",
#     ylabel="Declination",
#     # ylabel="Burst rate (1/hr)",
#     # ylim=(1e-4, 1),
#     # xlim=(0, 1),
#     xlog=False,
#     ylog=False,
#     # hspan=(0.0, 0.05),
#     savefig="Summary-n-vs-dec.pdf",
# )