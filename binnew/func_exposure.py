import pathlib
import argparse
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from cdshealpix.ring import healpix_to_skycoord

plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 15
plt.rcParams["savefig.dpi"] = 300


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--nside", type=int, default=4096)
    parser.add_argument("--every", type=int, default=20000)
    return parser.parse_args()

arguments = parse_arguments()

file = arguments.file
root = arguments.root
nside = arguments.nside
every = arguments.every

T = 977 #days

data = np.load(pathlib.Path(root, file)).get('exposure') * 4 / 3600
skcoord = healpix_to_skycoord(np.arange(data.shape[0])[::every], nside=nside)

summary = {
    "exposure": data[::every],# * np.cos(np.deg2rad(skcoord.dec)),
    "ra": skcoord.ra,
    "dec": skcoord.dec
}

# summary=pd.DataFrame(summary)
# summary=summary[(10 <= summary["dec"]) & (summary["dec"] <= 11)]

def figure(
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    data: pd.DataFrame,
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
    plt.figure(figsize=(10, 6))
    plt.scatter(data[xcol], data[ycol])
    if hspan != None:
        plt.axhspan(hspan[0], hspan[1], color="green", alpha=0.2, hatch="/")
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    if savefig != None:
        plt.savefig(pathlib.Path(root, savefig))
    else:
        plt.show()

def overlay(axis)-> None:
    # DYDX = lambda e, d, dd: dd*e*np.tan(d)/np.cos(d)
    Y = lambda e, d, dd: e*np.cos(d + dd)/np.cos(d)
    Q = 0.7
    delta_deg = np.deg2rad(10)
    M = summary['exposure']
    # X = np.linspace(-10, 89)
    X = summary['dec']
    # axis.plot(X, Y(M, np.deg2rad(X)))
    # print(Y(M, np.deg2rad(X)+delta_deg))
    # print(Y(M, np.deg2rad(X)-delta_deg))
    axis.vlines(X, Y(M, np.deg2rad(X).value, delta_deg), Y(M, np.deg2rad(X).value, -delta_deg), color='orange')
    # axis.plot(X, DYDX(M, X)*delta_deg)
    # axis.plot(X, M+DYDX(M, X)*delta_deg)
    # axis.plot(X, M-DYDX(M, X)*delta_deg)

figure(xcol="dec", ycol="exposure", xlabel="dec", ylabel="exposure (s)", data=summary, xlog=False, ylog=True, overlay=overlay)