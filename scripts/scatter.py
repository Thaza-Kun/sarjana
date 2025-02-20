# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
# ]
# ///
import argparse
import pathlib
from typing import List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", nargs="+", action="append", type=pathlib.Path, required=True
    )
    parser.add_argument("--xlin", type=str, action="append", required=False)
    parser.add_argument("--xlog", type=str, action="append", required=False)
    parser.add_argument("--y", type=str, required=True)
    parser.add_argument("--output", type=str, default="../output/")
    return parser.parse_args()


arguments = parse_arguments()

filenames: List[pathlib.Path] = [f[0] for f in arguments.csv]
xlin = arguments.xlin
xlog = arguments.xlog
ycol = arguments.y

xcols = []
if xlin:
    xcols.extend([(name, "linear") for name in xlin])
if xlog:
    xcols.extend([(name, "log") for name in xlog])

print(filenames)
f, axs = plt.subplots(1, len(xcols), sharey=True)
for filename in filenames:
    print(filename)
    ax = np.array(axs).reshape(-1)
    data = pd.read_csv(filename)
    for i, (xcol, scale) in enumerate(xcols):
        ax[i].scatter(x=data[xcol], y=data[ycol], label=filename.name)
        ax[i].set(xlabel=xcol)
        ax[i].set(ylabel=ycol)
        if scale == "log":
            ax[i].set(xscale="log")

plt.title("+".join([f.name for f in filenames]))
plt.legend()
plt.show()
