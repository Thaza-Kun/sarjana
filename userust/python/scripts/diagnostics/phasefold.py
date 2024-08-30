#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-05-10
# Python Version: 3.12

"""Diagnose phase folded data
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=float, default=2 * np.pi)
    parser.add_argument("--limit", type=float, default=0.0)
    parser.add_argument("--parent", type=pathlib.Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    period = arguments.period
    parent = arguments.parent
    name = arguments.name
    limit = arguments.limit

    def get_column(columnname: str) -> np.ndarray:
        return np.load(pathlib.Path(parent, name, f"{columnname}.npy"))

    time = get_column("mjd_400_barycentered")
    obs = get_column("fluence")
    time, obs = np.compress([o > limit for o in obs], [time, obs], axis=1)

    def fold_time(array: np.ndarray, period: float) -> np.ndarray:
        return (array / period) % 1

    phases = fold_time(time, period)
    ordered_index = np.lexsort([obs, phases])
    obs = obs[ordered_index]
    phases = phases[ordered_index]
    head = phases[0] - 0
    tail = 1 - phases[-1]
    combined = head + tail
    longest = max(np.diff(phases).max(), combined)
    print(longest)
    plt.scatter(phases, obs)
    plt.xlim(0, 1)
    plt.show()


def run():
    arguments = parse_arguments()
    main(arguments)


if __name__ == "__main__":
    run()
