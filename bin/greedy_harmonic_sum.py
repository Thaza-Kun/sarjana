#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-03-13
# Python Version: 3.12

"""Harmonic sum
"""

import argparse
import pathlib
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgram", "-p", help="Periodogram file", type=pathlib.Path, required=True
    )
    parser.add_argument(
        "--grid", "-g", help="Freq grid file", type=pathlib.Path, required=True
    )
    parser.add_argument("--harmonics", "-H", help="Harmonics", type=int, default=16)
    parser.add_argument(
        "--inverse", help="Flip power on y axis", type=bool, default=False
    )
    parser.add_argument("--outdir", help="Output dir", type=pathlib.Path, required=True)
    parser.add_argument("--method", help="Method name", type=str, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    grid = np.load(arguments.grid)
    power = np.load(arguments.pgram)
    power = -power if arguments.inverse else power

    def get_snr(sums: np.ndarray, mean: float, std: float) -> float:
        return np.max((sums - mean) / std)

    def compare_snr(
        snr: float, h: int, snr_max: float, h_m: float
    ) -> Tuple[float, float]:
        return np.nanmax([snr, snr_max]), np.nanmax([h, h_m])

    SNR = np.empty(len(grid))
    for i, f in tqdm(enumerate(grid)):
        let_sum = 0
        d = 0
        snr_max = 0
        h_m = 0
        # fundamental bin
        x_d = power[i]
        x_d_plus_1 = power[min(i + 1, len(grid) - 1)]
        if x_d_plus_1 > x_d:
            d += 1
            let_sum += x_d_plus_1
        else:
            let_sum += x_d
        snr_max = get_snr(let_sum, power.mean(), power.std())
        # higher harmonics
        for h in range(arguments.harmonics):
            x_d = power[min(int(h * i + d), len(grid) - 1)]
            x_d_plus_1 = power[min(int(h * i + d + 1), len(grid) - 1)]
            if x_d_plus_1 > x_d:
                d += 1
                let_sum += x_d_plus_1
            else:
                let_sum += x_d
            snr = get_snr(let_sum, power.mean(), power.std())
            snr_max, h_m = compare_snr(snr_max, h_m, snr, h)
        SNR[i] = snr_max
        h_0 = h_m
    print("Saving...")
    np.save(
        pathlib.Path(arguments.pgram.parent, f"{arguments.pgram.name}-SNR.npy"), SNR
    )

    plt.figure()
    plt.title(f"SNR spectrum of {arguments.method}")
    plt.semilogx(1 / grid, SNR)
    plt.savefig(pathlib.Path(arguments.outdir, f"{arguments.pgram.name}-SNR.png"))


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
