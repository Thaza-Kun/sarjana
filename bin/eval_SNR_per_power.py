#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-04-29
# Python Version: 3.12

"""Evaluating SNR per Power
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
        "--grid", help="Freq grid file", type=pathlib.Path, required=True
    )
    parser.add_argument("--snr", help="SNR file", type=pathlib.Path, required=True)
    parser.add_argument("--outdir", help="Output dir", type=pathlib.Path, required=True)
    parser.add_argument("--method", help="Method name", type=str, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    power = np.load(arguments.pgram)
    SNR = np.load(arguments.snr)
    grid = np.load(arguments.grid)

    snr_p_power = SNR / power
    power_p_snr = power / SNR
    snr_power = SNR * power

    np.save(
        pathlib.Path(
            arguments.pgram.parent, f"{arguments.pgram.name}-SNR-per-power.npy"
        ),
        snr_p_power,
    )
    np.save(
        pathlib.Path(arguments.pgram.parent, f"{arguments.pgram.name}-SNR-x-power.npy"),
        snr_power,
    )
    np.save(
        pathlib.Path(
            arguments.pgram.parent, f"{arguments.pgram.name}-power-per-SNR.npy"
        ),
        power_p_snr,
    )

    plt.figure()
    plt.title(f"SNR/power spectrum of {arguments.method}")
    plt.semilogx(1 / grid, snr_p_power)
    plt.savefig(
        pathlib.Path(arguments.outdir, f"{arguments.pgram.name}-SNR-per-power.png")
    )

    plt.figure()
    plt.title(f"power/SNR spectrum of {arguments.method}")
    plt.semilogx(1 / grid, power_p_snr)
    plt.savefig(
        pathlib.Path(arguments.outdir, f"{arguments.pgram.name}-power-per-SNR.png")
    )

    plt.figure()
    plt.title(f"SNR*power spectrum of {arguments.method}")
    plt.semilogx(1 / grid, power_p_snr)
    plt.savefig(pathlib.Path(arguments.outdir, f"{arguments.pgram.name}-SNR-power.png"))


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
