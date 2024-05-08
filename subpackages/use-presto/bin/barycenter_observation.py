#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-03-13
# Python Version: 3.7

"""Barycenters
"""

import argparse
import pathlib

import numpy as np

from presto.presto.prestoswig import barycenter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", "-p", help="Path to numpy files", type=pathlib.Path)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    for eventdir in arguments.path.iterdir():
        if (obspath := pathlib.Path(arguments.path, "observations.npy")).exists():
            topos = np.load(obspath)
            nn = len(topos)
            # # Vectors needed to be initialized
            bary = np.zeros(nn, "d")
            vel = np.zeros(nn, "d")
            # # Output written to file
            barycenter(topos, bary, vel, f"{ra}", f"{dec}", "CH", "DE200")
            np.save(
                pathlib.Path(eventdir, "mjd_400_observations_barycentered.npy"),
                bary,
                allow_pickle=False,
            )


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
