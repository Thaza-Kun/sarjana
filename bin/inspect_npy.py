#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-04-30
# Python Version: 3.12

"""Inspect an NPY file
"""

import argparse
import pathlib

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", "-f", help="NPY file", type=pathlib.Path, required=True
    )
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    item = np.unique(np.load(arguments.file).round(2))
    print(f"type\t: {item.dtype}")
    print(f"length\t: {len(item)}")
    print(f"Min\t: {item.min()}")
    print(f"Mean\t: {item.mean()}")
    print(f"Stdev\t: {item.std()}")
    print(f"Max\t: {item.max()}")
    print(f"Max-min\t: {item.max() - item.min()}")
    print(f"Av.Diff\t: {np.diff(item).mean()}")
    print(item)


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
