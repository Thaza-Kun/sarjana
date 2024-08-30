#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-03-15
# Python Version: 3.12

"""Extract observation times
"""

import argparse
import pathlib

import pandas as pd
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", "-f", help="Parquet source file", type=pathlib.Path)
    parser.add_argument("--out", "-o", help="Output folder", type=pathlib.Path)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    data = pd.read_parquet(arguments.file)
    dates = data["mjd_400"].unique()
    np.save(pathlib.Path(arguments.out, "observations.npy"), dates, allow_pickle=False)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
