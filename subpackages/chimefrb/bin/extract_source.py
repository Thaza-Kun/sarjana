#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-03-15
# Python Version: 3.12

"""Extract all source data into individual numpy arrays 
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
    frbs = data["repeater_name"].unique()
    for frb in frbs:
        if frb == "-9999":
            continue
        print(frb)
        info = data[data["repeater_name"] == frb]
        path = pathlib.Path(arguments.out, frb)
        path.mkdir(exist_ok=True, parents=True)
        for col in info.columns:
            print(f"{frb}::{col} ({info[col].dtype})")
            if info[col].dtype != object:
                npz = info[col].to_numpy()
                np.save(pathlib.Path(path, col), npz, allow_pickle=False)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
