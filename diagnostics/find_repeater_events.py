# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "seaborn",
# ]
# ///
""" """
import argparse
import pathlib
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    names = [
        "FRB20190804E",
        "FRB20190915D",
        "FRB20200223B",
        "FRB20201130A",
        "FRB20201221B",
    ]
    file = arguments.file
    catalog = pl.read_csv(file, null_values="-9999")
    f = catalog.filter(pl.col("repeater_name") == names[0])
    print(f.columns)
    print(
        f.select(
            pl.col(
                "tns_name",
                "dm_fitb",
                "width_fitb",
                "mjd_400",
                "high_freq",
                "peak_freq",
                "low_freq",
                "sub_num",
            )
        )
    )


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
