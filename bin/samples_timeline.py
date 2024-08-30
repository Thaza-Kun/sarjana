#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-06-14
# Python Version: 3.12

import argparse
import csv
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", help="CSV file of list of FRBs", type=pathlib.Path, required=True
    )
    parser.add_argument("--outdir", help="Output dir", type=pathlib.Path, required=True)
    return parser.parse_args()


def get_column(eventname: str, columnname: str, parent: str) -> np.ndarray:
    return np.load(pathlib.Path(parent, eventname, f"{columnname}.npy"))


def main(arguments: argparse.Namespace):
    data = pd.read_csv(
        arguments.csv,
        sep=r"\s*,\s*",
        na_values="-",
        comment="#",
        quotechar='"',
        quoting=csv.QUOTE_STRINGS,
        engine="python",
    )
    data.loc[:, "observations"] = data.apply(
        lambda x: get_column(x["eventname"], "mjd_400_barycentered", x["folder"]),
        axis=1,
    )
    data.loc[:, "eventcounts"] = data.apply(lambda x: len(x["observations"]), axis=1)
    data.loc[:, "begin"] = data.apply(lambda x: min(x["observations"]), axis=1)
    data.loc[:, "end"] = data.apply(lambda x: max(x["observations"]), axis=1)
    data_unique = data.sort_values(by="eventcounts", ascending=False)
    data = data_unique.explode("observations")
    # data = data.sort_values(by=["eventcounts", "observations"], ascending=[False, True])
    palette = sns.color_palette(n_colors=data["eventname"].nunique())
    ax = plt.figure(figsize=(15, 7))
    for i, row in enumerate(data_unique.itertuples()):
        plt.axhline(row.eventname, color="k", alpha=0.1)
        plt.hlines(row.eventname, row.begin, row.end, color=palette[i], alpha=0.3)
    # for (e, b, n) in data[['eventname', "begin", "end"]]:
    #     plt.vlines([e, e],[b, n], color='k', alpha=0.2)
    g = sns.scatterplot(
        data,
        x="observations",
        y="eventname",
        hue="eventname",
        legend=False,
        ax=ax.axes[0],
        palette=palette,
    )
    g.yaxis.tick_right()
    g.yaxis.label.set_text("")
    g.xaxis.label.set_text("MJD")
    g.set_title("Observation timeline of samples")
    plt.tight_layout()
    plt.savefig(pathlib.Path(arguments.outdir, "timeline.png"))
    # for row in data[["eventname", "folder"]].itertuples():
    #     print(get_column(row.eventname, "mjd_400_observations_barycentered", row.folder))


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
