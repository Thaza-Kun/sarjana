import os
from pathlib import Path
import warnings

import typer
import enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress FutureWarning messages
warnings.simplefilter(action="ignore", category=FutureWarning)

DATAPATH = os.getenv("DATAPATH")

assert DATAPATH is not None, "Please provide data path."

CATALOG_PATH = Path(DATAPATH, "raw", "catalog2023", "catalog2023_profile.parquet")
CATALOG_PATH_CSV = Path(DATAPATH, "raw", "catalog2023", "chimefrb2023repeaters.csv")
CATALOG1_PATH_PARQUET = Path(DATAPATH, "catalog_1.parquet")


class FRBName(enum.StrEnum):
    FRB20180916B = "FRB20180916B"  # (77)
    FRB20190915D = "FRB20190915D"  # 👍 (10)
    FRB20191106C = "FRB20191106C"  # 👎 (7)


def get_count_and_waiting_time(dataframe: pd.DataFrame, keep: list) -> pd.DataFrame:
    grouped = dataframe.groupby("repeater_name", group_keys=True)[keep].apply(
        lambda x: x
    )
    count = grouped.groupby("repeater_name")["mjd_400"].count().rename("count")
    grouped = pd.merge(grouped, count, on="repeater_name")
    grouped["wait"] = grouped.groupby("repeater_name")["mjd_400"].diff()
    grouped["log_wait"] = grouped["wait"].apply(lambda x: np.log10(x))
    return grouped


def main(
    # chosen_name: FRBName,
):
    cat1 = pd.read_parquet(CATALOG1_PATH_PARQUET)
    cat2023 = pd.read_csv(CATALOG_PATH_CSV).rename(columns={"tns_name": "eventname"})
    cat1 = pd.concat([cat1, cat2023])
    cat1_grouped = get_count_and_waiting_time(
        cat1.loc[cat1["repeater_name"] != "-9999"], keep=["eventname", "mjd_400", "dm"]
    )
    print(cat1_grouped)

    names = [n.name for n in FRBName]
    resetted = cat1_grouped.reset_index().sort_values(
        by="repeater_name", ascending=False
    )
    resetted = resetted.loc[resetted["repeater_name"].isin(names)]
    g = sns.displot(
        resetted,
        x="log_wait",
        hue="repeater_name",
        fill=False,
        bins=np.arange(-9, 5, 1),
        kind="hist",
    )
    g.set(
        xlim=(-8.5, 3.5),
        ylim=(0, 30),
        xlabel=r"$\log_{10}$ waiting time",
        ylabel="detections",
    )

    # grid = sns.JointGrid(
    #     resetted,
    #     x='log_wait',
    #     y='mjd_400',
    #     hue='repeater_name',
    # )
    # grid.plot_joint(sns.scatterplot)
    # grid.plot_marginals(sns.histplot,
    #     fill=False,
    #     binwidth=1)
    # plt.show()
    g.savefig("waiting.png")


if __name__ == "__main__":
    typer.run(main)
