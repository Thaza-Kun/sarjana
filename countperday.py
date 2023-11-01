import os
from pathlib import Path
import warnings

import typer
import enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from astropy.time import Time

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


def main(
    chosen_name: FRBName,
    period: int = typer.Option(default=...),
    activity: int = typer.Option(default=...),
    offset: int = typer.Option(0),
    alpha: float = typer.Option(0.5),
):
    chosen_name: str = chosen_name.value

    cat1 = pd.read_parquet(CATALOG1_PATH_PARQUET)
    cat2023 = pd.read_csv(CATALOG_PATH_CSV).rename(columns={"tns_name": "eventname"})
    data = pd.concat([cat1, cat2023])

    data["mjd"] = data["mjd_400"]

    selected = data[["repeater_name", "mjd"]].sort_values(by="mjd")
    selected[chosen_name] = (selected["repeater_name"] == chosen_name).astype(int)
    selected["datetime"] = Time(selected["mjd"], format="mjd").to_datetime()

    chosen = (
        selected.set_index("datetime").resample("d").sum()[chosen_name].reset_index()
    )
    interval = chosen["datetime"][period + offset :: period]
    interval_s = chosen["datetime"][period - activity + offset :: period]
    start = interval.to_list()[10]
    end = interval.to_list()[21]
    a = sns.relplot(
        chosen,
        x="datetime",
        y=chosen_name,
        kind="line",
        drawstyle="steps-post",
        aspect=3,
    )
    a.set(
        xlim=(start, end),
        ylim=(0, 5),
        ylabel="count",
        title=f"Number of events of {chosen_name} per day",
    )
    for i, b in zip(interval, interval_s):
        a.ax.axvspan(b, i, color="gray", alpha=alpha)
    a.figure.tight_layout()
    a.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f"{chosen_name}-countplot.png")
    pd.DataFrame(
        [
            {
                "eventname": chosen_name,
                "start_observation": chosen["datetime"].min(),
                "end_observation": chosen["datetime"].max(),
                "start_window": start,
                "end_window": end,
                "count": chosen[chosen_name].sum(),
            }
        ]
    ).to_csv(f"{chosen_name}-countperday.csv")


if __name__ == "__main__":
    typer.run(main)
