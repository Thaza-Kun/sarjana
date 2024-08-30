import os
from pathlib import Path
import warnings

import typer
import enum

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u

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
    period: float = typer.Option(default=...),
    tick: int = typer.Option(1),
):
    chosen_name: str = chosen_name.value

    cat1 = pd.read_parquet(CATALOG1_PATH_PARQUET)
    cat2023 = pd.read_csv(CATALOG_PATH_CSV).rename(columns={"tns_name": "eventname"})
    data = pd.concat([cat1, cat2023])

    data["mjd"] = data["mjd_400"]
    selected = data[["repeater_name", "mjd"]].sort_values(by="mjd")
    selected[chosen_name] = (selected["repeater_name"] == chosen_name).astype(int)
    selected["datetime"] = Time(selected["mjd"], format="mjd").to_datetime()

    data = selected.set_index("datetime").resample("d").sum()

    timeseries = TimeSeries(
        time=data.index,
        data={"detections": data[chosen_name].to_numpy().reshape(-1, 1)},
    )
    folded = timeseries.fold(period=period * u.day, wrap_phase=1, normalize_phase=True)
    phases = np.array(folded["time"])
    counts = np.array(folded["detections"]).flatten()
    phases = (
        pd.DataFrame({"phase": phases, "detections": counts})
        .groupby(pd.cut(phases, np.arange(0, 1, 0.05)))["detections"]
        .sum()
    )
    phases = phases.reset_index().rename(columns={"index": "phase_bin"})
    phases["phase"] = phases["phase_bin"].apply(lambda x: x.left)
    phases = pd.concat([phases, pd.DataFrame([{"detections": 0, "phase": 1.0}])])

    a = sns.relplot(
        phases, x="phase", y="detections", kind="line", drawstyle="steps-post"
    )
    YAXIS = np.arange(0, phases["detections"].max() + 10, tick)
    a.ax.set_yticks(YAXIS)
    a.set(ylim=(0, 10 * (1 + int(phases["detections"].max() / 10))))
    plt.title(chosen_name)
    plt.tight_layout()
    plt.savefig(f"{chosen_name}-phase-{period}.png")


if __name__ == "__main__":
    typer.run(main)
