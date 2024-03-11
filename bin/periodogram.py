from collections import defaultdict
import os
import enum
from pathlib import Path
from typing import Callable

import warnings

import pandas as pd
import numpy as np

from rich.progress import track

import typer

import seaborn as sns
import matplotlib.pyplot as plt

from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.timeseries import TimeSeries
import astropy.units as u

from datetime import datetime

from pdmpy import pdm


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


def calc_inactive_frac(timeseries: TimeSeries, trial_periods: np.ndarray):
    frac = []
    for period in trial_periods:
        folded_ = timeseries.fold(period=period, wrap_phase=1, normalize_phase=True)
        phases = np.array(folded_["time"])
        counts = np.array(folded_["detections"]).flatten()
        phases = (
            pd.DataFrame({"phase": phases, "detections": counts})
            .groupby(pd.cut(phases, np.arange(0, 1, 0.05)))["detections"]
            .sum()
        )
        phases = phases.reset_index().rename(columns={"index": "phase_bin"})
        phases["cumsum"] = phases["detections"].cumsum()
        inactive = 0
        state = 0
        prev = 0
        for current in phases["cumsum"]:
            if current == prev:
                state += 1
            else:
                prev = current
                inactive = state if state > inactive else inactive
                state = 0
        frac.append(inactive / len(phases))
    return frac


def LombScargle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> np.ndarray:
    LS = LombScargle(time, obs)
    power = LS.power(freq_grid)
    return power


def duty_cycle_periodogram(
    time: u.Quantity, obs: np.ndarray, freq_grid: u.Quantity
) -> np.ndarray:
    timeseries = TimeSeries(time=time, data={"detections": obs.reshape(-1, 1)})
    return calc_inactive_frac(timeseries, 1 / freq_grid)


def pdm_periodogram(
    time: np.ndarray, obs: np.ndarray, freq_grid: u.Quantity
) -> np.ndarray:
    if type(time) is Time:
        time = time.value
    f, theta = pdm(
        time,
        obs,
        f_min=freq_grid.min().value,
        f_max=freq_grid.max().value,
        delf=freq_grid.diff()[0].value,
    )
    return theta


def estimate_error_leave_one_out(
    chosen_name: str,
    detections: np.ndarray,
    selected: pd.DataFrame,
    freq_grid: u.Quantity,
    periodogram: Callable,
    phase_folding: bool,
    optimize: Callable,
    description: str,
) -> float:
    best_periods = []
    for idx in track(
        detections, description=f"Estimating Uncertainty ({description})..."
    ):
        if phase_folding:
            folded = selected.loc[selected.index != idx, ["mjd", chosen_name]]
            folded["datetime"] = Time(folded["mjd"], format="mjd").to_datetime()
        else:
            folded = selected
        reduced = (
            folded[folded.index != idx]
            .set_index("datetime")
            .resample("d")
            .sum(numeric_only=True)
        )

        time = Time(reduced.index.to_numpy())
        obs = reduced[chosen_name].to_numpy()

        power_ = periodogram(time, obs, freq_grid)
        best_periods.append(1 / freq_grid[optimize(power_)].value)

    return (
        pd.DataFrame(np.array(best_periods))
        .describe()
        .transpose()
        .to_dict(orient="records")[0]
    )["std"]


def calculate(
    chosen_name: FRBName,
    lombscargle: bool = typer.Option(False, "--ls"),
    dutycycle: bool = typer.Option(False, "--dc"),
    phase_disp_min: bool = typer.Option(False, "--pdm"),
    begin: datetime = typer.Option(None),
    end: datetime = typer.Option(None),
    n_0: int = typer.Option(5, "--n"),
):
    chosen_name: str = chosen_name.value

    cat1 = pd.read_parquet(CATALOG1_PATH_PARQUET)[
        ["eventname", "repeater_name", "mjd_400"]
    ]
    cat1["catalog"] = "Catalog 1"
    cat2023 = pd.read_csv(CATALOG_PATH_CSV)[
        ["tns_name", "repeater_name", "mjd_400"]
    ].rename(columns={"tns_name": "eventname"})
    cat2023["catalog"] = "Catalog 2023"
    data = pd.concat([cat1, cat2023])
    data["mjd"] = data["mjd_400"]

    selected = data[["repeater_name", "mjd", "catalog"]].sort_values(by="mjd")
    selected[chosen_name] = (selected["repeater_name"] == chosen_name).astype(int)
    selected["datetime"] = Time(selected["mjd"], format="mjd").to_datetime()

    if begin:
        selected = selected[(selected["datetime"] > Time(begin).to_datetime())]
    if end:
        selected = selected[(selected["datetime"] < Time(end).to_datetime())]

    target_df = selected.set_index("datetime").resample("d").sum()

    time = Time(target_df.index.to_numpy())
    obs = target_df[chosen_name].to_numpy()

    freq_min = 1 / (3 * u.day)
    freq_max = 1 / (len(target_df) * 0.5 * u.day)

    n_eval = int(n_0 * freq_min.value * len(target_df))

    freq_grid = np.linspace(freq_max, freq_min, n_eval)
    detections = selected.loc[selected[chosen_name] == 1].index

    print(len(target_df.index.to_list()), " days of observation.")
    print(len(detections), " detections.")

    if lombscargle:
        # Lomb-Scargle
        power = LombScargle_periodogram(time, obs, freq_grid)
        ls_period = 1 / freq_grid[np.nanargmax(power)]
        ls_fap = LombScargle(time, obs).false_alarm_probability(
            np.nanmax(power),
            method="bootstrap",
            minimum_frequency=freq_grid.min(),
            maximum_frequency=freq_grid.max(),
        )

        ## Leave-One-Out (Lomb-Scargle)
        ls_stdev = estimate_error_leave_one_out(
            chosen_name=chosen_name,
            detections=detections,
            selected=selected,
            freq_grid=freq_grid,
            periodogram=LombScargle_periodogram,
            phase_folding=False,
            optimize=np.nanargmax,
            description="Lomb-Scargle",
        )
        low_ = ls_period.value - ls_stdev * 2
        high_ = ls_period.value + ls_stdev * 2

        g = sns.lineplot(x=1 / freq_grid, y=power)
        g.axvline(ls_period.value, color="red", alpha=1)
        g.axvspan(min(low_, 1 / freq_grid.min().value), high_, alpha=0.3)
        g.set_xscale("log")
        g.set_xlabel("period")
        g.set_ylabel("LS Power")
        g.set_title(
            f"Lomb-Scargle Periodogram of {chosen_name} ({ls_period.value:.2f}±{2*ls_stdev:.2f} d)"
        )

        g.figure.savefig(f"{chosen_name}-periodogram-LS.png")
        plt.figure()

        ## SAVE
        np.save("freq-grid.npy", freq_grid.value)
        np.save(f"{chosen_name}-LS-power.npy", power.value)
        result = {
            "period": ls_period.value,
            "stdev": 2 * ls_stdev,
            "false_alarm_probability": ls_fap,
        }
        pd.DataFrame(result, index=[0]).to_csv(
            f"{chosen_name}-lomb-scargle.csv", index=False
        )

    if dutycycle:
        # Phase Folding
        selected_ = selected[["mjd", chosen_name]]
        selected_["datetime"] = Time(selected_["mjd"], format="mjd").to_datetime()

        reduced = selected_.set_index("datetime").resample("d").sum()

        frac = duty_cycle_periodogram(
            reduced.index, reduced[chosen_name].to_numpy(), freq_grid
        )
        max_inactive_idx = np.nanargmax(frac)
        PF_period = 1 / freq_grid[max_inactive_idx]

        ## Leave-One-Out (Phase-Folding)
        PF_stdev = estimate_error_leave_one_out(
            chosen_name=chosen_name,
            detections=detections,
            selected=selected,
            freq_grid=freq_grid,
            periodogram=duty_cycle_periodogram,
            phase_folding=True,
            optimize=np.nanargmax,
            description="Duty Cycle",
        )

        low_ = PF_period.value - (PF_stdev * 2)
        high_ = PF_period.value + (PF_stdev * 2)

        g = sns.lineplot(x=1 / freq_grid, y=frac)
        g.axvline(PF_period.value, color="red", alpha=1)
        g.axvspan(low_, high_, alpha=0.3)
        g.set_xscale("log")
        g.set_xlabel("Period (days)")
        g.set_ylabel("Inactive Fraction (%)")
        g.set_title(
            f"Phase Folding Periodogram of {chosen_name} ({PF_period:.2f}±{2*PF_stdev:.2f} d)"
        )

        g.figure.savefig(f"{chosen_name}-periodogram-PF.png")
        plt.figure()

        ## SAVE
        np.save(f"{chosen_name}-PF-power.npy", frac)
        result = {
            "period": PF_period.value,
            "stdev": 2 * PF_stdev,
            "duty_cycle": 1 - frac[max_inactive_idx],
        }
        pd.DataFrame(result, index=[0]).to_csv(
            f"{chosen_name}-duty-cycle.csv", index=False
        )

    if phase_disp_min:
        # Phase Dispersion Minimization
        time = target_df.index.to_numpy()

        theta = pdm_periodogram(time, obs, freq_grid)
        PDM_period = 1 / freq_grid[np.nanargmin(theta)]

        ## Leave One Out (PDM)
        PDM_stdev = estimate_error_leave_one_out(
            chosen_name=chosen_name,
            detections=detections,
            selected=selected,
            freq_grid=freq_grid,
            periodogram=pdm_periodogram,
            phase_folding=False,
            optimize=np.nanargmin,
            description="PDM",
        )
        low = PDM_period.value - PDM_stdev * 2
        high = PDM_period.value + PDM_stdev * 2

        g = sns.lineplot(x=1 / freq_grid, y=theta)
        g.axvline(PDM_period.value, color="red", alpha=1)
        g.axvspan(low, high, alpha=0.3)
        g.set_xscale("log")
        g.set_xlabel("Period (days)")
        g.set_ylabel("Theta")
        g.set_title(
            f"PDM Periodogram of {chosen_name} ({PDM_period:.2f}±{2*PDM_stdev:.2f} d)"
        )

        g.figure.savefig(f"{chosen_name}-periodogram-PDM.png")
        ## SAVE
        np.save(f"{chosen_name}-PDM-power.npy", theta)
        result = {
            "period": PDM_period.value,
            "stdev": 2 * PDM_stdev,
        }
        pd.DataFrame(result, index=[0]).to_csv(f"{chosen_name}-PDM.csv", index=False)


def graph(width: float, height: float, wspcae: float = typer.Option(0.15)):
    methods = defaultdict(defaultdict)
    freqgrid = np.load('./data/freq-grid.npy')
    for method in ["LS", "PF", "PDM"]:
        items = defaultdict(np.array)
        for name in FRBName._member_names_:
            items[name] = np.load(f'./data/{name}-{method}-power.npy')
        methods[method] = items
    print(methods['LS'][FRBName.FRB20190915D])
    print(freqgrid)
    fig, ax = plt.subplots(3,3, sharex=True, figsize=(width,height))
    fig.subplots_adjust(hspace=0)
    fig.suptitle("Periodograms of different methods and FRBs")
    fig.supxlabel('Period (days)')
    for x in ax.flatten():
        x.set_xscale('log')
    ax[0, 0].set_title(FRBName.FRB20180916B)
    ax[0, 1].set_title(FRBName.FRB20190915D)
    ax[0, 2].set_title(FRBName.FRB20191106C)
    
    for row, label, file, key in zip(range(3), ['Lomb-Scargle Power', 'Duty Cycle', 'Phase Dispersion (Theta)'], ['lomb-scargle', 'duty-cycle', 'PDM'], ['LS', 'PF', 'PDM']):
        ax[row, 0].set_ylabel(label)
        for col, name in zip(range(3),FRBName._member_names_):
            best = pd.read_csv(f'./data/{name}-{file}.csv')
            ax[row, col].plot(1/freqgrid, methods[key][name])
            ax[row, col].axvline(best['period'].item(), color='red')
            ax[row, col].axvspan(
                max(0, (best["period"] - (best['stdev'])).item()), 
                (best["period"] + (best['stdev'])).item(),
                alpha=0.2
            )
    
    plt.tight_layout()
    plt.savefig('periodograms.png')

if __name__ == "__main__":
    app = typer.Typer()
    app.command()(calculate)
    app.command()(graph)
    app()
