import json
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, TypeVar, Dict, Tuple

import logging

import numpy as np
import scipy
from pandas import DataFrame
import pandas as pd
from dataclasses import dataclass, asdict

from src.paths import data_path


@dataclass
class ParameterRecord:
    columns: List[str]
    min_cluster_size: int
    seed: int


@dataclass
class ScoreRecord:
    value: float
    metric: str


@dataclass
class WorkflowMetadata:
    timestamp: datetime
    parameters: ParameterRecord
    score: ScoreRecord
    name: str = ""


WorkflowResult = TypeVar("WorkflowResult", bound=Tuple[DataFrame, WorkflowMetadata])
DataFunc = TypeVar("DataFunc", bound=Callable[..., DataFrame])
FlowFunc = TypeVar("FlowFunc", bound=Callable[..., WorkflowResult])
StringOptions = TypeVar("StringOptions", bound=Optional[List[str]])

formatter = logging.Formatter("%(levelname)s::%(name)s::%(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

datalogger = logging.getLogger(f"{__name__}:data:")
datalogger.addHandler(stream_handler)


flowlogger = logging.getLogger(f"{__name__}:workflow:")
flowlogger.addHandler(stream_handler)


def logdata(
    msg: str,
    properties: StringOptions = None,
    show_info: bool = False,
) -> DataFunc:
    info: List[str] = [msg]

    def decorator(func: DataFunc) -> DataFunc:
        def wrapper(*args, **kwargs) -> DataFrame:
            data: DataFrame = func(*args, **kwargs)
            if properties:
                info.extend(
                    [f"{prop.title()}: {getattr(data, prop)}" for prop in properties]
                )
            if show_info:
                info.append("\n{}".format(data.info(verbose=True, null_counts=True)))
            datalogger.debug(" ".join(info))
            return data

        return wrapper

    return decorator


def logresult(result: WorkflowMetadata) -> None:
    workflow_logfile = Path(data_path, "workflow_log.json")
    with open(workflow_logfile, "r") as f:
        logs = json.load(f)
    logs.append(asdict(result))
    with open(workflow_logfile, "w") as f:
        json.dump(logs, f, indent=4)


def logflow(name: str, description: Optional[str] = None) -> FlowFunc:
    def decorator(func: FlowFunc) -> FlowFunc:
        def wrapper(*args, **kwargs) -> WorkflowResult:
            pd.options.mode.chained_assignment = None
            flowlogger.info("Running {}:: {}".format(name, description))
            debug = kwargs.get("debug", False)
            if debug:
                datalogger.setLevel(logging.DEBUG)
                flowlogger.setLevel(logging.DEBUG)
            data, metadata = func(*args, **kwargs)
            metadata.name = name
            logresult(metadata)
            return data, metadata

        return wrapper

    return decorator


H_0 = 67.4 * 1000 * 100  # cm s^-1 Mpc^-1
c = 29979245800  # cm s^-1
Mpc_to_cm = 3.085677581491367e24
Gyr_to_s = 3.15576e16
Hubble_time = 1 / (H_0 / Mpc_to_cm * Gyr_to_s)  # Gyr
Hubble_distance = c / H_0  # Mpc
Omega_b = 0.0224 / ((H_0) / 1000 / 100 / 100) ** 2
Omega_m = 0.315
Omega_Lambda = 0.685
f_IGM = 0.83
chi = 7 / 8
G = 6.6743e-8  # cm^3 g^-1 s^-2
m_p = 1.67262192e-24  # g
dm_factor = (
    3 * c * H_0 / (Mpc_to_cm) ** 2 * 1e6 * Omega_b * f_IGM * chi / (8 * np.pi * G * m_p)
)
DM_host_lab = 70.0  # pc cm^-3
DM_halo = 30.0


def comoving_distance_at_z(z):  # Mpc
    zp1 = 1.0 + z
    h0_up = np.sqrt(1 + Omega_m / Omega_Lambda) * scipy.special.hyp2f1(
        1 / 3, 1 / 2, 4 / 3, -Omega_m / Omega_Lambda
    )
    hz_up = (
        zp1
        * np.sqrt(1 + Omega_m * zp1**3 / Omega_Lambda)
        * scipy.special.hyp2f1(1 / 3, 1 / 2, 4 / 3, -Omega_m * zp1**3 / Omega_Lambda)
    )
    h0_down = np.sqrt(Omega_Lambda + Omega_m)
    hz_down = np.sqrt(Omega_Lambda + Omega_m * zp1**3)
    return Hubble_distance * (hz_up / hz_down - h0_up / h0_down)


def luminosity_distance_at_z(z):  # Mpc
    return (1.0 + z) * comoving_distance_at_z(z)
