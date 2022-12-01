import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from sarjana.data import datalogger
from sarjana.preamble import ExecutionOptions
from sarjana.utils.paths import datapath
from sarjana.utils.types import (
    WorkflowMetadata,
    WorkflowResult,
    DataFunc,
    FlowFunc,
    StringOptions,
)
from sarjana.workflows import flowlogger

formatter = logging.Formatter("%(levelname)s::%(name)s::%(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

if ExecutionOptions.Mode == "debug":
    datalogger.setLevel(logging.DEBUG)
    flowlogger.setLevel(logging.DEBUG)


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
    workflow_logfile = Path(datapath, "workflow_log.json")
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
