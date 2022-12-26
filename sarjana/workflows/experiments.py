from datetime import datetime
from pathlib import Path
from time import time
from typing import List, Optional

import numpy as np
import pandas as pd
import seaborn as sns

from sarjana.utils.paths import graph_path
from sarjana.data.collections import (
    load_FRBSTATS,
    load_hashimoto2022,
)
from sarjana.learning import (
    reduce_dimension_to_2,
    run_hdbscan,
)
from sarjana.learning.preprocessors import (
    separate_repeater_and_non_repeater,
    train_test_split_subset,
)
from sarjana.transients.parameters import (
    brightness_temperature,
    burst_energy,
    rest_frequency,
    rest_time_width,
)
from sarjana.utils.types import WorkflowMetadata, WorkflowResult
from sarjana.loggers.logger import logflow
from sarjana.loggers.workflow import flowlogger

from sarjana.workflows.replications import compare_with_chen2021


@logflow(
    name="FRBSTATS_UMAP-HDBSCAN",
    description="Running UMAP-HDBSCAN algorithm on FRBSTATS data",
)
def UMAP_HDBSCAN_FRBSTATS(
    min_cluster_size: int = 19,
    seed: int = 42,
    params: Optional[List] = None,
    filename_prefix: str = "FRBSTATS",
    **kwargs,
) -> WorkflowResult:
    if params is None:
        params = ["frequency", "dm", "flux", "width", "fluence", "redshift"]
    postfix: str = f"(mcs={min_cluster_size}_seed={seed})"
    data: pd.DataFrame = (
        load_FRBSTATS()
        .dropna(axis=0, subset=params)
        .astype({key: float for key in params})
    )
    data.loc[:, ["energy"]] = np.log10(burst_energy(data))
    data.loc[:, ["rest_frequency"]] = rest_frequency(data)
    data.loc[:, ["brightness_temperature"]] = np.log10(brightness_temperature(data))
    data.loc[:, ["rest_time_width"]] = rest_time_width(data)
    params.extend(
        ["energy", "rest_frequency", "brightness_temperature", "rest_time_width"]
    )

    repeating, non_repeating = separate_repeater_and_non_repeater(data=data)
    sample, test = train_test_split_subset(
        subsample=repeating, sidesample=non_repeating, test_size=0.1
    )
    data = reduce_dimension_to_2(
        sample=sample,
        params=params,
        drop_na=params,
        test=test,
        technique="UMAP",
        seed=seed,
    )
    sns.relplot(data=data, x="x", y="y", hue="label").savefig(
        Path(graph_path, f"{filename_prefix}_UMAP_{postfix}.png")
    )

    data = run_hdbscan(
        data=data, columns=["x", "y"], min_cluster_size=min_cluster_size, threshold=0.1
    )
    sns.relplot(data=data, x="x", y="y", hue="cluster").savefig(
        Path(
            graph_path,
            f"{filename_prefix}_UMAP_HDBSCAN_cluster{postfix}.png",
        )
    )
    sns.relplot(data=data, x="x", y="y", hue="group").savefig(
        Path(
            graph_path,
            f"{filename_prefix}_UMAP_HDBSCAN_{postfix}.png",
        )
    )
    data.loc[:, ["group"]] = [
        True if group == "repeater_cluster" else False for group in data["group"]
    ]
    start = time()
    from sklearn.metrics import fbeta_score

    end = time()
    flowlogger.debug(
        "Importing sklearn.metrics.fbeta_score took {} seconds".format(end - start)
    )
    score = fbeta_score(data["repeater"], data["group"], beta=2)
    metadata = WorkflowMetadata(
        parameters={
            "columns": params,
            "seed": seed,
            "min_cluster_size": min_cluster_size,
        },
        timestamp=datetime.now().strftime("%Y-%m-%d:%H:%M"),
        score={"value": score, "metric": "f2_score"},
    )
    return data, metadata


@logflow(
    name="CHIME/FRB_UMAP-HDBSCAN_model_independent",
    description='Replicating Chen et. al. (2021) "Uncloaking hidden repeating fast radio bursts with unsupervised machine learning" doi:10.1093/mnras/stab2994. With no model dependent parameters.',
)
def bo_han_chen2021_model_independent(
    min_cluster_size: int = 19,
    seed: int = 42,
    filename_prefix: str = "chen2021_model_independent",
    **kwargs,
) -> WorkflowResult:
    params: List[str] = [
        # Observational
        "bc_width",
        "width_fitb",
        "flux",
        "fluence",
        "scat_time",
        "sp_idx",
        "sp_run",
        "high_freq",
        "low_freq",
        "peak_freq",
    ]

    postfix: str = f"(mcs={min_cluster_size}_seed={seed})"
    dropna_subset = ["flux", "fluence"]
    data = load_hashimoto2022(
        source="Hashimoto2022_chimefrbcat1.csv",
    )
    repeating, non_repeating = separate_repeater_and_non_repeater(data=data)
    sample, test = train_test_split_subset(
        subsample=repeating, sidesample=non_repeating
    )
    data = reduce_dimension_to_2(
        sample=sample,
        params=params,
        drop_na=dropna_subset,
        test=test,
        technique="UMAP",
        seed=seed,
    )
    sns.relplot(data=data, x="x", y="y", hue="label").savefig(
        Path(graph_path, f"{filename_prefix}_UMAP_{postfix}.png")
    )

    data = run_hdbscan(
        data=data, columns=["x", "y"], min_cluster_size=min_cluster_size, threshold=0.1
    )
    sns.relplot(data=data, x="x", y="y", hue="group").savefig(
        Path(
            graph_path,
            f"{filename_prefix}_UMAP_HDBSCAN_{postfix}.png",
        )
    )
    _, score = compare_with_chen2021(
        data=data, filename_prefix=filename_prefix, filename_postfix=postfix
    )
    metadata = WorkflowMetadata(
        parameters={
            "columns": params,
            "seed": seed,
            "min_cluster_size": min_cluster_size,
        },
        timestamp=datetime.now(),
        score={"value": score, "metric": "f2_score"},
    )
    return data, metadata
