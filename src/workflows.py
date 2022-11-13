import logging
from pathlib import Path
from typing import List

import seaborn as sns
import pandas as pd

from paths import graph_path
from scan import (
    load_chen2021,
    load_data,
    reduce_dimension,
    run_hdbscan,
    separate_repeater_and_non_repeater,
    train_test_split_subset,
)


def replicate_chen2021() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logging.info(
        'Replicating Chen et. al. (2021) "Uncloaking hidden repeating fast radio bursts with unsupervised machine learning" doi:10.1093/mnras/stab2994'
    )
    min_cluster_size: int = 19
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
        # Model dependent
        "z",
        "logE_rest_400",
        "logsubw_int_rest",
    ]

    dropna_subset = ["flux", "fluence", "logE_rest_400"]
    data = load_data(
        source="Hashimoto2022_chimefrbcat1.csv",
        interval=("2018-07-25", "2019-07-01"),
    )
    logging.debug(f"Data loaded. Shape: {data.shape}. Columns: {data.columns}")
    repeating, non_repeating = separate_repeater_and_non_repeater(data=data)
    sample, test = train_test_split_subset(
        subsample=repeating, sidesample=non_repeating
    )
    data = reduce_dimension(
        sample=sample, params=params, drop_na=dropna_subset, test=test, technique="UMAP"
    )
    data = pd.concat([data, test])
    logging.debug(
        f"Data dimension reduced. Shape: {data.shape}. Columns: {data.columns}"
    )
    sns.relplot(data=data, x="x", y="y", hue="label").savefig(
        Path(graph_path, "replicate_chen2021_UMAP.png")
    )

    data = run_hdbscan(
        data=data, params=params, min_cluster_size=min_cluster_size, threshold=0.1
    )
    logging.debug(f"HDBSCAN Complete. Shape: {data.shape}. Columns: {data.columns}")
    sns.relplot(data=data, x="x", y="y", hue="group").savefig(
        Path(graph_path, "replicate_chen2021_UMAP_HDBSCAN.png")
    )
    chen_2021: pd.DataFrame = load_chen2021(
        rename_columns={"embedding_y": "y", "embedding_x": "x", "label": "group"}
    )
    chen_2021["group"] = chen_2021["group"].apply(lambda x: x[:-2])
    data["source"] = "this work"
    data = pd.concat([data, chen_2021])
    logging.debug(
        f"Data concatenated with chen et. al. (2021). Shape: {data.shape}. Columns: {data.columns}"
    )
    sns.relplot(data=data, x="x", y="y", hue="group", col="source").savefig(
        Path(graph_path, "replicate_chen2021_UMAP_HDBSCAN_compare.png")
    )
