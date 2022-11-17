import logging
from pathlib import Path
from typing import List, Optional

import seaborn as sns
import pandas as pd

from paths import graph_path, collected_data_path
from source.papers import (
    load_chen2021,
    load_hashimoto2022,
)
from learning import (
    reduce_dimension_to_2,
    run_hdbscan,
    separate_repeater_and_non_repeater,
    train_test_split_subset,
)


def get_chen2021_repeater_candidates(
    filename: Optional[str] = "chen2021_candidates",
) -> pd.DataFrame:
    data: pd.DataFrame = load_chen2021()[["tns_name", "classification", "group"]]
    data["cluster"] = data["group"].apply(lambda x: int(x.split("_")[-1]))
    data["candidate"] = [
        True if "repeater" in item else False for item in data["group"]
    ]
    data = data[data["candidate"] == True][["tns_name", "cluster"]]
    data.to_csv(Path(collected_data_path, f"{filename}.csv"), index=False)
    return data


def compare_with_chen2021(
    data: pd.DataFrame, filename_prefix: str, filename_postfix: str = ""
) -> pd.DataFrame:
    chen_2021: pd.DataFrame = load_chen2021(
        rename_columns={"embedding_y": "y", "embedding_x": "x"}
    )
    chen_2021["group"] = chen_2021["group"].apply(lambda x: x[:-2])
    data["source"] = "this work"
    data = pd.concat([data, chen_2021])
    logging.debug(
        f"Data concatenated with chen et. al. (2021). Shape: {data.shape}. Columns: {data.columns}"
    )
    sns.relplot(data=data, x="x", y="y", hue="group", col="source").savefig(
        Path(
            graph_path, f"{filename_prefix}_UMAP_HDBSCAN_compare_{filename_postfix}.png"
        )
    )
    return data


def replicate_chen2021_UMAP_HDBSCAN(
    min_cluster_size: int = 19,
    seed: int = 42,
    filename_prefix: str = "replicate_chen2021",
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logging.info(
        'Replicating Chen et. al. (2021) "Uncloaking hidden repeating fast radio bursts with unsupervised machine learning" doi:10.1093/mnras/stab2994'
    )
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
    data = load_hashimoto2022(
        source="Hashimoto2022_chimefrbcat1.csv",
        interval=("2018-07-25", "2019-07-01"),
    )
    logging.debug(f"Data loaded. Shape: {data.shape}. Columns: {data.columns}")
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
    data = pd.concat([data, test])
    logging.debug(
        f"Data dimension reduced. Shape: {data.shape}. Columns: {data.columns}"
    )
    postfix: str = f"(mcs={min_cluster_size}_seed={seed})"
    sns.relplot(data=data, x="x", y="y", hue="label").savefig(
        Path(graph_path, f"{filename_prefix}_UMAP_{postfix}.png")
    )

    data = run_hdbscan(
        data=data, columns=["x", "y"], min_cluster_size=min_cluster_size, threshold=0.1
    )
    logging.debug(f"HDBSCAN Complete. Shape: {data.shape}. Columns: {data.columns}")
    sns.relplot(data=data, x="x", y="y", hue="group").savefig(
        Path(
            graph_path,
            f"{filename_prefix}_UMAP_HDBSCAN_{postfix}.png",
        )
    )
    return compare_with_chen2021(
        data=data, filename_prefix=filename_prefix, filename_postfix=postfix
    )


def UMAP_HDBSCAN_no_model_dependent_params(
    min_cluster_size: int = 19,
    seed: int = 42,
    filename_prefix: str = "UMAP_HDBSCAN_no_model_dependent_params",
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO)
    logging.info(
        """Replicating Chen et. al. (2021) "Uncloaking hidden repeating fast radio bursts with unsupervised machine learning" doi:10.1093/mnras/stab2994. With no model dependent parameters.
        """
    )
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
    logging.debug(f"Data loaded. Shape: {data.shape}. Columns: {data.columns}")
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
    data = pd.concat([data, test])
    logging.debug(
        f"Data dimension reduced. Shape: {data.shape}. Columns: {data.columns}"
    )
    sns.relplot(data=data, x="x", y="y", hue="label").savefig(
        Path(graph_path, f"{filename_prefix}_UMAP_{postfix}.png")
    )

    data = run_hdbscan(
        data=data, columns=["x", "y"], min_cluster_size=min_cluster_size, threshold=0.1
    )
    logging.debug(f"HDBSCAN Complete. Shape: {data.shape}. Columns: {data.columns}")
    sns.relplot(data=data, x="x", y="y", hue="group").savefig(
        Path(
            graph_path,
            f"{filename_prefix}_UMAP_HDBSCAN_{postfix}.png",
        )
    )
    compare_with_chen2021(
        data=data, filename_prefix=filename_prefix, filename_postfix=postfix
    )
    return data
