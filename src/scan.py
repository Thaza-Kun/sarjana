from enum import Enum
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import seaborn as sns

from astropy.time import Time

import umap
from hdbscan import HDBSCAN
from sklearn.model_selection import train_test_split


from paths import external_data_path


def load_data(source: str, interval: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
    path = Path(external_data_path, source)
    logging.info(f"Loading data from {path}")
    catalog: pd.DataFrame = pd.read_csv(path)

    catalog["label"]: pd.Series = [
        "non-repeater" if row == "-9999" else "repeater"
        for row in catalog["repeater_name"]
    ]
    catalog["repeater"] = [
        False if name == "non-repeater" else True for name in catalog["label"]
    ]
    if interval:
        start: float = Time(interval[0]).mjd
        end: float = Time(interval[1]).mjd

        within_time: pd.Series = (start <= catalog["mjd_400"]) & (
            catalog["mjd_400"] <= end
        )
        return catalog[within_time]
    return catalog


def load_chen2021(rename_columns: Optional[dict] = None) -> pd.DataFrame:
    chen2021 = pd.read_csv(Path(external_data_path, "chen2021_classification.csv"))
    if rename_columns:
        chen2021 = chen2021.rename(columns=rename_columns)
    chen2021["source"] = "chen et al 2021"
    return chen2021


def separate_repeater_and_non_repeater(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    repeating: pd.DataFrame = data[(data["label"] == "repeater")]
    non_repeating: pd.DataFrame = data[(data["label"] == "non-repeater")]
    logging.info(f"Total repeaters\t: {len(repeating)}")
    logging.info(f"Total non-repeaters\t: {len(non_repeating)}")
    logging.info(f"Total sub-bursts\t: {len(repeating) + len(non_repeating)}")
    return repeating, non_repeating


def train_test_split_subset(
    subsample: pd.DataFrame,
    sidesample: Optional[pd.DataFrame] = None,
    test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"Spliting train ({100*(1-test_size)}%) and test ({test_size*100}%)")
    train, test = train_test_split(subsample, test_size=test_size)
    selected = pd.concat([train, sidesample])
    return selected, test


def reduce_dimension(
    sample: pd.DataFrame,
    params: List[str],
    drop_na: Optional[List[str]] = None,
    test: Optional[pd.DataFrame] = None,
    technique: str = "UMAP",
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    logging.info(f"Reducing dimension using '{technique.lower()}'")
    if technique.lower() == "umap":
        n_neighbors = kwargs.pop("n_neighbors", 8)
        n_components = kwargs.pop("n_components", 2)
        min_dist = kwargs.pop("min_dist", 0.1)
        model: umap.UMAP = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=min_dist,
            random_state=seed,
        )
    else:
        raise NotImplementedError
    sample = sample.dropna(subset=drop_na)
    map = model.fit(sample[params])
    sample["x"] = map.embedding_[:, 0]
    sample["y"] = map.embedding_[:, 1]
    data = sample

    if test is not None:
        test_map = map.transform(test[params])
        test["x"] = test_map[:, 0]
        test["y"] = test_map[:, 1]
        sample["label"] = [
            "repeater (train)" if name == "repeater" else name
            for name in sample["label"]
        ]
        test["label"] = "repeater (test)"
        data: pd.DataFrame = pd.concat([sample, test]).sort_values(by=["label"])
    return data


def run_hdbscan(
    data: pd.DataFrame,
    params: List[str],
    min_cluster_size: int = 19,
    threshold: float = 0.1,
) -> pd.DataFrame:
    logging.info(f"Running HDBSCAN with minimum cluster size {min_cluster_size}")
    model_ = HDBSCAN(min_cluster_size)
    data["cluster"] = model_.fit_predict(data[params])
    logging.info(f"HDBSCAN detected {data['cluster'].value_counts().shape[0]} clusters")
    data = data.sort_values(by="cluster", ascending=True)
    data["cluster"] = data["cluster"].astype(str)
    cluster = data.groupby("cluster").aggregate("mean", numeric_only=True)
    cluster["group"] = [
        "repeater_cluster" if row > threshold else "other_cluster"
        for row in cluster["repeater"]
    ]
    cluster.reset_index()
    data = data.merge(cluster["group"], on="cluster")
    return data
