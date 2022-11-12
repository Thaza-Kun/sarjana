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

data_path: Path = Path(".", "data")
external_data_path: Path = Path(data_path, "raw", "external")
graph_path: Path = Path(data_path, "graphs")

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

identifiers: List[str] = ["tns_name", "repeater_name"]

dropna_subset = ["flux", "fluence", "logE_rest_400"]


def load_data(
    source: str, columns: List[str], interval: Tuple[str, str]
) -> pd.DataFrame:
    path = Path(external_data_path, source)
    logging.info(f"Loading data from {path}")
    catalog: pd.DataFrame = pd.read_csv(path)

    start: float = Time(interval[0]).mjd
    end: float = Time(interval[1]).mjd

    within_time: pd.Series = (start <= catalog["mjd_400"]) & (catalog["mjd_400"] <= end)
    return catalog[columns][within_time]


def separate_repeater_from_non_repeater(
    catalog: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    repeating: pd.DataFrame = catalog[(catalog["repeater_name"] != "-9999")]
    non_repeating: pd.DataFrame = catalog[(catalog["repeater_name"] == "-9999")]
    logging.info(
        f"Total repeaters\t\t: {len(repeating)}\n",
        f"Total non-repeaters\t: {len(non_repeating)}\n",
        f"Total sub-bursts\t: {len(repeating) + len(non_repeating)}\n",
    )
    return repeating, non_repeating


def test_train_split_subset(
    subsample: pd.DataFrame,
    sidesample: Optional[pd.DataFrame] = None,
    test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info(f"Spliting train ({100*(1-test_size)}%) and test ({test_size*100}%)")
    train, test = train_test_split(subsample, test_size=test_size)
    selected = pd.concat([train, sidesample]).dropna(subset=dropna_subset)
    return selected, test


class DimRed(Enum):
    UMAP = "UMAP"


def reduce_dimension(
    sample: pd.DataFrame,
    test: pd.DataFrame,
    params: List[str],
    identifiers: List[str],
    technique: DimRed = DimRed.UMAP,
    plot_to: Optional[str] = None,
) -> pd.DataFrame:
    logging.info(f"Reducing dimension using {technique.value}")
    if technique == DimRed.UMAP:
        model: umap.UMAP = umap.UMAP(n_neighbors=8, n_components=2, min_dist=0.1)
    else:
        raise NotImplementedError
    map = model.fit(sample[params])
    test_map = map.transform(test[params])
    sample["x"] = map.embedding_[:, 0]
    sample["y"] = map.embedding_[:, 1]
    test["x"] = test_map[:, 0]
    test["y"] = test_map[:, 1]
    selected["label"] = [
        "non-repeater" if name == "-9999" else "repeater (train)"
        for name in selected["repeater_name"]
    ]
    test["label"] = "repeater (test)"
    data: pd.DataFrame = pd.concat([selected, test]).sort_values(by=["label"])
    if plot_to:
        sns.set_style("dark")
        sns.set_context("paper")
        sns.relplot(
            data=data,
            kind="scatter",
            x="x",
            y="y",
            hue="label",
            hue_order=["non-repeater", "repeater (train)", "repeater (test)"],
        ).savefig(Path(graph_path, plot_to))
    return data  # [[*params, *identifiers, "x", "y", "label"]]


def run_hdbscan(
    data: pd.DataFrame,
    params: List[str],
    min_cluster_size: int = 19,
    compare_with: Optional[str] = None,
    threshold: float = 0.1,
):
    logging.info(f"Running HDBSCAN with minimum cluster size {min_cluster_size}")
    model_ = HDBSCAN(min_cluster_size)
    data["cluster"] = model_.fit_predict(data[params])
    data = data.sort_values(by="cluster", ascending=True)
    data["cluster"] = data["cluster"].astype(str)
    data["repeater"] = [
        False if name == "non-repeater" else True for name in data["label"]
    ]
    print(f"Before agg {data.shape=}")
    cluster = data.groupby("cluster").aggregate("mean", numeric_only=True)
    cluster["group"] = [
        "repeater_cluster" if row > threshold else "other_cluster"
        for row in cluster["repeater"]
    ]
    cluster.reset_index()
    data = data.merge(cluster["group"], on="cluster")
    print(cluster.columns)
    print(f"After agg {cluster.shape=}")
    sns.relplot(data=data, kind="scatter", x="x", y="y", hue="cluster").savefig(
        Path(graph_path, "hdbscan.png")
    )
    sns.relplot(data=data, kind="scatter", x="x", y="y", hue="group").savefig(
        Path(graph_path, "hdbscan_group.png")
    )
    print(f"{data.shape=}")

    if compare_with:
        chen2021 = pd.read_csv(Path(external_data_path, compare_with)).rename(
            columns={
                "embedding_x": "x",
                "embedding_y": "y",
                "label": "group",
            }
        )
        chen2021["source"] = "chen et al 2021"
        this = data.rename(
            columns={
                "label": "repeater",
            }
        )
        this["source"] = "calculated"
        chen2021["group"] = chen2021["group"].apply(lambda x: x[:-2])
        graph_params: List[str] = ["source", "group", "x", "y"]
        data: pd.DataFrame = pd.concat([this[graph_params], chen2021[graph_params]])
        print(f"{chen2021.shape=}")
        print(f"{this.shape=}")
        print(f"{data.shape=}")
        sns.relplot(
            data=data,
            kind="scatter",
            x="x",
            y="y",
            hue="group",
            col="source",
        ).savefig(Path(graph_path, "hdbscan_compare.png"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    catalog = load_data(
        source="Hashimoto2022_chimefrbcat1.csv",
        columns=[*params, *identifiers],
        interval=("2018-07-25", "2019-07-01"),
    )
    print(f"{catalog.shape=}")
    repeating, non_repeating = separate_repeater_from_non_repeater(catalog=catalog)
    print(f"{repeating.shape=}")
    print(f"{non_repeating.shape=}")
    selected, test = test_train_split_subset(
        subsample=repeating, sidesample=non_repeating
    )
    print(f"{selected.shape=}")
    print(f"{test.shape=}")
    selected = reduce_dimension(
        sample=selected,
        test=test,
        params=params,
        identifiers=identifiers,
        plot_to="umap.png",
    )
    print(f"{selected.shape=}")
    run_hdbscan(
        data=selected, params=["x", "y"], compare_with="chen2021_classification.csv"
    )
