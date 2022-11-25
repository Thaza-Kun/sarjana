import logging
from typing import List

import pandas as pd
from hdbscan import HDBSCAN

from sarjana.utils.logger import logdata


@logdata("HDBSCAN Complete.", properties=["shape"], show_info=True)
def run_hdbscan(
    data: pd.DataFrame,
    columns: List[str],
    min_cluster_size: int = 19,
    threshold: float = 0.1,
) -> pd.DataFrame:
    """Run HDBSCAN of data on specified columns

    Args:
        data (pd.DataFrame): Data to be clustered
        columns (List[str]): List of columns to be considered in clustering
        min_cluster_size (int, optional): Minimum cluster size. Defaults to 19.
        threshold (float, optional): Fraction of repeaters needed to be considered repeater cluster. Defaults to 0.1.

    Returns:
        pd.DataFrame: Dataframe with cluster column
    """
    logging.info(f"Running HDBSCAN with minimum cluster size {min_cluster_size}")
    model_ = HDBSCAN(min_cluster_size)
    data.loc[:, ["cluster"]] = model_.fit_predict(data[columns])
    logging.info(f"HDBSCAN detected {data['cluster'].value_counts().shape[0]} clusters")
    data = data.sort_values(by="cluster", ascending=True)
    data.loc[:, ["cluster"]] = data["cluster"].astype(str)
    cluster = data.groupby("cluster").aggregate("mean", numeric_only=True)
    cluster["group"] = [
        "repeater_cluster" if row > threshold else "other_cluster"
        for row in cluster["repeater"]
    ]
    cluster.reset_index()
    data = data.merge(cluster["group"], on="cluster")
    return data
