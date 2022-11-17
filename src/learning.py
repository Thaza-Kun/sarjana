import logging
from typing import List, Optional, Tuple

import pandas as pd

import umap
from hdbscan import HDBSCAN
from sklearn.model_selection import train_test_split


def separate_repeater_and_non_repeater(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Separates repeater and non-repeater samples.

    Args:
        data (pd.DataFrame): Dataframe containing samples of repeating and non-repeating FRBs

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: repeating, non_repeating
    """
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
    """Split testing and training data set and recombine with sidesample if needed.

    Args:
        subsample (pd.DataFrame): The dataset that needs to be split.
        sidesample (Optional[pd.DataFrame], optional): The dataset that is part of the training dataset but not split. Defaults to None.
        test_size (float, optional): Size of test. Defaults to 0.1.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: training, testing
    """
    logging.info(f"Spliting train ({100*(1-test_size)}%) and test ({test_size*100}%)")
    train, test = train_test_split(subsample, test_size=test_size)
    if sidesample is not None:
        selected = pd.concat([train, sidesample])
        return selected, test
    return train, test


def reduce_dimension_to_2(
    sample: pd.DataFrame,
    params: List[str],
    drop_na: Optional[List[str]] = None,
    test: Optional[pd.DataFrame] = None,
    technique: str = "UMAP",
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Reduce dimensions of data to 2 using specified technique

    Args:
        sample (pd.DataFrame): Dataset to be reduced
        params (List[str]): List of columns to be reduced
        drop_na (Optional[List[str]], optional): List of columns to drop na values. Defaults to None.
        test (Optional[pd.DataFrame], optional): Test dataset. Defaults to None.
        technique (str, optional): Algorithms for dimensional reduction (Currently only supports UMAP). Defaults to "UMAP".
        seed (int, optional): Seed for reproducibility. Defaults to 42.

    Raises:
        NotImplementedError: Raised when a not yet implemented technique is chosen

    Returns:
        pd.DataFrame: Dataframe with labels and embedding coordinates
    """
    logging.info(f"Reducing dimension using '{technique.lower()}'")
    final_dimension = 2
    if technique.lower() == "umap":
        n_neighbors = kwargs.pop("n_neighbors", 8)
        min_dist = kwargs.pop("min_dist", 0.1)
        model: umap.UMAP = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=final_dimension,
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
    data["cluster"] = model_.fit_predict(data[columns])
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
