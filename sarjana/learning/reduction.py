import logging
from typing import List, Optional

import pandas as pd

from sarjana.utils.logger import logdata


@logdata("Dimension reduced.", properties=["shape"], show_info=True)
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
        drop_na (Optional[List[str]], optional): List of columns to drop na values. If None, will use the whole params list. Defaults to None.
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
        import umap

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
    drop_na = params if drop_na is None else drop_na
    sample = sample.dropna(subset=drop_na)
    test = test.dropna(subset=drop_na)
    model_ = model.fit(sample[params])
    sample.loc[:, ["x"]] = model_.embedding_[:, 0]
    sample.loc[:, ["y"]] = model_.embedding_[:, 1]
    data = sample

    if test is not None:
        test_map = model_.transform(test[params])
        test.loc[:, ["x"]] = test_map[:, 0]
        test.loc[:, ["y"]] = test_map[:, 1]
        sample.loc[:, ["label"]] = [
            "repeater (train)" if name == "repeater" else name
            for name in sample["label"]
        ]
        test.loc[:, ["label"]] = "repeater (test)"
        data: pd.DataFrame = pd.concat([sample, test]).sort_values(by=["label"])
    return data
