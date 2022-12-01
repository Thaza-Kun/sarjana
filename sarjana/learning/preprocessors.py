import logging
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from sarjana.loggers.logger import logdata


@logdata("Separating repeaters and non-repeaters.", show_info=False)
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


@logdata("Splitting training and testing data.", show_info=False)
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
