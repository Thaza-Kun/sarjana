import pandas as pd


def merge_embedding_into_profile(
    profiles: pd.DataFrame, embedding: pd.DataFrame
) -> pd.DataFrame:
    """Injects the embedding data from machine learning into a dataframe of flux profiles.

    Args:
        profiles (pd.DataFrame): A `.parquet` file containing a list of flux profiles.
        embedding (pd.DataFrame): A `.csv` file containing groups, classification, and embedding coordinates.

    Returns:
        pd.DataFrame: A merge of the two.
    """
    return profiles.merge(embedding, left_on="eventname", right_on="tns_name")
