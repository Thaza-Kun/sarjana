from pathlib import Path
import pandas as pd
from sarjana.optimize import fit_time_series

from sarjana.signal import is_multipeak, scattered_gaussian_signal


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


def generate_scattered_gaussian_model(data: pd.DataFrame) -> pd.DataFrame:
    data["multipeak"] = [is_multipeak(x.copy()) for x in data["ts"].values]
    single_peaks = data[data["multipeak"] == False]

    def try_fit(func, data) -> dict:
        try:
            return fit_time_series(
                func,
                data["plot_time"],
                data["ts"],
                params={
                    "amplitude": data["ts"].max(),
                    "peak_time": data["plot_time"][data["ts"].argmax()],
                },
            )
        except RuntimeError:
            return fit_time_series(
                func,
                data["plot_time"],
                data["model_ts"],
                params={
                    "amplitude": data["model_ts"].max(),
                    "peak_time": data["plot_time"][data["model_ts"].argmax()],
                },
            )

    single_peaks["params"] = single_peaks.apply(
        lambda x: try_fit(scattered_gaussian_signal, x), axis="columns"
    )
    single_peaks["model"] = "scattered_gaussian"
    return single_peaks
