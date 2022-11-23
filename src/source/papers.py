import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from astropy.time import Time

from paths import external_data_path


def load_FRBSTATS(source: str = "FRBSTATS2022-11-23_population.csv") -> pd.DataFrame:
    return pd.read_csv(Path(external_data_path, source))


def load_FRBSTATS_repeaters(
    source: str = "FRBSTATS2022-11-23_repeaters.csv",
) -> pd.DataFrame:
    return pd.read_csv(Path(external_data_path, source))


def load_hashimoto2022(
    source: str = "Hashimoto2022_chimefrbcat1.csv",
    interval: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """Returns a dataframe with calculated values from Hashimoto et. al. 2022 [doi:10.1093/mnras/stac065](doi:10.1093/mnras/stac065).

    Args:
        source (str, optional): CSV file of the data (if the name is different). Defaults to "Hashimoto2022_chimefrbcat1.csv".
        interval (Optional[Tuple[str, str]], optional): Select transients within the time frame. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with calculated values from Hashimoto 2022
    """
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
    """Returns a dataframe of clusters and UMAP embedding coordinates from Chen et. al. 2021 [doi:10.1093/mnras/stab2994](doi:10.1093/mnras/stab2994)

    Args:
        rename_columns (Optional[dict], optional): Rename columns. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe of clusters and UMAP embedding coordinates
    """
    chen2021 = pd.read_csv(Path(external_data_path, "chen2021_classification.csv"))
    if rename_columns:
        chen2021 = chen2021.rename(columns=rename_columns)
    chen2021["source"] = "chen et al 2021"
    return chen2021
