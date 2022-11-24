import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from astropy.time import Time

from paths import external_data_path
from utils import logdata


def load_FRBSTATS_repeaters(
    source: str = "FRBSTATS2022-11-23_repeaters.csv",
) -> pd.DataFrame:
    return pd.read_csv(Path(external_data_path, source))


@logdata(properties=["shape", "columns"])
def load_FRBSTATS(source: str = "FRBSTATS2022-11-23_population.csv") -> pd.DataFrame:
    """Returns a Dataframe from FRBSTATS CSV file and add relevant columns

    Args:
        source (str, optional): filename. Defaults to "FRBSTATS2022-11-23_population.csv".

    Returns:
        pd.DataFrame:   A DataFrame of basic values from FRBSTATS with the following columns:

        * given
            - frb: name of the frb
            - utc: time of observation, in utc
            - mjd: time of observation, in MJD
            - telescope: telescope used to observe
            - ra: right ascension
            - dec: declination
            - l: galactic longitude
            - b: galactic latitude
            - frequency: center frequency, in MHz
            - dm: dispersion measure, in pc cm^-3
            - flux: flux, in Jy
            - width: duration of burst, in ms
            - fluence: fluence, in Jy ms
            - snr: signal to noise ratio
            - reference: link to source
            - redshift: inferred redshift (method unclear)
            - redshift_measured: redshift of host if directly localized
            - ra_error: error of right ascension, in minute
            - dec_error: error of declination, in minute
            - dm_error: error of dispersion measure, in pc cm^-3
        * assigned
            - label: repeater or non-repeater
            - repeater: True or False
    """
    data = pd.read_csv(Path(external_data_path, source)).replace("-", None)
    # Labeling repeaters
    rptrs: pd.DataFrame = load_FRBSTATS_repeaters()
    data.loc[:, ["label"]] = [
        "repeater"
        if name in [*rptrs["name"].to_list(), *rptrs["samples"].to_list()]
        else "non-repeater"
        for name in data["frb"]
    ]
    data.loc[:, ["repeater"]] = [
        False if name == "non-repeater" else True for name in data["label"]
    ]
    return data


@logdata(properties=["shape", "columns"])
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


@logdata(properties=["shape", "columns"])
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
