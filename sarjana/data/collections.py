import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from astropy.time import Time

from sarjana.utils.paths import collected_datapath, external_datapath
from sarjana.utils.logger import logdata


def load_repeater_candidates() -> pd.DataFrame:
    """Returns a list of repeater candidates scraped from various sources.

    Current sources:
        - Chen et. al. 2021 [doi:10.1093/mnras/stab2994](doi:10.1093/mnras/stab2994)
        - Luo Jia-Wei et. al. 2022 [doi:10.1093/mnras/stac3206](doi:10.1093/mnras/stac3206)

    Returns:
        pd.DataFrame: A dataframe of candidates
    """
    chen2021_data: pd.DataFrame = pd.read_csv(
        Path(collected_datapath, "chen2021_candidates.csv")
    )
    chen2021_data["cluster"] = chen2021_data["cluster"].astype(int)
    luojiawei2022_data: pd.DataFrame = pd.read_csv(
        Path(collected_datapath, "luojiawei2022_candidates.csv")
    )
    return chen2021_data.merge(luojiawei2022_data, how="outer", on="tns_name").rename(
        columns={"cluster": "chen2021_cluster"}
    )


def load_strong_repeater_candidates(how: str = "luojiawei2022") -> pd.DataFrame:
    """Returns a list of repeater candidate with a strong likelihood

    Args:
        how (str, optional): Method to pick candidate. Defaults to "luojiawei2022".
            `"luojiawei2022"`: strong candidate list that is outlined in Luo Jia-Wei et. al. 2022 [doi:10.1093/mnras/stac3206](doi:10.1093/mnras/stac3206)
            `"zhuge2022"`: strong candidate list that is outlined in Zhu-Ge Jia-Ming et. al. 2022 [doi:10.48550/arXiv.2210.02471](doi:10.48550/arXiv.2210.02471)
            `"filter"`: a list of candidates that have no NaN in the columns defined

    Returns:
        pd.DataFrame: A dataframe of candidates
    """
    data: pd.DataFrame = load_repeater_candidates()
    if how == "filter":
        return data.dropna(
            subset=[
                "chen2021_cluster",
                "supervised_multimodel",
                "leave_one_out",
                "unsupervised_multimodel",
            ]
        )
    if how == "luojiawei2022":
        candidates: List[str] = [
            "FRB20190423B",
            "FRB20181017B",
            "FRB20190329A",
            "FRB20190423B",
            "FRB20190206A",
            "FRB20190527A",
            "FRB20190412B",
            "FRB20181231B",
            "FRB20190429B",
            "FRB20190109B",
        ]
    elif how == "zhuge2022":
        candidates: List[str] = [
            "FRB20181017B",
            "FRB20181030E",
            "FRB20181221A",
            "FRB20181229B",
            "FRB20181231B",
            "FRB20190109B",
            "FRB20190112A",
            "FRB20190129A",
            "FRB20190206A",
            "FRB20190218B",
            "FRB20190329A",
            "FRB20190409B",
            "FRB20190410A",
            "FRB20190412B",
            "FRB20190422A",
            "FRB20190423B",
            "FRB20190423B",
            "FRB20190429B",
            "FRB20190527A",
            "FRB20190609A",
        ]
    return data[data.tns_name.isin(candidates)]


def load_FRBSTATS_repeaters(
    source: str = "FRBSTATS2022-11-23_repeaters.csv",
) -> pd.DataFrame:
    return pd.read_csv(Path(external_datapath, source))


@logdata("Loading FRBSTATS data.", show_info=True)
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
            - snr: signal-to-noise ratio
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
    data = pd.read_csv(Path(external_datapath, source)).replace("-", None)
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


@logdata("Loading Hashimoto 2022.", properties=["shape", "columns"])
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
    path = Path(external_datapath, source)
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


@logdata("Loading Chen 2021.", properties=["shape", "columns"])
def load_chen2021(rename_columns: Optional[dict] = None) -> pd.DataFrame:
    """Returns a dataframe of clusters and UMAP embedding coordinates from Chen et. al. 2021 [doi:10.1093/mnras/stab2994](doi:10.1093/mnras/stab2994)

    Args:
        rename_columns (Optional[dict], optional): Rename columns. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe of clusters and UMAP embedding coordinates
    """
    chen2021 = pd.read_csv(Path(external_datapath, "chen2021_classification.csv"))
    if rename_columns:
        chen2021 = chen2021.rename(columns=rename_columns)
    chen2021["source"] = "chen et al 2021"
    return chen2021
