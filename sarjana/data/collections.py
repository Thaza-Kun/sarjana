import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from astropy.time import Time

from sarjana.utils.paths import collected_datapath, external_datapath
from sarjana.loggers.logger import logdata

# TYPES OF DATA TO LOAD
# 1.    Catalogue data
# 2.    Embedding data
# 3.    Profile data
# 4.    Candidates data


# TODO Standardize catalog
def load_catalog(filename: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a FRB catalog from a filename.

    Args:
        filename (str): Name of data file (in csv).
        columns (Optional[List[str]]): Columns to load. Defaults to None.

    Returns:
        pd.DataFrame: Available columns
            - tns_name      / frb
            -               / utc
            - mjd_400       / mjd
            - mjd_400_err
            - mjd_inf       / mjd
            - mjd_inf_err
            -               / telescope
            - repeater_name
            - ra            / ra
            - ra_err        / ra_error
            - ra_notes
            - dec           / dec
            - dec_err       / dec_error
            - dec_notes
            - gl            / l
            - gb            / b
            - high_freq
            - low_freq
            - peak_freq     / frequency
            -               / reference
            - exp_up
            - exp_up_err
            - exp_up_notes
            - exp_low
            - exp_low_err
            - exp_low_notes
            - bonsai_snr
            - bonsai_dm
            - low_ft_68
            - up_ft_68
            - low_ft_95
            - up_ft_95
            - snr_fitb      / snr
            - dm_fitb       / dm
            - dm_fitb_err   / dm_error
            - dm_exc_ne2001
            - dm_exc_ymw16
            - bc_width
            - scat_time
            - scat_time_err
            - flux          / flux
            - flux_err
            - flux_notes
            - fluence       / fluence
            - fluence_err
            - fluence_notes
            - sub_num
            - width_fitb    / width
            - width_fitb_err
            - sp_idx
            - sp_idx_err
            - sp_run
            - sp_run_err
            - chi_sq
            - dof
            - flag_frac
            - excluded_flag
            - subw_upper_flag
            - scat_upper_flag
            - spec_z
            - spec_z_flag
            - E_obs
            - E_obs_error
            - subb_flag
            - subb_p_flag
            - common_p_flag
            - delta_nuo_FRB
            - z_DM          / redshift
            - z_DM_error_p
            - z_DM_error_m
            - E_obs_400
            - E_obs_400_error_p
            - E_obs_400_error_m
            - logsubw_int_rest
            - logsubw_int_rest_error_p
            - logsubw_int_rest_error_m
            - z             / redshift_measured
            - z_error_p
            - z_error_m
            - logE_rest_400
            - logE_rest_400_error_p
            - logE_rest_400_error_m
            - logrhoA
            - logrhoA_error_p
            - logrhoA_error_m
            - logrhoB
            - logrhoB_error_p
            - logrhoB_error_m
            - weight_DM
            - weight_DM_error_p
            - weight_DM_error_m
            - weight_scat
            - weight_scat_error_p
            - weight_scat_error_m
            - weight_w_int
            - weight_w_int_error_p
            - weight_w_int_error_m
            - weight_fluence
            - weight_fluence_error_p
            - weight_fluence_error_m
            - weight
            - weight_error_p
            - weight_error_m
            - weighted_logrhoA
            - weighted_logrhoA_error_p
            - weighted_logrhoA_error_m
            - weighted_logrhoB
            - weighted_logrhoB_error_p
            - weighted_logrhoB_error_m

    """
    return pd.read_csv(filename, usecols=columns)


def load_2d_embedding(
    filename: str, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Load data from 2d embedding result of a dimensional reduction algorithm.

    Args:
        filename (str): Name of data file.
        columns (Optional[List[str]]): Columns to load. Defaults to None.

    Returns:
        pd.DataFrame: Available columns
            - tns_name (Name of FRB)
            - classification (FRB classification of the model. 1 = repeater, 0 = repeater candidate, -1 = non-repeater)
            - {algo}_group (The classification of an clustering algorithm {algo})
            - {algo}_x (The x coordinate of an embedding space by the dimensional reduction algorithm {algo})
            - {algo}_y (The y coordinate of an embedding space by the dimensional reduction algorithm {algo})
    """
    return pd.read_csv(filename, usecols=columns)


def load_profiles(
    filename: str,
    columns: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load the time series data of the selected source in a dataframe.

    Args:
        filename (str): Name of data file (in `.parquet`)
        columns (Optional[List[str]], optional): Columns to load. Defaults to None.
        sources (Optional[List[str]], optional): Source to load. Defaults to None.

    Returns:
        pd.DataFrame: Available columns
    """
    return pd.read_parquet(filename, columns=columns)


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
