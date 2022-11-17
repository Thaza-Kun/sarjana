import os
from pathlib import Path
from typing import List

import pandas as pd

from src.paths import collected_data_path


def load_repeater_candidates() -> pd.DataFrame:
    """Returns a list of repeater candidates scraped from various sources.

    Current sources:
        - Chen et. al. 2021 [doi:10.1093/mnras/stab2994](doi:10.1093/mnras/stab2994)
        - Luo Jia-Wei et. al. 2022 [doi:10.1093/mnras/stac3206](doi:10.1093/mnras/stac3206)

    Returns:
        pd.DataFrame: A dataframe of candidates
    """
    chen2021_data: pd.DataFrame = pd.read_csv(
        Path(collected_data_path, "chen2021_candidates.csv")
    )
    chen2021_data["cluster"] = chen2021_data["cluster"].astype(int)
    luojiawei2022_data: pd.DataFrame = pd.read_csv(
        Path(collected_data_path, "luojiawei2022_candidates.csv")
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
