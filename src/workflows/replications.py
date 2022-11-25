import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import seaborn as sns
from sklearn.metrics import fbeta_score

from src.learning import separate_repeater_and_non_repeater, train_test_split_subset, reduce_dimension_to_2, run_hdbscan
from src.paths import collected_data_path, graph_path
from src.source.papers import load_chen2021, load_hashimoto2022
from src.utils import logflow, WorkflowResult, WorkflowMetadata

def get_chen2021_repeater_candidates(
    filename: Optional[str] = "chen2021_candidates",
) -> pd.DataFrame:
    data: pd.DataFrame = load_chen2021()[["tns_name", "classification", "group"]]
    data["cluster"] = data["group"].apply(lambda x: int(x.split("_")[-1]))
    data["candidate"] = [
        True if "repeater" in item else False for item in data["group"]
    ]
    data = data[data["candidate"] == True][["tns_name", "cluster"]]
    data.to_csv(Path(collected_data_path, f"{filename}.csv"), index=False)
    return data


def compare_with_chen2021(
    data: pd.DataFrame, filename_prefix: str, filename_postfix: str = ""
) -> Tuple[pd.DataFrame, float]:
    chen_2021: pd.DataFrame = load_chen2021(
        rename_columns={"embedding_y": "y", "embedding_x": "x"}
    )
    chen_2021["group"] = chen_2021["group"].apply(lambda x: x[:-2])
    data["source"] = "this work"
    score_this = data.merge(chen_2021, on='tns_name')
    score = fbeta_score(score_this['group_y'], score_this['group_x'], beta=2, pos_label='repeater_cluster')
    data = pd.concat([data, chen_2021])
    logging.debug(
        f"Data concatenated with chen et. al. (2021). Shape: {data.shape}. Columns: {data.columns}"
    )
    sns.relplot(data=data, x="x", y="y", hue="group", col="source").savefig(
        Path(
            graph_path, f"{filename_prefix}_UMAP_HDBSCAN_compare_{filename_postfix}.png"
        )
    )
    return data, score


@logflow(
    name='chen2021_UMAP-HDBSCAN',
    description='Replicating Chen et. al. (2021) "Uncloaking hidden repeating fast radio bursts with unsupervised machine learning" doi:10.1093/mnras/stab2994'
)
def bo_han_chen_2021(
    min_cluster_size: int = 19,
    seed: int = 42,
    filename_prefix: str = "replicate_chen2021",
    **kwargs
) -> WorkflowResult:
    params: List[str] = [
        # Observational
        "bc_width",
        "width_fitb",
        "flux",
        "fluence",
        "scat_time",
        "sp_idx",
        "sp_run",
        "high_freq",
        "low_freq",
        "peak_freq",
        # Model dependent
        "z",
        "logE_rest_400",
        "logsubw_int_rest",
    ]

    dropna_subset = ["flux", "fluence", "logE_rest_400"]
    data = load_hashimoto2022(
        source="Hashimoto2022_chimefrbcat1.csv",
        interval=("2018-07-25", "2019-07-01"),
    )
    repeating, non_repeating = separate_repeater_and_non_repeater(data=data)
    sample, test = train_test_split_subset(
        subsample=repeating, sidesample=non_repeating
    )
    data = reduce_dimension_to_2(
        sample=sample,
        params=params,
        drop_na=dropna_subset,
        test=test,
        technique="UMAP",
        seed=seed,
    )
    postfix: str = f"(mcs={min_cluster_size}_seed={seed})"
    sns.relplot(data=data, x="x", y="y", hue="label").savefig(
        Path(graph_path, f"{filename_prefix}_UMAP_{postfix}.png")
    )

    data = run_hdbscan(
        data=data, columns=["x", "y"], min_cluster_size=min_cluster_size, threshold=0.1
    )
    sns.relplot(data=data, x="x", y="y", hue="group").savefig(
        Path(
            graph_path,
            f"{filename_prefix}_UMAP_HDBSCAN_{postfix}.png",
        )
    )
    _, score = compare_with_chen2021(
        data=data, filename_prefix=filename_prefix, filename_postfix=postfix
    )
    result = WorkflowMetadata(
        parameters={
            'columns': params,
            'seed': seed,
            'min_cluster_size': min_cluster_size
        },
        timestamp=datetime.now().strftime('%Y-%m-%d:%H:%M'),
        score={
            'value': score,
            'metric': 'f2_score'
        }
    )
    return data, result
