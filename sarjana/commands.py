from typing import Optional
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sarjana.tui import LinearProgress, DownloadProgress
from sarjana.collections import merge_embedding_into_profile
from sarjana.handlers import H5Waterfall, ParquetWaterfall

from sarjana.plotting import plot_flux_profile
from sarjana.download import manage_download_waterfall_data_task, compress_to_parquet


def plot_many_flux_profile_by_clustering_groups(
    profile: str, embedding: str, savefile: str, size: int, *, find_peaks: bool = False
) -> None:
    """
    TODO: DOCS
    """
    prof = pd.read_parquet(profile)
    emb = pd.read_csv(embedding)
    data = merge_embedding_into_profile(prof, emb)
    data = data.sort_values(by=["hdbscan_group", "eventname"])
    categories = data["hdbscan_group"].unique()
    with LinearProgress() as prg:
        for cat in categories:
            to_plot = data[data["hdbscan_group"] == cat].drop_duplicates(
                subset="eventname"
            )
            task = prg.add_task(cat, start=True, total=len(to_plot), filename=cat)
            for pos in range(0, len(to_plot), size):
                loop_num = int((pos / size) + 1)
                g = sns.FacetGrid(
                    current:=to_plot[pos : pos + size],
                    col="eventname",
                    col_wrap=5,
                    sharex=False,
                    sharey=False,
                )
                g.map(
                    plot_flux_profile,
                    "ts",
                    "model_ts",
                    "plot_time",
                    "dt",
                    find_peaks=find_peaks
                )
                g.fig.suptitle(cat + " " + str(loop_num))
                g.set_ylabels("flux (Jy)")
                g.set_xlabels("time (ms)")
                g.savefig(f"{savefile}-{cat}-{loop_num}.png")
                prg.update(task, advance=len(current['eventname']))


def download_waterfall_data_from_chimefrb_database(
    eventnames: list, tofile: str, path: str, limit: Optional[int] = None
) -> None:
    """
    TODO: DOCS
    """
    basepath = Path(path)
    baseurl = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data"
    try:
        if tofile is None:
            raise FileNotFoundError
        currently_available_names = pd.read_parquet(
            tofile, engine="pyarrow", columns=["eventname"]
        )["eventname"].tolist()
    except FileNotFoundError:
        currently_available_names = []

    names_to_download = [
        *{i.strip("\n") for i in eventnames if i not in currently_available_names}
    ]
    if limit:
        names_to_download = names_to_download[:limit]
    manage_download_waterfall_data_task(
        names_to_download,
        progress_manager=DownloadProgress(total=len(names_to_download) - 1),
        basepath=basepath,
        baseurl=baseurl,
        collect_to=tofile,
        DataHandler=H5Waterfall,
    )


def combine_multifile_into_single_parquet_file(
    eventnames: list, collectionfile: str, filepattern: str
) -> None:
    try:
        currently_available_names = pd.read_parquet(
            collectionfile, engine="pyarrow", columns=["eventname"]
        )["eventname"].tolist()
    except FileNotFoundError:
        currently_available_names = []
    names = [i.strip("\n") for i in eventnames if i not in currently_available_names]
    with LinearProgress() as prg:
        task = prg.add_task(
            "Removing RFI",
            filename=collectionfile,
            start=True,
            visible=True,
            total=len(names),
        ) 
        for name in names:
            compress_to_parquet(
                filepattern.format(name),
                tofile=collectionfile,
                DataHandler=ParquetWaterfall,
                packing_kwargs=dict(remove_interference=True, wfall="remove"),
            )
            prg.update(task, advance=1)
