import os
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from tqdm import tqdm

import dfdt

from sarjana.tui import LinearProgress, DownloadProgress
from sarjana.collections import merge_embedding_into_profile
from sarjana.handlers import H5Waterfall, ParquetWaterfall

from sarjana.plotting import plot_time_flux
from sarjana.download import manage_download_waterfall_data_task, compress_to_parquet

from sarjana.signal.properties import is_multipeak
from sarjana.signal.analyze import find_frequency_drift
from sarjana.signal import analyze


def find_frequency_drift_by_name(
    frbname: str, filepattern: str = "{}.parquet", path: Path = Path(".")
):
    burst = ParquetWaterfall(Path(path, filepattern.format(frbname)))
    return find_frequency_drift(burst.wfall)


def find_frequency_drift_acmc_all(
    n_mc: int = 10,
    n_dm: int = 10,
    event_total: int = 0,
    path: Path = Path(".", "results"),
):
    # burst parameters
    dm_uncertainty = 0.2  # pc cm-3

    # instrument parameters
    dt_s = 0.00098304
    df_mhz = 0.0244140625
    nchan = 16384
    freq_bottom_mhz = 400.1953125
    freq_top_mhz = 800.1953125

    ds = dfdt.DynamicSpectrum(dt_s, df_mhz, nchan, freq_bottom_mhz, freq_top_mhz)
    data = pd.read_parquet(
        Path(os.getenv("DATAPATH"), "raw", "cfod", "chimefrb_profile.parquet")
    )
    columnnames: list = [
        "subburstname",
        "constrained",
        "dfdt_data",
        "dfdt_mc",
        "dfdt_mc_low",
        "dfdt_mc_high",
        "duration_min",
    ]
    try:
        prev = pd.read_csv(Path(path, "dfdt.csv"))
        prevnames: set = {i.split("_")[0] for i in prev["subburstname"].to_list()}
    except FileNotFoundError:
        pd.DataFrame(columns=columnnames).to_csv(Path(path, "dfdt.csv"))
        prevnames: set = {}

    eventcount = 0
    for idx, event in enumerate(data["eventname"], start=1):
        print(f'{idx} / {len(data["eventname"])} -----------------------')
        burst = ParquetWaterfall(
            Path(os.getenv("DATAPATH"), "raw", "wfall", f"{event}_waterfall.h5.parquet")
        )
        if len(burst.peaks) > 1:
            if event not in prevnames:
                for i, (peak, width) in enumerate(
                    zip(burst.peaks[:-1], burst.widths[:-1]), 1
                ):
                    start = datetime.now()
                    subburstname = f"{burst.eventname}_{i}"
                    (
                        dfdt_data,
                        theta,
                        theta_sigma,
                    ) = analyze.frequency_drift_by_autocorrelation(
                        burst.wfall,
                        burst.eventname,
                        subburstname,
                        ds,
                        peak,
                        width,
                        fdir=path.as_posix() + "/",
                    )
                    (
                        constrained,
                        dfdt_mc,
                        dfdt_mc_low,
                        dfdt_mc_high,
                    ) = analyze.frequency_drift_by_acmc(
                        burst.wfall,
                        dm_uncertainty,
                        ds,
                        peak,
                        width,
                        dfdt_data,
                        theta,
                        theta_sigma,
                        burst.eventname,
                        subburstname,
                        mc_trials=n_mc,
                        dm_trials=n_dm,
                        fdir=path.as_posix() + "/",
                    )
                    dfdt_df = pd.read_csv(Path(path, "dfdt.csv"), index_col=0)
                    end = datetime.now()
                    duration_min: float = (end - start).seconds / 60
                    pd.concat(
                        [
                            dfdt_df,
                            pd.DataFrame(
                                [
                                    [
                                        subburstname,
                                        constrained,
                                        dfdt_data,
                                        dfdt_mc,
                                        dfdt_mc_low,
                                        dfdt_mc_high,
                                        duration_min,
                                    ]
                                ],
                                columns=columnnames,
                            ),
                        ]
                    ).to_csv(Path(path, "dfdt.csv"))
                eventcount += 1
                if (eventcount == event_total) and (event_total != 0):
                    break


def cluster_kmeans_on_bandwidth(
    catalog_file: str, wfall_catalog_file: str, saveto: str
):
    chimefrb_catalog_1 = pd.read_csv(Path(catalog_file))
    cat_wfall = pd.read_parquet(Path(wfall_catalog_file))
    cat_wfall["is_multipeak"] = [
        is_multipeak(x.copy()) for x in cat_wfall["model_ts"].values
    ]
    cat = pd.merge(
        cat_wfall[["is_multipeak", "eventname"]], chimefrb_catalog_1, left_on="eventname", right_on="tns_name"
    ).drop(columns="tns_name")
    cat["bandwidth"] = cat["high_freq"] - cat["low_freq"]
    clusterer = KMeans(n_clusters=2).fit(cat.loc[cat["is_multipeak"] == False, ["bandwidth"]])
    cat.loc[cat["is_multipeak"] == False, ["kmeans_label"]] = clusterer.predict(
        cat.loc[cat["is_multipeak"] == False, ["bandwidth"]]
        )
    cat['kmeans_label'].fillna(len(cat['kmeans_label'].unique()) -1, inplace=True)
    cat.to_parquet(Path(saveto))
    return cat


def find_frequency_drift_of_catalog(catalogfile: str, wfall_path: str):
    cat = pd.read_parquet(catalogfile, columns=["filename", "model_ts"])
    cat["is_multipeak"] = [is_multipeak(x.copy()) for x in cat["model_ts"].values]
    print(cat["is_multipeak"].describe())
    cat.loc[cat["is_multipeak"] == True, "dfdt"] = cat.apply(
        lambda x: find_frequency_drift(
            ParquetWaterfall(Path(wfall_path, x["filename"])).wfall
        ),
        axis="columns",
    )
    cat.loc[cat["is_multipeak"] == False, "dfdt"] = None
    return cat


# def plot_delay_spaces_frequency_drift():
#     ...


def plot_many_flux_profile_by_clustering_groups(
    profile: str,
    embedding: str,
    savefile: str,
    size: int,
    *,
    highlight_burst: bool = False,
) -> None:
    """
    TODO: DOCS
    """
    matplotlib.use("Agg")
    prof = pd.read_parquet(profile)
    emb = pd.read_csv(embedding)
    data = merge_embedding_into_profile(prof, emb)
    data["is_multipeak"] = [is_multipeak(x.copy()) for x in data["model_ts"].values]
    data = data.sort_values(by=["hdbscan_group", "is_multipeak", "eventname"])
    categories = data["hdbscan_group"].unique()
    data["fitting"] = [
        "cfod" if multipeak is True else "scattered_gaussian"
        for multipeak in data["is_multipeak"].values
    ]
    with LinearProgress() as prg:
        for cat in categories:
            to_plot = data[data["hdbscan_group"] == cat].drop_duplicates(
                subset="eventname"
            )
            task = prg.add_task(cat, start=True, total=len(to_plot), filename=cat)
            for pos in range(0, len(to_plot), size):
                try:
                    loop_num = int((pos / size) + 1)
                    g = sns.FacetGrid(
                        current := to_plot[pos : pos + size],
                        col="eventname",
                        col_wrap=5,
                        sharex=False,
                        sharey=False,
                    )
                    g.map(
                        plot_time_flux,
                        "ts",
                        "model_ts",
                        "plot_time",
                        "dt",
                        highlight_burst=highlight_burst,
                    )
                    g.fig.suptitle(cat + " " + str(loop_num))
                    g.set_ylabels("flux (Jy)")
                    g.set_xlabels("time (ms)")
                    g.add_legend()
                    g.savefig(f"{savefile}-{cat}-{loop_num}.png")
                except RuntimeError:
                    raise RuntimeError(current["eventname"].values)
                finally:
                    prg.update(task, advance=len(current["eventname"]))
                    plt.close(plt.gcf())  # Prevent keeping many figure open


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
