from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from cfod.analysis import waterfall

from sarjana.data.collections import load_2d_embedding, load_catalog, load_profiles

datasource = Path("/mnt/d/home/datasets/sarjana/")

cfod_path = Path(datasource, "./raw/cfod/")
external_data_path = Path(datasource, "./raw/external/")

chimefrb_profile = Path(cfod_path, "chimefrb_profile.parquet")
embedding_data = Path(external_data_path, "2d-embeddings_chen2022.csv")

profiles = load_profiles(chimefrb_profile)

embedding = load_2d_embedding(embedding_data)

catalog = load_catalog(Path(external_data_path, "catalog_Hashimoto2022.csv"))


def merge() -> pd.DataFrame:
    return profiles.merge(embedding, left_on="eventname", right_on="tns_name")


def plot(data: pd.DataFrame) -> None:
    item = data.loc[0, :]
    peak, width, snr = waterfall.find_burst(item['ts'])
    item['plot_time'] = item['plot_time'] - item['plot_time'][np.argmax(item['ts'])]
    item['plot_time'] = item['plot_time'] - item['dt']/2
    item['plot_time'] = np.append(item['plot_time'], item['plot_time'][-1] + item['dt'])
    item['ts'] = np.append(item['ts'], item['ts'][-1])
    item['model_ts'] = np.append(item['model_ts'], item['model_ts'][-1])
    g = sns.lineplot(item, x="plot_time", y="ts", drawstyle="steps-post")
    sns.lineplot(item, x="plot_time", y="model_ts", drawstyle="steps-post", ax=g)
    g.set_title(item["eventname"])
    g.axvspan(
        max(
            item['plot_time'].min(), 
            item['plot_time'][peak] + 0.5 * item['dt'] - (0.5 * width) * item['dt']
        ), 
        min(
            item['plot_time'].max(), 
            item['plot_time'][peak] + 0.5 * item['dt'] + (0.5 * width) * item['dt']
        ),
        facecolor="tab:blue", edgecolor=None, alpha=0.1
    )
    g.get_figure().savefig(f"{item['eventname']}.png")
