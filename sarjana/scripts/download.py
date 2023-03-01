"""
A rudimentary URL downloader (like wget or curl) to demonstrate Rich progress bars.
"""

import os.path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import signal
import threading
from typing import Iterable
import requests

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TransferSpeedColumn,
)
import pandas as pd
from cfod.routines.waterfaller import Waterfaller
import cfod
import numpy as np

done_event = threading.Event()


def handle_sigint(signum, frame):
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)


def copy_from_url(task_id: TaskID, url: str, path: str) -> None:
    """Copy data from a url to a local file."""
    progress.console.log(f"Requesting {url}")
    response = requests.get(url, stream=True)
    # This will break if the response doesn't contain content length
    progress.update(
        task_id, total=int(response.headers.get("Content-length")), visible=True
    )
    with open(path, "wb") as dest_file:
        progress.start_task(task_id)
        for data in response.iter_content(chunk_size=32768):
            dest_file.write(data)
            progress.update(task_id, advance=len(data))
            if done_event.is_set():
                return
    progress.remove_task(task_id)


def download(names: Iterable[str], dest_dir: str):
    """Download multiple files to the given directory."""

    with progress:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for name in names:
                filename = f"{name}_waterfall.h5"
                task_id = progress.add_task(
                    "download", filename=filename, start=False, visible=False
                )
                pool.submit(pipeline, filename, task_id)


class WaterfallData(Waterfaller):
    def _unpack(self):
        unnecessary_metadata = ["filename", "datafile"]
        self.datafile = self.datafile["frb"]
        self.eventname = self.datafile.attrs["tns_name"].decode()
        self.wfall = self.datafile["wfall"][:]
        self.model_wfall = self.datafile["model_wfall"][:]
        self.plot_time = self.datafile["plot_time"][:]
        self.plot_freq = self.datafile["plot_freq"][:]
        self.ts = self.datafile["ts"][:]
        self.model_ts = self.datafile["model_ts"][:]
        self.spec = self.datafile["spec"][:]
        self.model_spec = self.datafile["model_spec"][:]
        self.extent = self.datafile["extent"][:]
        self.dm = self.datafile.attrs["dm"][()]
        self.scatterfit = self.datafile.attrs["scatterfit"][()]
        self.dt = np.median(np.diff(self.plot_time))
        for metadata in unnecessary_metadata:
            self.__dict__.pop(metadata, None)

        self.wfall_shape = self.wfall.shape
        self.wfall = self.wfall.reshape((-1,))
        self.model_wfall = self.model_wfall.reshape((-1,))
        self.cal_wfall_shape = (
            self.cal_wfall.shape if getattr(self, "cal_wfall", None) else None
        )
        self.cal_wfall = (
            self.cal_wfall.reshape((-1,)) if getattr(self, "cal_wfall", None) else None
        )


def read_file_to_dataframe(filename: str) -> pd.DataFrame:
    """Reads the necessary metadata from a given filename

    Args:
        filename (str): filename

    Returns:
        pd.DataFrame: dataframe withour waterfall data
    """
    frb = WaterfallData(filename)
    return pd.DataFrame([frb.__dict__])


def append_to_main_parquet(filename) -> None:
    df = read_file_to_dataframe(filename)
    try:
        pqdf = pd.read_parquet(main_parquet_filename)
        pqdf = pd.concat([pqdf, df])
        pqdf.to_parquet(main_parquet_filename)
    except FileNotFoundError:
        df.to_parquet(main_parquet_filename)


main_parquet_filename = "chimefrb_profile.parquet"
available_names = pd.read_parquet(main_parquet_filename, columns=["eventname"])[
    "eventname"
].tolist()

names = [
    *{
        i["tns_name"]
        for i in cfod.catalog.as_list()
        if i["tns_name"] not in available_names
    }
]
baseurl = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data"

progress = Progress(
    TextColumn("{task.id:>3d}/"),
    TextColumn(f"{len(names)-1:>3d} "),
    TextColumn(
        "[bold blue]{task.fields[filename]}",
        justify="right",
    ),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeElapsedColumn(),
    "/",
    TimeRemainingColumn(),
    transient=True,
)


def pipeline(filename, task_id) -> None:
    url = "{}/{}".format(baseurl, filename)
    dest_path = os.path.join("./", filename)
    copy_from_url(task_id, url, dest_path)
    progress.console.log(f":inbox_tray: {dest_path} downloaded.")
    thread_lock = threading.Lock()
    thread_lock.acquire()
    progress.console.log(
        f":locked_with_key:{dest_path} acquired lock on ./{main_parquet_filename}"
    )
    try:
        append_to_main_parquet(filename)
        progress.console.log(
            f":pencil: {dest_path} copied to ./{main_parquet_filename}."
        )
        subprocess.run(["rm", filename])
        progress.console.log(f":heavy_large_circle: {dest_path} deleted.")
    except:
        progress.console.log(
            f":exclamation_mark: Error with {filename}. Saving for debugging..."
        )
    thread_lock.release()
    progress.console.log(
        f":unlocked: {dest_path} release lock on ./{main_parquet_filename}"
    )


from pathlib import Path

if __name__ == "__main__":
    # download(names, "./")
    for path in Path(".").iterdir():
        if path.suffix == ".h5":
            print(path)
            append_to_main_parquet(path)
