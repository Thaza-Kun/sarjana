"""
A rudimentary URL downloader (like wget or curl) to demonstrate Rich progress bars.
"""

import os.path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import signal
from threading import Event
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


done_event = Event()


def handle_sigint(signum, frame):
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)


def copy_from_url(task_id: TaskID, url: str, path: str) -> None:
    """Copy data from a url to a local file."""
    progress.console.log(f"Requesting {url}")
    response = requests.get(url, stream=True)
    # This will break if the response doesn't contain content length
    progress.update(task_id, total=int(response.headers.get("Content-length")))
    with open(path, "wb") as dest_file:
        progress.start_task(task_id)
        for data in response.iter_content(chunk_size=32768):
            dest_file.write(data)
            progress.update(task_id, advance=len(data))
            if done_event.is_set():
                return


def pipeline(filename, task_id) -> None:
    url = "{}/{}".format(baseurl, filename)
    dest_path = os.path.join("./", filename)
    copy_from_url(task_id, url, dest_path)
    append_to_main_parquet(filename)
    subprocess.run(["rm", filename])
    progress.console.log(
        f"Downloaded {dest_path}, copied to ./{main_parquet_filename}, and deleted {dest_path}"
    )


def download(names: Iterable[str], dest_dir: str):
    """Download multiple files to the given directory."""

    with progress:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for name in names:
                filename = f"{name}_waterfall.h5"
                task_id = progress.add_task("download", filename=filename, start=False)
                pool.submit(pipeline, filename, task_id)


def read_file_to_dataframe(filename: str) -> pd.DataFrame:
    """Reads the necessary metadata from a given filename

    Args:
        filename (str): filename

    Returns:
        pd.DataFrame: dataframe withour waterfall data
    """
    frb = Waterfaller(filename)

    frb.__dict__.pop("filename", None)
    frb.__dict__.pop("datafile", None)
    frb.__dict__.pop("wfall", None)
    frb.__dict__.pop("model_wfall", None)
    frb.__dict__.pop("cal_wfall", None)

    return pd.DataFrame([frb.__dict__])


def append_to_main_parquet(filename) -> None:
    df = read_file_to_dataframe(filename)
    try:
        pqdf = pd.read_parquet(main_parquet_filename)
        pqdf = pd.concat([pqdf, df])
        pqdf.to_parquet(main_parquet_filename)
    except FileNotFoundError:
        df.to_parquet(main_parquet_filename)


names = [i["tns_name"] for i in cfod.catalog.as_list()[0:2]]
baseurl = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/21.0007/data/waterfalls/data"
main_parquet_filename = "chimefrb_profile.parquet"
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
)

if __name__ == "__main__":
    download(names, "./")
