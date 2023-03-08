"""A script to download `.h5` files from CHIME/FRB website

(Currently only works in Linux as it depends on `cfod` which depends on `healpy`)
"""

from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor
import signal
import threading
from typing import Iterable, Protocol
import requests

import pandas as pd

# from cfod.routines.waterfaller import Waterfaller
# import cfod
# import numpy as np

# Typing
from rich.progress import Progress, TaskID

done_event = threading.Event()


def handle_sigint(signum, frame):
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)


# class WaterfallData(Waterfaller):
#     def _unpack(self):
#         unnecessary_metadata = ["filename", "datafile"]
#         self.datafile = self.datafile["frb"]
#         self.eventname = self.datafile.attrs["tns_name"].decode()
#         self.wfall = self.datafile["wfall"][:]
#         self.model_wfall = self.datafile["model_wfall"][:]
#         self.plot_time = self.datafile["plot_time"][:]
#         self.plot_freq = self.datafile["plot_freq"][:]
#         self.ts = self.datafile["ts"][:]
#         self.model_ts = self.datafile["model_ts"][:]
#         self.spec = self.datafile["spec"][:]
#         self.model_spec = self.datafile["model_spec"][:]
#         self.extent = self.datafile["extent"][:]
#         self.dm = self.datafile.attrs["dm"][()]
#         self.scatterfit = self.datafile.attrs["scatterfit"][()]
#         self.dt = np.median(np.diff(self.plot_time))
#         for metadata in unnecessary_metadata:
#             self.__dict__.pop(metadata, None)

#         self.wfall_shape = self.wfall.shape
#         self.wfall = self.wfall.reshape((-1,))
#         self.model_wfall = self.model_wfall.reshape((-1,))
#         self.cal_wfall_shape = (
#             self.cal_wfall.shape if getattr(self, "cal_wfall", None) else None
#         )
#         self.cal_wfall = (
#             self.cal_wfall.reshape((-1,)) if getattr(self, "cal_wfall", None) else None
#         )


class FRBDataHandler(Protocol):
    def _unpack(self) -> None:
        ...

    @property
    def __dict__(self) -> dict:
        ...


def read_frb_to_dataframe(filename: str, data_handler: FRBDataHandler) -> pd.DataFrame:
    """Reads the necessary metadata from a given filename

    Args:
        filename (str): filename
        data_handler (FRBDataHandler): a class to unpack data
    """
    frb = data_handler(filename)
    return pd.DataFrame([frb.__dict__])


def collect_data_into_single_file(
    collect_from: str, collect_to: str, data_handler: FRBDataHandler
) -> None:
    df = read_frb_to_dataframe(collect_from, data_handler=data_handler)
    try:
        pqdf = pd.read_parquet(collect_to)
        pqdf = pd.concat([pqdf, df])
        pqdf.to_parquet(collect_to)
    except FileNotFoundError:
        df.to_parquet(collect_to)


def run_download_from_url_task(
    task_id: TaskID, url: str, dest_file: str, progress_manager: Progress
) -> None:
    """Copy data from a url to a local file."""
    progress_manager.console.log(f"Requesting {url}")
    response = requests.get(url, stream=True)
    # This will break if the response doesn't contain content length
    progress_manager.update(
        task_id, total=int(response.headers.get("Content-length")), visible=True
    )
    with open(dest_file, "wb") as to_file:
        progress_manager.start_task(task_id)
        for data in response.iter_content(chunk_size=32768):
            to_file.write(data)
            progress_manager.update(task_id, advance=len(data))
            if done_event.is_set():
                return
    progress_manager.remove_task(task_id)


def run_download_and_collect_task(
    collect_to: str,
    basepath: Path,
    baseurl: str,
    collect_from: str,
    data_handler: FRBDataHandler,
    progress_manager: Progress,
    task_id: int,
) -> None:
    url = "{}/{}".format(baseurl, collect_from)
    dest_file = Path(basepath, collect_from)
    run_download_from_url_task(task_id, url, dest_file, progress_manager)
    progress_manager.console.log(f":inbox_tray: {dest_file} downloaded.")
    thread_lock = threading.Lock()
    thread_lock.acquire()
    progress_manager.console.log(
        f":locked_with_key:{dest_file} acquired lock on ./{collect_to}"
    )
    try:
        collect_data_into_single_file(collect_from, collect_to, data_handler)
        progress_manager.console.log(f":pencil: {dest_file} copied to ./{collect_to}.")
        subprocess.run(["rm", collect_from])
        progress_manager.console.log(f":heavy_large_circle: {dest_file} deleted.")
    except Exception:
        progress_manager.console.log(
            f":exclamation_mark: Error with {collect_from}. Saving for debugging..."
        )
    thread_lock.release()
    progress_manager.console.log(
        f":unlocked: {dest_file} release lock on ./{collect_to}"
    )


def manage_download_waterfall_data_task(
    names: Iterable[str], progress_manager: Progress, **kwargs
):
    """Download multiple files to the given directory."""
    expected_kwargs = run_download_and_collect_task.__annotations__
    generated_kwarg = ["collect_from", "task_id", "progress_manager"]
    for keyword in expected_kwargs.keys():
        if keyword not in [*kwargs.keys(), "return"] and keyword not in generated_kwarg:
            raise AttributeError(
                f"Expected {keyword} in function arguments {expected_kwargs.keys()} but only {kwargs.keys()} was given."
            )
    with progress_manager:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for name in names:
                filename = f"{name}_waterfall.h5"
                task_id = progress_manager.add_task(
                    "download", filename=filename, start=False, visible=False
                )
                pool.submit(
                    run_download_and_collect_task,
                    collect_from=filename,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    **kwargs,
                )
