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

# Typing
from rich.progress import Progress, TaskID
import rich

done_event = threading.Event()


def handle_sigint(signum, frame):
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)


class WaterfallDataHandler(Protocol):
    def _unpack(self) -> None:
        ...

    @property
    def dataframe(self) -> pd.DataFrame:
        ...


def read_frb_to_dataframe(filename: str, DataHandler: WaterfallDataHandler) -> pd.DataFrame:
    """Reads the necessary metadata from a given filename

    Args:
        filename (str): filename
        data_handler (FRBDataHandler): a class to unpack data
    """
    frb = DataHandler(filename)
    return frb.dataframe


def compress_to_parquet(
    fromfile: str, tofile: str, DataHandler: WaterfallDataHandler
) -> None:
    df = read_frb_to_dataframe(fromfile, DataHandler=DataHandler)
    try:
        data = pd.read_parquet(tofile, columns=["eventname"])
        del data
        df.to_parquet(tofile, engine="pyarrow")
    except FileNotFoundError:
        df.to_parquet(tofile, engine="pyarrow")


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
    DataHandler: WaterfallDataHandler,
    progress_manager: Progress,
    task_id: int,
) -> None:
    url = "{}/{}".format(baseurl, collect_from)
    dest_file = Path(basepath, collect_from)
    if collect_to is None:
        collect_to = Path(basepath, f"{collect_from}.parquet")
    run_download_from_url_task(task_id, url, dest_file, progress_manager)
    progress_manager.console.log(f":inbox_tray: {dest_file} downloaded.")
    thread_lock = threading.Lock()
    thread_lock.acquire()
    progress_manager.console.log(
        f":locked_with_key:{dest_file} acquired lock on ./{collect_to}"
    )
    try:
        compress_to_parquet(dest_file, collect_to, DataHandler)
        progress_manager.console.log(f":pencil: {dest_file} copied to ./{collect_to}.")
        subprocess.run(["rm", dest_file])
        progress_manager.console.log(f":heavy_large_circle: {dest_file} deleted.")
    except Exception as e:
        rich.print(Exception.__name__, e)
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
