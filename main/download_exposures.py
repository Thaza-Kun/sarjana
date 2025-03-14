#! usr/bin/env python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "rich",
# ]
# ///

# Author: Murthadza Aznam
# Date: 2024-07-05
# Python Version: 3.12

"""Download exposure data between ranges"""

import argparse
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import signal
import threading
from typing import Iterable
import requests

from rich.progress import (
    Progress,
    TaskID,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

done_event = threading.Event()


def handle_sigint(signum, frame):
    done_event.set()


signal.signal(signal.SIGINT, handle_sigint)


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
    progress_manager.console.log(f":inbox_tray: {dest_file} downloaded.")


def manage_download_exposure_data_task(
    dates: Iterable[datetime.datetime],
    progress_manager: Progress,
    basepath: Path,
    baseurl: str,
):
    """Download multiple files to the given directory."""
    # expected_kwargs = run_download_from_url_task.__annotations__
    # generated_kwarg = ["task_id", "progress_manager"]
    # for keyword in expected_kwargs.keys():
    #     if keyword not in [*kwargs.keys(), "return"] and keyword not in generated_kwarg:
    #         raise AttributeError(
    #             f"Expected {keyword} in function arguments {expected_kwargs.keys()} but only {kwargs.keys()} was given."
    #         )
    with progress_manager:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for date in dates:
                filename = f"exposure_{date.strftime("%Y%m%d")}_{(date + datetime.timedelta(days=1)).strftime("%Y%m%d")}_transit_L_beam_FWHM-600_res_4s_0.86_arcmin.npz"
                task_id = progress_manager.add_task(
                    "download",
                    filename=f"exp_{date.strftime("%Y%m%d")}_L",
                    start=False,
                    visible=False,
                )
                pool.submit(
                    run_download_from_url_task,
                    task_id=task_id,
                    url=f"{baseurl}/{filename}",
                    dest_file=Path(basepath, filename),
                    progress_manager=progress_manager,
                )
                filename = f"exposure_{date.strftime("%Y%m%d")}_{(date + datetime.timedelta(days=1)).strftime("%Y%m%d")}_transit_U_beam_FWHM-600_res_4s_0.86_arcmin.npz"
                task_id = progress_manager.add_task(
                    "download",
                    filename=f"exp_{date.strftime("%Y%m%d")}_U",
                    start=False,
                    visible=False,
                )
                pool.submit(
                    run_download_from_url_task,
                    task_id=task_id,
                    url=f"{baseurl}/{filename}",
                    dest_file=Path(basepath, filename),
                    progress_manager=progress_manager,
                )


def download(arguments: argparse.Namespace):
    basepath = arguments.out
    date_begin: datetime.datetime = arguments.begin
    date_end: datetime.datetime = arguments.end

    length = (date_end - date_begin).days
    baseurl = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/AstroDataCitationDOI/CISTI.CANFAR/23.0004/data/exposure/daily_exposure_maps"

    dates = [date_begin + datetime.timedelta(days=i) for i in range(length + 1)]
    progress = Progress(
        TextColumn("{task.id:>3d}/"),
        TextColumn(f"{length*2:>3d} "),
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
    manage_download_exposure_data_task(
        dates=dates,
        progress_manager=progress,
        basepath=basepath,
        baseurl=baseurl,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--begin", type=datetime.datetime.fromisoformat, required=True)
    parser.add_argument("--end", type=datetime.datetime.fromisoformat, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    download(arguments)
