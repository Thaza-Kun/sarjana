from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    DownloadColumn,
    TransferSpeedColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)


class DownloadProgress(Progress):
    def __init__(self, total: int) -> None:
        Progress.__init__(
            TextColumn("{task.id:>3d}/"),
            TextColumn(f"{total:>3d} "),
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

    def __enter__(self) -> "Progress":
        try:
            super().__enter__()
        except KeyboardInterrupt:
            self.stop()
        return self


class LinearProgress(Progress):
    def __init__(self) -> None:
        Progress.__init__(
            TextColumn(
                "[bold blue]{task.fields[filename]}",
                justify="right",
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            "•",
            TimeElapsedColumn(),
            "/",
            TimeRemainingColumn(),
        )
