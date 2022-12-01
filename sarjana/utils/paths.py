from pathlib import Path

from sarjana.preamble import ExecutionOptions


datapath: Path = Path(".", "data")
external_datapath: Path = Path(datapath, "raw", "external")
collected_datapath: Path = Path(datapath, "raw", "collected")
graph_path: Path = (
    Path(datapath, "graphs")
    if ExecutionOptions.Mode != "debug"
    else Path(datapath, "graphs", "debug")
)
