from pathlib import Path


datapath: Path = Path("..", "data")
external_datapath: Path = Path(datapath, "raw", "external")
collected_datapath: Path = Path(datapath, "raw", "collected")
graph_path: Path = Path(datapath, "graphs")
