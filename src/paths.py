from pathlib import Path


data_path: Path = Path(".", "data")
external_data_path: Path = Path(data_path, "raw", "external")
collected_data_path: Path = Path(data_path, "raw", "collected")
graph_path: Path = Path(data_path, "graphs")
