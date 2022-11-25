from pathlib import Path

APP_WORKING_DIRECTORY = Path.cwd() if Path('pyproject.toml') in [Path.cwd().iterdir] else Path('.')

DATAPATH = Path.joinpath(APP_WORKING_DIRECTORY, Path('data'))