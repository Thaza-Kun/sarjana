from typing import Optional
import typer
import rich

from sarjana.preamble import ExecutionOptions
from sarjana.workflows.merging import merge, plot

from sarjana.workflows.replications import bo_han_chen_2021
from sarjana.workflows.experiments import (
    UMAP_HDBSCAN_FRBSTATS,
    HDBSCAN_important_features,
)
from sarjana.data.collections import load_catalog

ExecutionOptions.Mode = "debug"

app = typer.Typer()

learning_func = (
    bo_han_chen_2021,
    UMAP_HDBSCAN_FRBSTATS,
    HDBSCAN_important_features,
)


@app.command()
def learn(
    func: int = typer.Argument(
        ...,
        help="The learning function to run on. ({})".format(
            ", ".join(
                [f"[{i}] {func.__name__}" for i, func in enumerate(learning_func)]
            )
        ),
    ),
    size: int = typer.Argument(19, help="Minimum size of cluster."),
    seed: Optional[int] = typer.Option(
        42, help="Seed to set on stochastic algorithms."
    ),
    debug: bool = typer.Option(False, help="Print debug log."),
):
    """An unsupervised machine learning workflow based on dimensional reduction and clustering algorithms."""
    data, result = learning_func[func](min_cluster_size=size, seed=seed, debug=debug)
    print("score: {}".format(result))


@app.command()
def info(
    func: int = typer.Argument(
        ...,
        help="Function id. ({})".format(
            ", ".join(
                [f"[{i}] {func.__name__}" for i, func in enumerate(learning_func)]
            )
        ),
    )
):
    """Get docstrings of specified function."""
    rich.print(learning_func[func].__doc__)


# 2. Read and display data
@app.command()
def inspect(
    # filename: str = typer.Argument(..., help="Data file to inspect")
):
    data = merge()
    rich.print(plot(data))


# 3. Get Data
@app.command()
def get():
    ...


if __name__ == "__main__":
    app()
