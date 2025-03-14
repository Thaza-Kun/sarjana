# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "polars",
#     "scikit-learn",
#     "umap-learn",
# ]
# ///
"""Evaluate UMAP-HDBSCAN Pipeline on a catalog"""
import argparse
import pathlib

import umap
import polars as pl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    catalog: pathlib.Path = arguments.catalog

    params = [
        "width",
        "flux",
        "fluence",
        "scat_time",
        "freq_high",
        "freq_low",
        "freq_peak",
        "z",
        "luminosity_log10",
        "temperature_log10",
    ]

    catalog = pl.read_csv(catalog).drop_nans(["luminosity_log10", "temperature_log10"])
    catalog = catalog.filter(~pl.any_horizontal(pl.selectors.numeric().is_infinite()))

    repeating = catalog.filter(pl.col("repeater_name").is_not_null()).with_columns(
        pl.lit(True).alias("repeater")
    )
    non_repeating = (
        catalog.filter(pl.col("repeater_name").is_null())
        .with_columns(pl.lit(False).alias("repeater"))
        .select(pl.exclude("repeater_name"))
    )

    test = repeating

    train = (
        repeating.group_by("repeater_name")
        .mean()
        .cast({"repeater": bool})
        .select(pl.exclude("eventname"))
        .rename({"repeater_name": "eventname"})
    )
    test, train = train_test_split(train, test_size=0.6)
    to_train = pl.concat([non_repeating, train])

    model = umap.UMAP(n_neighbors=8, n_components=2, min_dist=0.1)
    projection = model.fit(to_train[params])
    projection_test = projection.transform(test[params])
    to_train = to_train.with_columns(
        x = projection.embedding_[:, 0],
        y = projection.embedding_[:, 1]
    )
    test = test.with_columns(
        x = projection_test[:, 0],
        y = projection_test[:, 1]
    )
    print(test)
    print(to_train)

    plt.scatter(to_train.filter(pl.col('repeater').eq(False))['x'], to_train.filter(pl.col('repeater').eq(False))['y'], label='NR')
    plt.scatter(to_train.filter(pl.col('repeater').eq(True))['x'], to_train.filter(pl.col('repeater').eq(True))['y'], label='R (train)')
    plt.scatter(test['x'], test['y'], label='R (test)')
    plt.legend()
    plt.savefig(f'{arguments.out}/umap.png')


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
