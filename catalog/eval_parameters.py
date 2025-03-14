# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astropy",
#     "polars",
#     "scipy",
# ]
# ///
"""Evaluate parameters of catalog"""

import argparse
import pathlib

import polars as pl
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog1", type=pathlib.Path, required=True)
    parser.add_argument("--catalog2", type=pathlib.Path, required=True)
    parser.add_argument("--baseband", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    catalog: type = arguments.catalog1
    catalog2: type = arguments.catalog2
    baseband: type = arguments.baseband
    expects_catalog1 = {
        "dm_exc_ymw16": "dm_catalog",
        "bc_width": "width_catalog",
        "scat_time": "scat_time_catalog",
        "flux": "flux_catalog",
        "fluence": "fluence_catalog",
        "high_freq": "freq_high",
        "low_freq": "freq_low",
        "peak_freq": "freq_peak",
        "spec_z": "z_spec",
    }
    expects_catalog2 = {
        "dm_exc_1_ymw16": "dm_catalog",
        "bc_width": "width_catalog",
        "scat_time": "scat_time_catalog",
        "flux": "flux_catalog",
        "fluence": "fluence_catalog",
        "high_freq": "freq_high",
        "low_freq": "freq_low",
        "peak_freq": "freq_peak",
    }
    expects_baseband = {
        "DM": "DM",
        "dm_mw": "DM_MW",
        "Duration(s)": "width_baseband",
        "Scattering_600 MHz(ms)": "scat_time_baseband",
        "flux": "flux_baseband",
        "fluence": "fluence_baseband",
    }
    metadata_catalog1 = ["eventname", "repeater_name"]
    metadata_catalog2 = ["tns_name", "repeater_name"]
    metadata_baseband = ["tns_name"]
    data_catalog = (
        pl.read_csv(
            catalog,
            null_values="-9999",
            columns=[*metadata_catalog1, *expects_catalog1.keys()],
        )
        .rename(expects_catalog1)
        .rename({"eventname": "tns_name"})
        # .with_columns(pl.lit("2021").alias("catalog"))
    )
    data_catalog2 = (
        pl.read_csv(
            catalog2,
            null_values="-9999",
            columns=[*metadata_catalog2, *expects_catalog2.keys()],
        ).rename(expects_catalog2)
        # .with_columns(pl.lit("2024").alias("catalog"))
    )
    data_catalog = data_catalog.join(
        data_catalog2, on="tns_name", how="full", coalesce=True
    ).select(
        pl.col("tns_name", "z_spec"),
        pl.when(pl.col("repeater_name_right").is_not_null())
        .then(pl.col("repeater_name_right"))
        .otherwise(pl.col("repeater_name"))
        .alias("repeater_name"),
        pl.when(pl.col("dm_catalog_right").is_not_null())
        .then(pl.col("dm_catalog_right"))
        .otherwise(pl.col("dm_catalog"))
        .alias("dm_catalog"),
        pl.when(pl.col("width_catalog_right").is_not_null())
        .then(pl.col("width_catalog_right"))
        .otherwise(pl.col("width_catalog"))
        .alias("width_catalog"),
        pl.when(pl.col("scat_time_catalog_right").is_not_null())
        .then(pl.col("scat_time_catalog_right"))
        .otherwise(pl.col("scat_time_catalog"))
        .alias("scat_time_catalog"),
        pl.when(pl.col("flux_catalog_right").is_not_null())
        .then(pl.col("flux_catalog_right"))
        .otherwise(pl.col("flux_catalog"))
        .alias("flux_catalog"),
        pl.when(pl.col("fluence_catalog_right").is_not_null())
        .then(pl.col("fluence_catalog_right"))
        .otherwise(pl.col("fluence_catalog"))
        .alias("fluence_catalog"),
        pl.when(pl.col("freq_high_right").is_not_null())
        .then(pl.col("freq_high_right"))
        .otherwise(pl.col("freq_high"))
        .alias("freq_high"),
        pl.when(pl.col("freq_low_right").is_not_null())
        .then(pl.col("freq_low_right"))
        .otherwise(pl.col("freq_low"))
        .alias("freq_low"),
        pl.when(pl.col("freq_peak_right").is_not_null())
        .then(pl.col("freq_peak_right"))
        .otherwise(pl.col("freq_peak"))
        .alias("freq_peak"),
    )
    data_baseband = (
        pl.read_csv(
            baseband,
            null_values="-9999",
            columns=[*metadata_baseband, *expects_baseband.keys()],
        )
        .rename(expects_baseband)
        .with_columns((pl.col("DM") - pl.col("DM_MW")).alias("dm_baseband"))
        .select(
            pl.exclude("DM", "DM_MW"),
        )
    )
    A = 155.80
    Alpha = 4.58
    data_joined = (
        data_catalog.join(data_baseband, on="tns_name", how="left")
        .select(
            pl.col("tns_name").alias("eventname"),
            pl.col("repeater_name"),
            pl.col("freq_high"),
            pl.col("freq_low"),
            pl.col("freq_peak"),
            pl.col("z_spec"),
            pl.when(pl.col("dm_baseband").is_not_null())
            .then(pl.col("dm_baseband"))
            .otherwise(pl.col("dm_catalog"))
            .alias("dm"),
            pl.when(pl.col("dm_baseband").is_not_null())
            .then(pl.lit("baseband"))
            .otherwise(pl.lit("catalog"))
            .alias("dm_src")
            .cast(pl.Categorical),
            pl.when(pl.col("width_baseband").is_not_null())
            .then(pl.col("width_baseband"))
            .otherwise(pl.col("width_catalog"))
            .alias("width"),
            pl.when(pl.col("width_baseband").is_not_null())
            .then(pl.lit("baseband"))
            .otherwise(pl.lit("catalog"))
            .alias("width_src")
            .cast(pl.Categorical),
            pl.when(pl.col("flux_baseband").is_not_null())
            .then(pl.col("flux_baseband"))
            .otherwise(pl.col("flux_catalog"))
            .alias("flux"),
            pl.when(pl.col("flux_baseband").is_not_null())
            .then(pl.lit("baseband"))
            .otherwise(pl.lit("catalog"))
            .alias("flux_src")
            .cast(pl.Categorical),
            pl.when(pl.col("fluence_baseband").is_not_null())
            .then(pl.col("fluence_baseband"))
            .otherwise(pl.col("fluence_catalog"))
            .alias("fluence"),
            pl.when(pl.col("fluence_baseband").is_not_null())
            .then(pl.lit("baseband"))
            .otherwise(pl.lit("catalog"))
            .alias("fluence_src")
            .cast(pl.Categorical),
            pl.when(pl.col("scat_time_baseband").is_not_null())
            .then(pl.col("scat_time_baseband"))
            .otherwise(pl.col("scat_time_catalog"))
            .alias("scat_time"),
            pl.when(pl.col("scat_time_baseband").is_not_null())
            .then(pl.lit("baseband"))
            .otherwise(pl.lit("catalog"))
            .alias("scat_time_src")
            .cast(pl.Categorical),
        )
        .with_columns(
            pl.col("scat_time")
            .str.strip_chars_start("<")
            .str.strip_chars_start("~")
            .cast(pl.Float64),
            pl.when(pl.col("scat_time").str.starts_with("<"))
            .then(pl.lit("less_than"))
            .otherwise(
                pl.when(pl.col("scat_time").str.starts_with("~"))
                .then(pl.lit("approx"))
                .otherwise(pl.lit("exact"))
            )
            .alias("scat_time_measure"),
        )
        .with_columns(
            pl.when(pl.col("z_spec").is_not_null())
            .then(pl.col("z_spec"))
            .otherwise(
                # We follow https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2024.1371787/full to model z (Eq. 1)
                (pl.col("dm") / A) ** (1 / Alpha)
                - 1
            )
            .alias("z"),
            pl.when(pl.col("z_spec").is_not_null())
            .then(pl.lit("spectra"))
            .otherwise(pl.lit("dm"))
            .alias("z_src")
            .cast(pl.Categorical),
        )
        .with_columns(
            pl.col("z")
            .map_elements(
                lambda z: cosmo.angular_diameter_distance(z).to_value(),
                return_dtype=float,
            )
            .alias("Dist")
        )
        .with_columns(
            # We follow Zhu-Ge 2023 (Eq. 6)
            (pl.col("flux") * pl.col("freq_peak") * 4 * 3.1415 * pl.col("Dist"))
            .log10()
            .alias("luminosity_log10"),
            # We follow Di-Xiao 2022 (Eq. 1)
            (
                1.1e35
                * pl.col("flux")
                * (pl.col("Dist") ** 2)
                / (pl.col("freq_peak") * pl.col("width")) ** 2
            )
            .log10()
            .alias("temperature_log10"),
        )
        .select(pl.exclude("z_spec", "Dist"))
    )
    data_joined.write_csv(arguments.out)
    print(f"Saved to {arguments.out}")


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
