# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "pyarrow",
#     "seaborn",
# ]
# ///
""" """
import argparse
import pathlib
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    names = [
        "FRB20190804E",
        "FRB20190915D",
        "FRB20200223B",
        "FRB20201130A",
        "FRB20201221B",
    ]
    one = [
        "tns_name",
        "repeater_name",
        "ra_1",
        "ra_2",
        "dec_1",
        "dec_2",
        "gl",
        "gb",
        "exp_up",
        "exp_low",
    ]
    two = [
        "bonsai_snr",
        "bonsai_dm",
        "snr_fitb",
        "dm_fitb",
        "bc_width",
        "scat_time",
        "mjd_400",
    ]
    three = ["flux", "fluence", "width_fitb", "sp_idx", "bandwidth", "peak_freq"]
    file = arguments.file
    catalog = pl.read_csv(file, null_values="-9999").with_columns(
        (pl.col("high_freq") - pl.col("low_freq")).alias("bandwidth")
    )
    print(catalog.columns)
    catalog = catalog.select(
        pl.col([*one, *two, *three]),
    )
    mjd = catalog.group_by(pl.col("repeater_name")).agg(
        pl.col("mjd_400").min().alias("min_mjd"),
        (pl.col("mjd_400").max() - pl.col("mjd_400").min()).alias("diff_mjd"),
    )
    catalog = catalog.join(mjd, on="repeater_name").with_columns(
        ((pl.col("mjd_400") - pl.col("min_mjd")) / pl.col("diff_mjd")).alias(
            "time_evolution"
        )
    )

    # name = names[1]
    for name in names:
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(nrows=3, ncols=2, figure=fig)
        f = catalog.filter(pl.col("repeater_name") == name)
        for grid, colname in zip(gs, three):
            a = plt.subplot(grid)
            a.scatter(
                f.select(pl.col("time_evolution")).to_numpy(),
                f.select(pl.col(colname)).to_numpy(),
            )
            a.set_ylabel(colname)

        fig.suptitle(name)
        plt.tight_layout()
        plt.savefig(f"{arguments.out}/{name}_evolution.pdf")
        print(f"Saved to: {arguments.out}/{name}_evolution.pdf")

    # print(catalog)

    # sns.pairplot(catalog.to_pandas(), corner=True, y_vars=three, x_vars=["time_evolution"], hue="repeater_name")
    # plt.show()
    # catalog = catalog.with_columns(
    #     pl.when(pl.col("bonsai_dm").is_null()).then(pl.col('DM')).otherwise(pl.col("bonsai_dm")).alias("DM"),
    #     pl.when(pl.col("ra_1").is_null()).then(pl.when(pl.col('ra').is_null()).then(pl.col("ra_2")).otherwise(pl.col("ra"))).otherwise(pl.col("ra_1")).alias("RA"),
    #     pl.when(pl.col("dec_1").is_null()).then(pl.when(pl.col('dec').is_null()).then(pl.col("dec_2")).otherwise(pl.col("dec"))).otherwise(pl.col("dec_1")).alias("DEC"),
    # )


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
