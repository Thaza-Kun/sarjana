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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=pathlib.Path, required=True, help="Catalog file")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="Save directory")
    return parser.parse_args()


def main(arguments: argparse.Namespace) -> None:
    names = [
        "FRB20190804E",
        "FRB20190915D",
        "FRB20200223B",
        "FRB20201130A",
        "FRB20201221B",
    ]
    periods = [168.39, 13.94, 25.14, 11.38, 71.02]
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
    three = ["flux", "fluence", "width_fitb", "sp_idx"]
    three_err = ["flux_err", "fluence_err", "width_fitb_err", "sp_idx_err"]
    three_label = ["Flux [Jy]", "Fluence [Jy ms]", "Width [s]", "Spectral index"]
    four = ["bandwidth", "high_freq", "low_freq", "peak_freq"]
    file = arguments.file
    catalog = pl.read_csv(file, null_values="-9999").with_columns(
        (pl.col("high_freq") - pl.col("low_freq")).alias("bandwidth")
    )
    print(catalog.columns)
    catalog = catalog.select(
        pl.col([*one, *two, *three, *three_err, *four]),
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

    for name, period in zip(names, periods):
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(nrows=3, ncols=2)
        f = catalog.filter(pl.col("repeater_name") == name).with_columns(
            (
                np.random.normal(scale=0.001)
                + (((pl.col("time_evolution") * pl.col("diff_mjd")) / period) % 1)
            ).alias("phase")
        )
        for i, (grid, colname) in enumerate(zip(gs, three)):
            a = plt.subplot(grid)
            a.scatter(
                f.select(pl.col("phase")).to_numpy(),
                f.select(pl.col(colname)).to_numpy(),
            )
            if i < len(three_err):
                a.errorbar(
                    f.select(pl.col("phase")).to_numpy().flatten(),
                    f.select(pl.col(colname)).to_numpy().flatten(),
                    yerr=f.select(pl.col(three_err[i])).to_numpy().flatten(),
                    fmt="none",
                )
            a.set_ylabel(three_label[i])
            a.set_xlim(-0.01, 1.01)
        b = plt.subplot(gs[-2:])
        b.errorbar(
            f.select(pl.col("phase")).to_numpy().flatten(),
            f.select(pl.col("peak_freq")).to_numpy().flatten(),
            yerr=(
                f.select(pl.col("peak_freq") - pl.col("low_freq")).to_numpy().flatten(),
                f.select(pl.col("high_freq") - pl.col("peak_freq"))
                .to_numpy()
                .flatten(),
            ),
            fmt="none",
            zorder=1,
        )
        b.scatter(
            f.select(pl.col("phase")).to_numpy().flatten(),
            f.select(pl.col("high_freq")).to_numpy().flatten(),
            marker="_",
            color="C0",
            label="high",
        )
        b.scatter(
            f.select(pl.col("phase")).to_numpy().flatten(),
            f.select(pl.col("peak_freq")).to_numpy().flatten(),
            color="C0",
            label="peak",
        )
        b.scatter(
            f.select(pl.col("phase")).to_numpy().flatten(),
            f.select(pl.col("low_freq")).to_numpy().flatten(),
            marker="^",
            color="C0",
            label="low",
        )
        b.legend(loc="center")
        b.set_ylabel("Frequency [MHz]")
        b.set_ylim(400, 800)
        b.set_xlim(-0.01, 1.01)

        fig.suptitle(name)
        fig.supxlabel("Phase")
        plt.tight_layout()
        plt.savefig(f"{arguments.out}/{name}_evolution_folded.pdf")
        print(f"Saved to: {arguments.out}/{name}_evolution_folded.pdf")

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
