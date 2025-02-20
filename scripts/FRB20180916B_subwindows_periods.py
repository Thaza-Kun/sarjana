# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
# ]
# ///
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 2*12

data = pd.read_csv("./FRB20180916B_subwindows_periods.csv")

outdir = "../output/2024-09-07"

Yval = "n"
Ylabel = "Event count"
# Yval = "tau"
# Ylabel = r"$\frac{\tau}{n - 1}$ / Period"
# Ylabel = "D"
group = "this work"

plt.figure(figsize=(2*6.52, 2*4.15))
for i, subinterval in enumerate(
    data[(data["source"] == group)]["subinterval"].unique()
):
    plt.errorbar(
        x=data[(data["source"] == group) & (data["subinterval"] == subinterval)][
            "period"
        ],
        y = data[(data["source"] == group) & (data["subinterval"] == subinterval)][Yval],
        # y=1
        # / (
        #     data[(data["source"] == group) & (data["subinterval"] == subinterval)][
        #         "period"
        #     ]
        #     / (
        #         data[(data["source"] == group) & (data["subinterval"] == subinterval)][
        #             "tau"
        #         ]
        #         / (
        #             data[
        #                 (data["source"] == group) & (data["subinterval"] == subinterval)
        #             ]["n"]
        #             - 1
        #         )
        #     )
        # ),
        xerr=data[(data["source"] == group) & (data["subinterval"] == subinterval)][
            ["period_neg", "period_pos"]
        ].T,
        # fmt="o",
        fmt=["^", "o", "v", "P", "p", "d", "s", "X", "D", "o"][i],
        color=[
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#1f77b4",
            "#d62728",
        ][i],
        label=f"{int(100*subinterval)}%",
        capsize=0.1,
    )

# CHIME/FRB 2020
plt.axvline(16.35, color="green", alpha=0.5)
plt.axvspan(16.35 - 0.15, 16.35 + 0.15, facecolor='None', edgecolor="green", alpha=0.5, hatch="|")
plt.axvspan(16.35 - 0.15, 16.35 + 0.15, color="green", alpha=0.2)

# Sand et al 2023
plt.axvline(16.34, color="orange", alpha=0.5)
plt.axvspan(16.35 - 0.07, 16.35 + 0.07, facecolor='None', edgecolor="orange", alpha=0.5, hatch="/")
plt.axvspan(16.35 - 0.07, 16.35 + 0.07, color="orange", alpha=0.2)
plt.legend(title="Observation\n    Window", bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=2*10)
plt.xlabel("Period (d)")
plt.title("FRB20180916B")
plt.ylabel(Ylabel)
plt.tight_layout()
# plt.xlim(15.99, 16.7)
plt.savefig(f"{outdir}/FRB20180916B/{Yval}-vs-period.pdf")
