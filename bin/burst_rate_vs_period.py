import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15

data = pd.read_csv("./FRB20180916B-subwindows.csv")

Yval = "tau"
Ylabel = r"$\frac{\tau}{n - 1}$ / Period"
group = "this work"
# plt.scatter(
#     x=data[data["source"] == group]["period"],
#     y=data[data["source"] == group]["burst_rate"],
# )

plt.figure(figsize=(10, 7))
for i, subinterval in enumerate(
    data[(data["source"] == group)]["subinterval"].unique()
):
    plt.errorbar(
        x=data[(data["source"] == group) & (data["subinterval"] == subinterval)][
            "period"
        ],
        y=1/(data[(data["source"] == group) & (data["subinterval"] == subinterval)][
            "period"
        ]/(data[(data["source"] == group) & (data["subinterval"] == subinterval)][
            Yval
        ]/(data[(data["source"] == group) & (data["subinterval"] == subinterval)][
            "n"
        ]-1))),
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

plt.axvline(data[(data["source"] == group)]["period"].mean(), linestyle="-.")
# CHIME/FRB 2020
plt.axvline(16.35, color="green", alpha=0.5)
plt.axvspan(16.35 - 0.15, 16.35 + 0.15, color="green", alpha=0.2, hatch="|")
# Sand et al 2023
plt.axvline(16.34, color="orange", alpha=0.5)
plt.axvspan(16.35 - 0.07, 16.35 + 0.07, color="orange", alpha=0.2, hatch="/")
plt.legend(title="Observation window")
plt.xlabel("Period (d)")
plt.title("FRB20180916B")
plt.ylabel(Ylabel)
# plt.xlim(15.99, 16.7)
plt.savefig(f"../output/2024-08-04/FRB20180916B/{Yval}-vs-period.png")
