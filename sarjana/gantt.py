from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import datetime as dt

sns.set_theme()


def generate_gantt(
    progress: pd.DataFrame,
    milestones: Optional[pd.DataFrame] = None,
    *,
    label: Optional[str] = None,
    show: bool = False,
    savefile: Optional[str] = None,
) -> None:
    progress["start"] = pd.to_datetime(progress["start"])
    progress["end"] = pd.to_datetime(progress["end"])

    # Add duration
    progress.loc[:, ["duration"]] = (progress.end - progress.start).apply(
        lambda x: x.days + 1
    )

    # Sort in ascending order
    progress = progress.sort_values(by="start", ascending=True)

    thesis_start = progress["start"].min()
    thesis_end = progress["end"].max()
    thesis_duration = (thesis_end - thesis_start).days + 1

    # Add relative start
    progress.loc[:, ["rel_start"]] = progress["start"].apply(
        lambda x: (x - thesis_start).days
    )

    # Create custom x-ticks and x-tick labels
    x_ticks = [i for i in range(thesis_duration + 1)]
    x_labels = [(thesis_start + dt.timedelta(days=i)) for i in x_ticks]

    ticks = pd.DataFrame(data={"ticks": x_ticks, "labels": x_labels})
    tick_filter = ticks["labels"].apply(lambda x: x.is_month_start)
    tick_interval = ticks[tick_filter]["ticks"]
    tick_labels = ticks[tick_filter]["labels"].apply(lambda x: x.strftime("%b %Y"))

    now = (pd.Timestamp.today() - thesis_start).days

    palette = sns.color_palette()

    legend_colors = {
        phase: palette[idx] for idx, phase in enumerate(progress["phase"].unique())
    }

    progress["color"] = [legend_colors[key] for key in progress["phase"]]

    plt.figure(figsize=(15, 6))
    if label is not None:
        plt.title("Masters Study Progress ({})".format(label))
    else:
        plt.title("Masters Study Progress")
    plt.barh(
        y=progress["task"],
        left=progress["rel_start"],
        width=progress["duration"],
        color=progress["color"],
        label=progress["phase"],
    )
    plt.gca().invert_yaxis()
    plt.axvline(x=now, color="orange")
    plt.xticks(ticks=tick_interval, labels=tick_labels, rotation=90)
    plt.grid(axis="y")

    # Legends
    legend_elems = [
        Patch(facecolor=legend_colors[key], label=key) for key in legend_colors
    ]
    plt.legend(handles=legend_elems)

    if milestones is not None:
        milestones["date"] = pd.to_datetime(milestones["date"])
        milestones.loc[:, ["rel_date"]] = milestones["date"].apply(
            lambda x: (x - thesis_start).days
        )
        locations = [3, 4, 6, 7]
        idx = 0
        thesis_writing_location = len(progress["task"].unique()) - 1
        for date, name in zip(milestones["rel_date"], milestones["name"]):
            plt.plot(
                date,
                locations[idx],
                marker="*",
                color="#fdfd96",
                markersize=23,
                markeredgecolor="#c23b23",
                markeredgewidth=2,
            )
            center = int((len(name) * 5) / 2)
            plt.text(
                date - center,
                locations[idx] - 0.7,
                name,
            )
            idx += 1

    plt.tight_layout()

    if show:
        plt.show()

    if savefile:
        plt.savefig(savefile)
