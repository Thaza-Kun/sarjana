from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import datetime as dt

sns.set_theme()

filepath = Path(".", "data", "meta", "progress.csv")
data = pd.read_csv(filepath)

data.start = pd.to_datetime(data.start)
data.end = pd.to_datetime(data.end)

# Add duration
data.loc[:, ["duration"]] = (data.end - data.start).apply(lambda x: x.days + 1)

# Sort in ascending order
data = data.sort_values(by="start", ascending=True)

thesis_start = data.start.min()
thesis_end = data.end.max()
thesis_duration = (thesis_end - thesis_start).days + 1

# Add relative start
data.loc[:, ["rel_start"]] = data.start.apply(lambda x: (x - thesis_start).days)

# Create custom x-ticks and x-tick labels
x_ticks = [i for i in range(thesis_start.day, thesis_duration + 1)]
x_labels = [(thesis_start + dt.timedelta(days=i)) for i in x_ticks]

ticks = pd.DataFrame(data={"ticks": x_ticks, "labels": x_labels})
tick_interval = ticks[ticks.labels.apply(lambda x: x.is_month_start)].ticks
tick_labels = ticks[ticks.labels.apply(lambda x: x.is_month_start)].labels.apply(
    lambda x: x.strftime("%b %Y")
)

now = dt.date.today().day + thesis_start.day

plt.figure(figsize=(15, 6))
plt.title("Masters Study Progress")
plt.barh(
    y=data.task,
    left=data.rel_start,
    width=data.duration,
    color=data.color,
    label=data.phase,
)
plt.gca().invert_yaxis()
plt.axvline(x=now, color="orange")
plt.xticks(ticks=tick_interval, labels=tick_labels, rotation=90)
plt.grid(axis="y")

# Legends
legend_colors = {row.phase: row.color for row in data.itertuples()}
legend_elems = [Patch(facecolor=legend_colors[key], label=key) for key in legend_colors]
plt.legend(handles=legend_elems)

plt.tight_layout()

plt.savefig(Path(".", "_assets", "gantt-chart.png"))
