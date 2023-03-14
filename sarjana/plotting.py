from typing import Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sarjana.signal import find_burst


def plot_flux_profile(
    flux: pd.Series,
    model_flux: pd.Series,
    time: pd.Series,
    timedelta: pd.Series,
    axes: Optional[plt.Axes] = None,
    **kwargs,
) -> Any:
    """A single flux plot.

    Args:
        flux (pd.Series): Flux data as a time series
        model_flux (pd.Series): Denoised flux data as a time series
        time (pd.Series): Time axis
        timedelta (pd.Series): Differences between two units of time
        eventname (pd.Series): The name of the FRB
        axes (Optional[plt.Axes], optional): The axes to draw to. If None, it queries `plt.gca()`. Defaults to None.

    Returns:
        Any: A Seaborn plot.
    """
    axes = plt.gca() if axes is None else axes
    # Reshape data
    _time: np.ndarray = time.to_numpy()[0]
    _model_flux: np.ndarray = model_flux.to_numpy()[0]
    _flux: np.ndarray = flux.to_numpy()[0]
    _timedelta: float = timedelta.to_numpy()[0]

    # Find burst
    peak, width, _ = find_burst(_flux)

    # Resize time
    _time = _time - _time[np.argmax(_flux)]
    _time = _time - (_timedelta / 2)

    # Add one more step after final point
    _time = np.append(_time, _time[-1] + timedelta)
    _flux = np.append(_flux, _flux[-1])
    _model_flux = np.append(_model_flux, _model_flux[-1])

    g = sns.lineplot(
        x=_time,
        y=_flux,
        drawstyle="steps-post",
        ax=axes,
        color="orange",
    )
    sns.lineplot(x=_time, y=_model_flux, drawstyle="steps-post", ax=g)
    g.set(xlim=[_time[0], _time[-1]])

    # Color the burst event with a greyish box
    g.axvspan(
        max(
            _time.min(),
            _time[peak] + 0.5 * _timedelta - (0.5 * width) * _timedelta,
        ),
        min(
            _time.max(),
            _time[peak] + 0.5 * _timedelta + (0.5 * width) * _timedelta,
        ),
        facecolor="tab:blue",
        edgecolor=None,
        alpha=0.1,
    )
    return g
