from typing import Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sarjana.signal import (
    find_burst,
    find_peaks,
    scattered_gaussian_signal,
)
from sarjana.optimize import fit_time_series


def plot_flux_profile(
    flux: pd.Series,
    model_flux: pd.Series,
    time: pd.Series,
    timedelta: pd.Series,
    multipeak: pd.Series,
    *,
    axes: Optional[plt.Axes] = None,
    draw_peaks: bool = False,
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
        find_peaks (bool): Whether or not to find peaks

    Returns:
        Any: A Seaborn plot.
    """
    axes = plt.gca() if axes is None else axes
    # Reshape data
    _time: np.ndarray = time.to_numpy()[0]
    _model_flux: np.ndarray = model_flux.to_numpy()[0]
    _flux: np.ndarray = flux.to_numpy()[0]
    _timedelta: float = timedelta.to_numpy()[0]
    _multipeak: str = multipeak.item()

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
    if _multipeak is True:
        sns.lineplot(
            x=_time, y=_model_flux, drawstyle="steps-post", ax=g, label="cfod fit"
        )
    elif _multipeak is False:
        try:
            params = fit_time_series(
                scattered_gaussian_signal,
                _time,
                _flux,
                params=[1, 1, _flux.max(), _time[_flux.argmax()]],
            )
        except RuntimeError:
            params = fit_time_series(
                scattered_gaussian_signal,
                _time,
                _model_flux,
                params=[1, 1, _flux.max(), _time[_flux.argmax()]],
            )
        finally:
            _sigma = params.get("sigma", (1.0,))[0]
            _tau = params.get("tau", (1.0,))[0]
            _model_flux = scattered_gaussian_signal(
                _time, *[i[0] for i in params.values()]
            )
            sns.lineplot(
                x=_time,
                y=_model_flux,
                drawstyle="steps-post",
                color="green",
                ax=g,
                label="convolution",
            )
            axes.text(
                _time[-2],
                _flux.max() * 0.8,
                r"$\frac{\sigma}{\tau}$=" + f"{_sigma/_tau:.1e}",
                horizontalalignment="right",
            )
    elif _multipeak is None:
        ...
    else:
        raise KeyError(f"{_multipeak} is not an option.")
    g.set(xlim=[_time[0], _time[-1]])

    # Color the burst event with a greyish box
    halftick = 0.5 * _timedelta
    g.axvspan(
        max(
            _time.min(),
            _time[peak] + halftick - (0.5 * width) * _timedelta,
        ),
        min(
            _time.max(),
            _time[peak] + halftick + (0.5 * width) * _timedelta,
        ),
        facecolor="tab:blue",
        edgecolor=None,
        alpha=0.1,
    )
    var = np.nanstd(np.diff(_model_flux))
    if draw_peaks:
        peaks, _ = find_peaks(_model_flux, prominence=var)
        for spike in peaks:
            g.axvline(_time[spike] + 0.5 * _timedelta, color="red", linestyle=":")

    return g
