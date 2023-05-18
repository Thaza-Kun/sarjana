from typing import Optional, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sarjana.signal.properties import (
    find_peaks,
)
from sarjana.signal.optimize import fit_time_series
from sarjana.signal.properties import full_width_nth_maximum, is_multipeak
from sarjana.signal.template import scattered_gaussian_signal


def get_value_at_non_integer_index(index: float, initial: float, delta: float) -> float:
    return initial + (index * delta) - ((index * delta) % delta) + delta


def plot_frequency_flux(
    flux: pd.Series,
    model_flux: pd.Series,
    frequencies: pd.Series,
    *,
    axes: Optional[plt.Axes] = None,
    highlight_burst: bool = False,
    **kwargs,
):
    axes = plt.gca() if axes is None else axes
    _freq: np.ndarray = frequencies.to_numpy()[0]
    _model_flux: np.ndarray = model_flux.to_numpy()[0]
    _flux: np.ndarray = flux.to_numpy()[0]
    _multipeak: bool = is_multipeak(_flux)
    g = sns.lineplot(
        x=_freq,
        y=_flux,
        ax=axes,
        color="tab:grey",
    )
    sns.lineplot(x=_freq, y=_model_flux, ax=g, color="black")
    g.set(xlim=[_freq.min(), _freq.max()])
    return g


def plot_time_flux(
    flux: pd.Series,
    model_flux: pd.Series,
    time: pd.Series,
    timedelta: pd.Series,
    *,
    axes: Optional[plt.Axes] = None,
    highlight_burst: bool = False,
    **kwargs,
) -> Any:
    """A single timeseries flux plot.

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
    if isinstance(time, pd.Series):
        _time: np.ndarray = time.to_numpy()[0]
        _model_flux: np.ndarray = model_flux.to_numpy()[0]
        _flux: np.ndarray = flux.to_numpy()[0]
        _timedelta: float = timedelta.to_numpy()[0]
    else:
        _time = time
        _model_flux = model_flux
        _flux = flux
        _timedelta = timedelta
    _multipeak: bool = is_multipeak(_flux)

    # Resize time
    _time = _time - _time[np.argmax(_flux)] - (_timedelta / 2)

    # Add one more step after final point
    _time = np.append(_time, _time[-1] + timedelta)
    _flux = np.append(_flux, _flux[-1])
    _model_flux = np.append(_model_flux, _model_flux[-1])

    g = sns.lineplot(
        x=_time,
        y=_flux,
        drawstyle="steps-post",
        ax=axes,
        color="tab:grey",
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
                params={"amplitude": _flux.max(), "peak_time": _time[_flux.argmax()]},
            )
        except RuntimeError:
            params = fit_time_series(
                scattered_gaussian_signal,
                _time,
                _model_flux,
                params={"amplitude": _flux.max(), "peak_time": _time[_flux.argmax()]},
            )
        except not RuntimeError as e:
            raise e
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
    g.set(xlim=[_time[0], _time[-1]])

    if highlight_burst:
        # Color the burst event with a greyish box
        var = np.nanstd(np.diff(_model_flux))
        peaks, _ = find_peaks(_model_flux, prominence=var)
        # Find burst
        widths, width_heights, lefts, rights = full_width_nth_maximum(
            _model_flux, peaks, n=10
        )
        for i, spike in enumerate(peaks):
            g.axvline(_time[spike] + 0.5 * _timedelta, color="red", linestyle=":")
            g.axvspan(
                max(
                    _time.min(),
                    get_value_at_non_integer_index(
                        index=lefts[i], initial=_time.min(), delta=_timedelta
                    ),
                ),
                min(
                    _time.max(),
                    get_value_at_non_integer_index(
                        index=rights[i], initial=_time.min(), delta=_timedelta
                    ),
                ),
                facecolor="tab:blue",
                edgecolor=None,
                alpha=0.1,
            )

    return g
