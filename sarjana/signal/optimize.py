from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy

from sarjana.signal.template import gauss_2d

# TODO wrapper
# def fit_template(func):
#     def wrapper(*args, **kwargs):
#         return
#     return


def fit_time_series(
    func: Callable, time: np.ndarray, data: np.ndarray, params: Optional[dict] = None
) -> Dict[str, Tuple[float, float]]:
    """Fit the timeseries data to a given function.

    Args:
        func (Callable): the function to be used for fitting
        time (np.ndarray): the observation time
        data (np.ndarray): data to fit
        params (Optional[dict], optional): initial guess for the function parameters. Defaults to None.

    Returns:
        Dict[str, Tuple[float, float]]: {'param': (value, stdev)}
    """
    params_keys: List[str] = [*deepcopy(func.__annotations__).keys()][:-1]
    guesses = {key: 1 for key in params_keys[1:]}
    for key in params.keys():
        guesses[key] = params[key]
    _params, _pcov = scipy.optimize.curve_fit(func, time, data, p0=[*guesses.values()])
    _stdevs = np.sqrt(np.diag(_pcov))
    return {
        key: (optimal, stdev)
        for key, optimal, stdev in zip(params_keys, _params, _stdevs)
    }


def fit_tilted_2d_gaussian(
    xy: Tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    tilt_range: np.ndarray,
    sigma: Tuple[float, float] = (5, 10),
):
    vec = []
    for theta in tilt_range:
        print(f"{theta=}", end="\r")
        z_test = gauss_2d(
            xy,
            amplitude=np.nanmax(z),
            sigma=sigma,
            theta=theta,
            z_offset=np.nanmedian(z),
        )
        score = (z_test - z) ** 2
        vec.append(np.nansum(score))
    return vec


def fit_2d(
    func: Callable[..., np.ndarray],
    xy: Tuple[np.ndarray, np.ndarray],
    z: np.ndarray,
    params: Optional[dict] = None,
) -> Dict[str, Tuple[float, float]]:
    def flattened_func(*args, **kwargs):
        return func(*args, **kwargs).ravel()
    z = np.nan_to_num(z)
    params_keys: List[str] = [*deepcopy(func.__annotations__).keys()][:-1]
    guesses = {key: 1 for key in params_keys[1:]}
    for key in params.keys():
        guesses[key] = params[key]
    _params, _pcov = scipy.optimize.curve_fit(
        flattened_func, xy, z.ravel(), p0=[*guesses.values()]
    )
    _stdevs = np.sqrt(np.diag(_pcov))
    return {
        key: (optimal, stdev)
        for key, optimal, stdev in zip(params_keys, _params, _stdevs)
    }
