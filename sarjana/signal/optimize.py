from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy

from sarjana.signal.template.gauss2d import Gauss2D

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
    sigma: Tuple[float, float] = (5, 10),
    theta: float = 0.0,
) -> Tuple[float, Tuple[float, float]]:
    scores_theta = []
    x, y = xy
    center = (np.nanmin(x) + np.nanmax(x)) / 2, (np.nanmin(y) + np.nanmax(y)) / 2
    gauss = Gauss2D(
        amplitude=np.nanmax(z),
        sigma=sigma,
        theta=theta,
        offset=np.nanmedian(z),
        center=center,
    )
    lambda_parameter = 100
    h_theta = 0.01
    print(gauss.theta)
    print((1 - lambda_parameter) * gauss.d2func_dtheta2(x, y))
    for _ in range(100):
        left_hand_side = (1 - lambda_parameter) * gauss.d2func_dtheta2(x, y)
        right_hand_side = gauss.dfunc_dtheta(x, y) * (
            gauss(x, y) - gauss(x, y, modify={"theta": h_theta})
        )
        h_theta = right_hand_side / left_hand_side
    return gauss.theta, gauss.sigma


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
