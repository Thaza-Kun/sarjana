from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy


def fit_time_series(
    func: Callable, time: np.ndarray, data: np.ndarray, params: Optional[list] = None
) -> Dict[str, Tuple[float, float]]:
    """Fit the timeseries data to a given function.

    Args:
        func (Callable): the function to be used for fitting
        time (np.ndarray): the observation time
        data (np.ndarray): data to fit
        params (Optional[list], optional): initial guess for the function parameters. Defaults to None.

    Returns:
        Dict[str, Tuple[float, float]]: {'param': (value, stdev)}
    """
    params_keys: List[str] = [*deepcopy(func.__annotations__).keys()][:-1]
    _params, _pcov = scipy.optimize.curve_fit(func, time, data, p0=params)
    _stdevs = np.sqrt(np.diag(_pcov))
    return {
        key: (optimal, stdev)
        for key, optimal, stdev in zip(params_keys, _params, _stdevs)
    }
