from typing import Tuple

import numpy as np
import scipy

from scipy.signal import find_peaks


def remove_radio_frequency_interference(
    spec: np.ndarray, wfall: np.ndarray, model_wfall: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Set any frequency channel that has a higher variance than
    the mean variance (averaged across all frequency channels)
    to a np.nan.

    TODO DOCS FROM CFOD

    Parameters
    ----------
    spec : np.ndarray
    wfall : np.ndarray (2D array)
    model_wfall : np.ndarray (2D array)

    Returns
    -------
    Tuple[
        spec : np.ndarray
        wfall : np.ndarray (2D array)
        model_wfall : np.ndarray (2D array)
    ]
    """
    q1 = np.nanquantile(spec, 0.25)
    q3 = np.nanquantile(spec, 0.75)
    iqr = q3 - q1

    # additional masking of channels with RFI
    rfi_masking_var_factor = 3

    channel_variance = np.nanvar(wfall, axis=1)
    mean_channel_variance = np.nanmean(channel_variance)

    with np.errstate(invalid="ignore"):
        rfi_mask = (
            (channel_variance > rfi_masking_var_factor * mean_channel_variance)
            | (spec[::-1] < q1 - 1.5 * iqr)
            | (spec[::-1] > q3 + 1.5 * iqr)
        )
    wfall[rfi_mask, ...] = np.nan
    model_wfall[rfi_mask, ...] = np.nan
    spec[rfi_mask[::-1]] = np.nan

    return spec, wfall, model_wfall


def boxcar_kernel(width: int):
    """
    Returns the boxcar kernel of given width normalized by
    sqrt(width) for S/N reasons.

    TODO: DOCS FROM CFOD

    Parameters
    ----------
    width : int
        Width of the boxcar.
    Returns
    -------
    boxcar : np.ndarray
        Boxcar of width `width` normalized by sqrt(width).
    """
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_full_width_nth_maximum(y: np.ndarray, peaks: np.ndarray, n: float):
    return scipy.signal.peak_widths(y, peaks, rel_height=1 - (1 / n))


def find_burst(
    timeseries: np.ndarray, min_width: int = 1, max_width: int = 128
) -> Tuple[int, int, float]:
    """
    Find burst peak and width using boxcar convolution.

    TODO: DOCS FROM CFOD

    Parameters
    ----------
    ts : array_like
        Time-series.
    min_width : int, optional
        Minimum width to search from, in number of time samples.
        1 by default.
    max_width : int, optional
        Maximum width to search up to, in number of time samples.
        128 by default.

    Returns
    -------
    peak : int
        Index of the peak of the burst in the time-series.
    width : int
        Width of the burst in number of samples.
    snr : float
        S/N of the burst.

    """
    min_width = int(min_width)
    max_width = int(max_width)
    # do not search widths bigger than timeseries
    widths = list(range(min_width, min(max_width + 1, len(timeseries) - 2)))
    # envelope finding
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)
    for i in range(len(widths)):
        convolved = scipy.signal.convolve(
            timeseries, boxcar_kernel(widths[i]), mode="same"
        )
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]
    best_idx = np.nanargmax(snrs)
    return peaks[best_idx], widths[best_idx], snrs[best_idx]


def is_multipeak(timeseries: np.ndarray, prominence: str = "stdev") -> bool:
    if prominence == "stdev":
        nvar = np.nanstd(np.diff(timeseries))
    peak, _ = find_peaks(timeseries, prominence=nvar)
    return len(peak) != 1


def gauss(x: np.ndarray, amplitude: float, center: float, sigma: float):
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def scattered_gaussian_signal(
    time: np.ndarray,
    sigma: float,
    tau: float,
    amplitude: float = 1.0,
    peak_time: float = 0.0,
) -> np.ndarray:
    """The convolution between a gaussian distribution and a scattering function,
    $$
        \frac{e^{-t//\tau}}{\tau}U(t),
    $$
    where $U(t)$ is a step function which constrains the equation to time t > 0.
    The final equation is,
    $$
        I(t) = \frac{1}{2}\exp{-\frac{t}{\tau}
            + \frac{\sigma^2}{2\tau^2}}
                ( 1 + \text{erf} (
                    \frac{1}{\sqrt{2}\sigma} - \frac{\sigma}{\sqrt{2}\tau}
                    )
                )
    $$
    as described in [Oswald et. al. 2021](https://doi.org/10.1093/mnras/stab980).

    Args:
        time (np.ndarray): an array of observation time.
        sigma (float): the standard deviation of the data (the gaussian component)
        tau (float): the scattering time-scale (the scattering component)
        amplitude (float, optional): the amplitude of the signal. Defaults to 1.0.
        peak_time (float, optional): the peak of the signal. Defaults to 0.0.

    Returns:
        np.ndarray: y data
    """
    _time = time - peak_time
    first_term: np.ndarray = -(_time / tau) + (sigma**2) / (2 * (tau**2))
    second_term: np.ndarray = (_time / (np.sqrt(2) * sigma)) - (
        sigma / (np.sqrt(2) * tau)
    )
    return amplitude * 0.5 * np.exp(first_term) * (1 + scipy.special.erf(second_term))
