from typing import Tuple

import numpy as np
import scipy


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


def find_burst(
    ts: np.ndarray, min_width: int = 1, max_width: int = 128
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
    widths = list(range(min_width, min(max_width + 1, len(ts) - 2)))
    # envelope finding
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)
    for i in range(len(widths)):
        convolved = scipy.signal.convolve(ts, boxcar_kernel(widths[i]), mode="same")
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]
    try:
        best_idx = np.nanargmax(snrs)
    except ValueError:
        return peaks, widths, snrs
    return peaks[best_idx], widths[best_idx], snrs[best_idx]
