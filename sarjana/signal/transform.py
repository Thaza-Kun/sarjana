import numpy as np
import scipy

import copy

from typing import Tuple

from sarjana.signal.properties import full_width_nth_maximum, find_peaks


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


def autocorrelate_waterfall(wfall: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    timeseries = np.nansum(wfall, axis=0)
    timeseries_var = np.nanstd(np.diff(timeseries))
    peaks, _ = find_peaks(timeseries, prominence=timeseries_var)
    widths = full_width_nth_maximum(timeseries, peaks, n=10)
    width = np.array([*widths]).max()

    peak = peaks[0]
    window = 100

    # increase window for wide bursts
    while width > 0.5 * window:
        window += 100

    sub_factor = 64
    sub = np.nanmean(wfall.reshape(-1, sub_factor, wfall.shape[1]), axis=1)
    median = np.nanmedian(sub)
    sub[sub == 0.0] = median
    sub[np.isnan(sub)] = median

    waterfall = copy.deepcopy(sub[..., peak - window // 2 : peak + window // 2])

    # select noise before (and after) the burst (if necessary)
    noise_window = (peak - 3 * window // 2, peak - window // 2)

    if noise_window[0] < 0:
        difference = abs(noise_window[0])
        noise_window = (noise_window[0] + difference, noise_window[1] + difference)
        noise_waterfall = copy.deepcopy(
            np.roll(sub, difference, axis=1)[..., noise_window[0] : noise_window[1]]
        )
    else:
        noise_waterfall = copy.deepcopy(sub[..., noise_window[0] : noise_window[1]])

    ac2d = scipy.signal.correlate2d(
        waterfall, waterfall, mode="full", boundary="fill", fillvalue=0
    )

    ac2d[ac2d.shape[0] // 2, :] = np.nan
    ac2d[:, ac2d.shape[1] // 2] = np.nan

    scaled_ac2d = copy.deepcopy(ac2d)

    scaling_factor = np.nanmax(scaled_ac2d)
    scaled_ac2d = scaled_ac2d / scaling_factor

    noise_ac2d = scipy.signal.correlate2d(
        noise_waterfall, noise_waterfall, mode="full", boundary="fill", fillvalue=0
    )

    noise_ac2d[noise_ac2d.shape[0] // 2, :] = np.nan
    noise_ac2d[:, noise_ac2d.shape[1] // 2] = np.nan

    scaled_noise_ac2d = copy.deepcopy(noise_ac2d)
    scaled_noise_ac2d = scaled_noise_ac2d / scaling_factor

    return scaled_ac2d, scaled_noise_ac2d
