from typing import List, Optional
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import h5py
import pandas as pd

from sarjana.signal import transform, properties
from scipy.signal import peak_widths


def bin_freq_channels(data: np.ndarray, fbin_factor=4) -> np.ndarray:
    num_chan = data.shape[0]
    if num_chan % fbin_factor != 0:
        raise ValueError("frequency binning factor `fbin_factor` should be even")
    data = np.nanmean(
        data.reshape((num_chan // fbin_factor, fbin_factor) + data.shape[1:]), axis=1
    )
    return data


class CSVCatalog:
    def __init__(self, filename: str) -> None:
        self.dataframe = pd.read_csv(filename)


class ParquetWaterfall:
    def __init__(
        self,
        filename: str,
        columns: Optional[List[str]] = None,
    ) -> None:
        """
        Read Waterfall data from parquet and remove RFI.
        """
        self.filename = filename
        self.dataframe = pd.read_parquet(filename, columns=columns, engine="pyarrow")
        self._unpack()
        self.remove_rfi()
        self.wfall_dimension = "freq", "time"

    def plot_wfall(self, scatterfit: bool = False) -> "ParquetWaterfall":
        self.remove_rfi()
        extent = [*self.extent]
        fig = plt.figure(figsize=(6,6))
        ## Set up the image grid
        gs = gridspec.GridSpec(ncols=2, nrows=2, figure=fig, width_ratios=[3, 1],
                                height_ratios=[1, 3], hspace=0.0, wspace=0.0)
        data_im = plt.subplot(gs[2])
        data_ts = plt.subplot(gs[0], sharex=data_im)
        data_spec = plt.subplot(gs[3], sharey=data_im)


        ### time stamps relative to the peak
        peak_idx = np.argmax(self.ts)
        extent[0] = self.extent[0] - self.plot_time[peak_idx]
        extent[1] = self.extent[1] - self.plot_time[peak_idx]
        plot_time = self.plot_time.copy() - self.plot_time[peak_idx]

        # prepare time-series for histogramming
        plot_time -= self.dt / 2.
        plot_time = np.append(plot_time, plot_time[-1] + self.dt)

        cmap = plt.cm.viridis

        ### plot dynamic spectrum
        self.wfall[np.isnan(self.wfall)] = np.nanmedian(self.wfall)   # replace nans in the data with the data median
        # use standard deviation of residuals to set color scale
        vmin = np.nanpercentile(self.wfall, 1)
        vmax = np.nanpercentile(self.wfall, 99)

        data_im.imshow(self.wfall, aspect="auto", interpolation="none",
                        extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)

        ### plot time-series
        data_ts.plot(plot_time, np.append(self.ts, self.ts[-1]), color="tab:gray",
                        drawstyle="steps-post")

        ### plot spectrum
        data_spec.plot(self.spec, self.plot_freq, color="tab:gray")

        ### plot model time-series and spectrum
        if scatterfit:
            data_spec.plot(self.model_spec, self.plot_freq, color=cmap(0.25))
            data_ts.plot(plot_time, np.append(self.model_ts, self.model_ts[-1]),
                            color=cmap(0.25), drawstyle="steps-post", lw=2)
        else:
            data_spec.plot(self.model_spec, self.plot_freq, color=cmap(0.5))
            data_ts.plot(plot_time, np.append(self.model_ts, self.model_ts[-1]),
                            color=cmap(0.5), drawstyle="steps-post", lw=1)


        ## BEautify plot
        # remove some labels and ticks for neatness
        plt.setp(data_ts.get_xticklabels(), visible=False)
        data_ts.set_yticklabels([], visible=True)
        data_ts.set_yticks([])
        data_ts.set_xlim(extent[0], extent[1])
        plt.setp(data_spec.get_yticklabels(), visible=False)
        data_spec.set_xticklabels([], visible=True)
        data_spec.set_xticks([])
        data_spec.set_ylim(extent[2], extent[3])
        plt.setp(data_im.get_xticklabels(), fontsize=9)
        plt.setp(data_im.get_yticklabels(), fontsize=9)


        # #### highlighting the width of the pulse
        # data_ts.axvspan(max(plot_time.min(),
        #                         plot_time[peak] + 0.5 * self.dt \
        #                         - (0.5 * width) * self.dt),
        #                     min(plot_time.max(),
        #                         plot_time[peak] + 0.5 * self.dt \
        #                         + (0.5 * width) * self.dt),
        #                     facecolor="tab:blue", edgecolor=None, alpha=0.1)


        ##### add event ID and DM labels
        xlim = data_ts.get_xlim()
        ylim = data_ts.get_ylim()

        # add 20% extra white space at the top
        span = np.abs(ylim[1]) + np.abs(ylim[0])
        data_ts.set_ylim(ylim[0], ylim[1] + 0.2 * span)
        ylim = data_ts.get_ylim()

        ypos = (ylim[1] - ylim[0]) * 0.9 + ylim[0]
        xpos = (xlim[1] - xlim[0]) * 0.98 + self.extent[0]
        # data_ts.text(xpos, ypos, "{}\nDM: {:.1f} pc/cc\nSNR: {:.2f}".format(self.eventname, self.dm,self.snr), ha="right",
        data_ts.text(xpos, ypos, "{}\nDM: {:.1f} pc/cc".format(self.eventname, self.dm), ha="right",
                        va="top", fontsize=9)

        data_im.locator_params(axis="x", min_n_ticks=3)
        data_im.set_yticks([400, 500, 600, 700, 800])
        data_im.set_ylabel("Frequency [MHz]", fontsize=9)
        data_im.set_xlabel("Time [ms]", fontsize=9)


    def transpose(self) -> "ParquetWaterfall":
        self.wfall = np.transpose(self.wfall)
        self.cal_wfall = np.transpose(self.cal_wfall)
        self.wfall_dimension = self.wfall_dimension[::-1]
        return self

    def __getitem__(self, column: str):
        if (series := self.dataframe.get(column)) is None:
            return None
        return series.item()

    def remove_rfi(self) -> "ParquetWaterfall":
        (
            self.spec,
            self.wfall,
            self.model_wfall,
        ) = transform.remove_radio_frequency_interference(
            self.spec, self.wfall, self.model_wfall
        )
        self.ts = np.nansum(self.wfall, axis=0)
        self.model_ts = np.nansum(self.model_wfall, axis=0)
        self.wfall = bin_freq_channels(self.wfall, 32)
        self.wfall[np.isnan(self.wfall)] = np.nanmedian(self.wfall)
        self.model_wfall[np.isnan(self.model_wfall)] = np.nanmedian(self.model_wfall)
        self.no_rfi = True
        return self

    def pack(
        self, wfall: Optional[str] = None, remove_interference: bool = False
    ) -> pd.DataFrame:
        if remove_interference:
            if self.no_rfi is False:
                self.remove_rfi()
        if wfall == "remove":
            wfall_cols = [
                "wfall",
                "model_wfall",
                "cal_wfall",
                "_wfall_shape",
                "_cal_wfall_shape",
            ]
            for col in wfall_cols:
                self.__dict__.pop(col, None)
        elif wfall == "flatten":
            self.wfall_shape = self.wfall.shape
            self.wfall = self.wfall.reshape((-1,))
            self.model_wfall = self.model_wfall.reshape((-1,))
            self.cal_wfall_shape = (
                self.cal_wfall.shape if getattr(self, "cal_wfall", None) else None
            )
            self.cal_wfall = (
                self.cal_wfall.reshape((-1,))
                if getattr(self, "cal_wfall", None)
                else None
            )
        elif wfall is None:
            pass
        else:
            raise KeyError("wfall strategy not recognized")
        self.__dict__.pop("dataframe", None)
        return pd.DataFrame([self.__dict__])

    def _unpack(self) -> None:
        self._wfall_shape = self["wfall_shape"]
        self._cal_wfall_shape = self["cal_wfall_shape"]

        self.eventname = self["eventname"]
        self.wfall = self["wfall"]
        self.model_wfall = self["model_wfall"]
        self.plot_time = self["plot_time"]
        self.plot_freq = self["plot_freq"]
        self.ts = self["ts"]
        self.model_ts = self["model_ts"]
        self.spec = self["spec"]
        self.model_spec = self["model_spec"]
        self.extent = self["extent"]
        self.dm = self["dm"]
        self.scatterfit = self["scatterfit"]
        self.dt = self["dt"]
        self.wfall = np.reshape(self["wfall"], self._wfall_shape)
        self.model_wfall = np.reshape(self["model_wfall"], self._wfall_shape)
        self.cal_wfall = np.reshape(self["cal_wfall"], self._cal_wfall_shape)

        self.peaks, *_ = properties.find_peaks(
            deepcopy(self.model_ts), prominence=np.nanstd(np.diff(self.model_ts))
        )
        self.widths, *_ = peak_widths(deepcopy(self.model_ts), self.peaks)


class H5Waterfall:
    def __init__(self, filename: str) -> None:
        """
        Initialize the Waterfaller.

        TODO: DOCS FROM CFOD

        Parameters
        ----------
        filename : str
            h5 file, container the CHIME/FRB waterfall data.
        """
        self.filename = filename
        self.datafile = h5py.File(filename, "r")
        self._unpack()
        self.dataframe = pd.DataFrame([self.__dict__])

    def pack(
        self, wfall: Optional[str] = None, remove_interference: bool = False
    ) -> pd.DataFrame:
        if remove_interference:
            if self.no_rfi is False:
                (
                    self.spec,
                    self.wfall,
                    self.model_wfall,
                ) = transform.remove_radio_frequency_interference(
                    self.spec, self.wfall, self.model_wfall
                )
                self.ts = np.nansum(self.wfall, axis=0)
                self.model_ts = np.nansum(self.model_wfall, axis=0)
                self.no_rfi = True
        if wfall == "remove":
            wfall_cols = ["wfall", "model_wfall", "cal_wfall"]
            for col in wfall_cols:
                self.__dict__.pop(col, None)
        elif wfall == "flatten":
            self.wfall_shape = self.wfall.shape
            self.wfall = self.wfall.reshape((-1,))
            self.model_wfall = self.model_wfall.reshape((-1,))
            self.cal_wfall_shape = (
                self.cal_wfall.shape if getattr(self, "cal_wfall", None) else None
            )
            self.cal_wfall = (
                self.cal_wfall.reshape((-1,))
                if getattr(self, "cal_wfall", None)
                else None
            )
        elif wfall is None:
            pass
        else:
            raise KeyError("wfall strategy not recognized")
        return pd.DataFrame([self.__dict__])

    def _unpack(self) -> None:
        unnecessary_metadata = ["filename", "datafile"]
        self.datafile = self.datafile["frb"]
        self.eventname = self.datafile.attrs["tns_name"].decode()
        self.wfall = self.datafile["wfall"][:]
        self.model_wfall = self.datafile["model_wfall"][:]
        self.plot_time = self.datafile["plot_time"][:]
        self.plot_freq = self.datafile["plot_freq"][:]
        self.ts = self.datafile["ts"][:]
        self.model_ts = self.datafile["model_ts"][:]
        self.spec = self.datafile["spec"][:]
        self.model_spec = self.datafile["model_spec"][:]
        self.extent = self.datafile["extent"][:]
        self.dm = self.datafile.attrs["dm"][()]
        self.scatterfit = self.datafile.attrs["scatterfit"][()]
        self.dt = np.median(np.diff(self.plot_time))
        for metadata in unnecessary_metadata:
            self.__dict__.pop(metadata, None)
