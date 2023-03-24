from typing import List, Optional
import numpy as np

import h5py
import pandas as pd

from sarjana import signal

def bin_freq_channels(data: np.ndarray, fbin_factor=4) -> np.ndarray:
    num_chan = data.shape[0]
    if num_chan % fbin_factor != 0:
        raise ValueError("frequency binning factor `fbin_factor` should be even")
    data = np.nanmean(data.reshape((num_chan // fbin_factor, fbin_factor) + data.shape[1:]), axis=1)
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
        Read Waterfall data from parquet.
        """
        self.filename = filename
        self.dataframe = pd.read_parquet(filename, columns=columns, engine="pyarrow")
        self._unpack()
        self.wfall_dimension = "freq", "time"

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
        ) = signal.remove_radio_frequency_interference(
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
                ) = signal.remove_radio_frequency_interference(
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
