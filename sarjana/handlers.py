from typing import List, Optional
import numpy as np

import h5py
import pandas as pd

class ParquetWaterfall:
    def __init__(self, filename: str, columns: Optional[List[str]] = None) -> None:
        """
        Read Waterfall data from parquet.
        """
        self.filename = filename
        self.dataframe = pd.read_parquet(filename, columns=None, engine='pyarrow')
        self._unpack()

    def __getitem__(self, column: str):
        if (series := self.dataframe.get(column)) is None:
            return None
        return series.item()
        

    def _unpack(self) -> None:
        # for key in self.dataframe.keys():
        #     setattr(self, key, self.dataframe[key].item())
        self.eventname = self['eventname']
        self.wfall = self['wfall']
        self.model_wfall = self['model_wfall']
        self.plot_time = self['plot_time']
        self.plot_freq = self['plot_freq']
        self.ts = self['ts']
        self.model_ts = self['model_ts']
        self.spec = self['spec']
        self.model_spec = self['model_spec']
        self.extent = self['extent']
        self.dm = self['dm']
        self.scatterfit = self['scatterfit']
        self.dt = self['dt']
        self.wfall_shape = self['wfall_shape']
        self.wfall = self['wfall']
        self.model_wfall = self['model_wfall']
        self.cal_wfall_shape = self['cal_wfall_shape']
        self.cal_wfall = self['cal_wfall']


class H5Waterfall:
    def __init__(self, filename: str) -> None:
        """
        Initialize the Waterfaller.

        Parameters
        ----------
        filename : str
            h5 file, container the CHIME/FRB waterfall data.
        """
        self.filename = filename
        self.datafile = h5py.File(filename, "r")
        self._unpack()
        self.dataframe = pd.DataFrame([self.__dict__])

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

        self.wfall_shape = self.wfall.shape
        self.wfall = self.wfall.reshape((-1,))
        self.model_wfall = self.model_wfall.reshape((-1,))
        self.cal_wfall_shape = (
            self.cal_wfall.shape if getattr(self, "cal_wfall", None) else None
        )
        self.cal_wfall = (
            self.cal_wfall.reshape((-1,)) if getattr(self, "cal_wfall", None) else None
        )
