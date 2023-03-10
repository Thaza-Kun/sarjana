import numpy as np

import h5py


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

    def _unpack(self):
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
