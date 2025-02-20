#! usr/bin/env python

# Author: Murthadza Aznam
# Date: 2024-07-05
# Python Version: 3.12

"""Get exposure data from numpy file at a given index
"""

import argparse
import datetime
import pathlib

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from cdshealpix.ring import skycoord_to_healpix, vertices_skycoord
from userust import load_many_numpyz

import tqdm

# format: "exposure_{YYMMDD:start}_{YYMMDD:end}_beam_FWHM-600_res_{time resolution}s_0.86_arcmin.npz"
filename = "exposure_{}_{}_transit_{}_beam_FWHM-600_res_{}s_0.86_arcmin.npz"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ra", type=float, required=True)
    parser.add_argument("--nra", action="store_true")
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--ndec", action="store_true")
    parser.add_argument("--nside", type=int, help="Nside", default=4096)
    parser.add_argument("--dir", type=pathlib.Path, required=True)
    parser.add_argument("--at", type=datetime.datetime.fromisoformat, required=True)
    parser.add_argument("--res", type=float, help="Time resolution (in sec)", default=4)
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    RESOLUTION = arguments.res
    at: datetime.datetime = arguments.at
    dec = -arguments.dec if arguments.ndec else arguments.dec
    ra = -arguments.ra if arguments.nra else arguments.ra

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    hpx_idx = skycoord_to_healpix(coord, nside=arguments.nside)
    print(hpx_idx)

    files_U = [
        pathlib.Path(
            arguments.dir,
            filename.format(
                at.strftime("%Y%m%d"),
                (at + datetime.timedelta(days=1)).strftime("%Y%m%d"),
                "U",
                RESOLUTION,
            ),
        ).as_posix()
    ]
    files_L = [
        pathlib.Path(
            arguments.dir,
            filename.format(
                at.strftime("%Y%m%d"),
                (at + datetime.timedelta(days=1)).strftime("%Y%m%d"),
                "L",
                RESOLUTION,
            ),
        ).as_posix()
    ]

    print("Loading files")
    exposures_U = (
        np.array(load_many_numpyz(files_U, "exposure", hpx_idx[0] + 1))
        * RESOLUTION
        / 3600.0
    )
    exposures_L = (
        np.array(load_many_numpyz(files_L, "exposure", hpx_idx[0] + 1))
        * RESOLUTION
        / 3600.0
    )
    print(exposures_U.shape)
    print(exposures_L.shape)
    print(
        pd.DataFrame(
            {
                "date": at,
                "exposure_U": np.round(exposures_U, 3),
                "exposures_L": np.round(exposures_L, 3),
            }
        )
    )


if __name__ == "__main__":
    print(__doc__)
    arguments = parse_arguments()
    main(arguments)
