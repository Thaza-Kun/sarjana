import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u
from cdshealpix.ring import skycoord_to_healpix, vertices_skycoord


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--nside", type=int, help="Nside", default=4096)
    parser.add_argument("--ra", type=float, help="Right ascension", required=True)
    parser.add_argument(
        "--nra", action="store_true", help="Is ra negative?", required=False
    )
    parser.add_argument("--dec", type=float, help="Declination", required=True)
    parser.add_argument(
        "--ndec", action="store_true", help="Is dec negative?", required=False
    )
    return parser.parse_args()


def main(arguments: argparse.Namespace):
    dec = -arguments.dec if arguments.ndec else arguments.dec
    ra = -arguments.ra if arguments.nra else arguments.ra
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    hpx_idx = skycoord_to_healpix(coord, nside=arguments.nside)
    vert = vertices_skycoord(hpx_idx, nside=arguments.nside)
    vert_idx = skycoord_to_healpix(vert, nside=arguments.nside)
    print(hpx_idx)
    print(vert_idx.reshape(2, 2))
    print(ra)
    print((coord.ra - vert.ra).reshape(2, 2).deg)
    print(dec)
    print((coord.dec - coord.dec).deg)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
