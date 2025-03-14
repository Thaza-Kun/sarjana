# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
# ]
# ///
"""Calculate error on exposure based on declination error"""
import argparse
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure", type=float, required=True)
    parser.add_argument("--dec", type=float, required=True)
    parser.add_argument("--pdec", type=float, required=True)
    parser.add_argument("--ndec", type=float, required=True)
    parser.add_argument("--events", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    exposure = arguments.exposure
    dec = arguments.dec
    pdec = arguments.pdec
    ndec = arguments.ndec
    n = arguments.events

    def exposure_func(amplitude: float, degree: float, delta: float) -> float:
        """A * cos(deg) / cos(deg + delta)"""
        return amplitude * np.cos(degree) / np.cos(degree + delta)

    NEXP = exposure_func(exposure, np.deg2rad(dec), np.deg2rad(-ndec)) - exposure
    PEXP = exposure_func(exposure, np.deg2rad(dec), np.deg2rad(pdec)) - exposure
    _NEXP = exposure + NEXP
    _PEXP = exposure + PEXP
    print(f"{exposure} ({NEXP:.3f}, {PEXP:.3f})")
    rate = n / exposure
    print(f"{rate:.3f}, ({rate-(n/_NEXP):.3f}, {rate-(n/_PEXP):.3f})")
