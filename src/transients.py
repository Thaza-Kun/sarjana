from typing import Optional

# from


class FRB:
    def __init__(
        self,
        name: str,
        dm: float,
        redshift: Optional[float] = None,
        raj: Optional[float] = None,
        decj: Optional[float] = None,
        gl: Optional[float] = None,
        gb: Optional[float] = None,
    ):
        self.name = name
        self.dm = dm
        self.redshift = redshift

        self.right_ascension = raj
        self.declension = decj

        self.gal_long = gl
        self.gal_lat = gb

    @property
    def redshift_(self) -> float:
        if self.redshift:
            return self.redshift
        else:
            raise NotImplementedError
