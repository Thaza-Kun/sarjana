import numpy as np


from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple


@dataclass
class Gauss2D:
    amplitude: float = 1.0
    sigma: Tuple[float, float] = (1.0, 1.0)
    theta: float = 0.0
    offset: float = 0.0
    center: Tuple[float, float] = (0.0, 0.0)

    def __call__(self, x: np.ndarray, y: np.ndarray, modify: Optional[Dict[str, float]] = None) -> np.ndarray:
        X = x - self.center[0]
        Y = y - self.center[1]
        if modify is not None:
            for key in modify.keys():
                if key not in [i.name for i in fields(self)]:
                    raise KeyError("Parameter `{}` not in field".format(key))
                else:
                    setattr(self, key, getattr(self, key) + modify[key])
        return self.__func__(X, Y, self._a, self._b, self._b)

    def __func__(
        self, X: np.ndarray, Y: np.ndarray, A: float, B: float, C: float
    ) -> np.ndarray:
        return self.offset + self.amplitude * np.exp(
            -(A * (X**2) + 2 * B * X * Y + C * (Y**2))
        )

    def dfunc_dparam(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = x - self.center[0]
        Y = y - self.center[1]
        return self.dfunc_dtheta(X, Y) + self.dfunc_dsigx(X, Y) + self.dfunc_dsigy(X, Y)

    def d2func_dparam2(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        X = x - self.center[0]
        Y = y - self.center[1]
        # fmt: off
        return  (self.d2func_dtheta2(X, Y)          + self.d2func_dtheta_disgx(X, Y)    + self.d2func_dtheta_dsigy(X, Y)
                +   self.d2func_dsigx_dtheta(X, Y)  + self.d2func_dsigx2(X, Y)          + self.d2func_dsigx_dsigy(X, Y)
                +   self.d2func_dsigy_dtheta(X, Y)  + self.d2func_dsigy_dsigx(X, Y)     + self.d2func_dsigy2(X, Y))

    def dfunc_dtheta(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(X, Y, self.da_dtheta, self.db_dtheta, self.dc_dtheta)

    def dfunc_dsigx(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(X, Y, self.da_dsigx, self.db_dsigx, self.dc_dsigx)

    def dfunc_dsigy(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(X, Y, self.da_dsigy, self.db_dsigy, self.dc_dsigy)

    def d2func_dtheta2(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dtheta2, self.d2b_dtheta2, self.d2c_dtheta2
        )

    def d2func_dsigx2(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dsigx2, self.d2b_dsigx2, self.d2c_dsigx2
        )

    def d2func_dsigy2(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dsigy2, self.d2b_dsigy2, self.d2c_dsigy2
        )

    def d2func_dsigx_dtheta(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dsigx_dtheta, self.d2a_dsigx_dtheta, self.d2a_dsigx_dtheta
        )

    def d2func_dtheta_disgx(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dtheta_dsigx, self.d2a_dtheta_dsigx, self.d2a_dtheta_dsigx
        )

    def d2func_dsigy_dtheta(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dsigy_dtheta, self.d2a_dsigy_dtheta, self.d2a_dsigy_dtheta
        )

    def d2func_dtheta_dsigy(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dtheta_dsigy, self.d2a_dtheta_dsigy, self.d2a_dtheta_dsigy
        )

    def d2func_dsigx_dsigy(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dsigx_dsigy, self.d2a_dsigx_dsigy, self.d2a_dsigx_dsigy
        )

    def d2func_dsigy_dsigx(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.__func__(
            X, Y, self.d2a_dsigy_dsigx, self.d2a_dsigy_dsigx, self.d2a_dsigy_dsigx
        )

    @property
    def _a(self) -> float:
        return (np.square(np.cos(self.theta)) / 2 * (self.sigma[0] ** 2)) + (
            np.square(np.sin(self.theta)) / 2 * (self.sigma[1] ** 2)
        )

    @property
    def _b(self) -> float:
        return -(np.sin(2 * self.theta) / 4 * (self.sigma[0] ** 2)) + (
            np.cos(2 * self.theta) / 4 * (self.sigma[1] ** 2)
        )

    @property
    def _c(self) -> float:
        return (np.square(np.sin(self.theta)) / 2 * (self.sigma[0] ** 2)) + np.square(
            np.cos(self.theta) / 2 * (self.sigma[1] ** 2)
        )

    @property
    def da_dtheta(self) -> float:
        return -(np.sin(self.theta) / (self.sigma[0] ** 2)) + (
            np.cos(self.theta) / self.sigma[1]
        )

    @property
    def db_dtheta(self) -> float:
        return -(np.cos(2 * self.theta) / 2 * (self.sigma[0] ** 2)) - (
            np.sin(2 * self.theta) / 2 * (self.sigma[1] ** 2)
        )

    @property
    def dc_dtheta(self) -> float:
        return (np.cos(self.theta) / (self.sigma[0] ** 2)) - (
            np.sin(self.theta) / self.sigma[1]
        )

    @property
    def da_dsigx(self) -> float:
        return -np.square(np.cos(self.theta)) / (self.sigma[0] ** 3)

    @property
    def db_dsigx(self) -> float:
        return np.sin(2 * self.theta) / 2 * (self.sigma[0] ** 3)

    @property
    def dc_dsigx(self) -> float:
        return -np.square(np.sin(self.theta)) / (self.sigma[0] ** 3)

    @property
    def da_dsigy(self) -> float:
        return -np.square(np.sin(self.theta)) / (self.sigma[1] ** 3)

    @property
    def db_dsigy(self) -> float:
        return -np.cos(2 * self.theta) / 2 * (self.sigma[1] ** 3)

    @property
    def dc_dsigy(self) -> float:
        return -np.square(np.cos(self.theta)) / (self.sigma[1] ** 3)

    @property
    def d2a_dtheta2(self) -> float:
        return -(np.cos(self.theta) / self.sigma[0] ** 2) - (
            np.sin(self.theta) / self.sigma[1] ** 2
        )

    @property
    def d2b_dtheta2(self) -> float:
        return (np.sin(2 * self.theta) / self.sigma[0] ** 2) - (
            np.cos(2 * self.theta) / self.sigma[1] ** 2
        )

    @property
    def d2c_dtheta2(self) -> float:
        return -(np.sin(self.theta) / self.sigma[0] ** 2) - (
            np.cos(self.theta) / self.sigma[1] ** 2
        )

    @property
    def d2a_dsigx2(self) -> float:
        return 3 * np.square(np.cos(self.theta)) / (self.sigma[0] ** 4)

    @property
    def d2b_dsigx2(self) -> float:
        return -(3 / 2) * np.sin(2 * self.theta) / (self.sigma[0] ** 4)

    @property
    def d2c_dsigx2(self) -> float:
        return 3 * np.square(np.sin(self.theta)) / (self.sigma[0] ** 4)

    @property
    def d2a_dsigy2(self) -> float:
        return 3 * np.square(np.sin(self.theta)) / (self.sigma[1] ** 4)

    @property
    def d2b_dsigy2(self) -> float:
        return (3 / 2) * np.cos(2 * self.theta) / (self.sigma[1] ** 4)

    @property
    def d2c_dsigy2(self) -> float:
        return 3 * np.square(np.cos(self.theta)) / (self.sigma[0] ** 4)

    @property
    def d2a_dsigx_dtheta(self) -> float:
        return 2 * np.sin(2 * self.theta) / (self.sigma[0] ** 3)

    @property
    def d2a_dtheta_dsigx(self) -> float:
        return self.d2a_dsigx_dtheta

    @property
    def d2b_dsigx_dtheta(self) -> float:
        return np.cos(2 * self.theta) / (self.sigma[0] ** 3)

    @property
    def d2b_dtheta_dsigx(self) -> float:
        return self.d2b_dsigx_dtheta

    @property
    def d2c_dsigx_dtheta(self) -> float:
        return -2 * np.cos(self.theta) / (self.sigma[0] ** 3)

    @property
    def d2c_dtheta_dsigx(self) -> float:
        return self.d2c_dsigx_dtheta

    @property
    def d2a_dsigy_dtheta(self) -> float:
        return -2 * np.cos(self.theta) / (self.sigma[1] ** 3)

    @property
    def d2a_dtheta_dsigy(self) -> float:
        return self.d2a_dsigy_dtheta

    @property
    def d2b_dsigy_dtheta(self) -> float:
        return np.sin(2 * self.theta) / (self.sigma[1] ** 3)

    @property
    def d2b_dtheta_dsigy(self) -> float:
        return self.d2b_dsigy_dtheta

    @property
    def d2c_dsigy_dtheta(self) -> float:
        return 2 * np.sin(self.theta) / (self.sigma[1] ** 3)

    @property
    def d2c_dtheta_dsigy(self) -> float:
        return self.d2c_dsigy_dtheta

    @property
    def d2a_dsigx_dsigy(self) -> float:
        return 0.0

    @property
    def d2b_dsigx_dsigy(self) -> float:
        return 0.0

    @property
    def d2c_dsigx_dsigy(self) -> float:
        return 0.0

    @property
    def d2a_dsigy_dsigx(self) -> float:
        return 0.0

    @property
    def d2b_dsigy_dsigx(self) -> float:
        return 0.0

    @property
    def d2c_dsigy_dsigx(self) -> float:
        return 0.0