from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, final

import numpy as np
import scipy
import scipy.integrate
from numpy import typing as npt
from scipy.interpolate import PPoly


np_arr_f64 = npt.NDArray[np.float64]


@dataclass
class Fin(metaclass=ABCMeta):
    k: float
    h: float
    L: float
    Tb: float
    Tinf: float

    @abstractmethod
    def dP_dx(self, x: np_arr_f64) -> np_arr_f64:
        ...

    @abstractmethod
    def d2Ac_dx2(self, x: np_arr_f64) -> np_arr_f64:
        ...

    def deriv(self, x: np_arr_f64, y: np_arr_f64) -> np_arr_f64:
        y0, y1, _, y3, y4, y5 = y
        return np.vstack(
            [
                y1,
                -y4 * y1 / y3 + (self.h * y5) / (self.k * y3) * y0,
                y3,
                y4,
                self.d2Ac_dx2(x),
                self.dP_dx(x),
            ]
        )

    @final
    def solve(
        self,
        bc: Callable[[np_arr_f64, np_arr_f64], np_arr_f64],
    ) -> PPoly:
        x = np.linspace(0, self.L, 500)
        y = np.ones((6, x.size))

        self.res = scipy.integrate.solve_bvp(fun=self.deriv, bc=bc, x=x, y=y)
        return self.res.sol
