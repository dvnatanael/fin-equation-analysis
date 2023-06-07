from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, final

import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import PPoly

from .cross_section import CrossSection
from .types import np_arr_f64


@dataclass
class Fin(CrossSection, metaclass=ABCMeta):
    k: float
    h: float
    L: float
    Tb: float
    Tinf: float

    @abstractmethod
    def d2Ac_dx2(self, x: np_arr_f64, y: np_arr_f64) -> np_arr_f64:
        ...

    def deriv(self, x: np_arr_f64, y: np_arr_f64) -> np_arr_f64:
        y0, y1, _, y3, y4 = y
        return np.vstack(
            [
                y1,
                -y4 * y1 / y3 + (self.h * self.P(np.abs(y3))) / (self.k * y3) * y0,
                y3,
                y4,
                self.d2Ac_dx2(x, y),
            ]
        )

    @final
    def solve(
        self,
        bc: Callable[[np_arr_f64, np_arr_f64], np_arr_f64],
    ) -> PPoly:
        x = np.linspace(0, self.L, 5)
        y = np.ones((5, x.size))

        self.res = scipy.integrate.solve_bvp(
            fun=self.deriv, bc=bc, x=x, y=y, max_nodes=100000
        )
        return self.res.sol
