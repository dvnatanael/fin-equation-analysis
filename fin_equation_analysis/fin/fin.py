from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Callable, cast, final

import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import PPoly


@dataclass
class Fin(metaclass=ABCMeta):
    k: int | float
    h: int | float
    L: int | float
    Tb: int | float
    Tinf: int | float

    @abstractmethod
    def dP_dx(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def d2Ac_dx2(self, x: np.ndarray) -> np.ndarray:
        ...

    def deriv(self, x, y) -> np.ndarray:
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
        bc: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> tuple[PPoly, PPoly, PPoly, PPoly, PPoly, PPoly]:
        x = np.linspace(0, self.L, 500)
        y = np.ones((6, x.size))

        self.res = scipy.integrate.solve_bvp(fun=self.deriv, bc=bc, x=x, y=y)
        return cast(
            tuple[PPoly, PPoly, PPoly, PPoly, PPoly, PPoly],
            self.res.sol,
        )
