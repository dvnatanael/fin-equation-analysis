from abc import ABCMeta, abstractmethod
from typing import Callable, cast, final

import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import PPoly


class Fin(metaclass=ABCMeta):
    k: int | float
    h: int | float
    L: int | float
    Tb: int | float
    Tinf: int | float

    @abstractmethod
    def P(self, x: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def d2Ac_dx2(self, x: np.ndarray) -> np.ndarray:
        ...

    @final
    def governing_equation(self, x, y) -> np.ndarray:
        y0, y1, y2, y3 = y
        return np.vstack(
            [
                y1,
                -y3 * y1 / y2 + (self.h * self.P(x)) / (self.k * y2) * y0,
                y3,
                self.d2Ac_dx2(x),
            ]
        )

    @final
    def solve(
        self,
        bc: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> tuple[PPoly, PPoly, PPoly, PPoly]:
        x = np.linspace(0, self.L, 5)
        y = np.ones((4, x.size))

        self.res = scipy.integrate.solve_bvp(
            fun=self.governing_equation, bc=bc, x=x, y=y
        )
        return cast(tuple[PPoly, PPoly, PPoly, PPoly], self.res.sol)
