from dataclasses import dataclass
from typing import Callable, final

import numpy as np
import scipy
import scipy.integrate
from scipy.interpolate import PPoly

from .axial_profile import AxialProfile
from .cross_section import CrossSection
from .types import np_arr_f64


@dataclass
class Fin:
    """A class that represent a fin."""

    k: float
    h: float
    cross_section: CrossSection
    profile: AxialProfile

    def deriv(self, x: np_arr_f64, y: np_arr_f64) -> np_arr_f64:
        """Calculates the derivative of the system defined by the fin equation.

        Args:
            x: A 1-D independent variable with shape (m,).
            y: An 5-D vector-valued function, with shape (5, m).

        Returns:
            The derivative of y.
        """
        # set y0 = θ(x), y1 = θ'(x), y2 = V(x), y3 = Ac(x), y4 = Ac'(x)
        y0, y1, _, y3, y4 = y  # unpack y into n variables, each with shape (m,)
        return np.vstack(
            [
                y1,
                -y4 * y1 / y3
                + (self.h * self.cross_section.P(np.abs(y3))) / (self.k * y3) * y0,
                y3,
                y4,
                self.profile.d2Ac_dx2(x, y3),
            ]
        )

    @final
    def solve(
        self,
        bc: Callable[[np_arr_f64, np_arr_f64], np_arr_f64],
    ) -> PPoly:
        """Solves the general fin equation.

        Args:
            bc: A callable that given the boundary state,
                returns the residuals of the boundary conditions.

        Returns:
            A `PPoly` that represents y(x).
        """
        x = np.linspace(0, self.profile.L, 5)  # set the initial number of nodes as 5
        y = np.ones((5, x.size))

        self.res = scipy.integrate.solve_bvp(
            fun=self.deriv, bc=bc, x=x, y=y, max_nodes=100000
        )
        return self.res.sol
