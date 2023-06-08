from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import np_arr_f64


@dataclass
class AxialProfile(metaclass=ABCMeta):
    """An abstract class that represents an arbitrary axial fin profile."""

    L: float  # the characteristic length of the fin

    @abstractmethod
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.

        Args:
            x: The x-coorinates to evaluate the derivative on.
            y: The corresponding y-values of the system.

        Returns:
            The second derivative of the area.
        """
        ...


class UniformProfile(AxialProfile):
    """A class that represents a uniform profile."""

    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.

        d^2 Ac/dx^2 = 0
        """

        return np.zeros_like(x)


class LinearProfile(AxialProfile):
    """A class that represents a linear profile."""

    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.

        d^2 Ac/dx^2 = 2 Ac[0] / L^2
        """

        return np.full_like(x, 2 * Ac[0] / self.L**2)


class ParabolicProfile(AxialProfile):
    """A class that represents a parabolic profile."""

    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.

        d^2 Ac/dx^2 = 2 Ac[0] (2x^2 / L^2 - 2) / L^2
        """

        return 2 * (5 * x**2 / self.L**2 - 2) / self.L**2 * Ac[0]


class ExponentialProfile(AxialProfile):
    """A class that represents a exponential profile."""

    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.

        d^2 Ac/dx^2 = 4 Ac
        """

        return np.full_like(x, 4 * Ac)


class GaussianProfile(AxialProfile):
    """A class that represents a gaussian profile."""

    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.

        d^2 Ac/dx^2 = 16 x^2 - 4
        """

        return 16 * x**2 - 4


class CosineProfile(AxialProfile):
    """A class that represents a cosine profile."""

    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the second spatial derivative of the cross-sectional area.


        d^2 Ac/dx^2 = -0.5 (π / L)^2 Ac[0] * cos(πx/L)
        """

        return -((np.pi / self.L) ** 2) / 2 * Ac[0] * np.cos(np.pi * x / self.L)
