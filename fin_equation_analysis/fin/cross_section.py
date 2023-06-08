from abc import ABCMeta, abstractmethod

import numpy as np

from .types import np_arr_f64


class CrossSection(metaclass=ABCMeta):
    """An abstract class that represents an arbitrary cross section."""

    @abstractmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the perimeter of a cross section from its cross-sectional area.

        Args:
            Ac: The cross-sectional area.

        Returns:
            The perimeter of the cross section.
        """
        ...


class CircleCrossSection(CrossSection):
    """A class that represents a circular cross section."""

    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the perimeter of a circle.

        $P = 2 \sqrt{\pi Ac}$
        """

        return 2 * np.sqrt(np.pi * np.abs(Ac))


class RectangleCrossSectionMeta(type):
    """A metaclass that creates classes representing rectangular cross sections
    of varying widths, where it is assumed that t << w.
    """

    def __init__(self, name: str, w: float) -> None:
        super().__init__(self)

    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the perimeter of a rectangle.

        $P \approx 2w$
        """

        return np.full_like(Ac, 2 * self.w)


class RegularPolygonCrossSectionMeta(type):
    """A metaclass that generates classes that represents various regular polygons."""

    def __init__(self, name: str, n: int) -> None:
        super().__init__(self)

    def __new__(cls, name: str, n: int) -> type:
        kls = super().__new__(
            cls, name, (), {"n": n, "P": RegularPolygonCrossSectionMeta.P}
        )
        return kls

    @staticmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        """Calculates the perimeter of a regular polygon.

        $P = 2 \sqrt{n \tan{\frac{\pi}{n}} Ac}$
        """

        return 2 * np.sqrt(self.n * np.tan(np.pi / self.n) * Ac)


# The cross sections of common regular polygons are provided below
EquilateralTriangleCrossSection = RegularPolygonCrossSectionMeta(
    "EquilateralTriangleCrossSection", 3
)
SquareCrossSection = RegularPolygonCrossSectionMeta("SquareCrossSection", 4)
RegularPentagonCrossSection = RegularPolygonCrossSectionMeta(
    "RegularPentagonCrossSection", 5
)
RegulareHexagonCrossSection = RegularPolygonCrossSectionMeta(
    "RegulareHexagonCrossSection", 6
)
