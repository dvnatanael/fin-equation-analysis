from abc import ABCMeta, abstractmethod

import numpy as np

from .types import np_arr_f64


class CrossSection(metaclass=ABCMeta):
    @abstractmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        ...


class CircleCrossSection(CrossSection):
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * np.sqrt(np.pi * np.abs(Ac))


class RectangleCrossSection(CrossSection):
    def __init__(self, w: float) -> None:
        self.w = w

    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return np.full_like(Ac, 2 * self.w)


class RegularPolygonCrossSectionMeta(type):
    def __init__(self, name: str, n: int) -> None:
        super().__init__(self)

    def __new__(cls, name: str, n: int) -> type:
        kls = super().__new__(
            cls, name, (), {"n": n, "P": RegularPolygonCrossSectionMeta.P}
        )
        return kls

    @staticmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * np.sqrt(self.n * np.tan(np.pi / self.n) * Ac)


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
