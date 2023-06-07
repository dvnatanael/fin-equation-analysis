from abc import ABCMeta, abstractmethod

import numpy as np

from .types import np_arr_f64


class CrossSection(metaclass=ABCMeta):
    @abstractmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        ...


class Circle(CrossSection):
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * np.sqrt(np.pi * np.abs(Ac))


class RectangleMeta(type):
    def __init__(self, name: str, w: float) -> None:
        super().__init__(self)

    def __new__(cls, name: str, w: float) -> type:
        kls = super().__new__(cls, name, (), {"w": w, "P": RectangleMeta.P})
        return kls

    @staticmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return np.full_like(Ac, 2 * self.w)


class RegularPolygonMeta(type):
    def __init__(self, name: str, n: int) -> None:
        super().__init__(self)

    def __new__(cls, name: str, n: int) -> type:
        kls = super().__new__(cls, name, (), {"n": n, "P": RegularPolygonMeta.P})
        return kls

    @staticmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * np.sqrt(self.n * np.tan(np.pi / self.n) * Ac)


EquilateralTriangle = RegularPolygonMeta("EquilateralTriangle", 3)
Square = RegularPolygonMeta("Square", 4)
RegularPentagon = RegularPolygonMeta("RegularPentagon", 5)
RegulareHexagon = RegularPolygonMeta("RegulareHexagon", 6)
