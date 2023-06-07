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


class Square(CrossSection):
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 4 * np.sqrt(Ac)


class EquilateralTriangle(CrossSection):
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 6 * np.sqrt(Ac) / 3**0.25


class RegularHexagon(CrossSection):
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * np.sqrt(2 * Ac) * 3**0.25
