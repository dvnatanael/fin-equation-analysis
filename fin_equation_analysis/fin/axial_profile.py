from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np

from .types import np_arr_f64


@dataclass
class AxialProfile(metaclass=ABCMeta):
    L: float

    @abstractmethod
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        ...


class UniformProfile(AxialProfile):
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        return np.zeros_like(x)


class LinearProfile(AxialProfile):
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        return np.full_like(x, 2 * Ac[0] / self.L**2)


class ParabolicProfile(AxialProfile):
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * (5 * x**2 / self.L**2 - 2) / self.L**2 * Ac[0]


class ExponentialProfile(AxialProfile):
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        return np.full_like(x, 4 * Ac)


class GaussianProfile(AxialProfile):
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        return 16 * x**2 - 4


class CosineProfile(AxialProfile):
    def d2Ac_dx2(self, x: np_arr_f64, Ac: np_arr_f64) -> np_arr_f64:
        return -((np.pi / self.L) ** 2) / 2 * Ac[0] * np.cos(np.pi * x / self.L)
