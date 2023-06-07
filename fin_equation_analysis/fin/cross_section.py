from abc import ABCMeta, abstractmethod

import numpy as np

from .types import np_arr_f64


class CrossSection(metaclass=ABCMeta):
    @abstractmethod
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        ...


class CircularCrossSection(CrossSection):
    def P(self, Ac: np_arr_f64) -> np_arr_f64:
        return 2 * np.sqrt(np.pi * np.abs(Ac))
