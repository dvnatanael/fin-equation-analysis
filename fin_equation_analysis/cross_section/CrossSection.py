from abc import ABCMeta, abstractmethod

from sympy import Expr, Function


class CrossSection(metaclass=ABCMeta):
    @property
    @abstractmethod
    def P(self) -> Expr | Function:
        ...

    @property
    @abstractmethod
    def A_c(self) -> Expr | Function:
        ...


class UniformCrossSection(metaclass=ABCMeta):
    pass


class NonUniformCrossSection(metaclass=ABCMeta):
    pass
