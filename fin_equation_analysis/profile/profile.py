from abc import ABCMeta, abstractmethod

from sympy import Expr, Function


class Profile(metaclass=ABCMeta):
    @property
    @abstractmethod
    def P(self) -> Expr | Function:
        ...

    @property
    @abstractmethod
    def A_c(self) -> Expr | Function:
        ...


class UniformProfile(metaclass=ABCMeta):
    pass


class NonUniformProfile(metaclass=ABCMeta):
    pass
