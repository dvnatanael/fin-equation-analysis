from sympy import Expr, Rational, Symbol, pi, sqrt, symbols

from .profile import UniformProfile


class Triangle(UniformProfile):
    """A class that represents a fin with a uniform isosceles triangular profile
    of base `b` and height `h`.
    """

    def __init__(self) -> None:
        self.b: Symbol = symbols("b", positive=True)  # base
        self.h: Symbol = symbols("h", positive=True)  # height

    @property
    def P(self) -> Expr:
        return self.b + sqrt(self.b**2 + 4 * self.h**2)

    @property
    def A_c(self) -> Expr:
        return Rational(1, 2) * self.b * self.h


class Rectangle(UniformProfile):
    """A class that represents a fin with a uniform rectangular profile
    of thickness `t` and width `w`.
    """

    def __init__(self) -> None:
        self.w: Symbol = symbols("w", positive=True)  # width
        self.t: Symbol = symbols("t", positive=True)  # thickness

    @property
    def P(self) -> Expr:
        return 2 * (self.w + self.t)

    @property
    def A_c(self) -> Expr:
        return self.w * self.t


class ThinRectangle(Rectangle):
    """A class that represents a fin with a uniform rectangular profile
    where `t` << `w`.
    """

    @property
    def P(self) -> Expr:
        return 2 * self.w


class Hexagon(UniformCrossSection):
    """A class that represents a fin with a uniform hexagonal profile
    inscribed within a circle of radius `r`.
    """

    def __init__(self) -> None:
        self.r: Symbol = symbols("r", positive=True)  # radius of inscribing circle

    @property
    def P(self) -> Expr:
        return 6 * self.r

    @property
    def A_c(self) -> Expr:
        return Rational(3, 2) * sqrt(3) * self.r**2


class Circle(UniformProfile):
    """A class that represents a fin with a uniform circular profile of radius `r`."""

    def __init__(self) -> None:
        self.r: Symbol = symbols("r", positive=True)  # radius

    @property
    def P(self) -> Expr:
        return 2 * pi * self.r

    @property
    def A_c(self) -> Expr:
        return pi * self.r**2
