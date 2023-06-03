from sympy import Expr, Integer, Rational, Symbol, pi, sin, sqrt, symbols

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


class RegularPolygon(UniformCrossSection):
    """A class that represents a fin with a uniform regular polygon
    of equal sides inscribed in a circle of radius `r`.
    """

    def __init__(self, n: int | None = None) -> None:
        self.r: Symbol = symbols("r", positive=True)  # radius of inscribed circle

        self.n: Integer | None
        if n is None:
            self.n = None  # shape is a circle
        elif n < 3:
            raise ValueError(f"The number of sides must be at least 3, got {n}.")
        else:
            self.n = Integer(n)  # number of sides of polygon

    @property
    def P(self) -> Expr:
        if self.n is None:
            return 2 * pi * self.r
        else:
            return 2 * self.n * sin(pi / n) * self.r

    @property
    def A_c(self) -> Expr:
        if self.n is None:
            return pi * self.r**2
        else:
            return Rational(1, 2) * self.n * sin(2 * pi / self.n) * self.r**2


class Hexagon(UniformCrossSection):
    """A class that represents a fin with a uniform hexagonal profile
    inscribed within a circle of radius `r`.
    """

    def __init__(self) -> None:
        super().__init__(6)


class Circle(UniformProfile):
    """A class that represents a fin with a uniform circular profile of radius `r`."""

    def __init__(self, f: Expr | None = None) -> None:
        super().__init__()
