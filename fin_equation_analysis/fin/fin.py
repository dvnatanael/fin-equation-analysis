from typing import Any

from sympy import Equality, Expr, Function, Symbol, dsolve, symbols

from ..cross_section import CrossSection


class Fin:
    def __init__(self, cross_section: CrossSection) -> None:
        self.cross_section = cross_section
        self._vars: dict[str, Symbol | Expr | Function] = {
            "cross_section": cross_section
        }

        # variables
        self.T_b: Symbol = symbols("T_b", positive=True)
        self.T_inf: Symbol = symbols("T_inf", positive=True)
        self.k: Symbol = symbols("k", positive=True)
        self.h: Symbol = symbols("h", positive=True)
        self.x: Symbol = symbols("x", extended_nonnegative=True)
        self._vars |= dict(T_b=self.T_b, T_inf=self.T_inf, x=self.x, k=self.k, h=self.h)

        # functions
        self.T: Function = symbols("T", cls=Function, positive=True)(self.x)

        self.P: Expr | Function = cross_section.P
        if isinstance(self.P, Function):
            self.P = self.P(self.x)

        self.A_c: Expr | Function = cross_section.A_c
        if isinstance(self.A_c, Function):
            self.A_c = self.A_c(self.x)

        self._vars |= dict(T=self.T, P=self.P, A_c=self.A_c)

    @property
    def vars(self) -> dict[str, Symbol | Function]:
        return self._vars

    @property
    def equation(self) -> Equality:
        return Equality(
            self.T.diff(self.x, self.x)
            + 1 / self.A_c * self.A_c.diff(self.x) * self.T.diff(self.x)
            - (self.h * self.P) / (self.k * self.A_c) * (self.T - self.T_inf),
            0,
        )

    def solve(self, ics: Any | None) -> None:
        self.sol = dsolve(self.equation, self.T, ics=ics)
