from functools import partial
from typing import Any

from sympy import *  # nopycln: import


class FinEquation:
    def __init__(self) -> None:
        self._vars = {}

        # variables
        symbol = partial(symbols, real=True, imaginary=False, positive=True)
        self.T_b = symbol("T_b")
        self.T_inf = symbol("T_inf")
        self.x = symbol("x", extended_real=True, imaginary=False, nonnegative=True)
        self.k = symbol("k")
        self.h = symbol("h")
        self._vars |= dict(T_b=self.T_b, T_inf=self.T_inf, x=self.x, k=self.k, h=self.h)

        # functions
        self.T = symbol("T", cls=Function)(self.x)
        self.P = symbol("P", cls=Function, nonnegative=True)(self.x)
        self.A_c = symbol("A_c", cls=Function, nonnegative=True)(self.x)
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
        self.sol = dsolve(fin.equation, fin.T, ics=ics)


fin = FinEquation()
print(fin.equation)

w, t = symbols("w t", real=True, imaginary=False, nonnegative=True)
fin.P = 2 * w
fin.A_c = w * t
print(fin.equation)

L = symbols("L")
fin.solve(
    {
        fin.T.subs(fin.x, 0): fin.T_b,
        fin.T.subs(fin.x, L): fin.T_inf,
    }
)
sol = trigsimp((fin.sol.rhs - fin.T_inf) / (fin.T_b - fin.T_inf))
sol = powsimp(sol)
sol = factor(sol)
theta = symbols("theta/theta_b")
print(latex(Eq(theta, sol)))
