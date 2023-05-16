from sympy import *  # nopycln: import
from sympy import symbols

from fin_equation_analysis import profile
from fin_equation_analysis.fin import Fin


profile = profile.Circle()
fin = Fin(profile)
print(fin.equation)

L = symbols("L")
fin.solve(
    {
        fin.T.subs(fin.x, 0): fin.T_b,
        fin.T.subs(fin.x, L): fin.T_inf,
    }
)
sol = (fin.sol.rhs - fin.T_inf) / (fin.T_b - fin.T_inf)
T_bar = symbols(r"\bar{T}", cls=Function)(fin.x)
sol = Eq(T_bar, sol)
print(latex(simplify(sol)))
