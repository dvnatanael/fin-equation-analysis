# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: lines_to_next_cell
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: 2023s_bme3104
#     language: python
#     name: python3
# ---

# %%
from typing import Final

import numpy as np
from matplotlib import pyplot as plt
from ruamel.yaml import YAML

from fin_equation_analysis.fin.axial_profile import LinearProfile, UniformProfile
from fin_equation_analysis.fin.cross_section import CircleCrossSection
from fin_equation_analysis.fin.fin import Fin
from fin_equation_analysis.fin.types import np_arr_f64


# %%
yaml = YAML(typ="safe")
with open("config.yaml") as f:
    config = yaml.load(f)

k: Final[float] = config["material"]["k"]
h: Final[float] = config["material"]["h"]
r: Final[float] = config["geometry"]["r"]
L: Final[float] = config["geometry"]["L"]
T_b: Final[float] = config["environment"]["Tb"]
T_inf: Final[float] = config["environment"]["Tinf"]


# %%
def plot_results(
    fig: plt.Figure,
    sol: tuple[np_arr_f64, np_arr_f64, np_arr_f64, np_arr_f64, np_arr_f64],
    x: np_arr_f64,
    label: str,
) -> None:
    T, dT, V, Ac, dAc = sol
    axs = fig.axes
    axs[0].plot(x, 2 * np.sqrt(np.pi * Ac), label=label)
    axs[0].set_ylabel(r"$P$ [mm]")

    axs[1].plot(x, V, label=label)
    axs[1].set_ylabel(r"$V{(x)}$ [mm$^3$]")

    axs[2].plot(x, Ac, label=label)
    axs[2].set_ylabel(r"$A_c{(x)}$ [mm$^2$]")

    axs[3].plot(x, dAc, label=label)
    axs[3].set_ylabel(r"$A'_c{(x)}$ [mm]")

    axs[4].plot(x, T_inf + T, label=label)
    axs[4].set_ylabel(r"$T{(x)}$ [K]")

    axs[5].plot(x, dT, label=label)
    axs[5].set_ylabel(r"$T'{(x)}$ [K / mm]")

    axs[6].plot(x, -k * dT, label=label)
    axs[6].set_ylabel(r"$q''_x = -k T'{(x)}$ [W / mm$^2$]")

    axs[7].plot(x, -k * Ac * dT, label=label)
    axs[7].set_ylabel(r"$q_x = -k A_c T'{(x)}$ [W]")


# %% [markdown]
# The general fin equation is given by
#
# $$
# \frac{d^2 \theta}{dx^2}
# + \left( \frac{1}{A_c} \frac{d A_c}{dx} \right) \frac{d \theta}{dx}
# - \frac{h P}{k A_c} \theta
# = 0
# $$
# where $\theta{(x)} \equiv T{(x)} - T_{\infty}$.
#
# In state-space form,
# \begin{gathered}
#     \begin{align*}
#         y_0 &= \theta{(x)} & y_1 &= \theta'{(x)} = \frac{d \theta}{dx} \\
#         y_2 &= V{(x)} = \int_0^x A_c\,dx & y_3 &= A_c{(x)} \\
#         y_4 &= A_c'{(x)} = \frac{d A_c}{dx}
#     \end{align*}\\
#     \begin{equation*}
#         \frac{d \textbf{y}}{dx}
#         = \begin{bmatrix} y'_0 \\ y'_1  \\ y'_2 \\ y'_3 \\ y'_4 \end{bmatrix}
#         = \begin{bmatrix} \theta' \\ \theta'' \\ A_c \\ A'_c \\ A''_c \end{bmatrix}
#         = \begin{bmatrix}
#             y_1 \\
#             - \frac{y_4}{y_3} y_1 + \frac{h P}{k y_3} y_0 \\
#             y_3 \\
#             y_4 \\
#             \cdots
#         \end{bmatrix}
#     \end{equation*}
# \end{gathered}
# where for a polygons, $P = k \sqrt{A_c}$
# (ref: https://link.springer.com/chapter/10.1007/978-1-4899-2124-6_12)


# %%
def bc_uniform(ya, yb):
    return np.array(
        [
            # Temperature BCs
            ya[0] - (T_b - T_inf),  # θ(0) = T_b - T_inf
            h * yb[0] + k * yb[1],  # hθ(L) + kθ'(L) = 0; active tip
            # Geometry BCs
            ya[2],  # V(0) = 0
            ya[3] - np.pi * r**2,  # Ac(0) = πr²
            yb[3] - np.pi * r**2,  # Ac(L) = πr²
        ]
    )


# %%
def bc_linear(ya, yb):
    return np.array(
        [
            # Temperature BCs
            ya[0] - (T_b - T_inf),  # θ(0) = T_b - T_inf
            h * yb[0] + k * yb[1],  # hθ(L) + kθ'(L) = 0; active tip
            # yb[1],  # we can also use an adiabatic tip here, since Ac(L) = 0
            # Geometry BCs
            ya[2],  # V(0) = 0
            yb[2] - np.pi * r**2 * L,  # V(L) = πr²L
            yb[3] - 1e-12,  # Ac(L) = 0; reduces to a point
        ]
    )


# %%
ciruclar_uniform_fin = Fin(k, h, CircleCrossSection(), UniformProfile(L))
circular_linear_fin = Fin(k, h, CircleCrossSection(), LinearProfile(L))

# %%
x_plot = np.linspace(0, L, 100001)
sol_uniform = ciruclar_uniform_fin.solve(bc_uniform)(x_plot)
sol_linear = circular_linear_fin.solve(bc_linear)(x_plot)
print(circular_linear_fin.res.status)

fig, axs = plt.subplots(num=1, figsize=(10, 12), nrows=4, ncols=2, sharex="all")
plot_results(fig, sol_uniform, x_plot, "constant cross-section")
plot_results(fig, sol_linear, x_plot, "linear profile")

for ax in fig.axes:
    ax.set_xlabel(r"x [mm]")
    ax.grid(True)
    ax.legend()
plt.tight_layout()
