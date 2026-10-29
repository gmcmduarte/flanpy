# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Catalan numbers
#
# The running example without a parameter: binary trees counted by internal nodes,
# with generating function $F = 1 + z F^2$. Each build-order step adds a section.
# Run it as a script (`python examples/catalan.py`) or read it as an executed page.
#
# **Status**
#
# - Step 1 (`System`, `series`): done, the coefficients below.
# - Step 3 (elimination): the curve $G(z, w) = z w^2 - w + 1$.
# - Step 4 (singularity): $\rho = 1/4$, $a = F(\rho) = 2$.
# - Step 5 (Newton-Puiseux): $\alpha = 1/2$ in $F = 2 - 2 (1 - 4z)^{1/2} + \dots$
# - Step 6 (transfer): $[z^n] F \sim 4^n n^{-3/2} / \sqrt{\pi}$.
# - Steps 2 and 7 concern the parameter $k$ and do not apply.

# %% [markdown]
# ## Step 1: the system and its coefficients
#
# {class}`flanpy.System` checks that the iteration $F \leftarrow 1 + z F^2$ from 0
# converges; {func}`flanpy.series` returns the coefficients of $z^0, \dots, z^n$, with
# $n$ included. They are the Catalan numbers $\binom{2n}{n} / (n + 1)$
# {cite:p}`flajolet-sedgewick-2009`.

# %%
from math import comb

from sympy import symbols

from flanpy import System, series

z, F = symbols("z F")
catalan = System({F: 1 + z * F**2}, z=z)
coefficients = series(catalan, 15)[F]
assert coefficients == tuple(comb(2 * n, n) // (n + 1) for n in range(16))
print(coefficients)
