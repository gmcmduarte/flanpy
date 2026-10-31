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
# - Step 3 (elimination): done, the curve $G(z, F) = z F^2 - F + 1$ below.
# - Step 4 (singularity): done, $\rho = 1/4$, Case I, $a = F(\rho) = 2$, unique.
# - Step 5 (Newton-Puiseux): done, $F = 2 - 2 s^{1/2} + 2 s - \dots$ with $s = 1 - 4z$:
#   $\alpha = 1/2$, $b = 2$.
# - Step 6 (transfer): $[z^n] F \sim 4^n n^{-3/2} / \sqrt{\pi}$.
# - Step 2 (`KSet`): does not apply, there is no parameter (see below).
# - Step 7 (families in $k$): does not apply, there is no parameter.

# %% [markdown]
# ## Step 1: the system and its coefficients
#
# {class}`flanpy.System` checks that the iteration $F \leftarrow 1 + z F^2$ from 0
# converges; {func}`flanpy.series` returns the coefficients of $z^0, \dots, z^n$, with
# $n$ included. They are the Catalan numbers $\binom{2n}{n} / (n + 1)$
# {cite:p}`flajolet-sedgewick-2009`.

# %%
from fractions import Fraction
from math import comb

from sympy import symbols

from flanpy import System, eliminate, local_expansion, series, singularity

z, F = symbols("z F")
catalan = System({F: 1 + z * F**2}, z=z)
coefficients = series(catalan, 15)[F]
assert coefficients == tuple(comb(2 * n, n) // (n + 1) for n in range(16))
print(coefficients)

# %% [markdown]
# ## Step 2: conditions on $k$
#
# Catalan's equation has no parameter, so every result holds unconditionally and a
# {class}`flanpy.KSet` is not needed. The Broda 2020 example shows the sets of $k$ on
# which a discriminant is negative, zero and positive.

# %% [markdown]
# ## Step 3: the curve
#
# {func}`flanpy.eliminate` returns the minimal polynomial of the series, normalised.
# With one unknown nothing is eliminated: the curve is the equation itself,
# $G(z, F) = z F^2 - F + 1$. The coefficients of step 1 satisfy it.

# %%
curve = eliminate(catalan, F)
print(curve)
assert curve.as_expr() == z * F**2 - F + 1
c = coefficients  # z F^2 - F + 1 = 0, coefficient by coefficient:
assert c[0] == 1
assert all(sum(c[i] * c[n - 1 - i] for i in range(n)) == c[n] for n in range(1, 16))

# %% [markdown]
# ## Step 4: the dominant singularity
#
# The coefficients are non-negative, so by Pringsheim's theorem the radius of
# convergence $\rho$ is a singularity {cite:p}`flajolet-sedgewick-2009`, hence a root
# of $\operatorname{lc}_F G \cdot \operatorname{disc}_F G = z (1 - 4z)$. The root $0$
# is not a singularity of the series (it is a power series there), so $\rho = 1/4$.
# {func}`flanpy.singularity` finds it exactly and decides Case I: $F$ tends to the
# finite value $a = F(1/4) = 2$, a double root of $G(1/4, F) = (F - 2)^2 / 4$. Nothing
# else lies on the circle $|z| = 1/4$, so $\rho$ is the only dominant singularity.

# %%
s = singularity(curve)
print(s)
assert s.rho == Fraction(1, 4) and s.rho.polynomial.as_expr() == 4 * z - 1
assert s.case == "I" and s.a == 2
assert s.period == 1 and s.unique is True
assert curve.as_expr().subs(z, Fraction(1, 4)).factor() == (F - 2) ** 2 / 4

# %% [markdown]
# ## Step 5: the local expansion
#
# At $\rho = 1/4$, with $s = 1 - 4z$, the solution of $G = 0$ that is the series is
# $F = 2/(1 + \sqrt{s}) = 2 - 2 s^{1/2} + 2 s - 2 s^{3/2} + \dots$
# {cite:p}`flajolet-sedgewick-2009` (eq. (81), where $Y \sim \pm 4 Z^{1/2}$ with
# $Z = s/4$). {func}`flanpy.local_expansion` finds it from the Newton polygon of
# $G(\rho(1 - s), 2 + Y) = \tfrac14 (Y^2 - 4s - 4sY - sY^2)$: one edge, from $s^1$ to
# $Y^2$, so the exponent is $\alpha = 1/2$ and the ramification $\kappa = 2$. Of the two
# real branches $Y \sim \pm 2 s^{1/2}$, the series takes the negative one (it increases
# to $a = 2$), so $F = 2 - b\, s^{1/2} + \dots$ with $b = 2$
# {cite:p}`broda-2020-sigact` (eq. (20)).

# %%
e = local_expansion(s)
print(e)
assert (e.alpha, e.ramification) == (Fraction(1, 2), 2) and e.b == 2 and e.c is None
terms = e.terms(6)
print([(str(x), str(d)) for x, d in terms])
assert [x for x, _ in terms] == [Fraction(i, 2) for i in range(6)]
assert [d for _, d in terms] == [2 * (-1) ** i for i in range(6)]  # 2/(1 + √s)
