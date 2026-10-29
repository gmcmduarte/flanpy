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
# # Broda et al. 2020: regular expressions on average
#
# The running example with a parameter, the alphabet size $k$, following
# {cite:t}`broda-2020-sigact`. Each build-order step adds a section. Run it as a script
# (`python examples/broda2020.py`) or read it as an executed page.
#
# **Status**
#
# - Step 1 (`System`, `series`): done, the coefficients of $R_k$, $S_k$, $L_k$ below.
# - Step 2 (`KSet`): conditions on $k$, e.g. the discriminant $32 k^2 (k - 54)^3$ of
#   $p_k$, eq. (18).
# - Step 3 (elimination): eqs. (17)-(18) for $L_k$, computed from the system.
# - Step 4 (singularity): $\rho_k$, $a_k$, and the Case I / Case II decision.
# - Step 5 (Newton-Puiseux): $\alpha = 1/2$, eq. (32).
# - Step 6 (transfer): eqs. (6)-(8) for $R_k$ and $S_k$.
# - Step 7 (families in $k$): eqs. (35)-(38) and Table 1.

# %%
from fractions import Fraction
from math import comb, factorial

from sympy import symbols

from flanpy import System, series, series_in_k

z, k = symbols("z k")
R, S = symbols("R S")
B, Bb, M, Mb, L = symbols("B Bb M Mb L")
N = 30  # coefficients of z^0..z^N
K = 2  # alphabet size for the fixed-k checks

# %% [markdown]
# ## Step 1: systems and coefficients
#
# ### All regular expressions, $R_k$, and their stars, $S_k$
#
# Grammar (1), $r ::= \varepsilon \mid \sigma \mid r + r \mid r \cdot r \mid r^\star$,
# with size the number of symbols, gives $R_k = (k+1) z + 2 z R_k^2 + z R_k$ (p. 4).
# Marking stars and differentiating at $u = 1$ gives
# $S_k = 4 z S_k R_k + z R_k + z S_k$ (p. 6): $[z^n] S_k$ is the total number of stars
# in the expressions of size $n$.

# %%
rs = System(
    {R: (k + 1) * z + 2 * z * R**2 + z * R, S: 4 * z * R * S + z * R + z * S},
    z=z,
    k=k,
)
values = series(rs, N, k=K)
print("R_2:", values[R][:11])
print("S_2:", values[S][:11])

# %% [markdown]
# Eq. (3) gives $[z^n] R_k$ in closed form, and eq. (5) gives $[z^n] S_k$. For odd $n$,
# eq. (5) as printed has the prefactor $(n-1)/2^{n+2}$; the counts require
# $(n-1)/2^{n+1}$ (e.g. $k = 1$, $n = 3$: $\varepsilon^{\star\star}$ and
# $a^{\star\star}$ have 4 stars, the printed formula gives 2). This erratum is pending
# confirmation with the authors; the check below uses the corrected prefactor.


# %%
def eq3(n: int, k: int) -> Fraction:
    """[z^n] R_k, Broda et al. 2020, eq. (3), for n >= 1."""
    total = sum(
        Fraction(1, 2 * (n - j) + 1)
        * comb(2 * (n - j + 1), n - j + 1)
        * comb(n - j + 1, j)
        * (7 + 8 * k) ** j
        for j in range((n + 1) // 2 + 1)
    )
    return Fraction(total / 2 ** (n + 3))


def eq5(n: int, k: int) -> Fraction:
    """[z^n] S_k, Broda et al. 2020, eq. (5), for n >= 2 (odd case corrected)."""

    def term(j: int) -> Fraction:
        den = factorial(n - j - 1) * factorial(j) * factorial(n - 2 * j)
        return Fraction(Fraction(factorial(2 * n - 2 * j - 2), den) * (7 + 8 * k) ** j)

    if n % 2 == 0:
        head = Fraction(comb(n, n // 2), 2 ** (n + 2)) * (7 + 8 * k) ** (n // 2)
        return Fraction(
            head + Fraction(n - 1, 2 ** (n + 1)) * sum(map(term, range(n // 2)))
        )
    return Fraction(
        Fraction(n - 1, 2 ** (n + 1)) * sum(map(term, range((n - 1) // 2 + 1)))
    )


assert all(values[R][n] == eq3(n, K) for n in range(1, N + 1))
assert all(values[S][n] == eq5(n, K) for n in range(2, N + 1))

# %% [markdown]
# ### Letters in expressions in strong star normal form, $L_k$
#
# Grammar (9) gives eqs. (10)-(11) for $B_k$ and $\bar B_k$ (expressions whose
# language does or does not contain $\varepsilon$), eqs. (15)-(16) for the numbers of
# letters $M_k$ and $\bar M_k$, and $L_k = M_k + \bar M_k$, eq. (12).

# %%
ssnf = System(
    {
        B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
        Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
        L: M + Mb,
    },
    z=z,
    k=k,
)
lk = series(ssnf, N, k=K)[L]
print("L_2:", lk[:11])
print("L_k:", [c.as_expr() for c in series_in_k(ssnf, 5)[L]])

# %% [markdown]
# The paper eliminates the other unknowns to get the cubic eq. (17),
# $p_k(z) w^3 + k z ((k + 12) k z^2 + 4 k z + 1) w - k^2 z^2 (2 z + 1) = 0$, with
# $p_k(z) = 4 k (2 k + 27) z^3 + 36 k z^2 - k z - 4$, eq. (18). The coefficients
# satisfy it up to $z^N$:


# %%
def times(a: list[int], b: list[int]) -> list[int]:
    """Product of two power series, truncated after z^N."""
    out = [0] * (N + 1)
    for i, x in enumerate(a[: N + 1]):
        for j, y in enumerate(b[: N + 1 - i]):
            out[i + j] += x * y
    return out


w = [int(c) for c in lk]
p = [-4, -K, 36 * K, 4 * K * (2 * K + 27)]
q = [0, K, 4 * K**2, K**2 * (K + 12)]
r = [0, 0, K**2, 2 * K**2] + [0] * (N - 3)
cube = times(w, times(w, w))
lhs = [a + b - c for a, b, c in zip(times(p, cube), times(q, w), r, strict=True)]
assert lhs == [0] * (N + 1)
