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
# - Step 2 (`KSet`): done, the sets of $k$ where the discriminant $32 k^2 (k - 54)^3$
#   of $p_k$, eq. (18), is negative, zero and positive.
# - Step 3 (elimination): done, eq. (17) for $L_k$ computed from the system (for every
#   $k \ge 1$), and the curves of $R_k$ (radicand of eq. (2)) and $S_k$.
# - Step 4 (singularity): $\rho_k$, $a_k$, and the Case I / Case II decision.
# - Step 5 (Newton-Puiseux): $\alpha = 1/2$, eq. (32).
# - Step 6 (transfer): eqs. (6)-(8) for $R_k$ and $S_k$.
# - Step 7 (families in $k$): eqs. (35)-(38) and Table 1.

# %%
import time
from fractions import Fraction
from math import comb, factorial

from sympy import (
    Poly,
    diff,
    discriminant,
    expand,
    factor,
    resultant,
    simplify,
    sqf_part,
    sqrt,
    symbols,
)

from flanpy import KSet, System, eliminate, eliminate_in_k, series, series_in_k

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

# %% [markdown]
# ## Step 2: conditions on $k$
#
# The discriminant of $p_k$ in $z$ is $32 k^2 (k - 54)^3$, eq. (18).
# {meth}`KSet.where <flanpy.KSet.where>` gives the sets of $k \ge 1$ where it is
# negative, zero and positive; the root $k = 0$ lies outside the domain.

# %%
p_k = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4  # eq. (18)
disc = discriminant(p_k, z)
assert factor(disc) == 32 * k**2 * (k - 54) ** 3
negative, zero, positive = (KSet.where(disc, rel, k=k) for rel in ("<", "==", ">"))
assert negative == KSet([(1, 53)])
assert zero == KSet.of(54)
assert positive == KSet([(55, None)])
print("disc p_k < 0:", negative)
print("disc p_k = 0:", zero)
print("disc p_k > 0:", positive)

# %% [markdown]
# A sign can also change between two integers. **Derived by us, not printed in the
# paper:** let $D$ be the discriminant in $z$ of the squarefree part of
# $\operatorname{res}_w(G, \partial G / \partial w)$, for the curve $G$ of eq. (17).
# $D$ has the factor $k^2 - 34 k - 54$, whose root $17 + 7 \sqrt 7 \approx 35.52$ is
# not an integer, so the sign of $D$ changes between $k = 35$ and $k = 36$.

# %%
W = symbols("w")
G = expand(
    p_k * W**3
    + k * z * ((k + 12) * k * z**2 + 4 * k * z + 1) * W
    - k**2 * z**2 * (2 * z + 1)
)  # eq. (17)
D = discriminant(sqf_part(resultant(G, diff(G, W), W)), z)
print("D =", factor(D))
assert KSet.where(D, ">", k=k) == KSet([(1, 35), (55, None)])
assert KSet.where(D, "<", k=k) == KSet([(36, 53)])
assert KSet.where(D, "==", k=k) == KSet.of(54)
print("D > 0:", KSet.where(D, ">", k=k))
print("D < 0:", KSet.where(D, "<", k=k))

# %% [markdown]
# ## Step 3: elimination
#
# {func}`flanpy.eliminate_in_k` eliminates the other unknowns and returns the curve of
# one coordinate for symbolic $k$, as a {class}`flanpy.KCases`: a generic curve, and a
# separate piece for each $k$ where it changes.
#
# ### $R_k$ and $S_k$
#
# $R_k$ needs no elimination. The discriminant of its curve is the radicand
# $\Delta_k(z) = 1 - 2z - (7 + 8k) z^2$ of eq. (2), and eq. (2) is a root.
# $S_k$ depends on $R_k$, which a resultant eliminates.

# %%
r_cases = eliminate_in_k(rs, R)
print("R_k:", r_cases)
g_r = r_cases[1].as_expr()  # one piece, [1, ∞)
delta = 1 - 2 * z - (7 + 8 * k) * z**2
assert expand(discriminant(g_r, R) - delta) == 0
assert simplify(g_r.subs(R, (1 - z - sqrt(delta)) / (4 * z))) == 0  # eq. (2)
s_cases = eliminate_in_k(rs, S)
print("S_k:", s_cases)
assert r_cases.covered == s_cases.covered == KSet.all()

# %% [markdown]
# ### $L_k$: eq. (17)
#
# Eliminating $B_k$, $\bar B_k$, $M_k$ and $\bar M_k$ from the five equations gives
# **exactly** eq. (17) (no unit in $\mathbb Q(k)$), with $p_k$ of eq. (18) as leading
# coefficient, and for every $k \ge 1$. The value $k = 54$ is a candidate: the
# leading coefficients of two Gröbner basis elements vanish there. The curve is
# recomputed at $k = 54$, equals the generic one, and stays in the generic piece.

# %%
start = time.perf_counter()
l_cases = eliminate_in_k(ssnf, L)
print(f"eliminate_in_k(L_k): {time.perf_counter() - start:.1f} s")
print(l_cases)
assert len(l_cases.pieces) == 1 and l_cases.covered == KSet.all()
tail = l_cases.tail  # the generic piece
assert tail is not None
curve_l = tail[1]
assert expand(curve_l.as_expr() - G.subs(W, L)) == 0  # eq. (17), exactly
assert expand(Poly(curve_l.as_expr(), L).LC() - p_k) == 0  # eq. (18)

# %% [markdown]
# Spot check: the generic curve, certified at each $k$ with
# {meth}`Curve.specialize <flanpy.Curve.specialize>`, equals the curve computed from
# the system specialised at that $k$, and the series of step 1 satisfies it.

# %%
for n in (1, 2, 10, 50, 54):
    at_n = eliminate(ssnf, L, k=n)
    assert curve_l.specialize(n) == at_n
    g_n = Poly(at_n.as_expr(), L, z)
    w_n = [int(c) for c in series(ssnf, N, k=n)[L]]
    total, power = [0] * (N + 1), [1] + [0] * N
    for j in range(g_n.degree(L) + 1):
        a_j = [int(g_n.coeff_monomial(L**j * z**i)) for i in range(N + 1)]
        total = [x + y for x, y in zip(total, times(a_j, power), strict=True)]
        power = times(power, w_n)
    assert total == [0] * (N + 1)
print(f"eq. (17) at k = 1, 2, 10, 50, 54: satisfied by the series up to z^{N}")

# %% [markdown]
# The denominator series $F_k = z + B_k + \bar B_k$ of eq. (38) has a cubic curve:

# %%
F = symbols("F")
fk = System({B: ssnf.equations[B], Bb: ssnf.equations[Bb], F: z + B + Bb}, z=z, k=k)
print("F_k:", eliminate_in_k(fk, F))
