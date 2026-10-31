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
# - Step 4 (singularity): done, $\rho_k$ of $R_k$ (the positive root of $\Delta_k$,
#   §3.1, Case I) and of $L_k$ (the positive root of $p_k$, §3.4, Case II) for every
#   $k \ge 1$, $\rho_{54} = 2/45$; $F_k$ in Case I. $\rho_k$ is the only singularity
#   on $|z| = \rho_k$ for every $k$, including $k = 108$ (see below).
# - Step 5 (Newton-Puiseux): done, $\alpha = 1/2$ for $R_k$, $L_k$ (eq. (32)) and $F_k$
#   for every $k \ge 1$; $b_k$ of $R_k$ (§3.1), $c_k$ of $L_k$ (eq. (34)) and $b_k$ of
#   $F_k$ exactly, one piece $[1, \infty)$ each; the sign of eq. (37) (see below).
# - Step 6 (transfer): done, eq. (6) for $R_k$; eq. (8),
#   $[z^n] S_k / (n [z^n] R_k) \to \rho_k$ exactly (next term $\rho_k/(2n)$); Thm 3.2
#   for $L_k$ and $F_k$: $[z^n] L_k / (n [z^n] F_k) \to 2/(b_k c_k)$ exactly, one
#   piece $[1, \infty)$; Table 1's exact limits (see below).
# - Step 7 (families in $k$): eqs. (35)-(38) (expansions in $1/k$).

# %%
import time
import warnings
from fractions import Fraction
from math import comb, factorial

import mpmath
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

from flanpy import (
    KRoot,
    KSet,
    SingularityWarning,
    System,
    asymptotics_in_k,
    eliminate,
    eliminate_in_k,
    local_expansion,
    local_expansion_in_k,
    ratio,
    series,
    series_in_k,
    singularity,
    singularity_in_k,
)

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

# %% [markdown]
# ## Step 4: the dominant singularity
#
# {func}`flanpy.singularity_in_k` gives the radius of convergence $\rho_k$ for every
# $k \ge 1$, as a {class}`flanpy.KRoot` ("the $j$-th positive root of $f(z, k)$"), with
# Case I ($w(\rho_k) = a_k$ finite) or Case II ($w \to \infty$), piece by piece.
#
# ### $R_k$
#
# §3.1 (p. 7): $\rho_k$ is the positive root of $\Delta_k = 1 - 2z - (7 + 8k) z^2$, the
# radicand of eq. (2). At $\rho_k$ the square root vanishes, so
# $a_k = R_k(\rho_k) = (1 - \rho_k)/(4 \rho_k)$, which works out to
# $\sqrt{(k + 1)/2}$.

# %%
r_sing = singularity_in_k(r_cases)
print(r_sing)
((piece, s_r),) = r_sing.pieces
assert piece == KSet.all() and s_r.case == "I" and s_r.unique is True
assert s_r.rho == KRoot(delta, 0, k=k)
assert s_r.a == KRoot(2 * R**2 - k - 1, 0, k=k)
for n in (1, 2, 10, 50):
    rho_n, a_n = s_r.specialize(n).rho, s_r.specialize(n).a
    assert a_n is not None
    with mpmath.workdps(50):
        exact = (mpmath.sqrt(8 + 8 * n) - 1) / (7 + 8 * n)
        assert abs(rho_n.evalf(50) - exact) < mpmath.mpf(10) ** -45
        assert abs(a_n.evalf(50) - (1 - exact) / (4 * exact)) < mpmath.mpf(10) ** -45

# %% [markdown]
# ### $L_k$
#
# §3.4: the singularities of $L_k$ are zeros of $p_k$, and $\rho_k$ is its unique zero
# in $]0, 1[$. flanpy finds, for every $k \ge 1$, that $\rho_k$ is the positive root
# of $p_k$ and that $L_k$ is in Case II. The candidates are the roots of
# $\operatorname{res}_w(G, \partial G / \partial w) = k^3 z^3 p_k^2 q_k^2$, with a
# second cubic $q_k$. **Note:** §3.4 prints
# $\operatorname{res}_w(G, \partial G/\partial w) =
# z p_k(z) / (k^2 (4k^2 z^4 - 4k^2 z^3 + k(k + 24) z^2 + 4(k - 4) z - 4))$; this
# differs from the computed resultant $k^3 z^3 p_k^2 q_k^2$; the conclusion stands; to
# confirm with the authors.

# %%
q_k = (
    (2 * k**3 + 36 * k**2 + 108 * k) * z**3
    + (12 * k**2 + 72 * k) * z**2
    + 15 * k * z
    - 2
)
res = resultant(G, diff(G, W), W)
assert expand(res - k**3 * z**3 * p_k**2 * q_k**2) == 0
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("error", SingularityWarning)  # uniqueness is decided
    l_sing = singularity_in_k(l_cases)
print(f"singularity_in_k(L_k): {time.perf_counter() - start:.1f} s")
print(l_sing)
((piece, s_l),) = l_sing.pieces
assert piece == KSet.all() and s_l.rho == KRoot(p_k, 0, k=k)
assert s_l.case == "II" and s_l.period == 1 and s_l.unique is True

# %% [markdown]
# The positive root of $q_k$ is a candidate below $\rho_k$, but not a singularity of the
# series: the branch passes through it analytically (another pair of sheets meets
# there). For Table 1's $k$, $\rho_k$ agrees with the positive root of $p_k$ computed by
# mpmath, and with the coefficient ratio at $n = 1000$, corrected by
# $((n + 1)/n)^{-1/2}$ for $[z^n] L_k \sim C \rho_k^{-n} n^{-1/2}$ (eq. (32),
# $\alpha = 1/2$).

# %%
for n in (2, 10, 50):
    rho_n = s_l.specialize(n).rho
    (q_root,) = [r for r in Poly(q_k.subs(k, n), z).real_roots() if r > 0]
    assert q_root < rho_n.as_expr()
    with mpmath.workdps(50):
        roots = mpmath.polyroots(
            [int(c) for c in Poly(p_k.subs(k, n), z).all_coeffs()], extraprec=200
        )
        (positive,) = [
            mpmath.re(r)
            for r in roots
            if abs(mpmath.im(r)) < mpmath.mpf(10) ** -30 and mpmath.re(r) > 0
        ]
        assert abs(rho_n.evalf(50) - positive) < mpmath.mpf(10) ** -40
        c = series(ssnf, 1001, k=n)[L]
        quotient = mpmath.mpf(c[1000].numerator) / c[1000].denominator
        quotient /= mpmath.mpf(c[1001].numerator) / c[1001].denominator
        quotient *= (mpmath.mpf(1001) / 1000) ** mpmath.mpf(-0.5)
        error = abs(quotient / rho_n.evalf(50) - 1)
        assert error < mpmath.mpf(10) ** -6
    value = mpmath.nstr(rho_n.evalf(20), 15)
    print(f"k = {n}: rho_k = {value}, ratio error {mpmath.nstr(error, 2)}")

# %% [markdown]
# **$k = 54$**: $p_{54} = 2 (45z - 2)(18z + 1)^2$, so $\rho_{54} = 2/45$. The value 54
# is a root of the discriminant of $p_k$; it is checked separately and merges into the
# generic piece.
#
# **$k = 108$**: $p_{108} = 4 (27z + 1)(972z^2 - 1)$, so $-\rho_{108}$ is also a root of
# $p_{108}$ and lies on the circle $|z| = \rho_{108}$. It is not a singularity of
# $L_{108}$. flanpy proves this from the structure of the system: its recursive
# components, $\{B_k, \bar B_k\}$ (nonlinear) and $\{M_k, \bar M_k\}$ (linear), both
# have period 1 (Wielandt and the daffodil lemma; see the Notes of
# {func}`flanpy.singularity`). The coefficients agree: they show no $(-1)^n$ term.
#
# **Note on §3.4.** For $k > 54$ the paper argues that if $-\rho_k$ were a root, the
# third root $\mu_k$ would satisfy $\mu_k = -9/(2k + 27) < 0$ and
# $\mu_k = 1/(k(2k + 27)\rho_k^2) > 0$. In the text as we extracted it, the product of
# the roots, $1/(k(2k + 27))$, gives $\mu_k = -1/(k(2k + 27)\rho_k^2) < 0$ instead,
# with no contradiction, and $-\rho_k$ is a root exactly at $k = 108$. The conclusion
# (uniqueness for every $k$) stands; to check against the printed text, then raise
# with the authors.

# %%
at54 = s_l.specialize(54)
assert at54.rho == Fraction(2, 45) and at54.rho.polynomial.as_expr() == 45 * z - 2
assert expand(p_k.subs(k, 54) - 2 * (45 * z - 2) * (18 * z + 1) ** 2) == 0
assert expand(p_k.subs(k, 108) - 4 * (27 * z + 1) * (972 * z**2 - 1)) == 0
at108 = s_l.specialize(108)
assert at108.rho.polynomial.as_expr() == 972 * z**2 - 1 and at108.unique is True
mu = Fraction(-1, 27)  # the third root of p_108
assert (
    mu == Fraction(-9, 2 * 108 + 27) == -1 / (108 * (2 * 108 + 27) * Fraction(1, 972))
)
with mpmath.workdps(30):
    c = series(ssnf, 400, k=108)[L]
    rho = 1 / mpmath.sqrt(972)
    scaled = [c[n] * rho**n * mpmath.sqrt(n) for n in (398, 399, 400)]
    assert abs(2 * scaled[1] / (scaled[0] + scaled[2]) - 1) < mpmath.mpf(10) ** -5

# %% [markdown]
# ### $F_k$
#
# The denominator $F_k$ of eq. (38) has the same $\rho_k$ but is in Case I: $a_k$ is
# the positive root of a cubic factor of $\operatorname{res}_z(p_k, G)$.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("error", SingularityWarning)
    f_sing = singularity_in_k(eliminate_in_k(fk, F))
((f_piece, s_f),) = f_sing.pieces
assert f_piece == KSet.all()
assert s_f.rho == KRoot(p_k, 0, k=k) and s_f.case == "I" and s_f.unique is True
assert s_f.a is not None and s_f.a.polynomial.degree(F) == 3
print("F_k: a_k =", s_f.a)
print("F_2:", singularity(eliminate(fk, F, k=2)))

# %% [markdown]
# ## Step 5: the local expansion
#
# {func}`flanpy.local_expansion_in_k` expands each series at $\rho_k$ in powers of
# $s = 1 - z/\rho_k$ (eqs. (20)-(31)). The exponent $\alpha$ comes from the Newton
# polygon, and the constant ($b_k$ in Case I, $c_k$ in Case II) is a
# {class}`flanpy.KRoot` *relative to* $\rho_k$: "the positive root of
# $\Phi(y, \rho_k, k)$" (`over=rho_k`).
#
# ### $R_k$
#
# §3.1 (p. 7): $R_k \sim -\frac{\sqrt{2 - 2\rho_k}}{4\rho_k} (1 - z/\rho_k)^{1/2}$, so
# $\alpha = 1/2$ and $b_k = \sqrt{2 - 2\rho_k}/(4\rho_k)$.

# %%
r_local = local_expansion_in_k(r_sing)
((piece, e_r),) = r_local.pieces
print(piece, e_r)
assert piece == KSet.all() and e_r.alpha == Fraction(1, 2) and e_r.ramification == 2
assert e_r.b is not None
for n in (1, 2, 10, 50):
    with mpmath.workdps(50):
        rho_n = e_r.singularity.rho.at(n).evalf(50)
        closed = mpmath.sqrt(2 - 2 * rho_n) / (4 * rho_n)
        assert abs(e_r.b.at(n).evalf(50) - closed) < mpmath.mpf(10) ** -45

# %% [markdown]
# ### $L_k$: eqs. (32)-(34)
#
# $L_k$ is in Case II: $1/L_k = c_k s^{1/2} + \dots$, with $\alpha = 1/2$ from the
# Newton polygon of eq. (32), a single edge, and $c_k = \sqrt{2\rho_k H_z / H_{vv}}$ at
# $(\rho_k, 0)$ (eqs. (33)-(34)). flanpy finds one piece $[1, \infty)$, and $c_k$ is the
# positive root of $((k^2 + 12k)\rho_k^2 + 4k\rho_k + 1)\, y^2 -
# ((24k + 324)\rho_k^2 + 72\rho_k - 1)$, which is eq. (34) with
# $H_z = -p_k'(\rho_k)$. The cut $k = 54$ (where $p_{54}$ has a double root) is checked
# separately and merges; $c_{54} = \sqrt{15}/7$. The limit $c_k \sim 4 k^{-1/2}$
# (eq. (36)) is step 7.

# %%
l_local = local_expansion_in_k(l_sing)
((piece, e_l),) = l_local.pieces
assert piece == KSet.all() and e_l.singularity.case == "II"
assert e_l.alpha == Fraction(1, 2) and e_l.ramification == 2 and e_l.c is not None
y = symbols("y")
eq34 = ((k**2 + 12 * k) * z**2 + 4 * k * z + 1) * y**2 - (
    (24 * k + 324) * z**2 + 72 * z - 1
)
assert e_l.c == KRoot(eq34, 0, k=k, over=KRoot(p_k, 0, k=k))
for n in (2, 10, 50, 54, 108):
    c_n = e_l.c.at(n)
    with mpmath.workdps(50):
        r, kk = e_l.singularity.rho.at(n).evalf(50), mpmath.mpf(n)
        p_prime = 12 * kk * (2 * kk + 27) * r**2 + 72 * kk * r - kk
        h_vv = 2 * kk * r * ((kk + 12) * kk * r**2 + 4 * kk * r + 1)
        assert abs(c_n.evalf(50) - mpmath.sqrt(2 * r * p_prime / h_vv)) < 1e-40
        c_scaled = c_n.evalf(30) * mpmath.sqrt(kk) / 4
    value = mpmath.nstr(c_n.evalf(20), 15)
    print(f"k = {n}: c_k = {value}, c_k sqrt(k)/4 = {mpmath.nstr(c_scaled, 4)}")
assert e_l.c.at(54).polynomial.as_expr() == 49 * y**2 - 15

# %% [markdown]
# ### $F_k$: eq. (37) and a sign
#
# $F_k$ is in Case I with $\alpha = 1/2$ (eq. (37)); flanpy finds one piece and $b_k$
# exactly, relative to $\rho_k$. Eq. (20) writes $w = a - g(s) s^\alpha$ with
# $b = g(0)$, and $w$ increases to $a$, so $b_k > 0$; flanpy finds
# $b_k / \sqrt{k} \to 1$. So $|b_k| \sim k^{1/2}$ as eq. (37) says, but eq. (37) prints
# $b_k = -k^{1/2} + o(k^{1/2})$, and eq. (38) writes the ratio as $-2/(b_k c_k)$: both
# use $-b_k$ (the constant $\lambda$ of Thm 3.1) relative to eq. (20)'s $b$. The limit
# $1/2$ of eq. (38) is unaffected. To check against the printed eqs. (20), (37), (38),
# then raise with the authors.

# %%
f_local = local_expansion_in_k(f_sing)
((piece, e_f),) = f_local.pieces
assert piece == KSet.all() and e_f.alpha == Fraction(1, 2) and e_f.b is not None
assert e_f.b.over == KRoot(p_k, 0, k=k)
for n in (2, 10, 50, 108):
    b_n = e_f.b.at(n)
    assert b_n > 0 and b_n == local_expansion(s_f.specialize(n)).b
    print(
        f"k = {n}: b_k = {mpmath.nstr(b_n.evalf(20), 15)}, b_k/sqrt(k) = "
        f"{mpmath.nstr(b_n.evalf(20) / mpmath.sqrt(n), 6)}"
    )

# %% [markdown]
# ## Step 6: the coefficients
#
# {func}`flanpy.asymptotics_in_k` transfers the local expansions to the coefficients
# (Thm 3.1, FS Cor. VI.1): $[z^n] w \sim -b/\Gamma(-\alpha)\, \rho^{-n} n^{-\alpha-1}$
# in Case I and $1/(c\, \Gamma(\alpha))\, \rho^{-n} n^{\alpha-1}$ in Case II
# (Thm 3.2). The leading constants are exact monomials in the {class}`flanpy.KRoot` s
# of step 5, here with $\Gamma(1/2) = \sqrt\pi$.
#
# ### $R_k$: eq. (6)
#
# $[z^n] R_k \sim \sqrt{2 - 2\rho_k}/(8\rho_k\sqrt\pi)\, \rho_k^{-n} n^{-3/2}$, that is
# $\tfrac12 b_k/\Gamma(1/2)$; at $k = 2$, three non-zero terms are within $10^{-9}$ of
# the exact coefficient at $n = 1000$.

# %%
r_asym = asymptotics_in_k(r_local)
((piece, a_r),) = r_asym.pieces
print(piece, a_r.exponent, a_r.leading.rational, a_r.leading.gamma)
assert a_r.exponent == Fraction(-3, 2) and a_r.leading.factors == ((e_r.b, 1),)
for n in (1, 2, 10, 50):
    with mpmath.workdps(50):
        rho_n = e_r.singularity.rho.at(n).evalf(50)
        closed = mpmath.sqrt(2 - 2 * rho_n) / (8 * rho_n * mpmath.sqrt(mpmath.pi))
        assert abs(a_r.leading.at(n).evalf(45) - closed) < mpmath.mpf(10) ** -40
r_2 = a_r.specialize(K)
exact = series(rs, 1000, k=K)[R][1000]
errors = [abs(r_2.evalf(1000, count=c, dps=30) / exact - 1) for c in (1, 3, 5)]
print([f"{float(x):.1e}" for x in errors])
assert errors[2] < 1e-9

# %% [markdown]
# ### Stars: eq. (8)
#
# $S_k$ is in Case II ($\alpha = 1/2$), $R_k$ in Case I, at the same $\rho_k$. The
# probability that a symbol is a star is the ratio
# $[z^n] S_k / (n [z^n] R_k) \to \rho_k$ (eq. (8)). {func}`flanpy.ratio` gives the limit
# as the monomial $2 b_k^{-1} c_k^{-1}$ (the $\sqrt\pi$ cancel), which equals $\rho_k$
# exactly at every $k$ checked, and the next term: the ratio is
# $\rho_k (1 + \tfrac{1}{2n} + O(n^{-2}))$.

# %%
s_asym = asymptotics_in_k(local_expansion_in_k(singularity_in_k(s_cases)))
((piece, star_ratio),) = s_asym.combine(r_asym, ratio).pieces
print(piece, star_ratio.exponent, star_ratio.limit.rational, star_ratio.limit.gamma)
assert star_ratio.exponent == 1 and star_ratio.limit.gamma == ()
for n in range(1, 7):
    fixed = star_ratio.specialize(n)
    rho_n = e_r.singularity.rho.at(n)
    assert star_ratio.limit.at(n).algebraic() == rho_n == fixed.limit.algebraic()
    next_term = fixed.terms(3)[2][1].algebraic()
    with mpmath.workdps(40):
        assert abs(next_term.evalf(30) - rho_n.evalf(30) / 2) < 1e-25

# %% [markdown]
# ### Letters in ssnf: Thm 3.2, eq. (38) and Table 1
#
# $L_k$ counts the letters and $F_k$ the expressions of size $n$ (Case I), at the same
# $\rho_k$, so the probability that a symbol is a letter is
# $[z^n] L_k / (n [z^n] F_k) \to 2/(b_k c_k)$ (Thm 3.2). With eq. (20)'s $b_k > 0$ the
# limit is $2/(b_k c_k)$; eq. (38) writes $-2/(b_k c_k)$ with $-b_k$ (the sign
# convention of step 5, above). flanpy gives it exactly, one piece $[1, \infty)$, for
# every $k \ge 1$: at $k = 54$ it is $7/15$.
#
# Table 1 compares a simulation (10000 random expressions in ssnf of size $n = 1000$)
# with the approximation $\tfrac12 - \tfrac{1}{4\sqrt 2 \sqrt k}$ of eq. (38). The exact
# limits fit the simulation better at $k = 2$ ($0.4173$ vs the simulated $0.42$, while
# the approximation gives $0.375$); at $k = 10$ and $50$ the exact limits round to the
# simulated values, and the approximation is within $0.005$ of them. At $n = 1000$ the
# ratio is the limit plus a term in $1/n$, as computed below.

# %%
l_asym, f_asym = asymptotics_in_k(l_local), asymptotics_in_k(f_local)
((piece, letters),) = l_asym.combine(f_asym, ratio).pieces
assert piece == KSet.all() and letters.exponent == 1 and letters.limit.rational == 2
assert set(letters.limit.factors) == {(e_l.c, -1), (e_f.b, -1)}
simulated = {2: 0.42, 10: 0.44, 50: 0.47}
print(" k   exact limit      Table 1   eq. (38)   n^-1 term")
for n in (2, 10, 50, 54, 108):
    limit = letters.limit.at(n).algebraic()
    approx = 0.5 - 1 / (4 * mpmath.sqrt(2 * n))
    next_term = letters.specialize(n).terms(3)[2][1].algebraic()
    shown = simulated.get(n, "")
    print(
        f"{n:3}   {mpmath.nstr(limit.evalf(15), 12):14}   {shown!s:7}   "
        f"{mpmath.nstr(approx, 4):8}   {mpmath.nstr(next_term.evalf(15), 6)}"
    )
    if n in simulated:
        assert abs(limit.evalf(15) - simulated[n]) <= abs(approx - simulated[n]) + 0.005
assert letters.limit.at(54).algebraic() == Fraction(7, 15)
limit_2 = letters.limit.at(2).algebraic().evalf(20)
assert abs(limit_2 - mpmath.mpf("0.417256345028")) < 1e-12

# %% [markdown]
# At $n = 1000$ and $k = 2$, against the exact coefficients: the limit alone is off by
# $10^{-3}$, and with the $1/n$ term by $10^{-8}$.

# %%
fixed = letters.specialize(K)
exact = Fraction(series(ssnf, 1000, k=K)[L][1000]) / series(fk, 1000, k=K)[F][1000]
errors = [abs(fixed.evalf(1000, count=c, dps=30) / exact - 1) for c in (1, 3)]
print([f"{float(x):.1e}" for x in errors])
assert errors[0] < 1.2e-3 and errors[1] < 5e-8
