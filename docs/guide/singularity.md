---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Singularity analysis

Singularity analysis finds the dominant singularity $\rho$ of the series on a curve
and the value $a = w(\rho)$. Both are real algebraic numbers, and flanpy keeps them
exact. This page first covers those numbers, then the dominant singularity itself
(Pringsheim's theorem, Case I and Case II {cite:p}`broda-2020-sigact`), at a fixed $k$
and for symbolic $k$.

## Exact algebraic reals

An {class}`flanpy.AlgebraicReal` is a real root of a polynomial with integer
coefficients, given by its **minimal polynomial** and its **index** among the real
roots in increasing order (the convention of {class}`sympy.CRootOf`). The radius of
convergence of $R_k$ in {cite:t}`broda-2020-sigact` is the positive root of
$\Delta_k(z) = 1 - 2z - (7 + 8k) z^2$. At $k = 2$ the two real roots of
$23 z^2 + 2z - 1$ are negative and positive, so $\rho_2$ has index 1:

```{code-cell} python
from fractions import Fraction

from sympy import symbols

from flanpy import AlgebraicReal

z = symbols("z")
rho = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
print(rho)
print(repr(rho), rho.polynomial, rho.index)
print(AlgebraicReal(23 * z**2 + 2 * z - 1, 0))  # the other root
```

The pair (minimal polynomial, index) is canonical, so equality and hashing are exact.
The name of the variable does not matter, and a rational number equals the
corresponding `Fraction`:

```{code-cell} python
w = symbols("w")
print(rho == AlgebraicReal(23 * w**2 + 2 * w - 1, 1))
quarter = AlgebraicReal(4 * z - 1, 0)  # Catalan's radius
print(quarter == Fraction(1, 4), {quarter, Fraction(1, 4)})
```

## The polynomial must be irreducible

The polynomial is normalised silently: rational coefficients, a sign or a common
factor do not change the roots, so `1 - 4z` and `z/2 - 1/8` give the same number as
`4z - 1`. A polynomial that factors is rejected rather than replaced, because then the
index would not say which factor is meant. For example, $p_k$ of
{cite:t}`broda-2020-sigact` eq. (18) is irreducible for symbolic $k$, but at $k = 54$
it factors, and its positive root is rational:

```{code-cell} python
from sympy import factor

print(AlgebraicReal(1 - 4 * z, 0) == quarter)
p54 = 4 * 54 * (2 * 54 + 27) * z**3 + 36 * 54 * z**2 - 54 * z - 4
print(factor(p54))
try:
    AlgebraicReal(p54, 1)
except ValueError as error:
    print("ValueError:", error)
rho54 = AlgebraicReal(45 * z - 2, 0)
print(rho54 == Fraction(2, 45))
```

## Exact signs and comparisons

{meth}`~flanpy.AlgebraicReal.sign` gives the sign of a polynomial at the number.
It is zero exactly when the minimal polynomial divides it, which is the gcd test of
{cite:t}`broda-2020-sigact` (section 3.3). Otherwise the isolating interval is narrowed
until the sign is certain:

```{code-cell} python
print(rho.sign(1 - 2 * z - 23 * z**2))  # Δ_2(ρ_2) = 0
print(rho.sign(z - Fraction(16952, 10**5)), rho.sign(z - Fraction(16953, 10**5)))
print(rho54.sign(p54), rho54.sign(p54.diff(z)))  # a simple root of p_54
```

Comparisons with other algebraic reals, integers and fractions are exact too. Two
different numbers are told apart however close they are:

```{code-cell} python
sqrt2 = AlgebraicReal(z**2 - 2, 1)
near = AlgebraicReal(10**30 * z**2 - (2 * 10**30 + 1), 1)  # within 1e-30 of sqrt 2
print(sqrt2 < near, near == sqrt2, Fraction(1, 6) < rho < Fraction(1, 5))
print(sorted([Fraction(3, 2), near, sqrt2, 1]))
```

Floats are not compared (`sqrt2 < 1.5` raises `TypeError`): a float comparison would
not be exact.

## Isolating intervals and evaluation

The number keeps an isolating interval with rational endpoints. The interval contains
no other root of the polynomial and narrows when needed.
{meth}`~flanpy.AlgebraicReal.refine` narrows it on request, and
{meth}`~flanpy.AlgebraicReal.evalf` evaluates the number to any precision with mpmath:

```{code-cell} python
import mpmath

cbrt2 = AlgebraicReal(z**3 - 2, 0)
print(cbrt2.interval)
print(cbrt2.refine(Fraction(1, 10**12)).interval)
print(mpmath.nstr(rho.evalf(50), 50), float(rho))
```

`evalf(50)` returns an `mpmath.mpf` accurate to 50 digits. mpmath prints numbers at
its global precision (15 digits by default), so print with `mpmath.nstr` as above, or
inside `mpmath.workdps(50)`.

{meth}`~flanpy.AlgebraicReal.as_expr` gives the same number as a SymPy expression:

```{code-cell} python
print(rho.as_expr(), quarter.as_expr())
```

## The dominant singularity

{func}`flanpy.singularity` analyses the series $w(z)$ of the curve's system. Its
coefficients are non-negative, so its radius of convergence $\rho$ is a singularity
(Pringsheim's theorem), and a root of $\operatorname{lc}_w G \cdot \operatorname{disc}_w
G$. Catalan's $F = 1 + zF^2$ is in **Case I**: $F$ stays finite at $\rho = 1/4$, with
$a = F(1/4) = 2$:

```{code-cell} python
from flanpy import System, eliminate, singularity

F = symbols("F")
catalan = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
print(catalan)
print(catalan.rho == Fraction(1, 4), catalan.a == 2)
```

A root of $\operatorname{lc}_w G \cdot \operatorname{disc}_w G$ need not be a
singularity of the series. For the letters $L_k$ of {cite:t}`broda-2020-sigact` at
$k = 2$, the factor $q_2$ has a positive root below $\rho_2$ where the series is
analytic. flanpy locates $\rho$ among the candidates by exact comparisons, so it skips
that root. $L_k$ is in **Case II**: $L_k(x) \to \infty$ as $x \to \rho^-$, and
$\rho_2$ is a root of $p_2$, the leading coefficient of eq. (17):

```{code-cell} python
from sympy import Poly, discriminant

k = symbols("k")
B, Bb, M, Mb, L = symbols("B Bb M Mb L")
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
curve = eliminate(ssnf, L, k=2)
l2 = singularity(curve)
print(l2)
g = curve.as_expr()
print(factor(Poly(g, L).LC() * discriminant(g, L)))
q2_root = AlgebraicReal(188 * z**3 + 96 * z**2 + 15 * z - 1, 0)
print(q2_root, q2_root < l2.rho)
```

## Uniqueness

Transfer theorems need $\rho$ to be the only singularity on the circle
$|z| = \rho$. {attr}`~flanpy.Singularity.unique` is `True` when that is proved, `False`
when it is disproved, and `None`, with a {class}`~flanpy.SingularityWarning`, when it
is undecided. Binary trees $T = z + zT^2$ have only odd sizes (period 2), so $-\rho$ is
a singularity too:

```{code-cell} python
T = symbols("T")
binary = singularity(eliminate(System({T: z + z * T**2}, z=z), T))
print(binary.period, binary.unique)
```

Another candidate on the circle need not be a singularity. Uniqueness is then proved
from the structure of the system: if every recursive strongly connected component of
the closure has period 1, $\rho$ is the only dominant singularity (Wielandt's theorem
and the daffodil lemma; see the Notes of {func}`flanpy.singularity`). The rule looks
at every component, so a periodic component anywhere leaves the question open. For
$W = z^2 + T$, $W$ has period 1, but $T$'s component has period 2, and $W$ is indeed
singular at $-1/2$:

```{code-cell} python
import warnings

W = symbols("W")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    two = singularity(eliminate(System({T: z + z * T**2, W: z**2 + T}, z=z), W))
print(two.period, two.unique)
print(caught[0].message)
```

Systems with a negative coefficient are refused: Pringsheim's theorem needs
non-negative coefficients.

```{code-cell} python
from flanpy import UndecidedSingularityError

try:
    singularity(eliminate(System({F: 1 - z * F**2}, z=z), F))
except UndecidedSingularityError as error:
    print(error)
```

## Symbolic $k$

{func}`flanpy.singularity_in_k` takes the result of
{func}`~flanpy.eliminate_in_k` and describes $\rho_k$ for every $k \ge 1$, piece by
piece, as a {class}`flanpy.KRoot`: the $j$-th positive root of a polynomial
$f(z, k)$. The case, the period and uniqueness are constant on each piece. For $R_k$ of
{cite:t}`broda-2020-sigact`, $\rho_k$ is the positive root of
$\Delta_k = 1 - 2z - (7 + 8k) z^2$ for every $k$:

```{code-cell} python
from flanpy import KRoot, eliminate_in_k, singularity_in_k

R = symbols("R")
rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
r_cases = singularity_in_k(eliminate_in_k(rk, R))
print(r_cases)
print(r_cases[1].rho == KRoot(1 - 2 * z - (7 + 8 * k) * z**2, 0, k=k))
```

The pieces come from the real roots of finitely many polynomials in $k$, where the
description might change. Each interval between them is decided by one exact analysis
at an integer, and each integer root separately. Equal descriptions are merged. For
$L_k$, $\rho_k$ is the positive root of $p_k$, eq. (18), in Case II, and the only
dominant singularity, for every $k$. At $k = 108$, $p_{108} = 4(27z + 1)(972z^2 - 1)$,
so $-\rho_{108}$ is also a root and lies on the circle. It is not a singularity: the
recursive components of $L_k$'s system, $\{B, \bar B\}$ and $\{M, \bar M\}$, have
period 1.

```{code-cell} python
l_cases = singularity_in_k(eliminate_in_k(ssnf, L))
for piece, s in l_cases.pieces:
    print(piece, "|", s.rho, "| Case", s.case, "| unique:", s.unique)
print(l_cases[108].specialize(108).rho)
```

{meth}`Singularity.specialize <flanpy.Singularity.specialize>` gives the result at one
$k$, with the exact minimal polynomials there. At $k = 54$, $p_{54}$ factors and
$\rho_{54} = 2/45$:

```{code-cell} python
at54 = l_cases[54].specialize(54)
print(at54.rho, at54.rho.polynomial.as_expr())
```
