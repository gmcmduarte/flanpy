---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Singularity analysis

Singularity analysis finds the dominant singularity $\rho$ of the series on a curve
and the value $a = w(\rho)$. Both are real algebraic numbers, and flanpy keeps them
exact. This page covers those numbers. The dominant singularity itself (Pringsheim's
theorem, Case I and Case II {cite:p}`broda-2020-sigact`) follows in the next part of
build-order step 4.

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

## The polynomial must be minimal

The polynomial is checked: integer coefficients, primitive, positive leading
coefficient, and irreducible over $\mathbb{Q}$. A polynomial that factors is rejected
rather than silently replaced. For example, $p_k$ of {cite:t}`broda-2020-sigact`
eq. (18) is irreducible for symbolic $k$, but at $k = 54$ it factors, and its positive
root is rational:

```{code-cell} python
from sympy import factor

p54 = (4 * 54 * (2 * 54 + 27) * z**3 + 36 * 54 * z**2 - 54 * z - 4) / 2  # primitive
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
