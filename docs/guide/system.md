---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Systems and their coefficients

A {class}`flanpy.System` holds the equations that the symbolic method produces, and
{func}`flanpy.series` and {func}`flanpy.series_in_k` compute their exact power-series
coefficients. This page shows the API. The running examples show what it reproduces
from the papers.

## Building a system

Give a mapping from each unknown to its right-hand side, a polynomial in the
unknowns, the size variable `z` and, optionally, the parameter `k` (the alphabet size,
an integer $k \ge 1$):

```{code-cell} python
from sympy import symbols

from flanpy import System

z, k, F, R = symbols("z k F R")
catalan = System({F: 1 + z * F**2}, z=z)
rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)  # Broda 2020, (1)
print(catalan)
print(rk)
```

`System` is immutable and compares by content. Construction checks that the system
is well-posed: the iteration $F^{(0)} = 0$, $F^{(n+1)} = \Phi(z, k, F^{(n)})$
converges, for symbolic $k$. Its limit is the solution. The criterion and its sources
are in the {class}`~flanpy.System` reference.

## Fixed-point and polynomial forms

{attr}`~flanpy.System.equations` is the fixed-point form $F = \Phi(z, k, F)$, and
{meth}`~flanpy.System.polynomials` the polynomial form $F - \Phi = 0$ used for
elimination, over $\mathbb{Q}$ with `k` as the last generator:

```{code-cell} python
print(rk.equations)
print(rk.polynomials())
```

## Fixing the alphabet size

{meth}`~flanpy.System.specialize` substitutes a value $k \ge 1$. A system that is
well-posed for symbolic $k$ stays well-posed for every $k$:

```{code-cell} python
print(rk.specialize(2))
```

## When a system is rejected

A system whose iteration provably diverges raises
{class}`~flanpy.IllPosedSystemError`. The Catalan equation without its `z` is an
example:

```{code-cell} python
from flanpy import IllPosedSystemError, UndecidedSystemError

try:
    System({F: 1 + F**2}, z=z)
except IllPosedSystemError as error:
    print(error)
```

For systems with negative coefficients the check is only partial. When it cannot
decide, it raises {class}`~flanpy.UndecidedSystemError`, which never claims that the
system is ill-posed. Both errors derive from {class}`~flanpy.WellPosednessError`.

```{code-cell} python
try:
    System({F: z - F}, z=z)
except UndecidedSystemError as error:
    print(error)
```

Unknowns that are identically zero are kept as 0, with a
{class}`~flanpy.ZeroCoordinateWarning`:

```{code-cell} python
import warnings

G = symbols("G")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    System({F: z + G, G: G}, z=z)
print(caught[0].message)
```

## Coefficients

{func}`~flanpy.series` returns, for each unknown, the coefficients of
$z^0, \dots, z^n$, **with $n$ included**: `n + 1` exact numbers, `int` or `Fraction`.
For a system with `k`, pass the alphabet size:

```{code-cell} python
from flanpy import series, series_in_k

print(series(catalan, 8))
print(series(rk, 8, k=2))
```

{func}`~flanpy.series_in_k` keeps $k$ symbolic and returns polynomials in $k$. Use it
for small $n$ (up to a few hundred); for large $n$, fix $k$.

```{code-cell} python
print([c.as_expr() for c in series_in_k(rk, 4)[R]])
```

With python-flint installed (`pip install flanpy[fast]`) the computation is faster.
The results are the same.
