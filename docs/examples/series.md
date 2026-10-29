---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Worked example: coefficients of a system

A {class}`flanpy.System` holds the equations $F = \Phi(z, k, F)$ that the symbolic
method produces. {func}`flanpy.series` returns the exact coefficients of
$z^0, \dots, z^n$ of every unknown, **with $n$ included**. This page is executed in
every docs build, and its `assert` statements make the build fail if a number stops
reproducing.

## Catalan numbers

The generating function of binary trees satisfies $F = 1 + z F^2$.

```{code-cell} python
from math import comb

from sympy import symbols

from flanpy import System, series

z, k, F, R = symbols("z k F R")
catalan = System({F: 1 + z * F**2}, z=z)
coefficients = series(catalan, 10)[F]  # indices 0..10
assert coefficients == tuple(comb(2 * n, n) // (n + 1) for n in range(11))
coefficients
```

## Regular expressions over a two-letter alphabet

{cite:t}`broda-2020-sigact` count the regular expressions given by grammar (1),
$r ::= \varepsilon \mid \sigma \in \Sigma \mid r + r \mid r \cdot r \mid r^\star$,
by their number of symbols. Their generating function satisfies
$R_k = (k+1) z + 2 z R_k^2 + z R_k$, where $k = |\Sigma|$.

```{code-cell} python
rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
r2 = series(rk, 10, k=2)[R]
r2
```

Eq. (3) of the paper gives these numbers in closed form:

$$
[z^n] R_k(z) = \frac{1}{2^{n+3}} \sum_{j=0}^{\lfloor (n+1)/2 \rfloor}
\frac{1}{2(n-j)+1} \binom{2(n-j+1)}{n-j+1} \binom{n-j+1}{j} (7+8k)^j .
$$

```{code-cell} python
from fractions import Fraction


def eq3(n, k):
    total = sum(
        Fraction(1, 2 * (n - j) + 1)
        * comb(2 * (n - j + 1), n - j + 1)
        * comb(n - j + 1, j)
        * (7 + 8 * k) ** j
        for j in range((n + 1) // 2 + 1)
    )
    return total / 2 ** (n + 3)


assert all(r2[n] == eq3(n, 2) for n in range(1, 11))
```

With {func}`flanpy.series_in_k`, the same coefficients come out as polynomials in $k$:

```{code-cell} python
from flanpy import series_in_k

[c.as_expr() for c in series_in_k(rk, 4)[R]]
```
