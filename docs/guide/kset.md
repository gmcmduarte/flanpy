---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Conditions on the alphabet size

The alphabet size $k$ is the only parameter, an integer $k \ge 1$. Every condition on
$k$ that the pipeline produces is a sign or vanishing condition on a polynomial in $k$,
so the set of $k$ where a result holds is a finite union of integer points and integer
intervals. A {class}`flanpy.KSet` is such a set. This page shows the API. The Broda
2020 running example shows the sets that its discriminant gives.

## Building and printing

Give intervals `(lo, hi)`, with `hi = None` for $[lo, \infty)$, in any order. They are
merged into a canonical form, which is what equality and hashing compare:

```{code-cell} python
from flanpy import KSet

a = KSet([(55, None), (1, 50), (51, 53)])
print(a)
print(repr(a))
print(a.intervals)
print(KSet.of(54, 2), KSet.all(), KSet())
```

A one-point interval prints as $\{n\}$. Adjacent pieces are always merged, so
$\{54\} \cup [55, \infty)$ prints as $[54, \infty)$.

## Set operations

As for `frozenset`, with the complement taken in $k \ge 1$:

```{code-cell} python
b = KSet([(50, 60)])
print(a & b, "|", a | b, "|", a - b, "|", ~a)
print(54 in a, 10**9 in a, 0 in a)
print(KSet.of(2) <= a, a.isdisjoint(KSet.of(54)))
```

## From a condition

{meth}`KSet.where <flanpy.KSet.where>` gives the set of $k \ge 1$ where
`p(k) rel 0`, for a polynomial `p` in `k` with rational coefficients and `rel` one of
`"=="`, `"!="`, `"<"`, `"<="`, `">"`, `">="`. The computation is exact (real-root
isolation and exact signs at integers; no floating point):

```{code-cell} python
from sympy import symbols

k = symbols("k")
p = (k - 3) ** 2 * (2 * k**2 - 7 * k + 4)  # roots 3 (double), (7 ± sqrt 17)/4
for rel in ("<", "==", ">", "!="):
    print(rel, KSet.where(p, rel, k=k))
```

`rel` is a string: in Python, `p == 0` is a `bool` that compares expressions, not a
condition.

## Rational functions

For a fraction, pass the denominator as `den`. The result never contains a root of
`den`, even where the fraction in lowest terms is defined, because the denominator
records a division that the derivation performed:

```{code-cell} python
print(KSet.where(k - 1, ">", k=k, den=k - 1))
print(KSet.where(k - 3, "<", k=k, den=k - 5))
```

To use the lowest terms instead, cancel first:

```{code-cell} python
from sympy import cancel, fraction

num, den = fraction(cancel((k**2 - 1) / (k - 1)))
print(KSet.where(num, ">", k=k, den=den))
```

## Large $k$

A `KSet` is either finite or contains all $k$ from some point on.
{attr}`~flanpy.KSet.tail_start` says from where, which is what a limit as
$k \to \infty$ needs. {attr}`~flanpy.KSet.finite_part` holds the elements below the
tail:

```{code-cell} python
c = KSet.where((k - 4) * (k - 9) * (k - 20), ">=", k=k)
print(c, "| tail from", c.tail_start, "| below it:", c.finite_part)
print(list(c.finite_part))
```

A finite set can be iterated over; iterating over an infinite one raises
`ValueError`.
