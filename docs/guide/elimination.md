---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Elimination: one curve per coordinate

A {class}`flanpy.System` has several unknowns. The later steps (singularities,
Newton–Puiseux, transfer) work on **one** algebraic equation $G(z, w) = 0$ that the
series $w(z)$ of one unknown satisfies. {func}`flanpy.eliminate` computes it at a fixed
$k$, and {func}`flanpy.eliminate_in_k` for symbolic $k$. This page shows the API. The
running examples reproduce eq. (17) of {cite:t}`broda-2020-sigact`.

## A fixed $k$, or no $k$

The result is a {class}`flanpy.Curve`. Its polynomial is the minimal polynomial of
$w(z)$, normalised: integer coefficients with no common factor, and a positive
leading coefficient in lex order $w > z > k$.

```{code-cell} python
from sympy import symbols

from flanpy import System, eliminate, eliminate_in_k

z, k, F, G, A, B = symbols("z k F G A B")
catalan = eliminate(System({F: 1 + z * F**2}, z=z), F)
print(catalan)
print(repr(catalan))
```

Only the unknowns that $w$ depends on take part. Here $A_k = 2z + B_k$ counts regular
expressions, from {cite:t}`broda-2014-hitchhiker`, eqs. (7)–(8). $B$ is eliminated:

```{code-cell} python
hg = System({B: k * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}, z=z, k=k)
curve = eliminate(hg, A, k=2)
print(curve, "  valid for k in", curve.kset)
print(curve.system)  # the system at k = 2: its series is the branch on the curve
```

## Symbolic $k$: a {class}`~flanpy.KCases`

{func}`flanpy.eliminate_in_k` returns a {class}`flanpy.KCases`: a generic curve on a
{class}`flanpy.KSet` that contains all but finitely many $k$, and one piece for each
remaining $k$. The pieces cover every $k \ge 1$.

```{code-cell} python
cases = eliminate_in_k(hg, A)
print(cases)
generic_set, generic = cases.tail  # the piece that contains all large k
print(generic.polynomial)
```

The discriminant of this quadratic in $A$ is
$\Delta_k(z) = (4 - 8k) z^2 - 4z + 1$, eq. (9) of the Hitchhiker's Guide:

```{code-cell} python
from sympy import discriminant, expand

delta = discriminant(generic.as_expr(), A)
assert expand(delta - ((4 - 8 * k) * z**2 - 4 * z + 1)) == 0
print(delta)
```

## Exceptional $k$

The curve can change at a particular $k$. Here the coefficient $k - 1$ vanishes at
$k = 1$, the series there is $F = 1$, and the curve drops to degree 1:

```{code-cell} python
cases = eliminate_in_k(System({F: 1 + (k - 1) * z * F**2}, z=z, k=k), F)
print(cases)
print(cases[1], "|", cases[5])
```

Exceptional $k$ are found exactly, without sampling. The candidates are the integer
roots of polynomials in $k$ where any of these happens:

- a Gröbner basis or resultant stops specialising {cite:p}`kalkbrener-1997,gianni-1989`;
- the degree in $w$ drops;
- the curve stops being squarefree;
- its coefficients gain a common factor in $z$.

The curve is recomputed at each candidate. A candidate stays in the generic piece
when the recomputed curve agrees with the generic one. The second example has a
common factor $2z - 1$ at $k = 1$:

```{code-cell} python
system = System({F: 2 * z * F + z - 2 * z**2 + k * z * F**2 - 2 * z**2 * F**2}, z=z, k=k)
cases = eliminate_in_k(system, F)
print(cases)
```

On the generic piece, `G(z, w, n)` is proven annihilating, squarefree, of full degree,
and without a common factor in $z$. Irreducibility is proven for symbolic $k$. For each
$n$ it can fail only at finitely many $n$ {cite:p}`noether-1922,kaltofen-1995`, and those
are not computed. {meth}`Curve.specialize <flanpy.Curve.specialize>` certifies one
$n$, i.e. it returns the minimal polynomial there, which equals
`eliminate(system, w, k=n)`:

```{code-cell} python
generic = cases.tail[1]
assert all(generic.specialize(n) == eliminate(system, F, k=n) for n in range(2, 30))
print(generic.specialize(2))
```

## Choosing the factor

Elimination can produce a polynomial with several factors. Only one of them is
satisfied by the series. In this symmetric system $F = G$, and the eliminant has two
quadratic factors:

```{code-cell} python
from sympy import factor, resultant

sym = System({F: k * z + z * G**2, G: k * z + z * F**2}, z=z, k=k)
print(factor(resultant(F - k * z - z * G**2, G - k * z - z * F**2, G)))
print(eliminate_in_k(sym, F))
```

The factor is chosen exactly. For two factors $g$ and $h$,
$\operatorname{ord}_z h(z, w(z)) \le \deg_w g \deg_z h + \deg_w h \deg_z g$. So the
series modulo $z^{N+1}$, with $N$ that bound, tells the factors apart.

## Methods

`method="auto"` uses no elimination for one unknown, a resultant for two, and a lex
Gröbner basis (SymPy's F5B, with $k$ ordered last) otherwise. Both explicit methods
give the same curve:

```{code-cell} python
assert eliminate_in_k(sym, F, method="groebner") == eliminate_in_k(sym, F, method="resultant")
```

Elimination over the polynomial ring follows
{cite:t}`cox-little-oshea-2015`, Ch. 3. The design, with proofs and timings, is in
`dev/design/elimination.md`.
