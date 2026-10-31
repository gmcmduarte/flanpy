---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Coefficient asymptotics (transfer)

Singularity analysis turns the local expansion at $\rho$ (step 5) into an asymptotic
expansion of the coefficients {cite:p}`flajolet-sedgewick-2009` (Thm VI.4,
Thm VII.8). Each Puiseux term $d\, s^x$, $s = 1 - z/\rho$, transfers term by term
(Thm VI.1):

$$
[z^n]\, s^x = \rho^{-n}\, \frac{n^{-x-1}}{\Gamma(-x)}
\Bigl(1 + \frac{e_1(-x)}{n} + \frac{e_2(-x)}{n^2} + \dots\Bigr),
$$

where the $e_l$ are polynomials with rational coefficients. A term with $x$ a
non-negative integer gives nothing ($1/\Gamma(-x) = 0$): the analytic part of $w$ does
not count. {func}`flanpy.asymptotics` collects the terms by powers of $n$:

$$
[z^n]\, w = \rho^{-n} \sum_{j < \text{count}} D_j\, n^{-1-p_j}
+ O(\rho^{-n} n^{e}), \qquad p_j = e_0 + j/\kappa,
$$

with $e_0 = \alpha$ in Case I and $-\alpha$ in Case II. The leading term is
{cite:t}`broda-2020-sigact`'s Thm 3.2: $-b/\Gamma(-\alpha)\, \rho^{-n} n^{-\alpha-1}$
(Case I) and $1/(c\, \Gamma(\alpha))\, \rho^{-n} n^{\alpha-1}$ (Case II). For
$\alpha = 1/2$ these are the constants $b/(2\sqrt\pi)$ and $1/(c\sqrt\pi)$ of
{cite:t}`broda-2014-hitchhiker` (Prop. 4).

## Catalan

$F = 2 - 2 s^{1/2} + 2 s - 2 s^{3/2} + \dots$ with $s = 1 - 4z$ (step 5) gives the
classical expansion of the Catalan numbers,
$[z^n] F = 4^n / \sqrt\pi\, (n^{-3/2} - \tfrac98 n^{-5/2} + \tfrac{145}{128} n^{-7/2} -
\dots)$:

```{code-cell} python
from sympy import symbols

from flanpy import System, asymptotics, eliminate, local_expansion, series, singularity

z, F, n = symbols("z F n")
catalan = System({F: 1 + z * F**2}, z=z)
a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
print(a)
print(a.exponent, a.leading)
for power, d in a.terms(5):
    print(f"n^{power}: {d}")
a.as_expr(n, count=5)
```

`count` is the number of coefficients $D_j$, **zeros included**: `count=5` above has
three non-zero terms. Against the exact coefficient at $n = 1000$, each non-zero term
gains a factor $1/n$:

```{code-cell} python
exact = series(catalan, 1000)[F][1000]
for count in (1, 3, 5):
    print(count, float(abs(a.evalf(1000, count=count, dps=30) / exact - 1)))
```

## Reading the constants

Every $D_j$ is a {class}`~flanpy.GammaConstant`: an exact algebraic number
({meth}`~flanpy.GammaConstant.algebraic`, an {class}`~flanpy.AlgebraicReal`) times
$1/\Gamma(\nu)$ for one $\nu \in (0, 1)$, or no $\Gamma$ at all. The terms of one
$D_j$ have exponents $x$ in one class modulo 1, and $1/\Gamma(-x)$ is reduced to
$1/\Gamma(\nu)$ by $\Gamma(y + 1) = y\, \Gamma(y)$, so $\nu$ is the class of $-x$.
With $\kappa = 2$ the only base is $\Gamma(1/2) = \sqrt\pi$, which SymPy prints as
`sqrt(pi)`. Algebraic parts of degree at most 2 are written with radicals:

```{code-cell} python
motzkin = System({F: z + z * F + z * F**2}, z=z)
m = asymptotics(local_expansion(singularity(eliminate(motzkin, F))))
print(m.leading.as_expr(), "=", m.leading.algebraic(), "/ Γ(1/2)")
m.as_expr(n, count=3)
```

These are the terms printed in {cite:t}`flajolet-sedgewick-2009`, Example VII.18,
$3^{n+1/2}/(2\sqrt{\pi n^3})\, (1 - \tfrac{15}{16n} + \dots)$.

With $\kappa = 4$ other bases appear. $K = U/2 + K^2$, $U = z + U^2$, is
$K = (1 - s^{1/4})/2$, and $1/\Gamma(-1/4) = -\tfrac14 / \Gamma(3/4)$:

```{code-cell} python
U, K = symbols("U K")
quarter = System({U: z + U**2, K: U / 2 + K**2}, z=z)
q = asymptotics(local_expansion(singularity(eliminate(quarter, K))))
print(q.leading.gamma)
q.as_expr(n)
```

In Case II the exponent of $n$ can be positive. $W = (1 - 4z)^{-3/2}$ (Case II,
$\alpha = 3/2$) has $1/\Gamma(3/2) = 2/\sqrt\pi$:

```{code-cell} python
B, Y, W = symbols("B Y W")
three_halves = System({B: z + B**2, Y: 1 + 2 * B * Y, W: Y**3}, z=z)
asymptotics(local_expansion(singularity(eliminate(three_halves, W)))).as_expr(n, count=3)
```

A larger degree is shown as a `CRootOf`; {cite:t}`broda-2020-sigact`'s $R_k$ at
$k = 2$ has a leading constant of degree 4, $\sqrt{2 - 2\rho}/(8\rho\sqrt\pi)$
(eq. (6)):

```{code-cell} python
k, R = symbols("k R")
rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
r2 = asymptotics(local_expansion(singularity(eliminate(rk, R, k=2))))
print(r2.rho, r2.leading.evalf(20))
r2.leading.as_expr()
```

## The O-term

{meth}`~flanpy.Asymptotics.error_exponent` gives the exponent $e$ of the remainder
$O(\rho^{-n} n^e)$ after `count` coefficients. A power $n^{-1-p}$ with $p$ a
non-negative integer always has a zero coefficient, so it is skipped: for Catalan the
error after one term is $O(4^n n^{-5/2})$, not $O(4^n n^{-2})$. A coefficient that is
zero for another reason (the three after the leading one of $K$ above) does not
sharpen the exponent.

```{code-cell} python
print(a.error_exponent(1), q.error_exponent(1))
```

When $w$ is meromorphic at $\rho$ ($\kappa = 1$), the expansion in $n$ stops: a pole
of order $m$ gives a polynomial of degree $m - 1$, and the remainder is exponentially
smaller, $O((\rho + \varepsilon)^{-n})$. The exponent is then `None`:

```{code-cell} python
pole = System({Y: 1 + 2 * z * Y}, z=z)  # 1/(1 - 2z)
p = asymptotics(local_expansion(singularity(eliminate(pole, Y))))
print(p, p.error_exponent(1), p.evalf(100) == 2**100)
```

The $O$-terms are asymptotic statements (FS Thm VI.4), with no explicit constant.

## Numeric evaluation and its limits

{meth}`~flanpy.Asymptotics.evalf` evaluates the truncated expansion with mpmath, at
`dps` digits (and $\rho$ with enough extra digits for $\rho^{-n}$). The expansion is
asymptotic, not convergent: for small $n$, more terms need not help. Below, the
ratio to the exact coefficient for $n = 1, 2, 5, 10$: at $n = 1$ it keeps oscillating
however many terms are taken, while at $n = 10$ four non-zero terms (`count=7`) give
four digits.

```{code-cell} python
small = series(catalan, 10)[F]
for count in (1, 3, 5, 7, 9):
    print(count, [f"{float(a.evalf(m, count=count) / small[m]):.4f}" for m in (1, 2, 5, 10)])
```

## Refusals

The transfer needs $\rho$ to be the only singularity on the circle $|z| = \rho$
(step 4's `unique`). A series with period $p > 1$ has $p$ dominant singularities
and is refused; the sum over all of them (FS Thm VI.5) is not implemented:

```{code-cell} python
:tags: [raises-exception]

T = symbols("T")
odd = System({T: z + z * T**2}, z=z)  # only odd powers of z
asymptotics(local_expansion(singularity(eliminate(odd, T))))
```

When step 4 cannot decide uniqueness (`unique is None`, with a
{class}`~flanpy.SingularityWarning`), {func}`flanpy.asymptotics` refuses too: nothing
is guessed. An expansion for symbolic $k$ needs a value, `asymptotics(piece, k=n)`;
the leading constants for symbolic $k$ come in the next part of this step.
