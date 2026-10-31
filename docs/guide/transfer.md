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
is guessed.

## Ratios: averages

Averages are ratios of coefficients with the same $\rho$: a cost $C(z)$ over the counts
$F(z)$, $[z^n] C / (n [z^n] F)$. {func}`flanpy.ratio` divides two expansions:
$[z^n] A / [z^n] B = n^{\beta_A - \beta_B} (q_0 + q_1 n^{-1/\kappa} + \dots)$, with the
exact limit $q_0 = C_A/C_B$ (the $\Gamma$ factors divide) and the further $q_j$ from
the quotient of the two series, computed in one number field. {cite:t}`broda-2020-sigact`
§2: the probability that a symbol of a regular expression is a star tends to $\rho_k$
(eq. (8)); at $k = 2$ the ratio is $\rho (1 + \tfrac{1}{2n} + \dots)$:

```{code-cell} python
from flanpy import ratio

k, S = symbols("k S")
rs = System({R: (k + 1) * z + 2 * z * R**2 + z * R, S: z * R + z * S + 4 * z * S * R}, z=z, k=k)


def at_2(w):
    return asymptotics(local_expansion(singularity(eliminate(rs, w, k=2))))


stars = ratio(at_2(S), at_2(R))
print(stars.exponent, stars.limit, stars.limit.algebraic() == at_2(R).rho)
for power, q in stars.terms(3):
    print(f"n^{power}: {q}")
```

With different $\Gamma$ classes the limit keeps a $\Gamma$ part. $K = (1 - s^{1/4})/2$
over Catalan, both at $\rho = 1/4$:

```{code-cell} python
both = System({F: 1 + z * F**2, U: z + U**2, K: U / 2 + K**2}, z=z)


def at_quarter(w):
    return asymptotics(local_expansion(singularity(eliminate(both, w))))


quarter_over_catalan = ratio(at_quarter(K), at_quarter(F))
print(quarter_over_catalan.exponent)
quarter_over_catalan.limit.as_expr()
```

Constants are compared exactly: $\Gamma(\nu)$ with $\nu > 1/2$ is rewritten by reflection,
$\Gamma(\nu)\Gamma(1 - \nu) = \pi/\sin(\pi\nu)$, for `==` and hashing, so
$\sqrt\pi/(8\Gamma(3/4))$ and $\sqrt2\,\Gamma(1/4)/(16\sqrt\pi)$ are equal. A
denominator with non-zero coefficients in two $\Gamma$ classes (a pole and a square
root, say) gives terms $q_j$ that are sums of several $\Gamma$ monomials;
{meth}`~flanpy.Ratio.terms` then stops with `NotImplementedError` after the last
representable one, while {meth}`~flanpy.Ratio.evalf` still works.

Expansions at different $\rho$ are refused: their ratio has an exponential factor
$(\rho_B/\rho_A)^n$.

```{code-cell} python
:tags: [raises-exception]

ratio(a, m)  # Catalan (ρ = 1/4) over Motzkin (ρ = 1/3)
```

## Symbolic $k$

{func}`flanpy.asymptotics_in_k` maps the pieces of {func}`flanpy.local_expansion_in_k`
one to one. On a piece the exponent and the $O$-term are those of every $k$, and the
leading constant is a monomial in the {class}`~flanpy.KRoot` s of step 5:
$\alpha b_k/\Gamma(1 - \alpha)$ (Case I) or $c_k^{-1}/\Gamma(\alpha)$ (Case II), exposed as
{attr}`~flanpy.GammaConstant.rational`, {attr}`~flanpy.GammaConstant.factors` (pairs
`(KRoot, exponent)`) and {attr}`~flanpy.GammaConstant.gamma`.
{meth}`~flanpy.GammaConstant.at` evaluates it exactly at a fixed $k$;
`algebraic()`, `evalf` and `as_expr` need a fixed $k$, and so do the further terms
({meth}`~flanpy.Asymptotics.specialize`). For $R_k$ (Broda eq. (6)):

```{code-cell} python
from flanpy import asymptotics_in_k, eliminate_in_k, local_expansion_in_k, singularity_in_k

e_r = local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
cases = asymptotics_in_k(e_r)
((part, a_r),) = cases.pieces
lead = a_r.leading
print(part, a_r.exponent, lead.rational, lead.gamma)
print([m for _, m in lead.factors], lead.factors[0][0] == e_r[1].b)
print(lead.at(2) == r2.leading, a_r.specialize(2) == r2)
```

Ratios of two families: {meth}`KCases.combine <flanpy.KCases.combine>` intersects the
pieces and applies {func}`flanpy.ratio` on each. The stars of $R_k$ again, now for
every $k \ge 1$: the limit is the monomial $2 b_k^{-1} c_k^{-1}$, equal to $\rho_k$ at
each $k$:

```{code-cell} python
e_s = local_expansion_in_k(singularity_in_k(eliminate_in_k(rs, S)))
e_rs = local_expansion_in_k(singularity_in_k(eliminate_in_k(rs, R)))
star_cases = asymptotics_in_k(e_s).combine(asymptotics_in_k(e_rs), ratio)
((part, star),) = star_cases.pieces
print(part, star.limit.rational, [m for _, m in star.limit.factors])
print(all(star.limit.at(n).algebraic() == e_rs[n].singularity.rho.at(n) for n in range(1, 6)))
```

A piece where $\rho_k$ is not known to be the only dominant singularity makes
{func}`flanpy.asymptotics_in_k` raise, naming the piece; restrict the cases first
({meth}`KCases.restrict <flanpy.KCases.restrict>`). Expansions of these constants in
powers of $1/k$, and their limits as $k \to \infty$ (Broda eqs. (35)-(38)), are
step 7.
