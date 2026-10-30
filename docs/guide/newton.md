---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Local expansion (Newton–Puiseux)

At its dominant singularity $\rho$ (step 4), the series $w(z)$ has a **Puiseux
expansion** in fractional powers of $s = 1 - z/\rho$ {cite:p}`flajolet-sedgewick-2009`
(Thm VII.7). {func}`flanpy.local_expansion` computes it exactly, in the notation of
{cite:t}`broda-2020-sigact` (eqs. (20) and (31)):

- **Case I** ($w \to a < \infty$): $w = a + d_1 t + d_2 t^2 + \dots$ with
  $t = s^{1/\kappa}$. The exponent $\alpha$ is the first non-integer exponent with a
  non-zero coefficient, and $b > 0$ its coefficient with the sign changed:
  $w = (\text{analytic}) - b\, s^{\alpha} + \dots$
- **Case II** ($w \to \infty$): $1/w = c\, s^{\alpha} + \dots$, so
  $w = c^{-1} s^{-\alpha} + \dots$

These are the data the transfer theorems need (step 6): the coefficients behave like
$\rho^{-n} n^{-\alpha - 1}$ times a constant built from $b$ or $c$.

## Catalan

$F = 1 + zF^2$ has $\rho = 1/4$ and $a = 2$. With $s = 1 - 4z$,
$F = 2/(1 + \sqrt{s}) = 2 - 2 s^{1/2} + 2 s - 2 s^{3/2} + \dots$, so $\alpha = 1/2$,
$\kappa = 2$ and $b = 2$:

```{code-cell} python
from sympy import symbols

from flanpy import System, eliminate, local_expansion, singularity

z, F = symbols("z F")
e = local_expansion(singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)))
print(e)
print(e.alpha, e.ramification, e.b, e.c)
for exponent, coefficient in e.terms(5):
    print(f"s^{exponent}: {coefficient}")
```

The coefficients are {class}`~flanpy.AlgebraicReal`, exact. For Motzkin paths
($F = z + zF + zF^2$, $\rho = 1/3$) they are the expansion printed in
{cite:t}`flajolet-sedgewick-2009`, Example VII.18,
$1 - 3\delta + \tfrac92 \delta^2 - \tfrac{63}{8} \delta^3 + \dots$ with
$\delta = (1/3 - z)^{1/2} = s^{1/2}/\sqrt{3}$:

```{code-cell} python
motzkin = local_expansion(singularity(eliminate(System({F: z + z * F + z * F**2}, z=z), F)))
for exponent, coefficient in motzkin.terms(4):
    print(exponent, coefficient)
```

## The local problem and the Newton polygon

In Case I the curve is moved to the singular point,
$Q(s, Y) = G(\rho(1 - s), a + Y)$, whose coefficients are
$(-\rho)^i/(i!\,j!)\, \partial^i_z \partial^j_w G(\rho, a)$ (Broda eq. (21)). Which of
them vanish is decided exactly. When $a \in \mathbb{Q}(\rho)$ (every example of the
papers) it is a remainder modulo the minimal polynomial of $\rho$; otherwise Broda's
gcd test or the arithmetic of $\mathbb{Q}(\rho, a)$. The **Newton polygon** of $Q$ then
gives the possible exponents: each edge of slope $-\gamma$ carries the branches
$Y \sim c\, s^{\gamma}$, with $c$ a root of the edge's characteristic polynomial
{cite:p}`flajolet-sedgewick-2009` (pp. 499–500).

In Case II the reciprocal polynomial $H(z, v) = v^m G(z, 1/v)$ is used at $(\rho, 0)$
(Broda eq. (29)). For Broda's $L_k$ (eq. (32)) the polygon is a single edge from
$(0, 1)$ to $(2, 0)$, so $\alpha = 1/2$, and the constant is eq. (34),
$c_k = \sqrt{2 \rho_k H_z / H_{vv}}$:

```{code-cell} python
import mpmath

k, B, Bb, M, Mb, L = symbols("k B Bb M Mb L")
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
e = local_expansion(singularity(eliminate(ssnf, L, k=2)))
print(e)
mpmath.mp.dps = 30
rho, n = e.singularity.rho.evalf(30), 2
eq34 = mpmath.sqrt(
    (12 * n * (2 * n + 27) * rho**2 + 72 * n * rho - n)
    / (n * ((n + 12) * n * rho**2 + 4 * n * rho + 1))
)
print(mpmath.nstr(e.c.evalf(30), 25))
print(mpmath.nstr(eq34, 25))
```

The degree of $c_2$ is 6: it lies in $\mathbb{Q}(\rho_2, c_2)$, a quadratic extension of
the cubic field of $\rho_2$.

## Other exponents

The singular exponent is not always $1/2$. $K = U/2 + K^2$ with $U = z + U^2$ is
$K = (1 - s^{1/4})/2$, a quarter-root singularity like the supertrees of
{cite:t}`flajolet-sedgewick-2009` (Example VII.20). In Case II, a pole has $\alpha = 1$
(every rational series, e.g. $1/(1 - 2z)$), and $(1 - 4z)^{-3/2}$ has $\alpha = 3/2$:

```{code-cell} python
U, K, Y, W = symbols("U K Y W")
quarter = System({U: z + U**2, K: U / 2 + K**2}, z=z)
print(local_expansion(singularity(eliminate(quarter, K))))
print(local_expansion(singularity(eliminate(System({Y: 1 + 2 * z * Y}, z=z), Y))))
three_halves = System({B: z + B**2, Y: 1 + 2 * B * Y, W: Y**3}, z=z)
print(local_expansion(singularity(eliminate(three_halves, W))))
```

For a series with non-negative coefficients in Case I, the exponent is always
$1/2^m$ with $b > 0$: this follows from {cite:t}`pivoteau-salvy-2025` (Thm 8.5 and its
proof), and flanpy asserts it. A cube root, or a $(3/2)$-type singularity in Case I,
cannot occur.

## Which branch is $w$

Several branches of the curve can pass through $(\rho, a)$. The coefficients of the
branch of $w$ are real (it is real on $(\rho - \varepsilon, \rho)$), so only the *real*
roots of the characteristic polynomials matter. $w$ increases to $a$, so its leading
coefficient is negative in Case I (and positive for $1/w$ in Case II). When several real
branches share that sign, a **rank test** decides. Step 4 already has a rational
$x_0 < \rho$ and a certified enclosure of $w(x_0)$ between rational separators. The
rank of $w(x_0)$ among the real roots of $G(x_0, \cdot)$ in its window is the rank of
$w$ among the branches at $(\rho, a)$, sorted by their value as $s \to 0^+$. The proofs
are in the Notes of {func}`flanpy.local_expansion`.

## Exact numbers beyond radicals

All the arithmetic is in real number fields $\mathbb{Q}(\theta)$, with $\theta$ an
{class}`~flanpy.AlgebraicReal`, and roots of polynomials over them come from Trager's
norms {cite:p}`trager-1976,loos-1983`. Nothing needs to be expressible in radicals.
For the trees $T = z(1 + T^2 + T^5)$, $\rho$ is a root of a quintic whose Galois group
is $S_5$, yet $b$ comes out exactly (degree 10). It equals
$\sqrt{2\varphi(\tau)/\varphi''(\tau)}$ with $\varphi(y) = 1 + y^2 + y^5$ and $\tau = a$
{cite:p}`flajolet-sedgewick-2009` (Thm VII.3):

```{code-cell} python
T = symbols("T")
s5 = singularity(eliminate(System({T: z * (1 + T**2 + T**5)}, z=z), T))
e = local_expansion(s5)
print(s5.rho.polynomial.as_expr())
print(e.b)
tau = s5.a.evalf(30)
print(mpmath.nstr(mpmath.sqrt(2 * (1 + tau**2 + tau**5) / (2 + 20 * tau**3)), 20))
```

## Symbolic $k$

A piece of {func}`flanpy.singularity_in_k` is expanded at one $k$ with `k=`:

```{code-cell} python
from flanpy import eliminate_in_k, singularity_in_k

R = symbols("R")
rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
piece = singularity_in_k(eliminate_in_k(rk, R))[2]
print(local_expansion(piece, k=2))
```

Expansions for every $k$ at once ($b_k$ and $c_k$ described on pieces of $k$) come with
the next part of this step.
