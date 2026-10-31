# Design note: `transfer`

Status: approved 2026-10-31 (decisions in Section 12). Chunk 2 (fixed `k`) in progress.

Build-order step 6. From a `LocalExpansion` (step 5): the **asymptotic expansion of
the coefficients** `[z^n] w` (FS Thm VI.4, Thm VII.8). That means the leading term
`C ρ^{-n} n^β` with `C` exact (an algebraic number times a `Γ` factor), as many further
terms in powers of `n^{-1/κ}` as asked for, the `O`-term, numeric evaluation, and
**ratios** of two such expansions with the same `ρ` (Broda Thm 3.2's averages). For
symbolic `k`: the leading constants on step 5's pieces, exactly, in terms of `ρ_k`,
`b_k`, `c_k`. After this step comes release 0.1.0: equations to asymptotics at a
fixed `k`, end to end.

Spike: `dev/spikes/transfer_spike.py`, sections `gamma`, `fixed`, `broda`, `ratio`,
`symk` (15.8 s, mostly a slow SymPy check of FS Note VI.3; 3.9 s, 5.3 s, 0.6 s,
11.1 s). Numbers marked "spike" come from it (Apple M1, Python 3.13, SymPy 1.14.0,
python-flint 0.9.0, mpmath 1.3.0).

**Sources read for this note** (numbers as printed):

- Flajolet–Sedgewick 2009 (FS): Thm VI.1 (standard scale, p. 381) and eq. (13) (`e_1`,
  `e_2`, `e_3`), its footnote 2 (`e_k` is a polynomial in `α` of degree `2k` divisible
  by `α(α − 1)…(α − k)`; `1/Γ(α) = 0` for `α ∈ Z_{<=0}`), the proof (eqs. (14)–(19),
  Hankel contour); Note VI.3 (p. 383, `e_k = Σ_{ℓ=k}^{2k} λ_{k,ℓ} (α − 1)…(α − ℓ)`,
  `λ_{k,ℓ} = [v^k t^ℓ] e^t (1 + vt)^{−1−1/v}`); Def. VI.1 (`Δ(φ, R)`, p. 389); Thm VI.3
  (big-Oh and little-oh transfer, p. 390) and its proof (contour `γ_1..γ_4`, constants
  `K`, `J_n`, `r`, `θ`; pp. 390–392); Cor. VI.1 (p. 392); Note VI.10 (large negative
  exponents); the summary (29)–(31) and Figure VI.7; Thm VI.4 (single singularity,
  p. 393); Thm VI.5 (multiple singularities, p. 398); Thm VII.8 (algebraic asymptotics,
  eqs. (82)–(83), p. 501) and Example VII.18 (Motzkin, p. 502).
- Broda et al. 2020 (Broda): Thm 3.1 (FS Cor. VI.1 restated: `f ~ λ(1 − z/ρ)^α` gives
  `λ n^{−α−1} ρ^{−n}/Γ(−α)`), §3.1 eqs. (6)–(8) (`R_k`, `S_k`, `[z^n] S_k/(n [z^n] R_k)
  → ρ_k`), eq. (31), Thm 3.2, eqs. (34)–(38), Table 1 (simulated `0.42, 0.44, 0.47`,
  approximation `1/2 − 1/(4√(2k))` = `0.375, 0.444, 0.475` for `k = 2, 10, 50`, `n = 1000`).
- Hitchhiker's Guide (HG, Broda et al. 2014): Prop. 4 (p. 89: `a − b√(1 − z/ρ) + o(·)`,
  `b ≠ 0` gives `b ρ^{−n} n^{−3/2}/(2√π)`; `a/√(1 − z/ρ) + o(·)` gives
  `a ρ^{−n} n^{−1/2}/√π`), eqs. (14)–(16) (p. 91: `[z^n] F_k ~ (2k)^{1/4}√ρ_k
  ρ_k^{−n} n^{−3/2}/√π`, `[z^n] A_k ~ (2k)^{1/4}√ρ_k ρ_k^{−(n+1)}(n + 1)^{−3/2}/(4√π)`).
  The PDF's text layer is font-shifted; read after decoding.
- `dev/design/newton.md` §5.3–5.5 (how many terms; the line with step 6; the sign
  convention of Broda eqs. (37)–(38)) and §6.4 (what step 7 gets).
- DLMF 5.11.8 (Stirling's series for `ln Γ(z + h)`), for the `e_l` recurrence (§2.1).

## 1. Input, hypotheses, scope

### 1.1 Input

A `LocalExpansion` (fixed `k`), or a `LocalExpansion[KRoot]` piece / a
`KCases[LocalExpansion]` (symbolic `k`) from step 5. It carries the singularity (`ρ`,
case, `a`, period, `unique`), `α`, `κ`, `b` or `c`, and, at a fixed `k`, the exact
Puiseux terms `terms(N)` in the branch's number field `L` (private `_State`).

### 1.2 Hypotheses: why `unique = True` is enough

FS Thm VI.4 needs `w` analytic in a `Δ`-domain at `ρ`, and an expansion there in the
scale `S`. For our `w` both hold as soon as step 4 says `unique = True`:

- `w` is algebraic, so its analytic continuation can be singular only at the finitely
  many candidates (roots of `lc_w G · disc_w G`, FS Lemma VII.4, step 4 §3.1).
- `unique = True` means `ρ` is the only singularity on `|z| = ρ`. Every other point of
  the circle has a disc of analyticity. By compactness they cover the circle outside a
  small disc at `ρ`, and with the open disc they give analyticity in `|z| < ρ + ε`
  minus a small disc at `ρ`.
- Near `ρ`, the Newton–Puiseux expansion (FS Thm VII.7) converges in a slit disc, so
  `w` is analytic in a `Δ`-domain `ρ · Δ(φ, R)` for any `φ` and some `R > 1`.
- Step 5's expansion is a sum of `d_x s^x`, `s = 1 − z/ρ`, `x ∈ (1/κ)Z`. Each term is
  in the scale `S` of FS eq. (29), and the remainder after any finite number of terms
  is `O(s^{x*})` in the slit disc.

This is FS Thm VII.8 (82) with one dominant singularity (its proof cites Thm VI.4).
Step 6 adds no condition, and step 4's `unique` is the only gate.

**Refusals** (step 4 Q3, newton.md §5.4: "step 6 refuses `unique != True`"):

- `unique is False` (step 4 proves it only through a period `p > 1`): `ValueError`,
  "`w` has `p` dominant singularities (period `p`): the transfer needs FS Thm VI.5".
- `unique is None` (a periodic component somewhere, the circle test undecided):
  `ValueError`, "whether `ρ` is the only dominant singularity is undecided (step 4)".
  Never guessed.

**What multiple singularities would take (later).**

- **Periodic `w`** (the only `False` case step 4 produces). The support of `w` lies in
  `r + pZ`, so `w = z^r V(z^p)`. `V` has the curve `Res_z(G(z, z^r v), z^p − x)`, with
  the right factor selected. Steps 4–6 then run on `V` at `ρ^p`, and give
  `[z^n] w = [x^{(n−r)/p}] V` for `n ≡ r (mod p)`, and `0` otherwise. This needs real
  arithmetic only, and is the cheap route.
- **General** (FS Thm VI.5, Thm VII.8 (83)): local expansions at every `ζ_j`,
  `|ζ_j| = ρ`, in complex number fields, and a sum of `ζ_j^{−n} σ_{j,n}`. Step 5's
  fields are real only (Lemma R), so this is a larger piece of work. It is needed for
  the `None` cases, for example `W = z^2 + T` (truth `False`) and PS Example 23.

### 1.3 Polynomial, rational, integer `α`

- **Polynomial `w`**: step 4 already raises `ValueError` (no singularity). Step 6 never
  sees it.
- **Case I** always has `α = 1/κ`, `κ = 2^m >= 2` (PS Thm 8.5, asserted in step 5). The
  integer exponents form a function analytic at `ρ`, and they contribute nothing to the
  expansion in `n` (§2.2).
- **Case II, integer `α = m`** (poles). The terms `d_{−j} s^{−j}`, `1 <= j <= m`,
  transfer **exactly**: `[z^n] s^{−j} = ρ^{−n} binom(n + j − 1, j − 1)`, a polynomial in
  `n` of degree `j − 1`. In the `n`-expansion they are the powers `n^{m−1}, …, n^0`, with
  finitely many non-zero coefficients (`e_l(j) = 0` for `l >= j`, FS footnote 2). If
  `κ = 1`, `w` is meromorphic at `ρ`, so `[z^n] w = ρ^{−n} P(n) + O((ρ + ε)^{−n})` with
  `P` exact. If `w` is rational the `O`-term holds for every `ε` below the next pole's
  distance. This is FS Thm VII.8's "polar case, terminating form". Spike: `1/(1 − 2z)`
  gives `D_0 = 1` and all further `D_j = 0`.
- **Case II with non-integer `α >= 1`** (`s^{−3/2}`): nothing special. The leading power
  is `n^{α−1}`, and FS Thm VI.1 holds for every `α ∉ Z_{<=0}`.

## 2. The expansion in `n`

### 2.1 One term (FS Thm VI.1)

With `s = 1 − z/ρ`, `[z^n] f(z/ρ) = ρ^{−n} [z^n] f(z)`, and for `x ∉ Z_{>=0}`:

```
[z^n] s^x = ρ^{−n} n^{−x−1} / Γ(−x) · (1 + Σ_{l>=1} e_l(−x) n^{−l})          (FS Thm VI.1, α = −x)
```

The polynomials `e_l(a)` are exact, in `Q[a]`. Write `Γ(n + a)/Γ(n + 1) =
n^{a−1} exp(Σ_j s_j n^{−j})`, where by Stirling's series (DLMF 5.11.8)

```
s_j = (−1)^{j+1} (B_{j+1}(a) − B_{j+1}(1)) / (j (j + 1))     (Bernoulli polynomials)
```

Then `e_0 = 1`, `l e_l = Σ_{j=1}^{l} j s_j e_{l−j}` (from `E' = S'E`). This takes a few ms
for `l <= 12`, and the result is cached.

Spike (`gamma`):

- `e_1..e_3` are FS eq. (13). Footnote 2 holds for `l <= 8`: degree `2l`, divisible by
  `a(a − 1)…(a − l)`. `e_4 = a(a − 1)(a − 2)(a − 3)(a − 4)(15a^3 − 30a^2 + 5a + 2)/5760`.
- `e_1..e_4` agree with FS Note VI.3, **with a sign `(−1)^ℓ`** in the sum, which the
  extracted text lacks. The Hankel integral of `t^ℓ (−t)^{−a} e^{−t}` gives
  `(−1)^ℓ (a − 1)…(a − ℓ)/Γ(a)`. By hand for `e_1`: `λ_{1,1} = −1`, `λ_{1,2} = 1/2`,
  and only `+(a − 1) + (a − 1)(a − 2)/2 = a(a − 1)/2` is eq. (13). This reading was
  confirmed independently in review (`λ_{1,1} = −1`, `λ_{1,2} = 1/2`). It is our
  reading, not an erratum (a low-priority CLAUDE.md open item: check the printed page).
  We use the Bernoulli recurrence and do not depend on it.
- Numerically, `e_0..e_8` at `n = 1000` reproduce `Γ(n + a)/Γ(n + 1)` to `1e−26..1e−32`
  for `a ∈ {1/2, −1/2, 1/4, 3/2, −5/4}`.

### 2.2 Summing the terms

Step 5 gives `w = Σ_i d_i s^{x_i}`, `x_i = x_start + i/κ`, where `x_start = 0` (I) or `−α`
(II). Let `e_0` be the first singular exponent (`α` in Case I, `−α` in Case II). Then

```
[z^n] w = ρ^{−n} Σ_{j>=0} D_j n^{−1−e_0−j/κ} + O((ρ + ε)^{−n}),
D_j = Σ_{x_i <= p_j, p_j − x_i ∈ Z} d_i e_{p_j − x_i}(−x_i) / Γ(−x_i),   p_j = e_0 + j/κ.
```

Only exponents `x_i ≡ p_j (mod 1)` meet in `D_j`, so `D_j` has one `Γ` class (§3).
Two facts follow (spike, every example):

- **Integer class** (`p_j ∈ Z`): only integer `x_i` contribute. Those `>= 0` have
  `1/Γ(−x_i) = 0`. Those `< 0` (Case II poles) contribute only the powers `n^{>=0}`
  (§1.3). So **`D_j = 0` whenever `p_j` is an integer `>= 0`**. For `κ = 2`, Case I,
  every other `D_j` is zero (Catalan `D = 1, 0, −9/8, 0, 145/128` times `1/√π`).
- Each `D_j` lies in step 5's field `L` (all `d_i` do), up to the `Γ` factor. It is
  computed with `L`'s exact arithmetic and turned into an `AlgebraicReal` once
  (`to_real`). This takes 1–60 ms for `D_0..D_5` (spike, after step 5).

### 2.3 How many Puiseux terms

`J` coefficients `D_0..D_{J−1}` need the Puiseux exponents `x <= e_0 + (J − 1)/κ`, that is
`N = κ(e_0 − x_start) + J` terms of step 5. This is **`N = J + 1` in Case I** (the
analytic constant `a` comes first, and `κα = 1`), and **`N = J` in Case II**. The `e_l`
needed are `l <= (J − 1)/κ`. This settles newton.md §5.3's "how many terms".

In the API a number of terms is called **`count`** (`n` is the coefficient index), and
`count = J` counts the `D_j` **including the zeros** (structural or not): for Catalan,
`count = 5` is `D_0..D_4`, three of them non-zero. Step 5's `LocalExpansion.terms`
takes `count` too (renamed from `n` in chunk 2; positional calls are unaffected).

### 2.4 The `O`-term

After `J` coefficients the remainder is `O(ρ^{−n} n^{−1−e_0−J/κ})` (FS Thm VI.4, with
`τ = s^{e_0 + J/κ}`). Since `n^{−1−p}` with `p` an integer `>= 0` has `D = 0` (§2.2), the
true remainder can be smaller. The stated exponent is always valid, and a sharper one
is printed only when the zero is structural. In other words: if `p_J` is an integer
`>= 0`, the exponent moves on to `p_{J+1}`. A `D_j` that is zero by accident (`quarter`:
`K = (1 − s^{1/4})/2` has `D_1 = D_2 = D_3 = 0`) is shown as `0` but not used to sharpen
the bound.

Spike (`fixed`), relative error at `n = 1000` against `series`, after 1, 2 and 3 **non-zero**
terms. For `κ = 2` that is `count = 1, 3, 5` (the `O`-term drops by `1/n` per non-zero
term); for `α = 1/4` (`κ = 4`) the second non-zero term is `D_4`, `count = 5`:

| example | `ρ`, case, `α` | 1 non-zero | 2 non-zero | 3 non-zero | `D_0` (times `1/Γ(ν)`) |
|---|---|---|---|---|---|
| Catalan | 1/4, I, 1/2 | 1.1e-3 | 1.1e-6 | 1.1e-9 | `1` |
| Motzkin | 1/3, I, 1/2 | 9.4e-4 | 9.9e-7 | 9.9e-10 | `√3/2` (FS Ex. VII.18) |
| `R_2` | deg 2, I, 1/2 | 8.3e-4 | 8.7e-7 | 9.0e-10 | Broda eq. (6) |
| HG `A_2` | 1/6, I, 1/2 | 9.4e-4 | 9.9e-7 | 9.9e-10 | HG eq. (16) (`√3/2`) |
| constructed | deg 4, I, 1/2 | 2.7e-4 | 2.9e-6 | 2.2e-8 | deg 8 |
| S5 tree | deg 5, I, 1/2 | 1.7e-5 | 1.9e-8 | 3.0e-10 | deg 10 |
| `F_2` | deg 3, I, 1/2 | 8.0e-4 | 8.3e-7 | 8.5e-10 | deg 6 |
| `L_2` | deg 3, II, 1/2 | 2.1e-4 | 3.6e-8 | 2.7e-11 | deg 6 |
| `s^{−3/2}` | 1/4, II, 3/2 | 3.7e-4 | 5.5e-8 | 8.8e-12 | `2` (`1/Γ(3/2) = 2/√π`) |
| `α = 1/4` | 1/4, I, 1/4 | 1.6e-4 | 5.1e-8 | — | `1/8` (`ν = 3/4`) |
| pole | 1/2, II, 1 | 0 | — | — | `1` (exact, terminating) |

Motzkin's first four terms are FS Example VII.18's printed
`3^{n+1/2}/(2√π n^{3/2}) (1 − 15/(16n) + 505/(512n^2) − 8085/(8192n^3))`, exactly (spike
asserts the ratios `D_2/D_0`, `D_4/D_0`, `D_6/D_0`).

### 2.5 Which form to evaluate: the `n`-expansion, not `Γ` ratios

Step 5's test helper summed `d_x [z^n] s^x` with the **exact** `Γ` ratios
`Γ(n − x)/(Γ(−x) Γ(n + 1))`. With the same Puiseux terms, the `n`-expansion is as
accurate or better at `n = 1000` in every example (Catalan `1.1e-9` vs `1.3e-8` with 6
Puiseux terms; `L_2` `2.7e-11` vs `8.8e-11`). It is also what the papers print (FS
(82), Broda Thm 3.2, HG Prop. 4). The `Γ`-ratio form is exact only for terminating
cases (`α = 1/4`, `s^{−3/2}`, the pole), where the `n`-expansion with enough terms is
also very close (`s^{−3/2}`: 3 terms `8.8e-12`). **Recommendation: the public numeric
evaluation is the `n`-expansion.** The `Γ`-ratio form is not offered.

## 3. Exact constants

### 3.1 One `Γ` base per class

Every `D_j` is `A_j / Γ(ν_j)`: `A_j` is algebraic (in `L`), and `ν_j ∈ (0, 1)` is the
class of `−p_j` mod 1, or there is no `Γ` (integer class). The reduction is the
recurrence `Γ(y + 1) = y Γ(y)`, so `1/Γ(−x) = r/Γ(ν)` with `r ∈ Q`:
`1/Γ(−1/2) = (−1/2)/Γ(1/2)`, `1/Γ(−3/2) = (3/4)/Γ(1/2)`, `1/Γ(3/2) = 2/Γ(1/2)`,
`1/Γ(−1/4) = (−1/4)/Γ(3/4)`, `1/Γ(−5/4) = (5/16)/Γ(3/4)` (spike, to 45 digits). Then:

- **Case I leading constant**: `−b/Γ(−α) = α b/Γ(1 − α)` (Broda Thm 3.2; `α = 1/2`:
  `b/(2√π)`, HG Prop. 4).
- **Case II leading constant**: `1/(c Γ(α))`, i.e. `1/(c (ν)_p Γ(ν))` with
  `α = ν + p` (`α = 1/2`: `1/(c√π)`; integer `α`: `1/(c (α − 1)!)`, no `Γ`).

### 3.2 What "exact" means

- **`κ = 2`**: the only base is `Γ(1/2) = √π`. A constant is `A/√π` (or `A`, integer
  class) with `A` algebraic. Since `π` is transcendental, this form is canonical, and
  equality of two constants is equality of the `A`s (exact, `AlgebraicReal`).
- **`κ = 4`**: the bases are `Γ(1/4)`, `Γ(1/2)`, `Γ(3/4)`. Reflection
  (`Γ(ν)Γ(1 − ν) = π/sin(πν)`) gives `Γ(3/4) = π√2/Γ(1/4)`. Everything is algebraic
  times `π^{a/2} Γ(1/4)^b`. `π` and `Γ(1/4)` are algebraically independent (a known
  result of G. V. Chudnovsky; reference to check, e.g. *Contributions to the Theory of
  Transcendental Numbers*, AMS Math. Surveys 19, 1984), so this form is canonical too.
  The docs do not state this, so nothing is cited.
- **`κ = 2^m >= 8`** (allowed by PS Thm 8.5, met by no example): reflection and Legendre
  duplication (`Γ(ν)Γ(ν + 1/2) = 2^{1−2ν}√π Γ(2ν)`) relate the bases, and the
  algebraic relations among `Γ(j/8)` are not all known. The representation is then
  **a** form, not a canonical one. Equality is structural, and two constants with
  different `Γ` parts are reported "not structurally equal", never "different".

**Recommendation (Q3)**: store the `Γ` part as printed in FS and Broda, a monomial
`Π Γ(ν)^{m_ν}` with `ν ∈ (0, 1)`. So the `α = 1/4` constant reads `1/(8 Γ(3/4))`, not
`Γ(1/4)/(8√2 π)`. Comparisons and ratios use a canonical key: reflection to
`ν <= 1/2` plus `π`, with the `sin(πν)` factors (algebraic) moved into `A`. For
`κ <= 4` the key decides equality.

### 3.3 How a user reads them

- `as_expr()` gives SymPy: `A · Π sympy.gamma(ν)^m`. SymPy rewrites `gamma(1/2)` as
  `sqrt(pi)` (checked; `gamma(3/2) = sqrt(pi)/2`, `gamma(3/4)` stays). So Catalan's
  leading term reads `4**n/(sqrt(pi)*n**(3/2))` (checked:
  `sqrt(3)*3**n/(2*sqrt(pi)*n**(3/2))` for Motzkin's).
- The algebraic part is an `AlgebraicReal`. Its `as_expr` today is
  `CRootOf(4*y**2 - 3, 1)`, not `sqrt(3)/2`. **Recommendation (Q4)**: `as_expr` writes
  degree-1 and degree-2 algebraic parts in radicals (exact: the quadratic formula, with
  the sign picked by the isolating interval), and `CRootOf` otherwise. General radicals
  remain a step 7 display question (newton.md Q3).
- `str()` prints `ρ^{−n} n^{β} · C (1 + …)` with `C` as above, and `evalf` gives digits.

## 4. Error terms: asymptotic `O(·)` only (recommended)

The `O`-term of §2.4 is an asymptotic statement, as in FS Thm VI.4, Thm VII.8 and Broda
Thm 3.2. An **explicit** bound `|[z^n] w − ρ^{−n} Σ_{j<J} D_j n^{…}| <= K ρ^{−n}
n^{−1−p_J}` for all `n >= n_0` would need the constants of FS Thm VI.3's proof:

1. **An explicit `Δ`-domain**: `R` and `φ`. `R` can be any radius below the least
   modulus of the other candidates (exact, from step 4's candidate set), and `φ` is
   free.
2. **A bound on the remainder** `|w − Σ_{x<x*} d_x s^x| <= K_1 |s|^{x*}` on the part of
   the domain near `ρ`. This needs a majorant for the Puiseux series, i.e. its radius
   (distance to the nearest other candidate) and bounds on the coefficients (Cauchy
   estimates from a bound on `|w|`).
3. **A bound on `|w|`** on the rest of the contour: root bounds of `G(z, ·)` (Cauchy's
   bound in terms of `g_i/g_m`), valid where `lc_w G ≠ 0`.
4. **The contour integrals** `γ_1..γ_4` with explicit `K`, `J_n <= ∫_1^∞ t^{−α}
   e^{−t cos θ} dt`-type constants, and the outer circle `r^{−n}`. These are elementary
   but tedious.

All four steps are feasible with what steps 3–5 give, but the constants would be very
pessimistic, and no paper of ours uses them. **Recommendation (Q5): `O`-terms only in
step 6.** Tests check the expansions against exact `series` coefficients (§9).
Explicit bounds would be a later, optional feature, for example for certified tables.

## 5. Numeric evaluation

`Asymptotics.evalf(n, *, count=1, dps=15)` evaluates `ρ^{−n} Σ_{j<J} D_j n^{−1−p_j}`
with mpmath:

- `D_j` from `AlgebraicReal.evalf` at `dps + 5`, and `1/Γ(ν)` by `mpmath.gamma`.
- `ρ^{−n}`: a relative error `δ` in `ρ` becomes `n δ` in `ρ^{−n}`, so `ρ` is evaluated
  at `dps + ⌈log10 n⌉ + 5` digits (spike: `60 + len(str(n))`).
- The expansion is asymptotic, not convergent. Docs say so: for small `n`, more terms
  can be worse.

The step-5 slow test (`test_coefficients_at_1000`, `test_broda_coefficients_at_1000`,
`transfer(e, n, count)` helper) moves to `tests/test_transfer.py` and uses `evalf`
instead of the helper. The bounds are re-measured (3 times the §2.4 table; `F_k`, `L_k`
for `k ∈ {2, 10, 50, 54, 108}`). The hypothesis test in `test_newton.py`
(`test_random_systems`, the 5% check at `n = 300`) uses `evalf(n, count=1)`. The helper
is deleted.

## 6. Ratios of two expansions

### 6.1 Same `ρ`

`ratio(A, B)` for `[z^n] A / [z^n] B`, with `A`, `B` two `Asymptotics` at the same `ρ`
(exact `AlgebraicReal` equality):

```
[z^n] A / [z^n] B = n^{β_A − β_B} (q_0 + q_1 n^{−1/κ} + q_2 n^{−2/κ} + …),
q_0 = D^A_0 / D^B_0,  q = series division of (D^A_j) by (D^B_j)      (κ = lcm of the two)
```

- `q_0` is the **exact limit** (with the `n^{β_A−β_B}` factor). The `Γ` parts divide.
  For Broda's averages they cancel (both `ν = 1/2`), so `q_0` is algebraic.
- The `D^A_j` lie in `L_A` and the `D^B_j` in `L_B`. The division needs one field
  containing both: the real roots of `m_{θ_B}` over `L_A` (step 5's
  `_NumberField.real_roots`), pick the one equal to `θ_B`, embed. This reuses step 5's
  machinery and needs no new algorithm (spike `composite`, 0.02–0.12 s).
- With different `Γ` classes, `q_j` keeps a `Γ` monomial (§3.2).

### 6.2 Results (spike `broda`, `ratio`)

**Broda Thm 3.2, eq. (38), Table 1**: `[z^n] L_k/(n [z^n] F_k) → 2/(b_k c_k)` (with
newton.md §5.5's sign convention), exactly, then `q_1 = 0` and a term in `n^{−1}`:

| `k` | limit `2/(b_k c_k)` (min. poly) | value | eq. (38) approx | Table 1 sim. | actual `n = 1000` | `n^{−1}` coeff. |
|---|---|---|---|---|---|---|
| 2 | `31y^3 − 35y^2 + 14y − 2` | 0.417256 | 0.375 | 0.42 | 0.41767829 | 0.421941 |
| 10 | `47y^3 − 67y^2 + 34y − 6` | 0.443252 | 0.444 | 0.44 | 0.44370254 | 0.450121 |
| 50 | `127y^3 − 227y^2 + 134y − 26` | 0.465747 | 0.475 | 0.47 | 0.46621969 | 0.473185 |
| 54 | `15y − 7` (**7/15**) | 0.466667 | 0.476 | — | — | 0.474074 |
| 108 | `27y^2 − 36y + 11` | 0.474217 | 0.483 | — | — | 0.481125 |

At `n = 1000` the limit alone is off by `1.0e-3` (relative), and with the `n^{−1}` term by
`7.4e-9, 1.6e-8, 1.8e-8`. Table 1's simulated values agree with the exact limits to the
two printed digits. The approximation `1/2 − 1/(4√(2k))` is visibly worse at `k = 2`.

**Broda §2, eq. (8)**: `[z^n] S_k/(n [z^n] R_k) → ρ_k` **exactly** (`AlgebraicReal`
equality, `k = 1..6`), next term `ρ_k/(2n)`. So the ratio is `ρ_k (1 + 1/(2n) + O(n^{−2}))`.
The system is `R = (k + 1)z + 2zR^2 + zR`, `S = zR + zS + 4zSR` (Broda's grammar (1),
differentiated in `u`).

**Observation (spike)**: `2/(b_k c_k) ∈ Q(ρ_k)` at every sampled `k` (1, 2, 3, 5, 10,
50, 53, 55, 108). For example `k = 2`: `31ρ^2/13 + 3ρ/26 + 4/13`; `k = 50`:
`3175ρ^2 + 75ρ/2 − 8`; `k = 108`: `2/3 − 6ρ`. This is expected: `L_k` marks letters in
`F_k`, and the mean of a marked parameter has limit `−ρ'(1)/ρ(1)` (perturbation of
`ρ(u)`), which is rational in `ρ` and `k`. It matters for step 7 (§7.3).

### 6.3 Refusals

- **`ρ_A ≠ ρ_B`**: `ValueError`, "the ratio has an exponential factor `(ρ_B/ρ_A)^n`;
  not represented". **Recommendation (Q6)**: refuse rather than carry the factor. The
  limit is then `0` or `∞`, no paper needs the factor, and the user can read it off
  `A.rho`, `B.rho`.
- One of the two not `unique`: already refused by `asymptotics`.
- At a fixed `k` against a symbolic piece, or different `k`: `ValueError`.

## 7. Symbolic `k`

### 7.1 On a piece

On a piece of `local_expansion_in_k`, `α`, `κ` and the case are constant, `ρ_k` is a
`KRoot`, and `b_k` (I) or `c_k` (II) is a `KRoot` relative to `ρ_k`. The **leading
term** is exact:

- Case I: `ρ_k^{−n} n^{−α−1} · α b_k / Γ(1 − α)`;
- Case II: `ρ_k^{−n} n^{α−1} · 1/(c_k (ν)_p Γ(ν))`.

The algebraic part is a **monomial** `q · Π X_i^{m_i}` (`q ∈ Q`, `X_i` `KRoot`s,
`m_i ∈ Z`): `(α) · b_k` or `(1/(ν)_p) · c_k^{−1}`. We do not build a new `KRoot` for
`1/c_k`. Its positive index would be the number of positive roots of `Φ(·, ρ_k, k)`
minus `1 + j`, and that count need not be constant on the piece, so it would add cuts.
The monomial needs no new polynomial and no new cut.

- `at(n)` / `specialize(n)` evaluate the monomial exactly at `k = n` (field arithmetic
  in `Q(b_n)` or `Q(c_n)`) and must equal the fixed-`k` `Asymptotics` (spike `symk`:
  `b_k.at(n)`, `c_k.at(n)` and `2 b_k^{−1} c_k^{−1}` at `n` equal the fixed-`k` values
  for `n ∈ {1, 2, 3, 10, 50, 54, 55, 108}`).
- Further terms `D_1, D_2, …` for symbolic `k` would need step 5's Hensel lifting over
  `Q(k)(ρ_k, b_k)`, which step 5 does not provide (`terms` needs a fixed `k`). **Out of
  scope.** `terms(count)` on a symbolic piece raises `ValueError` ("use `specialize(n)`"),
  as `LocalExpansion.terms` does.

### 7.2 Families, pieces and refusals

- `asymptotics_in_k(e)` maps step 5's pieces one to one. There are **no new cuts**: the
  leading data are functions of the step 5 description.
- `unique` is constant on a piece (step 4). **Recommendation (Q7)**: if any piece has
  `unique != True`, raise `ValueError` naming those pieces' `KSet`s, and suggest
  `cases.restrict(...)` to the good ones. The alternative is to move them to
  `uncovered`, but `uncovered` means "undecided", while `unique = False` is decided.
- **Ratios in `k`**: `KCases.combine(other, ratio)` (public since step 4) intersects
  the pieces. `ratio` on two pieces requires the same `ρ_k`, by structural `KRoot`
  equality (irreducible `f`, same index; two different normalised irreducible `f` share
  a root at finitely many `k` only, so equal values there are reported as different
  `ρ_k` → `ValueError`). The limit is the monomial `q_0 = (q_A/q_B) Π X^{m}`. `L_k`/`F_k`:
  `2 b_k^{−1} c_k^{−1}` on `[1, ∞)`.

### 7.3 What step 7 needs

On the tail piece: `α`, `κ` (constants), `ρ_k` (`KRoot(f, j)`), and the leading
constant or the ratio limit as a monomial in `ρ_k`, `b_k`, `c_k`. Step 7 expands each
`KRoot` in `1/k` (Newton polygon in `(y, 1/k)`; for relative roots, substitute
`ρ_k`'s expansion into `Φ`). It then multiplies the expansions of the monomial, which
gives eq. (35) (`ρ_k`), (36) (`c_k ~ 4k^{−1/2}`), (37) (`b_k ~ k^{1/2}`) and (38)
(`2/(b_k c_k) = 1/2 − 1/(4√2 √k) + o(k^{−1/2})`).

Given §6.2's observation, **step 7 could instead reduce the limit to `A(ρ_k, k)`,
`A ∈ Q(k)[z]`**. Both `Φ` are binomial, `D y^2 − N` (`b_k^2 = N_b/D_b`, `c_k^2 =
N_c/D_c` at `z = ρ_k`), so `(2/(b_k c_k))^2 = 4 D_b D_c/(N_b N_c)` lies in `Q(k)(ρ_k)`
exactly. Then solve `A^2 ≡ 4 D_b D_c/(N_b N_c) (mod f)`, with the sign fixed at one
sample. The `A_n(ρ_n)` computed at samples (spike)
give a candidate by rational reconstruction in `k`, verified exactly mod `f`. Then
eq. (38) needs only `ρ_k`'s expansion (eq. (35)). **Q8**: leave this to step 7
(recommended), or add it to step 6's `ratio` for symbolic `k`?

## 8. Output types and public API

```python
class GammaConstant[X: (AlgebraicReal, KRoot)]:
    # value: rational * prod X_i^m_i * prod Gamma(nu)^m_nu (Section 3.2)
    # symbolic k (chunk 3): the monomial in KRoots
    rational: Fraction
    factors: tuple[tuple[X, int], ...]
    # 0 < nu < 1, as printed in FS and Broda; a canonical key for == and hash
    gamma: tuple[tuple[Fraction, int], ...]

    # fixed k: rational * prod X^m exactly, the Gamma part excluded
    def algebraic(self) -> AlgebraicReal: ...
    # the full value, Gamma part included
    def evalf(self, dps: int = 15) -> mpmath.mpf: ...
    # radicals for degree <= 2 (Q4), sympy.gamma
    def as_expr(self) -> sp.Expr: ...
    # symbolic k only
    def at(self, k: int) -> GammaConstant[AlgebraicReal]: ...

    # ==, hash, *, / (used by ratio); immutable, picklable


class Asymptotics[X: (AlgebraicReal, KRoot)]:
    expansion: LocalExpansion[X]  # singularity, rho, case, alpha, kappa come from it
    exponent: Fraction  # beta: [z^n] w ~ C rho^-n n^beta; -alpha-1 (I), alpha-1 (II)
    leading: GammaConstant[X]  # C

    # fixed k: (power of n, D_j) for j < count, zeros included; exact, cached
    def terms(
        self, count: int
    ) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]: ...
    # the O-term after count terms (Section 2.4)
    def error_exponent(self, count: int) -> Fraction: ...
    def evalf(self, n: int, *, count: int = 1, dps: int = 15) -> mpmath.mpf: ...
    def as_expr(self, n: sp.Symbol, *, count: int = 1) -> sp.Expr: ...
    def specialize(self, k: int) -> Asymptotics[AlgebraicReal]: ...

    # rho, k, kset: forwarded


class Ratio[X: (AlgebraicReal, KRoot)]:
    numerator: Asymptotics[X]
    denominator: Asymptotics[X]
    exponent: Fraction  # [z^n] A / [z^n] B ~ limit · n^exponent
    limit: GammaConstant[X]  # q_0

    def terms(
        self, count: int
    ) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]: ...
    def evalf(self, n: int, *, count: int = 1, dps: int = 15) -> mpmath.mpf: ...
    def specialize(self, k: int) -> Ratio[AlgebraicReal]: ...


def asymptotics(
    expansion: LocalExpansion[Any], *, k: int | None = None
) -> Asymptotics[AlgebraicReal]: ...


def asymptotics_in_k(
    expansions: LocalExpansion[KRoot] | KCases[LocalExpansion[Any]],
) -> KCases[Asymptotics[KRoot]]: ...


def ratio(numerator: Asymptotics[X], denominator: Asymptotics[X]) -> Ratio[X]: ...
```

- **Names (Q1)**: the nouns follow steps 3–5 (`singularity`, `local_expansion`):
  `asymptotics`, `asymptotics_in_k`, `Asymptotics`, `ratio`, `Ratio`, `GammaConstant`.
  Alternatives: `coefficient_asymptotics`, or `transfer`/`Transfer` (the method's name,
  CLAUDE.md's module name). The module stays `_transfer.py`.
- `asymptotics(e, k=n)` on a symbolic piece works on `e.specialize(n)`, as in steps 4–5.
  `asymptotics_in_k` is a `map` over the step 5 pieces (no `flat_map`: no new cuts).
- **No `ratio_in_k`**: `KCases.combine(other, ratio)` does it, and the guide shows it.
- **Value semantics** as in steps 4–5: immutable (`__slots__`, `__setattr__` raises),
  equality and hash on `(expansion, …)` and on `GammaConstant`'s canonical key,
  pickling (cached `D_j` recomputed), `repr` re-creatable in spirit, and `str` as
  §3.3.
- **Errors**: `TypeError` (not a `LocalExpansion` / `Asymptotics`), `ValueError` (`k`
  given or missing as in steps 4–5; `unique != True` (§1.2); different `ρ` (§6.3);
  `terms` on a symbolic piece). **Q2**: no new exception class (recommended). The
  alternative is `TransferError(ValueError)` for the `unique` refusals.
- **No end-to-end shortcut** (`asymptotics(system, w, k=…)`) in this step: the chain
  `eliminate → singularity → local_expansion → asymptotics` is four calls and shows
  each step's result. Reconsider for 0.1.0 if the README needs it (Q9).
- Private: `_e_polys(l)` (cached), `_inverse_gamma(x)` (`1/Γ(x) = r/Γ(ν)`),
  `_n_expansion(state, J)`, `_composite(x, y)` (one field for two elements; may move to
  `_field.py`).

## 9. Tests (`tests/test_transfer.py`, `tests/test_transfer_in_k.py`)

**Private pieces**: `e_l` against FS eq. (13) (`l <= 3`), footnote 2 (degree `2l`,
roots `0..l`, `l <= 8`), and Note VI.3 with the sign (`l <= 2`; the SymPy series is
slow); `1/Γ(x) = r/Γ(ν)` for `x ∈ {±1/2, ±3/2, −1/4, −5/4, 7/4}` and integers
(`hypothesis`: random `x` with denominator `<= 8`, against mpmath to 40 digits); the
composite field (`√2` with `√3`, cubic with quadratic).

**Quadratic first (HG style), exact:**
- Catalan: `D = 1, 0, −9/8, 0, 145/128, 0, −1155/1024` times `1/√π` (the known expansion
  of `C_n`), i.e. `terms(7)`; `as_expr` gives `4**n/(sqrt(pi)*n**(3/2))`;
  `error_exponent(1) = −5/2` (`p_1 = 1` is a structural zero).
- Motzkin: FS Example VII.18's printed terms, exactly.
- HG `A_k`, Broda `R_k`, `k = 1..20`: leading constant = HG eq. (16) /
  Broda eq. (6) (HG Prop. 4), to 40 digits.

**Exponents and fields:** `α = 1/4`: `D_0 = 1/(8 Γ(3/4))`, `ν = 3/4` (the class rule;
`Γ(1/4)` canonical key); pole `1/(1 − 2z)`: `D = 1, 0, 0, …` (terminating) and
`evalf` exact; `s^{−3/2}`: `D_0 = 2/√π`, `D_2 = (3/8) D_0`; constructed (deg 8) and S5
tree (deg 10): `D_0` against `local_expansion` (`α b/Γ(1 − α)`).

**Acceptance (Broda):**
- `F_k`, `L_k`, `k ∈ {2, 10, 50, 54, 108}`, `n = 1000` (`slow`): `evalf` with 1, 2, 3
  non-zero terms (`count = 1, 3, 5`) against `series`, bounds 3 times the §2.4 errors (`F_2`
  `8.0e-4, 8.3e-7, 8.5e-10`; `L_2` `2.1e-4, 3.6e-8, 2.7e-11`; others measured in chunk 2).
- Thm 3.2 / eq. (38) / Table 1: `ratio(asymptotics(L_k), asymptotics(F_k))` has
  `exponent = 1` and `limit` = the minimal polynomials of §6.2 (exact), `0.4173, 0.4433,
  0.4657` (4 digits) against Table 1's `0.42, 0.44, 0.47`, and `7/15` at `k = 54`; at
  `n = 1000` (`slow`), the limit plus the `n^{−1}` term is within `5e-8` of the exact
  ratio.
- §2, eq. (8): `ratio(asymptotics(S_k), asymptotics(R_k)).limit == ρ_k` exactly
  (`k = 1..10`), and the next term is `ρ_k/2`.
- Symbolic: `asymptotics_in_k` on `L_k`, `F_k`, `R_k`, `A_k`: one piece `[1, ∞)`; the
  leading monomials (`c_k^{−1}/√π`, `b_k/(2√π)`); `at(n)` equals fixed `k` for
  `k = 1..60` and 108 (`slow`, reusing step 5's samples); `L_k`/`F_k` via
  `KCases.combine(…, ratio)`: `2 b_k^{−1} c_k^{−1}`, `at(n)` = fixed `k`; the `K =
  kU/4 + K^2` family: three pieces `{1}`, `{2}` (`α = 1/4`, `Γ(3/4)`), `[3, ∞)`, all
  `unique` (checked).

**Refusals:** `T = z + zT^2` (`unique = False`, period 2) → `ValueError` naming the
period; `W = z^2 + T` (`unique = None`) → `ValueError` "undecided"; a symbolic family
with a bad piece → `ValueError` naming its `KSet`; `ratio` with different `ρ` (Catalan
vs Motzkin) → `ValueError`; `terms` on a symbolic piece → `ValueError`.

**Properties (`hypothesis`)**, from step 5's random small systems (aperiodic,
`unique`): `evalf(300, count=1)` within 5% (moved from `test_newton.py`); `terms` agree
with step 5's `α`, `b`, `c` (`D_0` formula); `D_j = 0` for integer `p_j >= 0`.

**API:** value semantics, `specialize`, `terms` caching, `GammaConstant` arithmetic and
key, `KCases` mapping (`uncovered` propagated), the docstring examples (EX01).

## 10. Docs plan, and the 0.1.0 release

- **Docstrings** (NumPy, doctested `Examples`, EX01): `asymptotics` (Catalan; cites FS
  Thm VI.1, VI.4, VII.8, Broda Thm 3.2, HG Prop. 4; Notes: §1.2's argument, the `e_l`
  recurrence (DLMF 5.11.8), the `O`-term rule), `Asymptotics` (`leading`, `terms`,
  `evalf`, `as_expr`), `GammaConstant`, `ratio` (`S_k/R_k → ρ_k` at `k = 2`),
  `asymptotics_in_k` (`R_k`).
- **Guide page `docs/guide/transfer.md`** (executed): Catalan from the local expansion
  to `[z^n] F ~ 4^n n^{−3/2}/√π (1 − 9/(8n) + …)`, against `series`; how to read the
  constants (`Γ` classes, `α = 1/4`); the `O`-term; `evalf` (and its limits); refusals;
  ratios (Broda §2); symbolic `k` and `KCases.combine`.
- **`examples/catalan.py`**, "Step 6": `[z^n] F ~ 4^n n^{−3/2}/√π`, the next terms, and
  the error at `n = 1000`; Status: step 6 done (0.1.0).
- **`examples/broda2020.py`**, "Step 6": `R_k` (eq. (6)), eq. (8) (`S_k/R_k → ρ_k`),
  Thm 3.2 for `L_k` and `F_k`, eq. (38) and Table 1 (exact limits vs simulated vs
  approximation, the sign convention of newton.md §5.5); Status: step 6 done.
- **`docs/refs.bib`**: DLMF (chapter 5; online, no DOI, cite by URL and equation).
  Chudnovsky only if the docs state the `κ = 4` canonicity, and then from a source we
  can verify (e.g. the 1984 AMS Math. Surveys 19 book), not from memory.
- **CLAUDE.md**: a "Step 6" block under Decisions; layout (`_transfer.py`); open items
  (FS Note VI.3 sign reading, low priority).

**Release 0.1.0** (CLAUDE.md, Build order; a **separate side task and PR after this
one**):

1. **Trusted publishing**: `.github/workflows/release.yml` on tags `v*`: build, `twine
   check`, `pypa/gh-action-pypi-publish` with OIDC (`id-token: write`), GitHub
   environment `pypi`. Guilherme must add the pending publisher on pypi.org (owner,
   repository, workflow, environment) and create the environment. The token in
   `~/.pypirc` can then be revoked. CLAUDE.md "Releasing" is rewritten.
2. **README**: status (steps 1–6), install (`pip install flanpy`, `[fast]`), a
   10-line Catalan example from `System` to `asymptotics`, links to the docs and the
   two running examples, how to cite (`CITATION.cff`).
3. **`CHANGELOG.md`** (Keep a Changelog): 0.1.0 with steps 1–6 and the public API
   list. 0.0.1.dev0 is the name reservation.
4. **Metadata**: version `0.1.0` in `pyproject.toml` and `CITATION.cff` (`date-released`),
   classifiers (`Development Status :: 3 - Alpha`, topics), project URLs. RTD: activate
   the `stable` version / tag builds.
5. Tag `v0.1.0` from `main` after CI, the release workflow publishes, and then a bump to
   `0.2.0.dev0`.

## 11. Chunk plan

| chunk | content | PR |
|---|---|---|
| 1 | this note + spike | draft PR "Step 6: transfer" |
| 2 | `_transfer.py` at a fixed `k`: `e_l`, `Γ` reduction, `GammaConstant`, `Asymptotics`, `asymptotics`, `terms`, `error_exponent`, `evalf`, `as_expr`, refusals; tests (closed forms, refusals, slow `n = 1000` replacing step 5's helper); `catalan.py` step 6; guide part 1 | same branch |
| 3 | `ratio` (composite field, `Ratio`), symbolic `k` (`asymptotics_in_k`, monomials, `at`, `combine`); Broda acceptance (Thm 3.2, Table 1, eq. (8)); `broda2020.py` step 6; guide part 2; CLAUDE.md | same branch |
| — | release 0.1.0 (§10) | separate PR, after this one is merged |

## 12. Decisions (approved 2026-10-31)

- **Q1**: yes: `asymptotics`, `asymptotics_in_k`, `Asymptotics`, `ratio`, `Ratio`, `GammaConstant`; module `_transfer.py`.
- **Q2**: `ValueError`, no new exception class.
- **Q3**: store `Γ(ν)`, `0 < ν < 1`, as FS and Broda print it; the reflection key only for `==` and `hash`.
- **Q4**: radicals for algebraic parts of degree <= 2 in `as_expr`, `CRootOf` otherwise.
- **Q5**: `O`-terms only; explicit bounds are a later feature (Section 4 stays as the plan).
- **Q6**: refuse different `ρ`.
- **Q7**: refuse, naming the bad pieces' `KSet`s.
- **Q8**: step 7 (Section 7.3 stays as its plan).
- **Q9**: no shortcut now; revisit when writing the 0.1.0 README.
- **Q10**: chunks as in Section 11; 0.1.0 as a separate PR after this one.
- **Review changes**: a number of terms is `count`, never `n`, and counts the `D_j` including zeros (Sections 2.3, 2.4, 8, 9; `LocalExpansion.terms` renamed in chunk 2); `GammaConstant.value()` is `algebraic()` (the `Γ` part excluded; `evalf` is the full value), and the field of algebraic factors is `factors`; the FS Note VI.3 sign reading confirmed in review (a low-priority open item); Chudnovsky cited only from a verified source and only if the docs state the `κ = 4` canonicity.
