# Design note: `singularity`

Status: approved 2026-10-30 with amendments (answers in Section 11). Chunk 2
(`AlgebraicReal`, the circle test, `KCases.flat_map`) implemented on branch `singularity`.

Build-order step 4. From a `Curve` `G(z, w, k) = 0` (step 3) and its branch, the series
`w(z)` of `curve.system`, find the **dominant singularity** `ρ` of the branch (not of the
curve), decide **Case I** (`w(ρ) = a` finite) or **Case II** (`w → ∞`), and give `ρ` and `a` as
**exact algebraic reals**. For symbolic `k`, describe `ρ_k` as "the `j`-th positive root
of `m_ρ(z, k)`" on a `KSet`, with every exceptional `k` instantiated (a `KCases`, nothing
dropped).

Spike: `dev/spikes/singularity_spike.py`, with sections `fixed`, `lk`, `ratio`, `symk` and
`signed` (about 22 s in total). Numbers marked "spike" come from it (Apple M1,
Python 3.13, SymPy 1.14.0, python-flint 0.9.0, mpmath 1.3.0).

**Sources read for this note** (theorem numbers as printed):

- Flajolet–Sedgewick 2009 (FS): Thm IV.6 (Pringsheim, p. 240); Def. IV.5 and Lemma IV.1
  (period, Daffodil lemma, p. 266); §VII.7 (pp. 493–506): Def. VII.6, eq. (78)
  (exceptional set), Lemma VII.4 (location of singularities, p. 496), Thm VII.7
  (Newton–Puiseux, p. 498), Thm VII.8 (algebraic asymptotics, p. 501), Example VII.19
  (non-crossing forests: the branch is the middle sheet, p. 502), Note VII.36
  (algorithm ACA, p. 504) and Prop. VII.8 (decidability of connections, p. 505).
  Also Thm VII.6 (DLW, p. 489).
- Pivoteau–Salvy 2025 (PS, arXiv:2508.20008v1). The paper targets labelled structures,
  but from §3 on everything is stated for analytic systems with non-negative Taylor
  coefficients, so it covers our polynomial systems. Used: Def. 3.1 (well-founded),
  Def. 3.2 (irreducible), Prop. 3.3, Lemma 3.5 (convexity), Lemma 3.6 (components),
  Def. 3.7 (characteristic system), Thm 3.8, Cor. 3.9–3.10, Prop. 3.11 (no non-negative
  solution beyond `ρ`), Thm 3.14, Thm 4.1 (Newton iteration converges iff `a ≤ ρ`),
  Thm 4.4 (a posteriori bound), Thm 5.1 (Algorithm Radius), Thm 6.2 (periods,
  Bell–Burris–Yeats), Thm 6.3 (nonlinear irreducible: dominant singularities
  `ρ e^{2iπj/q}`), Thm 6.4 (linear case), Lemma 6.5 (Daffodil), Lemma 6.7 (Wielandt),
  Thm 7.1 and Algorithm Periods (§7.1), Prop. 8.2 (DLW for components), Thm 8.5,
  Lemma B.2 (irreducible: all coordinates finite at `ρ` unless linear), Lemma B.5
  (`Λ < 1` below `ρ`), Cor. B.6.
- Broda et al. 2020 §§3–4. **There is no §5**: §4 is the survey table (Table 2). §3.3
  covers Pringsheim, Cases I/II, eq. (19), `m_ρ` from `res_w(G, ∂G/∂w)`, `m_a` from
  `res_z`, and Case II, where `m_ρ | lc_w G` (citing Hille 1962, Thm 12.2.1). §3.4 covers
  `L_k`: the printed `res_w`, `p_k` with a unique zero in `]0, 1[`,
  `disc p_k = 32 k^2 (k − 54)^3`, and uniqueness on `|z| = ρ_k` for `k < 54`, `= 54`,
  `> 54`. §3.1 gives `ρ_k` of `R_k` as the positive root of
  `Δ_k = 1 − 2z − (7 + 8k) z^2` (unnumbered, p. 7). Also eqs. (17), (18), (32)–(38).
- Hitchhiker's Guide (HG, Broda et al. 2014): eq. (9) `Δ_k = (4 − 8k) z^2 − 4z + 1`,
  `ρ_k = 1/(2(√(2k) + 1))` (unnumbered, after (9)), Prop. 4 (square-root transfer).

## 1. Input, output, scope

### 1.1 Input

- A `Curve` (step 3), or a `KCases[Curve]` from `eliminate_in_k`. The branch is the
  `w`-coordinate of `curve.system`'s solution. On `curve.kset` the step 3 invariants hold:
  annihilating, no content in `z`, squarefree, full `w`-degree (`elimination.md` §1.4).
- **The closure.** Only the unknowns `w` depends on matter (`elimination.md` §1.1;
  private `_closure`). Every property below (non-negativity, radius, period,
  irreducibility) is a property of the closure, not of the whole system. Example: `L` in
  the `ssnf` system depends on `(B, B̄, M, M̄)`; `R` in `(R, S)` depends only on `R`.

### 1.2 What step 4 returns

For each `k` (fixed, or per piece of `k`):

- `ρ`: the radius of convergence of `w(z)`, which is also its dominant singularity on
  `(0, ∞)`; an exact algebraic real; `m_ρ` is its minimal polynomial;
- `case`: `"I"` if `lim_{x→ρ⁻} w(x) = a < ∞`, `"II"` if it is `+∞` (Broda 2020 §3.3);
- `a` (Case I only): exact, with minimal polynomial `m_a`;
- `period` `p`: `w(z) = z^r V(z^p)` (FS Def. IV.5);
- `unique`: whether `ρ` is the only singularity on `|z| = ρ` (`True` / `False` / `None` =
  undecided, with a warning; Section 5).

### 1.3 Scope: non-negative closures

Everything is decided when the closure is **non-negative**. That is the `system.md` test
(`k = 1 + t`, all coefficients `>= 0`) applied to the closure, which makes it hold for
every real `k >= 1`. Then Pringsheim applies, and so does the PS 2025 theory. All
generating functions in the reference papers are of this kind, cost series (`d/du` at
`u = 1`) included.

**Signed closures are refused** in this step with `UndecidedSingularityError`, which
lists the candidates. The reason, with the spike's example `F = 1 − zF^2`: `G = zF^2 + F − 1`
has candidates `{0, −1/4}` and none on `(0, ∞)`, yet the radius is `1/4` and the dominant
singularity is `−1/4`. Without positivity the dominant singularity can sit anywhere on its
circle, and deciding which candidate the branch actually reaches is FS's *connection
problem* (Note VII.36). Section 3.4 lists what could be added later. This is question Q1.

## 2. Exact algebraic reals

### 2.1 Representation

`AlgebraicReal` is an immutable value: **minimal polynomial** `m ∈ Z[x]` (irreducible over
`Q`, primitive, positive leading coefficient) plus **index** `i` among the real roots of
`m` in increasing order. `(m, i)` is canonical, so equality and hashing are exact. A
private cache holds the current **isolating interval** `[lo, hi]` with rational
endpoints. It contains exactly that root of `m`, and refinement only ever narrows it. The
pair "minimal polynomial + isolating interval" of CLAUDE.md is `(m, cached interval)`.
The index makes the identity independent of how far the interval has been refined.

### 2.2 Operations steps 4–6 need

| operation | how | step |
|---|---|---|
| isolate the real (positive) roots of `f ∈ Q[x]` | factor over `Q`, `Poly.intervals` per factor, refine overlaps | 4 |
| refine to width `ε` | `Poly.refine_root` | 4–6 |
| `sign p(θ)` for `p ∈ Q[x]` | **exact**: `p(θ) = 0` iff `m | p` (`m` irreducible); otherwise refine until `p` has no root in `[lo, hi]` (Sturm count), then evaluate at `lo` | 4–5 |
| compare `θ, η` | equal iff same `(m, i)`; else refine both until the intervals are disjoint (terminates: distinct reals) | 4 |
| evaluate | refine to `10^-(dps+5)` and convert to `mpmath.mpf` | 4–6 |
| minimal polynomials of `θ^2`, `−θ`, `1/θ` | resultants (`Res_x(m(x), y − x^2)`, ...) then factor | 4 |
| norm `Res_x(m(x), G(x, w))` (the fibre over `θ`) | SymPy resultant | 4 |
| arithmetic `θ ± η`, `θη`, `θ/η`, roots of `P(θ, η, y)` | resultants + factor + isolation | 6 (b, c) |
| polynomial zero test at a pair `(ρ, a)` | Broda's gcd test (eqs. (22)–(27)) or a primitive element | 5 |

Step 4 needs only the first seven rows. Arithmetic is added in step 6, when `b` and `c`
need it.

### 2.3 What exists, what we write

- **SymPy** provides exact real root isolation (`Poly.intervals`, continued fractions,
  the same code as `KSet.where`), `refine_root`, `count_roots` (Sturm), `factor_list`
  over `Q` and `Q[k]`, `resultant`, `discriminant`, and `CRootOf` (indexed roots with
  cached intervals and mpmath evaluation). Complex root isolation exists
  (`intervals(all=True)`) but is slow. Number fields (`QQ.algebraic_field`) are slow and
  not needed in step 4.
- **python-flint 0.9** gives `fmpz_poly.complex_roots()`, which returns *certified*
  Arb balls and is fast. `fmpz_poly.real_roots()` raises `NotImplementedError` (checked),
  which confirms the step 2 note that flint has no exact real isolation. Flint is useful
  only as an optional accelerator for the numerical identification in the circle test
  (Section 5.1).
- **We write**: the `AlgebraicReal` class; exact sign and comparison; positive-root
  isolation with rational separators; the window test (Section 4); the circle test
  (Section 5); and the certified comparison of Section 3.3. About 300 lines of private
  helpers in `_algebraic.py`.
- `AlgebraicReal.as_expr()` returns the equal `CRootOf`, for SymPy interop.

### 2.4 The coefficient-field interface

CLAUDE.md asks for the Puiseux coefficient field (step 5) to sit behind a swappable
interface: SymPy radicals now, an explicit number field `Q(θ)` later, for characteristic
polynomials whose roots are not radicals. `AlgebraicReal` is the `θ` of that explicit
field, and it is also what step 4 returns. **Step 4 adds no protocol.** There is one
implementation and nothing to swap (same rule as step 3's backends). The protocol
(`_Field`: elements, `+ − × ÷`, `is_zero`, sign under the real embedding, roots of a
polynomial) comes with step 5, with `AlgebraicReal` as the generator of its `Q(θ)`.

## 3. The dominant singularity of the branch

### 3.1 Candidates

`R(z) = lc_w(G) · disc_w(G)`, squarefree part over `Q` (fixed `k`). Its roots are FS's
exceptional set (eq. (78): `Res_w(G, ∂_w G) = ± lc_w G · disc_w G`) together with the
points at infinity (ACA step (P1)). **FS Lemma VII.4**: the branch can be continued
analytically along any path from `0` that avoids them, so every singularity of `w` is a
root of `R`. Content factors are not candidates: the step 3 invariant "no content in `z`"
holds on `curve.kset` (`X_prim`). `z = 0` can be a root (`lc = z` for Catalan, `R_k` and
`A_k`), but the branch is a power series there. If `w` is not a polynomial, its radius is
finite: an algebraic function that is entire is a polynomial (Puiseux at `∞` bounds it by
`C|z|^m`, then Liouville). A polynomial branch has no singularity, and `singularity`
raises `ValueError` for it.

### 3.2 A candidate is not a singularity of the branch, even for Broda's `L_k`

`disc_L G = −k^3 z^3 p_k(z) q_k(z)^2` with

```
q_k(z) = (2k^3 + 36k^2 + 108k) z^3 + (12k^2 + 72k) z^2 + 15k z − 2,
```

and `q_k` has a **positive root below `ρ_k` for every `k`** (spike: `k = 1`: 0.0859 vs
`ρ_1 = 0.2548`; `k = 54`: 0.00223 vs `2/45`). The branch passes analytically through it
(the other sheets meet there: `ord disc = 2`). So "the smallest positive candidate" is
wrong for the acceptance example itself. Broda 2020 §3.4 prints
`res_w(G, ∂G/∂w) = z p_k(z) / (k^2 (4k^2 z^4 − 4k^2 z^3 + k(k + 24) z^2 + 4(k − 4) z − 4))`,
a rational function. The computed resultant of eq. (17) is the polynomial
`k^3 z^3 p_k^2 q_k^2` (spike). **The printed formula differs from the computed resultant
`k^3 z^3 p_k^2 q_k^2`; the conclusion ("the singularities of `L_k` must be zeros of
`p_k`") stands (Section 3.3, `k = 1..60`); to confirm with the authors.** The examples use
this wording. It is not an erratum (answer to Q5).

Two more examples. The constructed one (spike): `F = z + zF^2 + zG`, `G = z + zG^2` has
candidates `0.2144 < 0.4064 < 0.5`, and `ρ_F = 0.4064` is the **second** one (the first
comes from the sheet through the other branch of `G`). In FS Example VII.19 the branch is
the *middle* sheet of three, and FS settle which sheet reaches `ξ` by positivity.

### 3.3 Non-negative closures: certified comparison with `ρ` (the core algorithm)

**Pringsheim** (FS Thm IV.6): `ρ` is a singularity, hence `ρ` is a positive root of `R`.
So `ρ` is one of the positive candidates `σ_1 < ... < σ_s`. It is found by deciding
`q < ρ` or `q > ρ` at rationals `q` strictly between consecutive candidates, by binary
search (`⌈log2 s⌉` decisions; one for `L_k`, two for the constructed example). This
avoids the connection problem entirely. Each decision is a **certificate checked in exact
rational arithmetic**.

Let `Y = H(z, Y)` be **the closure of `w`** (Section 1.1), with zero coordinates removed.
It is polynomial with non-negative coefficients at the given `k`, and well-founded in the
sense of PS Def. 3.1 (that is the PSS 2012 criterion of `system.md`). Let `J = ∂H/∂Y`
and `Λ` the spectral radius.

**The certificates run on the closure only, never on the whole system.** Its radius is
the radius of `w` (positivity: `w` dominates termwise a positive multiple of
`z^p U` for every `U` it depends on, so its radius is at most theirs, and `w` is analytic
wherever the closure is). An unknown outside the closure can have a smaller radius and
would answer "`q > ρ`" wrongly. The chunk 1 spike had this bug: with
`U = 1 + 10zU^2` (radius `1/40`) added to the constructed example, `compare` answered
"above" for `q ∈ (0.214, 0.406)`, and the `x0` assertion failed. It is fixed in the
spike, and the system is in the tests (Section 8).

- **`q < ρ` certificate**: a rational vector `u >= 0` with `H(q, u) <= u` componentwise.
  *Proof.* The Kleene iterates `Y_n` (polynomials with non-negative coefficients
  increasing to `Y` coefficientwise, PS Prop. 3.3) satisfy `Y_n(q) <= u` by induction
  and monotonicity of `H`. So `Y(q) = lim Y_n(q) <= u` (monotone convergence), the
  series converges at `q`, and `q <= ρ`. `q` is not a candidate, so `q < ρ`.
  *Existence when `q < ρ`*: `Λ(J(q, Y(q))) < 1` (PS Lemma B.5 on each irreducible
  component via Lemma 3.6; non-recursive blocks have a zero diagonal block, and `J` is
  block-triangular). So `v = (I − J)^{-1} 1 > 0` and `Jv = v − 1`, and
  `u = Y(q) + δv` gives `H(q, u) = u − δ·1 + O(δ²) < u` for small `δ`. A rational `u`
  near it works by continuity.
- **`q > ρ` certificate**: a vector `ℓ` built from `0` by sound steps, with
  `Λ(J(q, ℓ)) >= 1`. Sound steps are Kleene steps `ℓ ← H(q, ℓ)`, or Newton steps
  `ℓ ← ℓ + (I − J(ℓ))^{-1}(H(ℓ) − ℓ)` taken only while `Λ(J(ℓ)) < 1`, each rounded
  *down* **and clamped from below by the previous iterate**:
  `ℓ_{n+1} = max(round_down(N(ℓ_n)), ℓ_n)` componentwise (the spike's
  `max(down(·), old)`). So `ℓ_n >= 0` and `ℓ_n` is non-decreasing. The proof needs both:
  `ℓ >= 0` keeps `(q, ℓ)` in the orthant where `J(q, ℓ) >= 0` and convexity applies, and
  the clamp keeps rounding from undoing progress. Taking a max of two vectors that are
  both `<= Y(q)` keeps the bound. *Proof.* If `q <= ρ`, then `q < ρ` and, by induction,
  `ℓ <= Y(q)`. For a
  Newton step, `Y − N(ℓ) = (I − J(ℓ))^{-1}[H(Y) − H(ℓ) − J(ℓ)(Y − ℓ)] >= 0` by
  convexity (PS Lemma 3.5) and `(I − J)^{-1} >= 0`. Then
  `Λ(J(q, ℓ)) <= Λ(J(q, Y(q))) < 1` (monotonicity of `Λ` on non-negative matrices,
  PS Lemma 6.7). Contradiction. `Λ >= 1` is decided exactly: `J` is a non-negative
  rational matrix, so `Λ` is its largest real eigenvalue (Perron–Frobenius), and
  `Λ >= 1` iff its characteristic polynomial has a root in `[1, ∞)` (Sturm count).
- **Termination.** Exactly one of the two holds. For `q < ρ`, the floating-point
  Newton approximation of `Y(q)` converges (PS Thm 4.1, first item) and the `u` above is
  found. For `q > ρ`, exact Newton steps reach `Λ >= 1` after finitely many steps
  (PS Thm 4.1, third item, polynomial case). With rounding, the implementation doubles
  the precision on each round. The only `q` where neither certificate may appear are the
  finitely many radii of irreducible components at which `Λ` tends to exactly 1. After a
  budget, the implementation moves `q` within its gap. *The termination argument with
  rounding is ours and is sketched here; the full version goes in the docstring (Q6).*
- **Why not PS Algorithm Radius as is** (Thm 5.1). It works per strongly connected
  component, with a characteristic system (Def. 3.7) per component and an oracle for
  equalities. Here the curve already supplies an exact finite candidate set, so only
  comparisons with rationals are needed, and those never require an equality oracle.

Spike timings for the whole fixed-`k` analysis: Catalan, `R_k`, `A_k`, the periodic
`T = z + zT^2`: 0.01–0.03 s; constructed 0.05 s; `F_k` 0.05 s; `L_k` 0.18–0.25 s.
`L_k`, `k = 1..60`: every `ρ_k` is the positive root of `p_k`, Case II, and agrees with
mpmath roots of `p_k` to 40 digits (12 s in total).

### 3.4 Not chosen, or left for later

- **Curve-only tracking along `[0, ρ]`.** That means a branch's rank among the real
  sheets, carried across each real candidate by local analysis (a CAD of the real curve,
  or Puiseux at every candidate). It works for signed systems on the real axis, but needs
  step 5's Newton–Puiseux at algebraic points to decide nodes against branch points.
  Positivity makes it unnecessary.
- **Signed closures** (Q1). Decidable later without new theory: `deg_w G = 1` (poles =
  roots of `lc`) and `deg_w G = 2` (closed form: odd-order roots of the discriminant are
  branch points of both sheets). In general: FS Note VII.36 (ACA, connection by certified
  numerics with separation bounds) or rigorous numerical analytic continuation of the
  D-finite `w` (Mezzarobba's work; not read yet).

## 4. Case I / Case II, `a`, `m_ρ`, `m_a`

### 4.1 The window test (exact, rational)

1. `N(w) = Res_z(m_ρ(z), G(z, w)) ∈ Q[w]`. Its real roots include those of `G(ρ, ·)`.
   Take rational separators `q_0 < r_1 < q_1 < ... < r_s < q_s` between its real roots.
2. Pick a rational `x0` in `(σ_{i−1}, ρ)` above every root of `G(·, q_j) ∈ Q[z]` that is
   below `ρ`, for all `j`. Then on `[x0, ρ)` no real sheet crosses a separator (because
   `G(x, q_j) ≠ 0`) and none comes from infinity (`lc ≠ 0` there: no candidate). So each
   window `(q_{j−1}, q_j)` holds the same number of real sheets on all of `[x0, ρ)`.
3. Certified enclosure `ℓ_w <= w(x0) <= u_w` from the two certificates of Section 3.3 at
   `x0 < ρ`. Refine until `G(x0, ·)` has exactly one root in `[ℓ_w, u_w]` (its roots are
   simple: `x0` is not a candidate). That root is `w(x0)`, and its window is the
   branch's window.
4. `w` is increasing on `[0, ρ)`, so its limit exists. If the window is the top one
   `(q_s, ∞)`, the limit is `+∞` (no root of `G(ρ, ·)` is there): **Case II**. Otherwise
   the limit is a root of `G(ρ, ·)` in that window, which is the unique root `r_j` of `N`
   there: **Case I**, `a = r_j`. The bottom window would mean `−∞`, which is impossible
   for a non-negative series (asserted).

**Consistency check** (internal assertion, numerical): in Case I,
`G(ρ, a) = ∂_w G(ρ, a) = 0` (Broda eq. (19)). Otherwise the implicit function theorem
makes the branch analytic at `ρ`, which contradicts Pringsheim.

### 4.2 Minimal polynomials

- `m_ρ` is the irreducible factor of `R` over `Q` that vanishes at `ρ`. It is unique,
  because distinct irreducible factors share no root. In Case II, `m_ρ | lc_w G` (Broda
  §3.3, Hille Thm 12.2.1; FS footnote 13 to Thm VII.7). For `L_k` it is `p_k`, eq. (18):
  `lc_L G = p_k` exactly.
- `m_a` is the irreducible factor of `N` with `a` as root. Broda uses
  `Res_z(G, ∂_w G)` instead. Both contain `m_a`, and `N` has lower degree and depends
  only on the actual `ρ`. All the "exact gcd tests" are divisibility tests by an
  irreducible polynomial (`sign_of` in Section 2.2).

### 4.3 What the spike finds

| example | `ρ` | case | `a` |
|---|---|---|---|
| Catalan `F = 1 + zF^2` | `1/4` | I | `2` |
| `R_k`, `k = 2` | root of `23z^2 + 2z − 1` (`Δ_k`) | I | `√6/2` (`= (1 − ρ)/(4ρ)`) |
| HG `A_k`, `k = 2` | `1/6` (`= 1/(2(√4 + 1))`) | I | `4/3` |
| constructed | root of `32z^4 + 16z^3 − 8z^2 − 4z + 1` (2nd candidate) | I | `1.2303` |
| `F_k = z + B + B̄`, `k = 2` | root of `p_2` | I | root of a cubic |
| `L_k`, `k = 2` | root of `p_2` (2nd candidate) | II | — |
| `L_k`, `k = 54` | `2/45` (`p_54 = 2(45z − 2)(18z + 1)^2`) | II | — |

`L_k`: `lc = p_k`, and `G` has no `L^2` term, so the degree drops by 2 at `ρ_k`.
`ord_ρ disc = 1`, so the two sheets that escape form a 2-cycle at infinity, the shape that
eq. (32) and `α = 1/2` describe (step 5).

## 5. Uniqueness of the dominant singularity

### 5.1 Exact circle test

Every singularity is a candidate (Lemma VII.4), so: **no other root of `R` on `|z| = ρ`
implies that `ρ` is unique, with proof.** The test is exact. For each irreducible factor
`f` of `R`, `P_f(y) = Res_z(f(z), z^{deg f} f(y/z))` has as roots all the products
`σ_i σ_j` of roots of `f`. A root `σ ≠ ρ` with `|σ| = ρ` makes `ρ^2 = σσ̄` such a product.
So test `gcd(P_f, m_{ρ²}) = 1`, with `m_{ρ²} = Res_x(m_ρ(x), y − x^2)`. **The factor
`f = z` of `R` is skipped**: `0` is never on the circle, and `P_z` is degenerate
(`Res_z(z, y)` is a constant). For `f = m_ρ`,
take `P_f / Res_z(f, y − z^2)` (pairs `i ≠ j`) and separately test whether `−ρ` is a root
of `f`. If a gcd is non-trivial, numerics identify which roots lie on the circle (the
spike's periodic example finds `−1/2`). Spike: `L_k`, `k = 1..60`: no other candidate on
the circle, which agrees with Broda's argument (§3.4).

### 5.2 Period

PS §7.1 (Algorithm Periods; rules from Bell–Burris–Yeats) for polynomial systems: start
from the valuations (from the series), then iterate the gcd rules to a fixed point:
`Π(sum) = gcd(Π's, valuation differences)`, `Π(product) = gcd(Π's)`. If `p > 1`, then
`w = z^r V(z^p)` and every `ρ e^{2iπj/p}` is a singularity (Pringsheim for `V`). Spike:
`T = z + zT^2` has period 2, `±1/2`. All other examples have period 1. Tests check the
period against the support of `series(·, 60)`.

### 5.3 Decision

| situation | `unique` | status |
|---|---|---|
| no other candidate on the circle | `True` | proved (Lemma VII.4) |
| `p > 1` | `False` | proved (rotations are singular) |
| `p = 1`, other candidates on the circle, closure nonlinear irreducible | `True` | proved (PS Thm 6.3) |
| otherwise | `None` | `SingularityWarning` naming the candidates (CLAUDE.md, Rigor) |

When `p > 1`, the full set of dominant singularities is known to be exactly the rotations
if no other candidate lies on the circle, or by PS Thm 6.3. Step 6 assumes uniqueness
(transfer at a single point), so it will refuse `unique` other than `True` until
multi-point transfer exists (FS Thm VI.5).

### 5.4 What a full proof would need later

For a candidate `σ` on the circle that is not a rotation: decide whether the branch is
singular at `σ`. Positivity gives `|w(σx)| <= w(ρx)`. So in Case I, `w` is bounded near
`σ`, and only finite branch points matter. With step 5's Newton–Puiseux at `σ` (complex
algebraic), "every sheet over `σ` is analytic" proves `σ` regular, "every sheet is
singular" proves it singular, and a mixed fibre needs the branch identified at `σ`:
certified evaluation along the ray (ACA steps (D1)–(D4)), or PS Thm 6.4 for linear
components.

## 6. Symbolic `k`

### 6.1 The description

On a piece, `ρ_k` = **the `j`-th positive real root of `f(z, k)`**, where `f` is an
irreducible factor over `Q(k)` of `R̃ = ∏` (distinct irreducible factors of
`lc_w G · disc_w G` of positive `z`-degree). The case is constant on the piece, and in
Case I, `a_k` = the `j'`-th positive root of an irreducible factor `g(w, k)` of
`Ñ = Res_z(f, G)`. "Positive root, counted from 0" is stable where "the `i`-th real root"
is not: for `L_k`, `ρ_k` is the only real root of `p_k` for `k <= 53` and the largest of
three for `k >= 55`.

`f(·, n)` need not be the minimal polynomial of `ρ_n`: `p_54 = 2(45z − 2)(18z + 1)^2`, so
`m_{ρ_54} = 45z − 2`. That is Hilbert irreducibility again, as in step 3. The description
stays valid, and `Singularity.specialize(n)` gives the exact minimal polynomial at one
`n`.

### 6.2 Where the description can change: the projection set `P`

```
P = { lc_z(R̃),  disc_z(R̃),  R̃'(0, k)            (R̃' = R̃ without the factor z)
      cont_z(lc_w G),  cont_z(disc_w G)            (curve non-degenerate at real k)
      lc_w(Ñ),  disc_w(Ñ),  Ñ(0, k)                 (Case I only)
      E(k),                                        (uniqueness, Section 6.4)
      c(k) for each monomial coefficient of the closure }   (support, hence period)
```

The last two rows matter only for `unique` and `period`. `c(k) >= 0` for `k >= 1`, and
it vanishes where the support changes (for example `k − 1` at `k = 1`).

**Proposition.** Let `I ⊂ [1, ∞)` be an open real interval on which no polynomial of `P`
vanishes. Then on `I` the description of Section 6.1 (`f`, `j`, the case, `g`, `j'`) is
constant, and so is `unique`.

*Argument* (ours; to be written in full in the chunk 4 docstring):

1. For every real `k ∈ I`: the closure is non-negative (the `k = 1 + t` test covers real
   `k`) and well-posed (a pass for symbolic `k` holds at every `k`, `system.md` §5.7). The
   curve has full degree and is squarefree (contents in `P`), so `ρ_k` is a positive root
   of `R̃(·, k)` (Section 3.1).
2. The roots of `R̃(·, k)` are distinct (`disc_z`), keep their number (`lc_z`) and none
   crosses `0` (`R̃'(0, k)`). Real roots cannot become complex without colliding. So the
   positive roots are `s` continuous functions `σ_1(k) < ... < σ_s(k)`, each a root of
   one fixed factor.
3. **`k ↦ ρ_k` is continuous on `I`.** Both certificates of Section 3.3 are *strict
   inequalities in finitely many polynomial expressions of `(q, k)`*, so they are open
   conditions. `q < ρ_{k0}` gives `u` with `H(q, u, k0) < u` strictly (the `δ` argument),
   which also holds near `k0`, so `ρ` is lower semicontinuous. `q > ρ_{k0}` gives a
   Kleene iterate with `Λ > 1` strictly (possible for all but finitely many `q`), which
   also holds near `k0`, so `ρ` is upper semicontinuous.
4. A continuous function on a connected `I` that takes its values in
   `{σ_1(k), ..., σ_s(k)}` (distinct continuous functions) equals one `σ_i` throughout.
5. **The case and `a_k`: the window index is locally constant.** Fix `k0 ∈ I` and the
   data of the window test there: separators `q_j` for `Ñ(·, k0)` and a point `x0`.
   (a) The roots of `Ñ(·, k)` are distinct with `lc_w Ñ ≠ 0` (`P`), so they move
   continuously, and for `k` near `k0` the same rationals `q_j` separate them, one root
   per window as at `k0`.
   (b) `G(x, q_j, k0) ≠ 0` on the compact `[x0, ρ_{k0}]` (at `ρ_{k0}` because `q_j` is
   not a root of `G(ρ, ·)`). So it is non-zero on `[x0, ρ_k]` for `k` near `k0`
   (`ρ_k` continuous, item 3). Likewise `σ_{i−1}(k) < x0 < ρ_k`, so `[x0, ρ_k)` holds
   no candidate. The test's hypotheses therefore hold at `k` with the same data.
   (c) `w(x0, k)` is continuous in `k`. The coefficients `w_n(k)` are polynomials with
   non-negative coefficients in `t = k − 1`, hence non-decreasing in `k`, so on
   `[k0 − ε, k0 + ε]` they are bounded by `w_n(k0 + ε)`, and
   `Σ w_n(k0 + ε) x0^n < ∞` because `x0 < ρ_{k0+ε}`. Uniform convergence follows
   (M-test). `w(x0, k)` is never a `q_j` (by (b)), so it stays in its window.
   Hence the window index is locally constant, so constant on the connected `I`, and so
   is the case. In Case I, `a_k` is the root of `Ñ(·, k)` in that window. It is
   continuous, it stays in one irreducible factor `g` (roots of distinct factors never
   meet: `disc_w Ñ ≠ 0`), and it keeps its positive index `j'` (`g`'s positive roots
   stay distinct and away from `0`: `Ñ(0, k) ≠ 0`). The full proof goes in the chunk 4
   docstring.

This **needs no monodromy argument** and no degree-drop conditions. An earlier draft of
this note used both. Continuity of `ρ_k` plus distinctness of the candidates is enough.

### 6.3 Real cuts, exceptional integers, samples, merge

- The **real** roots of `P` in `[1, ∞)` cut it into intervals, not just the integer
  roots. A non-integer root between `n` and `n + 1` can change the description between
  them. `L_k` has cuts at `35.52` (`k^2 − 34k − 54`, from `disc_z q_k`, where two
  negative roots of `q_k` appear), `42.38`, `54`, `58.61`, `108`, `112.57` and `146.13`.
- **Exceptional `k`** = the integer cuts (`L_k`: 54 and 108). Each is instantiated:
  fixed-`k` algorithm on the curve at `n`.
- **Each interval containing integers is decided by one sample integer** (the
  proposition). Squarefree `P` means a sign change at each cut, so the intervals come
  from `KSet.where` sign pieces.
- **Merge**: adjacent pieces, and exceptional points, whose descriptions are equal
  (same `f`, `j`, case, `g`, `j'`) merge, the way `eliminate_in_k` merges. **Nothing is
  dropped**: every `k` of the input `kset` lands in exactly one piece.

Spike: `R_k`, `A_k`: no cuts, one piece (0.1 s). `F_k`: cuts 54, 76.29, 108, 146.13, one
piece (0.8 s). `L_k`: 7 cuts and 10 samples, which all give "1st positive root of `p_k`,
Case II", so **one piece `[1, ∞)`** (3.5 s).

### 6.4 `E(k)` and `k = 54`

`E(k) = Res_y(m_{ρ²}(y, k), P_f(y, k))` over the factors `f` (Section 5.1 with `k`
symbolic), times `Res_z(f_ρ, f_ρ(−z))`. It vanishes where some other candidate might meet
`|z| = ρ_k`. It is coarse (it also catches real pairs with `σ_i σ_j = ρ_k^2`), which only
adds samples. If `E ≡ 0` (for example `−ρ_k` is always a candidate, periodic families),
the sample decides through the period.

**`k = 54`** for `L_k`: `p_54 = 2(45z − 2)(18z + 1)^2`. The complex pair of `p_k`
(`k <= 53`) collides at `−1/18`, which is also a root of `q_54`
(`Res_z(p_k, q_k) = −8k^3(k − 54)^3(k + 2)^3`), and splits into two negative roots for
`k >= 55`. None of this touches `ρ_54 = 2/45`. It is a simple root, the certified
bracketing gives Case II, and `|−1/18| ≠ 2/45`, so it is unique (this matches Broda §3.4,
"`k = 54`" paragraph). 54 is exceptional (a cut), is instantiated, and merges.
`Singularity.specialize(54)` reports `m_ρ = 45z − 2`.

## 7. Output types and public API

```python
class AlgebraicReal:  # fixed number (Section 2)
    polynomial: sympy.Poly  # minimal polynomial over Q, primitive, lc > 0
    index: int  # among its real roots, increasing, from 0
    interval: tuple[Fraction, Fraction]  # current isolating interval

    def refine(self, eps) -> AlgebraicReal: ...
    def evalf(self, dps=15) -> mpmath.mpf: ...  # also __float__
    def sign(self, p) -> int: ...  # exact sign of p(self)
    def as_expr(self) -> sympy.CRootOf: ...

    # __eq__/__hash__ exact on (polynomial, index); <, <= exact; immutable; picklable


class KRoot:  # "the positive_index-th positive root of polynomial(x, k)" (Section 6.1)
    polynomial: sympy.Poly  # irreducible over Q(k), gens (x, k)
    positive_index: int  # among the positive roots, from 0 (not AlgebraicReal.index)

    def at(self, k: int) -> AlgebraicReal: ...  # exact; factors polynomial(x, k)


class Singularity:
    curve: Curve
    rho: AlgebraicReal | KRoot
    case: Literal["I", "II"]
    a: AlgebraicReal | KRoot | None  # None in Case II
    period: int
    unique: bool | None
    k: sympy.Symbol | None
    kset: KSet

    def specialize(self, k: int) -> Singularity: ...  # exact minimal polynomials at k


def singularity(curve: Curve, *, k: int | None = None) -> Singularity: ...
def singularity_in_k(curves: Curve | KCases[Curve]) -> KCases[Singularity]: ...


class UndecidedSingularityError(ValueError): ...  # signed closure (Q1)


class SingularityWarning(UserWarning): ...  # uniqueness undecided
```

- `singularity` works at a fixed `k`, mirroring `series` / `eliminate`. With `k=n` on a
  symbolic curve it uses `curve.specialize(n)`, so `ρ` and `a` carry their minimal
  polynomials over `Q`. A polynomial branch raises `ValueError`.
- `singularity_in_k` returns a `KCases` covering the `kset` of each input curve. A
  `KCases[Curve]` input is handled piece by piece.
- `m_ρ`, `m_a` are `rho.polynomial`, `a.polynomial`. There are no separate fields (small
  API); the docstrings give Broda's names.
- **`KCases.flat_map(f: T -> KCases[U]) -> KCases[U]`: yes, now.** `singularity_in_k`
  on a `KCases[Curve]` is exactly this (each curve's analysis splits again on `k`, and
  each result is restricted to its piece), and steps 5–6 chain the same way. It is
  backward compatible, as `elimination.md` §6 planned. `uncovered` of the result =
  input `uncovered` ∪ each piece's `uncovered`.
- Private modules: `_algebraic.py` (AlgebraicReal, KRoot, root isolation, window and
  circle tests), `_singularity.py` (`Singularity`, certificates, candidates, period,
  projection set). CLAUDE.md's planned `singularity/` package becomes these two modules
  (the layout of steps 1–3).

## 8. Tests (`tests/test_algebraic.py`, `tests/test_singularity.py`)

**Algebraic reals:** construction from `(poly, index)`; exact equality and hashing; sign
at a multiple of `m` (zero) and near roots; comparison of two close roots of different
polynomials; `evalf` against mpmath to 50 digits; `as_expr` equals the `CRootOf`;
pickling. `hypothesis`: random squarefree integer polynomials, isolated roots against
`nroots`.

**Quadratic first (Hitchhiker's Guide style):**

- Catalan: `ρ = 1/4`, Case I, `a = 2`, `period = 1`, `unique = True`.
- HG `A_k`, `k = 1..20`: `ρ_k = 1/(2(√(2k) + 1))` (positive root of eq. (9)), Case I,
  `a_k = (8ρ_k^2 − 2ρ_k + 1)/(4ρ_k)` (eq. (8) with `√Δ = 0`); symbolic: one piece,
  `KRoot` of `(8k − 4)z^2 + 4z − 1`.
- Broda `R_k`, `k = 1..20`: `ρ_k` = positive root of `Δ_k` (§3.1, p. 7), Case I,
  `a_k = (1 − ρ_k)/(4ρ_k)` (eq. (2) with `√Δ = 0`); symbolic: one piece.

**Acceptance (Broda `L_k`):**

- `singularity_in_k(eliminate_in_k(ssnf, L))` is **one piece `[1, ∞)`**: `rho` =
  `KRoot(p_k, 0)` with `p_k` = eq. (18), Case II, `unique = True`. 54 is exceptional and
  merged. The cut `k = 17 + 7√7 ≈ 35.52` (from `disc_z q_k`) is present. The other cuts
  depend on how coarse `E` is and are not pinned.
- `k = 1..60` (54 included): `cases[n].specialize(n)` equals `singularity(curve, k=n)`;
  `ρ_n` agrees with mpmath roots of `p_n` to 40 digits; `m_{ρ_54} = 45z − 2`.
- The skipped candidate: the positive root of `q_k` is a candidate below `ρ_k` for every
  tested `k` (private helper), and is not returned.
- `slow`: `[z^n] L_k / [z^{n+1}] L_k · ((n + 1)/n)^{−1/2}` at `n = 1000` matches `ρ_k`
  to `1e-6` relative for `k ∈ {2, 10, 50, 54}` (spike: `2.1–2.4e-7`; plain ratio
  `5e-4`).
- `F_k` (eq. (38) denominator): Case I, one piece, `a_k` a `KRoot` of a cubic factor of
  `Ñ`.

**Candidate not a singularity:** the constructed `F = z + zF^2 + zG`, `G = z + zG^2`:
`ρ` is the 2nd of 3 positive candidates. Plus `L_k` above.

**Closure only:** the same system with `U = 1 + 10zU^2` added (radius `1/40`, outside
the closure of `F`) gives the same `ρ_F ≈ 0.4064`, and the certificates at
`q ∈ (0.214, 0.406)` say `q < ρ`.

**Certificates near `ρ`** (Q6): for `L_2` and the constructed example, rationals
`q = ρ ± 2^-60` (rounded to a rational on the right side of `ρ`) get the right
certificate on both sides.

**Signed:** `F = 1 − zF^2` raises `UndecidedSingularityError` listing `{0, −1/4}`.

**Uniqueness:** periodic `T = z + zT^2`: `period = 2`, `unique = False`, `−1/2` found
by the circle test. A `p = 1` case with a non-rotation candidate on the circle, built in
chunk 3, gives `unique=None` with `SingularityWarning` (or `True` through PS Thm 6.3 when
the closure is nonlinear irreducible). Period against series support for every example.

**Certificates (private):** for `L_k`, `k = 2`, rationals just below and above `ρ` get
the right certificate; soundness spot check: every `u` certificate dominates
`series(·, 200)` evaluated at `q`.

**Properties (`hypothesis`):** random small non-negative systems (as in step 3). `ρ` is
a root of `R`; the coefficient ratio at `n = 200` is within 10% of `ρ` (aperiodic
cases); Case II iff `w(x)` exceeds any bound as `x → ρ⁻` along the certified enclosures.

**API:** errors (polynomial branch, signed, wrong `k`), value semantics of the three
classes, `KCases.flat_map` (restriction, `uncovered` propagation).

## 9. Docs plan

- **Docstrings** (NumPy, doctested Examples, EX01): `AlgebraicReal` (Catalan's `1/4`,
  `R_k`'s `ρ_2`), `KRoot`, `Singularity`, `singularity` (Catalan; cites FS Thm IV.6,
  Lemma VII.4, PS Prop. 3.11 / Thm 4.1 / Lemma 3.5), `singularity_in_k` (`R_k`; cites the
  Section 6.2 proposition), `KCases.flat_map`, the error and the warning.
- **Guide page `docs/guide/singularity.md`** (executed): exact algebraic reals; the
  dominant singularity of Catalan and `L_2`, showing the skipped `q_2` candidate; Case
  I/II; `singularity_in_k` for `L_k` and reading the `KCases`; `specialize(54)`;
  uniqueness and the periodic example; the signed refusal.
- **`examples/catalan.py`**, "Step 4": `ρ = 1/4`, `a = 2`, Case I, unique (asserts).
  Status: step 4 done.
- **`examples/broda2020.py`**, "Step 4": `R_k` (`ρ_k` the positive root of `Δ_k`, §3.1;
  eq. (8) context), `L_k`: one piece `[1, ∞)`, `ρ_k` the unique positive root of `p_k`
  (§3.4), Case II, the `q_k` candidate, `k = 54`, mpmath and `n = 1000` spot checks for
  `k ∈ {2, 10, 50}` (Table 1's `k`), the note on §3.4's printed resultant (Q5); `F_k`:
  Case I. Status: step 4 done.
- **`docs/refs.bib`**: nothing new is cited beyond FS, PS 2025, Broda 2020 and HG (all
  present). Hille 1962 is cited only through Broda.
- **CLAUDE.md**: layout (`_algebraic.py`, `_singularity.py`), a "Step 4" block under
  Decisions, the python-flint `complex_roots` fact, the open item ticked.

## 10. Chunk plan

Step 4 is split (design question 9):

| chunk | content | PR |
|---|---|---|
| 1 | this note + spike | draft PR "Step 4: singularity" |
| 2 | `AlgebraicReal` + `_algebraic.py` helpers (isolation, sign, compare, separators, circle test) + `KCases.flat_map`; tests; guide page part 1 | same branch |
| 3 | `singularity` at fixed `k` for non-negative closures: certificates, window test, Case I/II, period, uniqueness, errors; tests (all examples, `L_k` for `k = 1..60`, slow ratio test); catalan.py step 4 | same branch |
| 4 | `singularity_in_k`, `KRoot`, projection set, cuts, merge; `L_k` acceptance for all `k`; broda2020.py step 4; guide part 2; CLAUDE.md | same branch |

Chunks 2–3 are useful on their own (fixed `k`), and chunk 4 carries the only new
argument (Section 6.2). One PR, as in earlier steps, reviewed per chunk.

## 11. Decisions (approved 2026-10-30)

- **Q1 Scope**: yes. Non-negative closures only; signed closures raise
  `UndecidedSingularityError`. Its message says that the `k = 1 + t` test is
  *sufficient, not necessary*: a system it rejects may still have non-negative series.
  `deg_w G <= 2` and ACA are later work.
- **Q2 Names**: yes, with `KRoot.index` renamed **`positive_index`**.
  `AlgebraicReal.index` counts real roots, and one name must not mean two things.
- **Q3**: yes, undecided uniqueness is a warning with `unique=None`. Step 6 refuses
  `unique != True`.
- **Q4 Chunks**: as in Section 10.
- **Q5**: worded as "differs from the computed resultant `k^3 z^3 p_k^2 q_k^2`; the
  conclusion stands; to confirm with the authors", not as an erratum (Section 3.2).
- **Q6**: yes. The full termination argument goes in the chunk 3 docstring, with tests
  at `q` within `2^-60` of `ρ` on both sides (Section 8).

Amendments folded in after approval: the certificates run on the closure of `w`
(Section 3.3, with the spike fixed and the `U = 1 + 10zU^2` test); rounded Newton steps
clamped by the previous iterate (Section 3.3); the factor `z` skipped in the circle test
(Section 5.1); constancy of the case and `a_k` (Section 6.2, item 5); `positive_index`
(Section 7).
