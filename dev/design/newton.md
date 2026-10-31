# Design note: `newton`

Status: approved 2026-10-30 (decisions in Section 11). Implemented: chunk 2 (`_field.py`,
Section 4.4), chunk 3 (`local_expansion` at a fixed `k`, Section 5.6) and chunk 4
(`local_expansion_in_k`, `KRoot(over=)`, Section 6.5). Step 5 is complete.

Build-order step 5. From a `Singularity` (step 4): the **local expansion** of the branch
`w(z)` at its dominant singularity `ρ`. That means the exponent `α`, the ramification
`κ`, the leading constant (`b` in Case I, `c` in Case II, Broda 2020 eqs. (20), (31))
**exactly**, and as many further Puiseux terms as asked for, with exact coefficients. For
symbolic `k`: the same on the pieces of step 4, split further where the local picture
changes, with `b_k`, `c_k` as `KRoot`s and every exceptional `k` instantiated (a
`KCases`, nothing dropped).

Spike: `dev/spikes/newton_spike.py`, sections `field`, `fixed`, `broda`, `symk`, `relative`
(0.2 s, 6.5 s, 5.4 s, 20 s, 13 s). Numbers
marked "spike" come from it (Apple M1, Python 3.13, SymPy 1.14.0, python-flint 0.9.0,
mpmath 1.3.0).

**Sources read for this note** (numbers as printed):

- Flajolet–Sedgewick 2009 (FS), §VII.7: Thm VII.7 (Newton–Puiseux, p. 498) and its
  footnote 13 (`k0 < 0`: `κ = 1` possible, polar; `k0 >= 0`: a singularity needs
  `κ >= 2`); the Catalan expansion (81) and Newton's polygon method (pp. 499–500);
  Thm VII.8 (algebraic asymptotics, eqs. (82)–(83), p. 501); Example VII.18 (Motzkin,
  printed expansion in `δ = (1/3 − z)^{1/2}`, p. 502); Example VII.19 (non-crossing
  forests: the "middle" branch, the sign of `β` chosen by positivity, pp. 502–503);
  Example VII.20 (supertrees, exponent `1/4`, pp. 503–504); Note VII.36 (ACA, steps (P2),
  (P3), (D1)–(D4), III; pp. 504–505). Also Thm VII.3 (smooth implicit-function schema,
  `γ = √(2r G_z/G_ww)`, p. 468), Thm VI.1, Cor. VI.1 (p. 392), Thm VI.4 (p. 393) and
  Thm VI.5 (p. 398), for the line with step 6.
- Broda et al. 2020 §3.3–3.4: Thm 3.1, Case I/II, eq. (19), (20) (`s = 1 − z/ρ`,
  `w = a − g(s) s^α`, `b = g(0)`), (21)–(22) (derivative support), the gcd test (text
  after (22)), (23)–(26) (Newton polygon over Puiseux series), (27)–(28), (29)–(31)
  (Case II via the reciprocal polynomial), Thm 3.2, (32)–(38), Table 1.
- Hitchhiker's Guide (HG, Broda et al. 2014): Prop. 4 (p. 89: `a − b√(1 − z/ρ)` and
  `a/√(1 − z/ρ)`), eqs. (8)–(16) (p. 91: `A_k`, `Δ_k`, `ρ_k`, `F_k = 1 − √Δ_k`,
  eq. (14) `F_k = 1 − 2 (2k)^{1/4} √ρ_k √(1 − z/ρ_k) + o(·)`).
- Pivoteau–Salvy 2025 (PS, arXiv:2508.20008v1), §8: Def. 8.1 (valuation at `σ`),
  Prop. 8.2 (DLW for a nonlinear component with Puiseux inputs, eqs. (19)–(20)),
  Def. 8.3, Thm 8.5 (classification at `ρ`, `r = 2^m`) **and the induction hypothesis of
  its proof** (the coefficient of `Z^{1/r}` is negative), Thm 8.6, Algorithms
  Irr-Localbehavior and NR-Localbehavior, Thm 8.7.
- `dev/spikes/np_spike.py` (CLAUDE.md, Decisions): `α` from the derivative support and
  gcd tests, and the Newton-polygon recursion.

## 1. Input, output, scope

### 1.1 Input

A `Singularity` (fixed `k`) or a `KCases[Singularity]` / `Singularity[KRoot]` (symbolic
`k`) from step 4. It carries the curve `G` (step 3 invariants: irreducible, squarefree,
full `w`-degree on `kset`), `ρ`, the case, `a`, period and `unique`. **The scope is step
4's**: non-negative closures. Signed systems never reach step 5 (step 4 refuses them).

Step 5 does **not** require `unique = True`. The local expansion at `ρ` is valid
whatever happens elsewhere on the circle. Step 6 refuses `unique != True` (step 4 Q3).

### 1.2 What step 5 returns

With `s = 1 − z/ρ` (Broda eq. (20); FS's scale `(1 − z/ρ)^α`) and `t = s^{1/κ}` (the
positive real root for `s > 0`), FS Thm VII.7 gives, in a slit neighbourhood of `ρ`:

```
Case I:   w = a + Σ_{n>=1} d_n t^n          (κ >= 2, footnote 13)
Case II:  w = Σ_{n>=n0} d_n t^n, n0 < 0     (via v = 1/w = Σ_{n>=p} e_n t^n, p > 0)
```

- `κ`: the ramification (FS's "branching type").
- `α`: in Case I, **the least exponent `e = n/κ ∉ Z` with `d_n ≠ 0`**, and `b = −d_{ακ}`,
  so `w = (analytic at ρ) − b s^α + ...`. In Case II, `v = c s^α (1 + ...)`, so
  `w = (1/c) s^{−α} (1 + ...)`. These are Broda's `α`, `b`, `c` (eqs. (20), (31),
  Thm 3.2), with one change in Case I. Broda takes "the first positive exponent", but
  the transfer of an integer exponent is 0 (`1/Γ(−e) = 0`), so the first exponent with a
  non-zero transfer is the one that matters. The two definitions agree in our scope
  (Section 1.3).
- The coefficients `d_n` (or `e_n`), **exact**, as many as asked for.

### 1.3 What positivity predicts (PS Thm 8.5)

PS Thm 8.5 classifies the behaviour at `ρ` of every coordinate of a constructive system.
Its proof is an induction over the strongly connected components, and its hypothesis is
that every non-constant coordinate with a finite limit has a **negative** coefficient of
`Z^{1/r}`. That proof uses only sums, products, positive constants, linear components and
Prop. 8.2. So, on our reading, it also covers our polynomial systems with non-negative
rational coefficients; to be re-checked when the chunk 3 docstring cites it.
Consequences for `w`:

- **Case I**: `w = a − b t + ...` with `b > 0` and `κ = r = 2^m`, `m >= 1`. So
  `α = 1/κ ∈ {1/2, 1/4, 1/8, ...}` and the first term is already the singular one.
- **Case II**: `α ∈ (1/2^m) Z_{>0}`. `α >= 1` is possible: a pole `1/(1 − 2z)` (`α = 1`;
  any rational `w`, so every regular language) or `(1 − 4z)^{−3/2}` (`α = 3/2`, spike).
  Broda writes "for some `0 < α < 1`" before eq. (29). That restriction is not needed:
  eq. (31) holds for every `α > 0`.

So **a cube-root or a `(3/2)`-type singularity in Case I cannot occur** for a series of
our scope. The algorithm below does not rely on this. It is general Newton–Puiseux, and
PS Thm 8.5 enters only as a check (Q5). **Re-checked in chunk 3 against the printed
proof** (PS §8.2.1 and Appendix D): the hypotheses hold for our scope, with one step
the printed proof leaves implicit (the regular nonlinear case, filled in Section 5.6).
So `α = 1/κ`, `κ = 2^m >= 2` in Case I is now an internal `RuntimeError` assertion, and
stays a test-suite property too.
Exponents `1/3` and `3/2` (Case I) are therefore tested at the private curve level
(Section 8), and `1/4` end to end.

## 2. The local problem and the Newton polygon

### 2.1 The polynomial at the singular point

- **Case I.** `Q(s, Y) = G(ρ(1 − s), a + Y) = Σ q_ij s^i Y^j` with
  `q_ij = (−ρ)^i/(i! j!) · ∂^i_z ∂^j_w G(ρ, a)`. This is Broda eq. (21) with
  `Y = −g(s) s^α`. `q_00 = 0` (`G(ρ, a) = 0`) and `q_01 = 0` (eq. (19)).
- **Case II.** The reciprocal polynomial `H(z, v) = v^m G(z, 1/v)` (eq. (29)), at
  `(ρ, 0)`: `Q(s, V) = H(ρ(1 − s), V)`. Here `∂^j_v H(z, 0) = j! · g_{m−j}(z)`, where
  `g_i` is the coefficient of `w^i` in `G`. So **`q_ij` lies in `Q(ρ)`**:
  `q_ij = (−ρ)^i/i! · g_{m−j}^{(i)}(ρ)`, and there is no `a`.

**Newton polygon** (FS pp. 499–500; Broda eqs. (23)–(26)). The points are `(j, i)` with
`q_ij ≠ 0`. Only the columns `j <= r` matter, where `r = ord_Y Q(0, Y)` is the
multiplicity of `a` as a root of `G(ρ, ·)` (resp. of `0` for `H(ρ, ·)`). `r` counts the
branches through the point. Keep the lowest point of each column and take the lower convex
hull. Every edge `E` with slope `−γ < 0` (`γ = p/q` in lowest terms) carries the branches
`Y ~ c s^γ`. Here `c` ranges over the non-zero roots of the **characteristic polynomial**
`φ_E(c) = Σ_{(j,i)∈E} q_ij c^{j − j_E}`, and `φ_E(c) = ψ_E(c^q)` (lattice points on `E`
are `q` apart).

Broda's points `(j − 1, i)` (eq. (22)) are the same polygon shifted by one column. Two
readings of the printed text need care. Neither changes the paper's results, where the
polygon has a single edge.

- Eq. (27) sums over **all** of `D`. It should sum over the points of `D` on the edge.
  Eq. (33) does exactly that for `L_k`.
- "`α` is the symmetric of the minimum value of slopes" holds when there is one edge.
  With several edges, the steepest one belongs to the branch with the **largest**
  exponent, which need not be `w`'s. Example: `Q = (Y − s)(Y^2 − s)` has edges of slopes
  `−1` (the analytic branch `Y = s`) and `−1/2` (`Y = ±√s`). The rule returns `α = 1`,
  while the singular branch has `1/2`. Which edge carries `w` is the subject of Section 3.

### 2.2 Which derivatives vanish: exact zero tests, mostly without field arithmetic

The polygon needs, for each column `j <= r`, the least `i` with `q_ij ≠ 0`, and `r`
itself. Each such question is "does `D(ρ, a) = 0`?" for `D = ∂^i_z ∂^j_w G ∈ Q[z, w]`.
Cascade, cheapest first:

1. **`D ≡ 0`**: zero.
2. **`a ∈ Q(ρ)`, `a = A(ρ)` with `A ∈ Q[z]`** (Section 2.3), and always in Case II
   (`a = 0`): `D(ρ, A(ρ)) = 0` iff `m_ρ` divides the remainder of `D(z, A(z))` modulo
   `m_ρ`. This is one polynomial remainder, exact, with no inverse and no algebraic
   numbers. It is cheaper than item 3, which needs a resultant.
3. **Broda's gcd test** (text after eq. (22)): `gcd(Res_w(G, D), m_ρ) = 1` proves
   `D(ρ, a) ≠ 0`. Anything else is inconclusive. Use it when `a ∉ Q(ρ)`, to avoid
   building `Q(ρ, a)`.
4. **Exact in `Q(ρ, a)`** (Section 4): the zero test of an element of a number field.

Spike: on all examples (Catalan, Motzkin, `R_2`, `A_2`, `α = 1/4`, pole, `α = 3/2`,
constructed, S5 tree, `L_k`, `F_k`), the hull columns are decided by items 1–2 and agree
with item 3. Item 3 is never inconclusive there, and `a ∈ Q(ρ)` in every example.
**Hence `α` needs no algebraic-number arithmetic in every example we have.** Only
remainders modulo `m_ρ` are used, plus one inverse in `Q(ρ)` to get `A` (Section 2.3).

### 2.3 Is `a ∈ Q(ρ)`?

In Case I, `a` is a multiple root of `G(ρ, ·)`. Let `h = gcd(G(ρ, ·), G_w(ρ, ·))` over
`Q(ρ)` (Euclid in `Q(ρ)[w]`).

- If `deg h = 1`, then `a = −h_0/h_1 ∈ Q(ρ)`. This is the generic square-root case:
  `a` is the only double root.
- If the squarefree part of `h` has degree 1, then `h = (w − a)^e` and
  `a = −h_{e−1}/e ∈ Q(ρ)`. Example: `α = 1/4`, where `a = 1/2` has multiplicity 4.
- Otherwise `G(ρ, ·)` has several multiple roots, conjugate over `Q(ρ)`. Then `a` is
  identified among the real roots of `h` over `Q(ρ)` (Section 4.2, equality with step
  4's `a`), and the field is `Q(ρ, a) = Q(θ)`.

For symbolic `k`, the first subresultant `S_1 = s_11(z) w + s_10(z)` of `G` and `G_w`
gives the same thing: `a_k = −s_10(ρ_k)/s_11(ρ_k)` wherever `s_11(ρ_k) ≠ 0` and
`lc_w G(ρ_k) ≠ 0`. The polynomials `Res_z(f, s_11)` and `Res_z(f, lc_w G)` join the cut
set (Section 6).

### 2.4 The square-root (binomial) fast path

If `w`'s edge has **no interior lattice point**, it runs from `(0, i_0)` to `(r, 0)`
with `gcd(i_0, r) = 1`. Then `φ_E` is a binomial: `c^r = −q_{i_0 0}/q_{0 r}`, and the
constant is a real `r`-th root of an element of `Q(ρ, a)`. Its minimal polynomial over
`Q` is one norm away (Section 4.3), with no field built for `c`. The square-root case is
`i_0 = 1`, `r = 2`. With `G_z(ρ, a) ≠ 0` and `G_ww(ρ, a) ≠ 0`:

```
Case I:   b^2 = 2ρ G_z(ρ, a) / G_ww(ρ, a)      (FS Thm VII.3, γ; Broda (27) with D on the edge)
Case II:  c^2 = 2ρ H_z(ρ, 0) / H_vv(ρ, 0)      (Broda eqs. (33)–(34))
```

The sign is fixed by Section 3 (`b > 0`, `c > 0`). This is the HG closed form: for a
quadratic `G = A w^2 + B w + C` with discriminant `Δ`,
`b = √(−ρ Δ'(ρ)) / (2 A(ρ))`, which is HG eq. (14) for `F_k`. Broda notes that "it is
enough to know that `∂G/∂z` and `∂²G/∂w²` are non-zero for all `k`". These are exactly
the two zero tests the fast path needs. It covers every example in the papers.

Case II **generically gives a pole**. If `lc_w G` has a simple root at `ρ` and
`g_{m−1}(ρ) ≠ 0`, then `r = 1`, the edge is `(0, 1)–(1, 0)` and `α = 1`. The square
root of `L_k` comes from `H_v ≡ 0` (eq. (32) has no `v^1` term).

## 3. Which Puiseux branch is `w`

### 3.1 Real branches, and why no complex numbers are needed

**Lemma R.** Every coefficient `d_n` (Section 1.2) of `w` is real. *Proof.* Thm VII.7
holds in a disc around `ρ` slit along `[ρ, ρ + ε)`, so `(ρ − ε, ρ)` is not on the slit.
On it, `s ∈ (0, ε/ρ)` and `t = s^{1/κ} > 0` (the determination fixed in Section 1.2).
`t ↦ Σ d_n t^n = w(ρ(1 − t^κ))` is a convergent Laurent series in `t` that is real for
`t ∈ (0, δ)`, because `w` is real on `[0, ρ)`. So its coefficients are real. ∎

So `w`'s leading coefficient `c` is a **real** root of some `φ_E`, and all later
coefficients are real. Each real root of each `φ_E` (real `s^γ > 0`) is one real branch
over `(ρ − ε, ρ)`. For even `q` the roots `±c` give the two determinations, `t` and
`−t`. All arithmetic is over **real** number fields (Section 4). No complex algebraic
number is ever needed, unlike ACA (P2), which expands every branch.

### 3.2 The sign rule

`w` is non-negative and not a polynomial, so it is strictly increasing on `[0, ρ)`.

- **Case I**: `w(x) < a` for `x < ρ`, so `Y = w − a < 0` and `c < 0`.
- **Case II**: `v = 1/w > 0` and `v → 0+`, so `c > 0`.

(FS Example VII.19 uses the same argument: "the determination with a minus sign must be
adopted [...] since otherwise one would get negative asymptotic estimates".) In every
example of the spike, the sign rule leaves exactly one real branch. But it is not a proof
in general: two real branches through `(ρ, a)` can both have `Y < 0`.

### 3.3 The rank test (exact; decides every case)

Step 4's window test already computes the data needed (private; refactored to return
it): a rational `x0 < ρ` with no candidate in `[x0, ρ)`, rational separators
`q_0 < ... < q_s` of the real roots of `N = Res_z(m_ρ, G)`, the window `W` of `w` (a
window `(q_{j−1}, q_j) ∋ a` in Case I, the top window `(q_s, ∞)` in Case II), and a
certified enclosure `[ℓ, u] ∋ w(x0)` holding exactly one root of `G(x0, ·)`. Let
`n_W` be the number of real roots of `G(x0, ·)` in `W`, and `r_w` the number in `W`
below `w(x0)`. Both are exact Sturm counts.

**Claim 1.** On `[x0, ρ)`, the real roots of `G(x, ·)` in `W` are `n_W` continuous
functions `y_1 < ... < y_{n_W}`, and `w = y_{r_w + 1}`. *Proof.* There is no candidate in
`[x0, ρ)`, so the roots are simple, `lc_w G ≠ 0`, and the roots move continuously
without meeting. `G(x, q_j) ≠ 0` on `[x0, ρ]` (step 4's choice of `x0`), so no root
crosses a separator. `w` is a continuous root, so it is one `y_i` with a fixed index. ∎

**Claim 2.** As `x → ρ−`, every `y_i` tends to `a` (Case I) or to `+∞` (Case II). The
`y_i` are exactly the real branches of Section 3.1 at `(ρ, a)`, with `Y` of either sign
in Case I and `V > 0` in Case II. *Proof.* In Case I, `y_i` is bounded in `W`, and its
limit points are roots of `G(ρ, ·)` in `W̄`. The only one is `a`: `W` holds a single root
of `N`, and the `q_j` are not roots. In Case II, there is no root of `G(ρ, ·)` in
`[q_s, ∞)`, so `y_i → +∞`. Conversely, a real branch near `a` (resp. `+∞`) lies in `W`
for `x` close to `ρ`. Distinct branches are distinct functions (`G` squarefree). ∎

So: **sort the real branches by their value as `s → 0+` and take the one of rank
`r_w`** (reversed for `V = 1/w` in Case II). Two branches are ordered by the sign of the
first non-zero term of their difference: first by the sign of `c`, then by the exponent
(`c s^γ`, `c' s^{γ'}` with the same sign: for `c < 0`, the larger `γ` is closer to 0),
then by `c` itself (an exact comparison of two real algebraic numbers), and on a tie (a
multiple root of `φ_E`) by deeper terms of the Newton–Puiseux recursion. Distinct
branches separate after finitely many terms, because `G` is squarefree, so this
terminates.

**Two candidates sharing `α`.** Two real branches with the same `γ` and `c` of the same
sign are separated by `c` (if `c ≠ c'`) or deeper (if `c = c'`). The rank test needs no
special case. When they agree up to the `s^α` term (same `α` and `b`), the leading
asymptotics coincide, but later terms do not, and the test still picks the right one.

**Checks** (internal assertions): `n_W` equals the number of real branches at `(ρ, a)`
(Case I) or with `V > 0` (Case II). The chosen branch satisfies the sign rule
(`b > 0`, `c > 0`; proved in Section 3.2). `γ = 1/2^m` in Case I (Section 1.3) is a
test-suite property until chunk 3 settles it (Q5). Spike: `n_W` matches on every example.
Case II needs the sign filter: `W = Y^3` has 2 real branches, `V = ±s^{3/2}`, and 1
sheet in the top window.

**Recommendation: always run the rank test.** It costs one Sturm count on data step 4
already has. The sign rule then becomes an assertion. Compare ACA (P3)/(D1)–(D4), which
matches branches numerically with separation bounds; positivity and step 4's
certificates make the matching exact here.

## 4. The coefficient field

### 4.1 Interface (`_Field`, private protocol)

CLAUDE.md asks for the Puiseux coefficient field behind a swappable interface, and the
step 4 note deferred it to this step (`singularity.md` §2.4). The operations step 5
needs:

| operation | used by |
|---|---|
| `+ − ×`, inverse, powers | `q_ij`, Hensel lifting, series inversion (Case II) |
| `is_zero` (exact) | Newton polygons at every level |
| `sign` under the real embedding (exact) | the sign rule, branch order |
| rational enclosure of an element | identifications (below) |
| `to_real(x) -> AlgebraicReal` | `b`, `c`, `d_n` as public values |
| `real_roots(φ ∈ K[c]) -> [(L, ι: K → L, c ∈ L)]` | leading coefficients; `a ∉ Q(ρ)` |
| equality of two reals living in different fields | branch order |

### 4.2 Implementation now: one real number field `Q(θ)`

`_NumberField`: `θ` an `AlgebraicReal` with minimal polynomial `m_θ`. Elements are
polynomials of degree `< deg m_θ`; arithmetic is modulo `m_θ`, with the inverse from the
extended gcd. The zero test is exact (reduce). The sign of `p(θ)` is `θ.sign(p)` (step 4,
exact). The polynomial arithmetic goes through the `_rings` backends: `fmpq_poly` (with
`xgcd`) under python-flint, SymPy `QQ[x]` (`gcdex`) otherwise, so CI tests both.

- **`to_real(p)`**: the minimal polynomial divides the norm `Res_x(m_θ(x), y − p(x))`.
  Factor it over `Q`, then refine `θ` and enclose `p(θ)` by rational interval Horner
  until the enclosure holds exactly one root of the norm (Sturm counts). That root is
  `p(θ)`, as an `AlgebraicReal`.
- **`real_roots(φ)`** (Trager 1976, norms for algebraic factoring; Loos 1983 for real
  algebraic numbers). Choose a small integer `t` with
  `N_t(y) = Res_x(m_θ(x), φ(x, y − t x))` squarefree. For each irreducible factor `f`
  and each real root `η` of `f`: the gcd over `Q(η)` of `m_θ(x)` and `φ(x, η − t x)` is
  linear, `x − R(η)`. Keep `η` iff `R(η) = θ`; this is exact, because `R(η)` is a real
  root of `m_θ` and its enclosure refines into or away from `θ`'s isolating interval.
  Then `L = Q(η) = Q(θ, c)`, `θ = R(η)`, `c = η − t R(η)`, and `ι` maps `K` into `L`.
  The tower is flattened at every step.

**Pitfall found by the spike.** When testing "`x = target`" with `x` a root of the
target's minimal polynomial, refine **only `x`'s enclosure**. The target's isolating
interval must stay fixed. Refining both in lockstep can loop forever: the enclosure is
wider than the target interval by a constant factor.

Field degrees in the spike: `[Q(ρ, a, c) : Q]` = 2 for Motzkin and `A_2`, 4 for `R_2`,
6 for `L_k` and `F_k` (`k = 2, 10, 50`), 2 at `k = 54`, 4 at `k = 108`, 8 constructed,
10 S5 tree. One analysis takes 0.01–0.10 s (the `fixed` examples) and 0.2–0.3 s (`L_k`, `F_k`), after step 4. Spike `field`:
`c^10 − √2 c^2 − 1` over `Q(√2)` gives 2 real roots of degree 20, in 0.2 s.

### 4.3 Why not SymPy radicals

1. **Zero tests.** The polygon (every level) and the branch order need exact zero and
   sign tests. SymPy decides zero-ness of radical expressions heuristically
   (`simplify`): the spike's `iszero` is not a decision procedure. Reducing modulo
   `m_θ` is one.
2. **`ρ` itself is often not in radicals.** Spike, S5 tree `T = z(1 + T^2 + T^5)`:
   `m_ρ = 3233z^5 − 2250z^4 − 27z^3 + 1600z^2 − 256` has Galois group of order 120
   (`S_5`), and `sympy.roots` returns no radical root. `b` (degree 10) still comes out
   exactly and agrees with FS Thm VII.3's `√(2φ(τ)/φ''(τ))` to 40 digits.
3. Eq. (35)'s `√2`s are a *display* of a Puiseux expansion in `1/k`. That is step 7
   output, not an engine.

So `_NumberField` is the only implementation. Radical forms are a display question for
step 7: for example, show a degree-2 `AlgebraicReal` as `p + q√d`. The protocol stays,
so a faster backend (for example a flint number field `nf`, or Sage) can be swapped in.

**Non-radical characteristic polynomials.** Section 1.3 fixes the slope of `w`'s edge
(`−1/2^m` in Case I), not its lattice points. `φ_E = ψ_E(c^κ)` could have a `ψ_E` of
any degree if the edge were long. In every example the edge is binomial (Section 2.4),
so in practice non-radicality comes from `ρ` and `a`, not from `φ_E`. The general path
is still tested at the private level. `Q = Y^10 − s^4 Y^2 − s^5` at a
rational point has the edge `(0, 5)–(2, 4)–(10, 0)` and
`φ(c) = c^10 − c^2 − 1 = ψ(c^2)` with `ψ(u) = u^5 − u − 1` (Galois group `S_5`). Its 2
real roots `±√u_0` live in a field of degree 10.

### 4.4 Implementation (chunk 2)

`src/flanpy/_field.py`, tests in `tests/test_field.py` (both backends).

- **`Q[x]` arithmetic**: a small adapter `_QQx` in `_field.py` (`fmpq_poly`, `xgcd`,
  `fmpq_mat.charpoly` under flint; SymPy sparse `QQ[x]`, `gcdex`, `DomainMatrix.charpoly`
  otherwise), selected by `_rings.backend()`. It is not added to `_rings`: that module
  builds the scalar rings of the series recurrences (`_Ring`: zero, one, conversions), and
  division, extended gcd and characteristic polynomials do not fit that record.
- **`to_real`**: the characteristic polynomial of the multiplication by `x` on `Q(θ)` is
  `m_x^{[Q(θ):Q(x)]}`, so its squarefree part is `m_x`, with no factorisation over `Q`.
  The root is identified by refining the enclosure of `x` only (the pitfall).
- **Equality across fields** (`_Element.__eq__`): by `to_real` (canonical minimal
  polynomial and index), cached per element. A common field `Q(θ_1, θ_2)` would need a
  primitive element (a Trager norm of degree up to `d_1 d_2`) per comparison; `to_real`
  costs one characteristic polynomial of degree `d`. Hashing uses `to_real` too, so it
  agrees with `==`. Order across fields refines both enclosures (distinct reals).
- **`real_roots`**: Yun over `K`, then Trager per squarefree part, as in Section 4.2;
  a linear factor over `K` (rational roots, `0`, roots already in `K`) keeps `L = K`;
  over `K = Q` the roots are isolated directly. Results ascend, with multiplicities.
- **Timings** (Apple M1, cold, roots plus `to_real` plus one inverse): `c^10 − c^2 − 1`
  over `Q` 0.004 s (flint) / 0.006 s (pure Python); `c^10 − √2 c^2 − 1` over `Q(√2)`
  0.039 s / 0.047 s.
- **References**: Trager 1976 and Loos 1983 are cited in the private docstrings only.
  Their `docs/refs.bib` entries come with the first docs page that cites them (chunk 3
  guide), DOIs checked then.

## 5. The leading constant, how many terms, and the line with step 6

### 5.1 The leading constant, exactly

`b` (Case I, `b = −c_w > 0`) or `c` (Case II, `c > 0`) is `to_real` of the chosen root:
an `AlgebraicReal` with its minimal polynomial over `Q`. Fast path (Section 2.4): one
norm. Spike values, all matching an independent closed form (40 digits unless stated):

| example | `α` | constant | closed form |
|---|---|---|---|
| Catalan | 1/2 | `b = 2` | FS (81): `Y ~ ±4Z^{1/2}`, `Z = s/4` |
| Motzkin | 1/2 | `b = √3` | FS Ex. VII.18; next 4 terms too (below) |
| `R_2` | 1/2 | `b = 1.90062…` (deg 4) | Broda §3.1: `√(2 − 2ρ_k)/(4ρ_k)` |
| HG `A_2` | 1/2 | `b = √3` | `(2k)^{1/4}/(2√ρ_k)`, from HG eq. (14) and `A_k = (F_k + 8z^2 − 2z)/(4z)` |
| `K = U/2 + K^2`, `U = z + U^2` | 1/4 | `b = 1/2`, `κ = 4` | `K = (1 − s^{1/4})/2` exactly |
| `Y = 1 + 2zY` | 1 (II) | `c = 1` | `1/(1 − 2z) = s^{−1}` |
| `W = Y^3`, `Y = 1 + 2BY`, `B = z + B^2` | 3/2 (II) | `c = 1` | `(1 − 4z)^{−3/2}` |
| constructed `F = z + zF^2 + zG` | 1/2 | `b = 1.97710…` (deg 8) | (none; coefficient check) |
| S5 tree | 1/2 | `b = 0.62746…` (deg 10) | FS Thm VII.3 |
| `L_k`, `k ∈ {2, 10, 50, 54, 108}` | 1/2 (II) | `c_k` (deg 6; 2 at 54; 4 at 108) | eq. (34), all digits |
| `F_k`, same `k` | 1/2 | `b_k` (deg 6; 2 at 54; 4 at 108) | (none in the paper; see 5.4) |

Motzkin, in full: the spike's terms `−√3 t + 3/2 t^2 − 1.51554 t^3 + 3/2 t^4 −
1.50201 t^5` (`t = s^{1/2}`) are FS Example VII.18's printed
`1 − 3δ + 9/2 δ^2 − 63/8 δ^3 + 27/2 δ^4 − 2997/128 δ^5` after `δ = t/√3`.

### 5.2 Further terms

After the first level, if `c` is a **simple** root of `φ_E`, put `s = t^q`,
`Y = t^p (c + Y_1)`: then `Q_1(t, Y_1) = t^{−β} Q(t^q, t^p (c + Y_1))` has
`∂Q_1/∂Y_1(0, 0) ≠ 0`. So `Y_1 ∈ L[[t]]` comes from undetermined coefficients (one
linear solve per term), in the **same** field `L`. This is FS p. 500, "after a few
initial stages of iteration, the method of indeterminate coefficients can always be
eventually applied". A multiple root restarts the polygon on `Q_1` (the `np_spike`
recursion), and new fields may appear. In our scope the root is simple whenever the
edge is binomial (Section 2.4: `c^r = u ≠ 0` has simple roots). Case II inverts
`v = t^p(c + ...)` as a power series in `L`.

### 5.3 How many terms step 6 needs (FS Thm VII.8)

Eq. (82): `f_n ~ ρ^{−n} Σ_{j>=j0} d_j n^{−1−j/κ}`. Each term `d s^e` with
`e ∉ Z_{>=0}` transfers to `d ρ^{−n} n^{−e−1}/Γ(−e) (1 + Σ_l c_l n^{−l})` (FS Thm VI.1).
The integer-exponent terms together form a function analytic at `ρ`, which contributes
`O((ρ + ε)^{−n})` given uniqueness (FS Thm VI.4). So:

- **The leading term** (Broda Thm 3.2, HG Prop. 4) needs only `α` and `b` or `c`.
- **`J` terms in powers of `n^{−1/κ}`** need the Puiseux terms with exponents in
  `[α, α + J/κ)`, that is `J` terms after the first singular one, plus the `Γ`
  expansions (step 6).

Spike check (exact terms, transfer by `Γ` ratios, at `n = 1000`): Catalan 4 terms
`3.8e-6`, Motzkin 5 terms `1.1e-8`, `R_2` `2.9e-6`, `A_2` `3.3e-6`, constructed
`3.9e-6`, S5 tree `5.2e-7`, `F_k` (`k = 2, 10, 50`) 3 terms `2.8e-6`; `L_k` leading term
only `3.4–3.6e-4` (Case II terms beyond the first are not inverted in the spike). Step 6
turns these into transfer tests.

### 5.4 The line between step 5 and step 6

| step 5 (`newton`): the function near `ρ` | step 6 (`transfer`): the coefficients |
|---|---|
| local problem, polygon, branch, `κ`, `α` | `Γ` factors, FS Thm VI.1 expansions in `n` |
| `b` or `c` exactly; the square-root closed form `b^2 = 2ρG_z/G_ww` (fast path) | Broda Thm 3.2, HG Prop. 4, FS Thm VII.8 |
| exact Puiseux terms on demand | how many terms for a target error |
| valid whatever `unique` is | refuses `unique != True`; multi-point transfer later (FS Thm VI.5) |
| `KRoot`s for `b_k`, `c_k` | asymptotic formulas in `n`, numeric evaluation |

CLAUDE.md's pipeline item 5 ("Transfer ... includes the square-root special case") keeps
the transfer of `a − b√(1 − z/ρ)` (HG Prop. 4) in step 6. The constant `b` moves to step
5.

### 5.5 Broda eqs. (37)–(38): a sign convention

The spike gives `b_k > 0` for `F_k` (as eq. (20) requires: `w = a − b s^{1/2}`
increases to `a`), with `b_k/√k = 1.244, 1.122, 1.056, 1.054, 1.038` at
`k = 2, 10, 50, 54, 108`. So **`|b_k| ~ k^{1/2}` as eq. (37) says, but with the
opposite sign**. Eq. (37) prints `b_k = −k^{1/2} + o(k^{1/2})`, and eq. (38) writes the
ratio as `−2/(b_k c_k)`. With eq. (20)'s `b`, Thm 3.2 gives
`[z^n] L_k / (n [z^n] F_k) → (1/(c Γ(1/2))) / (b/(2√π)) = 2/(b c)`. So (37)–(38)
consistently use `λ = −b` (the constant of Thm 3.1). The limit `1/2` is unaffected.
Suggested wording, as for the earlier items: "eqs. (37)–(38) use `−b_k`; the result
stands; to confirm with the authors". The exact limits `2/(b_k c_k)` are
`0.41726, 0.44325, 0.46575` at `k = 2, 10, 50`. Table 1's simulations give
`0.42, 0.44, 0.47`, closer than the approximation `1/2 − 1/(4√(2k))`
(`0.375, 0.444, 0.475`). At `k = 54`: `b = 2√15`, `c = √15/7`, ratio exactly `7/15`.
This belongs to the step 6–7 examples, not to step 5.

### 5.6 Implementation (chunk 3)

`src/flanpy/_newton.py` (`LocalExpansion`, `local_expansion`), tests in
`tests/test_newton.py`; `_singularity.py` returns the window data (`_Window`,
`_window_of`; step 4 results unchanged, its tests pass as they were).

- **Local problem** (2.1): `q_ij` by expanding `G(ρ(1 - s), a + Y)` (or `H`) in `K`.
  `a ∈ Q(ρ)` by Section 2.3's three branches (the third, `Q(ρ, a)` by `real_roots`, is
  tested on `(w^2 - 2)^2 + (z - 1) w` at `ρ = 1`; no example reaches it).
- **Zero-test cascade** (2.2): each hull entry is decided by the first rule that
  applies: `zero polynomial`, `remainder mod m_rho` (`K = Q(ρ)`), `gcd test`
  (Broda, proves non-zero), `field`. A private counter (`_newton._STATS`) records the
  rule. In every example the second rule decides; the last two are tested by forcing
  `K ≠ Q(ρ)` on `F_2`.
- **Branches**: the real roots of each edge's `φ`; a binomial edge at the top level
  uses the fast path (2.4: `AlgebraicReal`s from the norm `m_u(y^d)`, the field built
  only for `terms`); other edges use `_NumberField.real_roots`; a multiple root
  restarts the polygon. The rank test is always on; the sign rule and `n_W` are
  assertions (`RuntimeError`).
- **Terms**: Hensel lifting by undetermined coefficients, incremental (powers of `V`
  kept), in the field of the branch; Case II inverts `V` as a series. `terms(count)` gives
  `count` consecutive exponents, zeros included, from `0` (I) or `-α` (II).
- **PS Thm 8.5 for our scope.** The proof of Thm 8.5 uses sums, products, linear
  components and Prop. 8.2, whose proof (Appendix D) needs non-negative Taylor
  coefficients, irreducibility and polynomial dependence on the inputs, never integer
  coefficients; positive rational constants keep a negative `Z^{1/r}` coefficient
  negative. The implicit step: in the regular nonlinear case (`Λ_H(ρ) < 1`), the
  `Z^{1/r}` coefficient is `−(I − J(ρ))^{-1} W` with `(I − J(ρ))^{-1} > 0` entrywise
  (`J` irreducible, non-negative, `Λ < 1`) and `W >= 0`, `W ≠ 0` (PS eq. (35)), so it
  is negative. The argument is in the Notes of `local_expansion`.
- **`n = 1000` (slow tests; transfer by `Γ` ratios, a test helper).** Catalan
  `3.76e-6` (5 terms), Motzkin `1.14e-8` (6), `R_2` `2.93e-6`, `A_2` `3.26e-6`,
  constructed `3.94e-6`, S5 tree `5.22e-7` (5 each), `F_k` `2.84e-6`, `2.82e-6`,
  `2.78e-6` (`k = 2, 10, 50`; 4 terms). `L_k`, with `1/V` inverted: 3 or 4 terms
  (singular exponents `−1/2`, `1/2`) `9.9e-8`, an `O(n^{-2})` error like `F_k`'s 4
  terms (`1/2`, `3/2`) but with a smaller constant; 5 terms (`−1/2`, `1/2`, `3/2`)
  `8.9e-11` (`k = 2, 10, 50`), `O(n^{-3})`. Bounds: 3 times these.
- **Timings** (Apple M1, after step 4; `local_expansion` + `terms(6)`, flint / pure
  Python): Catalan 0.005 + 0.001 s / 0.005 + 0.001 s; Motzkin 0.005 + 0.003 /
  0.005 + 0.003; `R_2` 0.012 + 0.019 / 0.010 + 0.022; `A_2` 0.006 + 0.002 /
  0.007 + 0.003; `α = 1/4` 0.007 + 0.001 / same; pole 0.003 + 0.001 / 0.003 + 0.002;
  `α = 3/2` 0.006 + 0.001 / 0.006 + 0.002; constructed 0.019 + 0.037 / 0.021 + 0.044;
  S5 tree 0.015 + 0.059 / 0.017 + 0.069; `L_2` 0.023 + 0.011 / 0.024 + 0.018; `F_2`
  0.018 + 0.033 / 0.020 + 0.038; `L_50` 0.025 + 0.011 / 0.026 + 0.018; `F_50`
  0.019 + 0.027 / 0.021 + 0.032; `L_108` 0.028 + 0.014 / 0.028 + 0.019. Step 4
  dominates (`singularity(L_k)` about 0.2 s).

## 6. Symbolic `k`

### 6.1 The description on a piece

Input: a piece of `singularity_in_k`, on which `ρ_k = KRoot(f, j)`, the case is fixed,
and in Case I `a_k = KRoot(g, j')`. On a sub-piece, step 5 describes:

- `κ`, `α` and the edge `E` of `w` (constants);
- `a_k = A(ρ_k, k)` with `A ∈ Q(k)[z]` (Section 2.3, the subresultant). This is
  required for symbolic `k`; otherwise `NotImplementedError` naming the piece (Q4);
- the leading constant **relative to `ρ_k`**: "the `j''`-th positive root of
  `Φ(y, ρ_k, k)`", where `Φ` is `φ_E(±y)` with `a = A(z)` substituted and denominators
  cleared. In the fast path, `Φ = den(z, k) y^r − num(z, k)` and `j'' = 0`.

**Why relative to `ρ_k`, and not an absolute `KRoot` over `Q(k)`.** The obvious
absolute description is `KRoot(h, j)` with `h` an irreducible factor over `Q(k)` of the
norm `Res_z(f, Φ)`. It is exact, but **its positive index changes for `L_k`** (spike
`symk`). `h` has degree 6 and is even in `y`, and
`h(0, k) = 128 (k − 54)(2k + 27)`. Below 54 the conjugates of `ρ_k` are complex; above
it they are real (negative) and contribute a second positive root **below** `c_k`. So
`c_k = KRoot(h, 0)` on `[1, 54]` and `KRoot(h, 1)` on `[55, ∞)` (checked `k = 1..60`,
100, 108, 200). It would also cut at `disc_y h` (`k = 4`, where
`h(y, 4) = 64 (y^2 − 10)^2 (9y^2 − 35)`, and 54). Those cuts come from the *conjugates*,
not from `ρ_k`. The relative description cuts only where something happens at `ρ_k`
itself:

```
L_k:  c_k = the positive root of ((k^2 + 12k) ρ_k^2 + 4k ρ_k + 1) y^2 − ((24k + 324) ρ_k^2 + 72 ρ_k − 1)
      (eq. (34)); cuts Res_z(p_k, num) = −128 (k − 54)^3 (2k + 27),
      Res_z(p_k, den) = k^2 (k − 54)^2 (k + 2)^2   ->  only k = 54
```

Spike `relative`: this matches the fixed-`k` `c_k` for `k = 1..60` and 108, so **one
piece `[1, ∞)`** (54 instantiated and merged). For `F_k`: `a_k = −s_10/s_11 (ρ_k)` with
`Res_z(p_k, s_11) = −2^25 k^2 (k − 54)^2`, and the zero tests of `G_z`, `G_ww` at
`(ρ_k, a_k)` cut only at 54. The absolute `KRoot(h, 0)` (`h` of degree 6 in `y`, 3 in
`k`) holds for `k = 1..60` but cuts at `19.42`, `54`, `75.08` (`disc_y h`). Relative:
one piece.

It is also the form step 7 needs. Broda's eq. (36) is obtained by substituting the
Puiseux expansion of `ρ_k` (eq. (35)) into eq. (34). `np_spike.py` does exactly that, and
that is the relative description. `at(n)` stays exact: `ρ_n` is an `AlgebraicReal`, and
the positive roots of `Φ(y, ρ_n, n)` come from `_NumberField.real_roots` over `Q(ρ_n)`,
as an `AlgebraicReal` over `Q` (Section 4.2). Q2 asks how to expose it.

### 6.2 Where the description can change: the cut set `P5`

On a piece of step 4 (description `f`, `j`, case, `g`, `j'` constant), `P5` is the set
of the following polynomials in `k` (each a resultant with `f` in `z`, numerators after
substituting `a = A(z)`):

```
P5 = { Res_z(f, s_11), Res_z(f, lc_w G)               (a_k = A(ρ_k), Case I)
       Res_z(f, D_ij)  for every point (j, i) on the generic lower hull, columns j <= r,
                       including (r, 0) (multiplicity of a) and the j = 0 point
       Res_z(f, disc_y Φ), Res_z(f, Φ(0))             (positive roots of Φ distinct, nonzero)
       Res_z(f, den)                                  (the description's denominators) }
```

Points strictly above the generic hull do not matter. Points below it vanish
identically (that is why it is the generic hull). **Proposition.** On an open real
interval `I ⊂ [1, ∞)` inside a step 4 piece, where no polynomial of `P5` vanishes,
`κ`, `α`, `E`, the branch of `w` and `j''` are constant, and the leading constant is the
`j''`-th positive root of `Φ(·, ρ_k, k)`. *Argument* (ours; full proof in the chunk 4
docstring):

1. The hull entries are continuous in `k` (`ρ_k` is continuous on the step 4 piece,
   and `A` has no pole) and do not vanish on `I`. The identically-zero ones stay zero.
   So the polygon, `r`, `α`, `κ` and `E` are constant.
2. The real roots of `φ_E(·; ρ_k, a_k, k)` stay distinct and non-zero
   (`disc_y`, `Φ(0)`), so they move continuously, and the real branches keep their
   signs and their order at the first level.
3. The rank `r_w` of Section 3.3 is constant on `I`. This is the argument of
   `singularity.md` §6.2, item 5. `x0` and the separators work on a neighbourhood of
   each `k0` (open conditions); `w(x0, k)` is continuous (M-test); the roots of
   `G(x0, ·, k)` stay simple (`x0` is not a candidate). So the count of roots below
   `w(x0, k)` in the window is locally constant, hence constant on the connected `I`.
4. So the selected branch is the same first-level root throughout. Its constant is a
   simple root of `Φ(·, ρ_k, k)` with continuous coefficients, hence continuous
   (implicit function theorem). It is always a positive root, the positive roots are
   distinct and continuous, so its index `j''` is constant (`singularity.md` §6.2,
   item 4).

**Ties at the first level** (two real branches of `w`'s window with the same `γ` and
`c` at a sample `k`) would need the deeper recursion to be constant in `k`. That is not
covered: `NotImplementedError` naming the piece, never a silent guess (Q4). No example
needs it.

### 6.3 Cuts, exceptional integers, samples, merge

As in `singularity.md` §6.3. The real roots of `P5` in each step 4 piece cut it. Each
interval holding integers is decided by one sample (the fixed-`k` algorithm at an
integer of the interval, then lifted: `Φ`, `j''` read off at the sample). Every integer
cut is instantiated (`local_expansion` of `singularity.specialize(n)`). Equal
descriptions merge. The input piece's `kset` is covered exactly. Spike: `L_k` and `F_k`
are one piece `[1, ∞)`, 54 instantiated and merged. `R_k`, `A_k` have no cut. Timing:
the symbolic norms and resultants for `F_k` take 0.3 s. The spike's 60 fixed-`k`
analyses take 13–14 s (`L_k`) and 6 s (`F_k`), but `local_expansion_in_k` needs only the
samples.

### 6.4 What step 7 gets

On the tail piece (`KCases.tail`): `α` and `κ` (constants), `ρ_k = KRoot(f, j)` (step 4),
`a_k = A(ρ_k)` (Case I), and the constant as the relative root of `Φ(y, ρ_k, k)`. That
is everything Broda eqs. (35)–(37) use: Puiseux of `f` in `1/k`, substitution into
`Φ`, Newton polygon in `(y, 1/k)`. Step 5 does no expansion in `1/k`.

### 6.5 Implementation (chunk 4)

`KRoot(..., over=rho_k)` in `_algebraic.py`; `local_expansion_in_k` and
`_FamilyExpansion` in `_newton.py`; tests in `tests/test_newton_in_k.py` and
`tests/test_algebraic.py` (relative `KRoot`).

- **`KRoot(over=)`** (Q2 (a)): `Φ` in `(y, z, k)`, stored with integer coefficients,
  primitive, positive leading coefficient (`y > z > k`); not required irreducible;
  `positive_index` counts the distinct positive roots of `Φ(·, ρ_n, n)`, computed by
  `_NumberField(ρ_n).real_roots`; equality, hash, pickling and `repr` include `over`.
  Existing `KRoot`s are unchanged (their tests pass as they were).
- **The local polynomial for symbolic `k`**: `s_11^m G(z(1 - s), A + Y)` (Case I,
  `A = -s_10/s_11`, so `A + Y = (-s_10 + s_11 Y)/s_11`) or `H(z(1 - s), Y)` (Case II),
  a polynomial in `(s, Y, z, k)`; an entry is generically zero iff
  `Res_z(f, q_ij) ≡ 0`.
- **The cut set includes step 4's `P1 ∪ P2(f)`** (recomputed through `_Family`), not
  only `P5`. The proposition needs them: continuity and simplicity of `ρ_k`, `a_k`,
  and step 4's window data near each `k0` (item 4 of the proof). The cost is a few
  more samples.
- **`Φ` is reduced**: content in `(z, k)` removed, coefficients reduced modulo `f`
  over `Q(k)`, denominators cleared. For `L_k` this gives eq. (34) exactly:
  `((k^2 + 12k) z^2 + 4k z + 1) y^2 - ((24k + 324) z^2 + 72z - 1)`. Without it, `F_k`'s
  `Φ` had degree 21 in `z`.
- **Results**: `L_k`, `F_k`, `R_k`, `A_k`: one piece `[1, ∞)`; `L_k` instantiates 54
  and 108 (roots of `Res_z(p_k, num)` and of step 4's set) and merges both. `at(n)`
  equals the fixed-`k` constant for `k = 1..60` and 108 (`L_k`, `F_k`; slow test, 20
  s) and the closed forms for `k = 1..20` (`R_k`, `A_k`). The family
  `K = kU/4 + K^2`, `U = z + U^2`: step 4 gives `{1}`, `[2, ∞)`; step 5 gives `{1}`
  (`α = 1/2`, `b = 1/√32`), `{2}` (`α = 1/4`, `κ = 4`, `b = 1/2`), `[3, ∞)`
  (`α = 1/2`, `Φ = (4k - 8) y^2 - (k - 1)`: `b_3 = 1/√2`, `b_4 = √6/4`). At `k = 2`
  the vertex `(2, 0)` vanishes (`Res_z(f, q_02) ∝ k^8 (k - 2)^13`: `a_2` becomes a
  4-fold root), and so do the vertex `(0, 1)` (`(k - 2)^12`), `Res_z(f, s_11)`
  (`(k - 2)^3`) and step 4's `disc_z R̃` (`(k - 2)^4`: `ρ_2 = 1/4` meets the radius of
  `U`; step 4 merged it).
- **Timings** (Apple M1, after step 4): `local_expansion_in_k` `L_k` 0.68 s / 0.70 s,
  `F_k` 0.83 s / 0.85 s (flint / pure Python).

## 7. Output types and public API

```python
class LocalExpansion[X: (AlgebraicReal, KRoot)]:
    singularity: Singularity[X]  # rho, case, a, period, unique, k, kset come from it
    alpha: Fraction  # Section 1.2 (Broda's alpha; Case II: w ~ (1/c) s^-alpha)
    ramification: int  # kappa: expansion in powers of s^(1/kappa), s = 1 - z/rho
    b: X | None  # Case I: w = (analytic) - b s^alpha + ...; None in Case II
    c: X | None  # Case II: 1/w = c s^alpha + ...; None in Case I

    def terms(self, n: int) -> list[tuple[Fraction, AlgebraicReal]]: ...

    # fixed k only: the first n terms (exponent, coefficient) of w in powers of s^(1/kappa),
    # from the constant term a (Case I) or from s^-alpha (Case II); exact, cached

    def specialize(self, k: int) -> LocalExpansion[AlgebraicReal]: ...

    # k, kset: properties forwarded from the singularity


def local_expansion(
    singularity: Singularity[Any], *, k: int | None = None
) -> LocalExpansion[AlgebraicReal]: ...


def local_expansion_in_k(
    singularities: Singularity[KRoot] | KCases[Singularity[Any]],
) -> KCases[LocalExpansion[KRoot]]: ...
```

- Names mirror Broda (`α`, `b`, `c`) and step 4 (`a`, `rho`). `b` and `c` are separate
  fields because they mean different things (eq. (20) vs (31)), as `a` is `None` in
  Case II.
- `local_expansion(s, k=n)` on a symbolic singularity works on `s.specialize(n)`, as
  `singularity(curve, k=n)` does. `local_expansion_in_k` is a `flat_map` over the step 4
  pieces (`KCases.flat_map`, step 4).
- `b`, `c` for symbolic `k` (Q2): **(a) recommended**, `KRoot` gains an optional
  `over=` base, "the `positive_index`-th positive root of `polynomial(x, ρ_k, k)`" with
  `ρ_k` the base `KRoot`. This is backward compatible, and `at(n)` returns an
  `AlgebraicReal`. **(b)** A new class for the relative root. **(c)** Absolute
  `KRoot(h, j)` only, accepting the extra pieces of Section 6.1 (`L_k`: two pieces).
- Errors: `TypeError` (not a `Singularity`), `ValueError` (wrong `k`, as step 4),
  `NotImplementedError` for the two documented gaps (`a_k ∉ Q(k)(ρ_k)` and first-level
  ties, symbolic `k` only), naming the piece's `KSet` and saying that fixed `k` works.
  No new exception class. A violated sign rule is a `RuntimeError` (a bug); PS Thm 8.5
  follows Q5.
- Private: the polygon, the edge's `Φ`, the field `L` and the rank data stay private
  attributes (`_edge`, `_field`). Step 6 (transfer) and step 7 (families) use them
  inside the package. They become public only if users need them.
- Modules (flat, CLAUDE.md layout): `_field.py` (`_Field` protocol, `_NumberField`,
  Trager real roots, enclosures; it outgrows `_algebraic.py`, already 1124 lines) and
  `_newton.py` (local problem, polygon, branches, rank test, Hensel lifting,
  `LocalExpansion`, the two functions, the cut set). `_singularity.py`: the window test
  returns its data (private refactor, no behaviour change). `_algebraic.py`: `KRoot`
  `over=` if Q2 (a).

## 8. Tests (`tests/test_field.py`, `tests/test_newton.py`)

**Field (private):** arithmetic and inverse in `Q(√2)` and in a quintic field; exact
sign; `to_real` against `CRootOf`; `real_roots` of `c^10 − c^2 − 1` over `Q`
(2 roots, degree 10, `S_5`), of `c^10 − √2 c^2 − 1` over `Q(√2)` (degree 20), and of a
polynomial with rational and repeated roots; the enclosure pitfall (Section 4.2) as a
regression test. `hypothesis`: random small fields and elements, `to_real(x)` against
mpmath, and `x · x^{−1} = 1`. Both backends (flint and pure Python).

**Private polygon and branches** (curve level, no `System`): `Q = Y^3 − s` (`α = 1/3`),
`Q = Y^2 − s^3` (a `3/2` type), `(Y − s)(Y^2 − s)` (two edges; the "minimum slope" rule
would pick the wrong one), `Y^10 − s^4 Y^2 − s^5` (non-radical edge), and a curve with
two real branches of the same sign through the point, decided by the rank test with
given window data. Also a tie broken at the second level.

**Quadratic first (HG style):**
- Catalan: `α = 1/2`, `κ = 2`, `b = 2`; `terms` = `2, −2, 2, −2, …` (exact,
  `F = 2/(1 + √s)`).
- Motzkin (FS Example VII.18): the five printed terms, exactly.
- HG `A_k`, `k = 1..20`: `b_k = (2k)^{1/4}/(2√ρ_k)`, equivalently
  `b_k^2 = (1 − 2ρ_k)/(8ρ_k^2)` (from HG eq. (14)); symbolic: one piece.
- Broda `R_k`, `k = 1..20`: `b_k = √(2 − 2ρ_k)/(4ρ_k)` (§3.1); symbolic: one piece.

**Acceptance (Broda `L_k`):**
- `local_expansion_in_k(singularity_in_k(eliminate_in_k(ssnf, L)))` is **one piece
  `[1, ∞)`**: Case II, `α = 1/2` (eq. (32)), `κ = 2`, `c_k` = the positive root of
  eq. (34)'s relation. 54 is instantiated and merged (Q2 (a); with (c): two pieces).
- `k ∈ {2, 10, 50, 54, 108}` and `k = 1..60`: `c_k` equals eq. (34) evaluated with
  mpmath to 40 digits, and the symbolic description at `n` equals the fixed-`k` result.
  `c_54 = √15/7` (`49y^2 − 15`), `c_108` of degree 4.
- Eq. (36) is a limit (step 7); here only recorded: `c_k √k / 4 = 0.964, 1.005, 1.016,
  1.016` at `k = 2, 10, 50, 108` (not monotone, so no trend is asserted).

**`F_k`** (Case I, eq. (37)): `α = 1/2`, `b_k > 0`, `b_k/√k` = `1.244, 1.122, 1.056`
at `k = 2, 10, 50` (positivity; the sign note of Section 5.5); one piece `[1, ∞)`.

**Exponents and fields:** `K = U/2 + K^2`: `α = 1/4`, `κ = 4`, `b = 1/2`,
`terms` = `1/2, −1/2, 0, 0, …`; pole `Y = 1 + 2zY`: Case II, `α = 1`, `c = 1`;
`W = Y^3`: Case II, `α = 3/2`, `c = 1`; constructed `F = z + zF^2 + zG`: `b` of degree 8;
S5 tree: `ρ`, `b` not in radicals (`b` of degree 10 against FS Thm VII.3).

**Branch checks:** `n_W` equals the number of real branches (sign-filtered in Case II)
on every example; the sign rule agrees with the rank test.

**Coefficients (`slow`, to become transfer tests in step 6):** at `n = 1000`, the
transfer of the exact terms by `Γ` ratios (a test helper) matches `series` to the spike's
errors: Catalan `3.8e-6`, Motzkin `1.1e-8`, `R_2` `2.9e-6`, `A_2` `3.3e-6`, constructed
`3.9e-6`, S5 tree `5.2e-7`, `F_k` (`k = 2, 10, 50`) `2.8e-6`, `L_k` (leading term)
`3.6e-4` (bounds with a factor 3 of slack).

**Properties (`hypothesis`):** random small non-negative systems (as in step 4): Case I
gives `α = 1/2^m` and `b > 0`; `κ` is a power of 2; at `n = 300` the leading transfer is
within 5% (aperiodic, `unique`).

**API:** errors, value semantics (equality, hashing, pickling, immutability),
`specialize`, `terms` caching, `KCases` coverage (`uncovered` propagated).

## 9. Docs plan

- **Docstrings** (NumPy, doctested `Examples`, EX01): `LocalExpansion` (Catalan:
  `alpha`, `b`, `terms(4)`), `local_expansion` (Catalan; cites FS Thm VII.7, Broda
  eqs. (20)–(31), PS Thm 8.5; Notes: Lemma R, the sign rule and the rank test with
  their proofs), `local_expansion_in_k` (`R_k`; Notes: the Section 6.2 proposition),
  `KRoot(over=...)` if Q2 (a).
- **Guide page `docs/guide/newton.md`** (executed): the local problem and polygon for
  Catalan and `L_2`; `α`, `b`, `c`; `terms`; the `α = 1/4` and pole examples; which
  branch (sign rule, rank); symbolic `k` for `L_k` (one piece; eq. (34)); what step 6
  will do with it.
- **`examples/catalan.py`**, "Step 5": `α = 1/2`, `b = 2`, the first terms; Status: step
  5 done.
- **`examples/broda2020.py`**, "Step 5": `R_k` (`b_k` closed form, §3.1); `L_k`:
  eq. (32)'s `α = 1/2`, `c_k` against eq. (34) for `k ∈ {2, 10, 50, 54, 108}`, one piece
  for all `k`; `F_k`: `b_k` and the eq. (37) sign note. Status: step 5 done.
- **`docs/refs.bib`**: Trager 1976 (SYMSAC, algebraic factoring) and possibly Loos 1983;
  DOIs checked against Crossref when added.
- **CLAUDE.md**: a "Step 5" block under Decisions; layout (`_field.py`, `_newton.py`);
  open items: Broda eqs. (37)–(38) sign convention (for the authors); the readings of
  eq. (27) and of the "minimum slope" rule (presentation, not errata).

## 10. Chunk plan

| chunk | content | PR |
|---|---|---|
| 1 | this note + spike | draft PR "Step 5: newton" |
| 2 | `_field.py`: `_Field`, `_NumberField` (flint and pure Python), `to_real`, Trager `real_roots`; tests | same branch |
| 3 | `local_expansion` at fixed `k`: local problem, zero-test cascade, polygon, real branches, rank test (window data from step 4), Hensel lifting, Case II inversion, `LocalExpansion`; tests (all fixed-`k` examples, private curves, slow `n = 1000`); `catalan.py` step 5; guide part 1 | same branch |
| 4 | `local_expansion_in_k`: `a_k = A(ρ_k)`, relative constants (Q2), `P5`, cuts, samples, merge; `L_k`/`F_k` acceptance; `broda2020.py` step 5; guide part 2; CLAUDE.md | same branch |

## 11. Decisions (approved 2026-10-30)

- **Q1**: yes: `local_expansion`, `local_expansion_in_k`, `LocalExpansion` (`alpha`, `ramification`, `b`, `c`, `terms(count)` (renamed from `terms(n)` in step 6), `specialize`).
- **Q2**: (a) `KRoot(..., over=rho_k)`; chunk 4 pins down the generators `(y, z, k)`, `positive_index` over the distinct positive roots of `Φ(·, ρ_k, k)` (`Φ` need not be irreducible), equality and hash, exact `at(n)` equal to the fixed-`k` `AlgebraicReal`, and `repr`.
- **Q3**: yes: `_NumberField` only, behind the `_Field` protocol; radicals are a step 7 display question.
- **Q4**: yes: `NotImplementedError` naming the piece (its `KSet`) and saying that fixed `k` works.
- **Q5**: rank test always on; the sign rule (`b > 0`, `c > 0`) an assertion (proved, Section 3.2); `α = 1/2^m` (PS Thm 8.5) a `RuntimeError` assertion only after chunk 3 re-checks PS's hypotheses against the printed proof for our scope, a test-suite property until then (never a failure on user input).
- **Q6**: (37)–(38) as a sign convention, in CLAUDE.md's open items; eq. (27) and "minimum slope" as presentation notes in this note only; Table 1's exact limits stay in Section 5.5 for the step 6–7 examples.
- **Q7**: yes, chunks as in Section 10.
