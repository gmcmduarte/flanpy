# Design note: `elimination`

Status: approved 2026-10-30 with amendments A–F (chunk 1, folded in; answers in
Section 9); implemented in chunk 2.

Build-order step 3. From a `System` `F = Phi(z, k, F)` (step 1) and one of its unknowns
`w`, compute one algebraic equation `G(z, w, k) = 0` that the series `w(z)` satisfies,
with the set of `k` where it is valid. Exceptional `k` are handled by instantiation, and
the result is a `KCases` (step 2).

Spike: `dev/spikes/elimination_spike.py` (all cases below, with timings). Numbers marked
"spike" come from it (Apple M1, Python 3.13, SymPy 1.14.0, python-flint 0.9.0).

## 1. Input and output

### 1.1 Input

- A `System` and an unknown `w` of it (a `sympy.Symbol` in `system.unknowns`; anything
  else is a `ValueError`). To eliminate for an expression such as `M + M̄`, add an
  auxiliary unknown (`L = M + M̄`, as Broda 2020 eq. (12) does).
- **Closure first.** Only the unknowns that `w` depends on, transitively through the
  right-hand sides, take part. That set `U` is closed under dependency, so its equations
  form a well-posed system on their own, whose solution is the restriction of the full
  one (the iteration on `U` is the restriction of the full iteration). It has the same
  `w(z)`, hence the same answer, with fewer unknowns. Examples: `R_k` needs only `R`
  in the `(R, S)` system; `B_k` needs `(B, B̄)` of the five unknowns of `L_k`.
- **Zero coordinates** (`System._active`) are substituted by 0 before elimination. If
  `w` itself is one, the answer is `G = w`, with no elimination.

### 1.2 What `G` is

Let `w(z) ∈ Q[k][[z]]` be the `w`-coordinate of the solution. **`G` is the minimal
polynomial of `w(z)` over `Q(k)(z)`, with denominators cleared and normalised
(Section 1.3).** It exists because `w(z)` is algebraic: it is a root of every non-zero
element of the elimination ideal `I ∩ Q[w, z, k]`, where `I = (F_i − Phi_i)`.

Facts used later:

- `G` is irreducible over `Q(k)` and in fact **absolutely irreducible**: irreducible in
  `K[z, w]`, where `K` is the algebraic closure of `Q(k)`. This is what Noether's
  theorem (Section 2.5) needs. Proof: `G` has no factor in `K[z]` (its content in `z`
  is the same over `Q(k)` and over `K`, and it is 1). Let `h ∈ K[z, w]` be an
  irreducible factor with `h(z, w(z)) = 0` in `K[[z]]`. One exists because `K[[z]]` is a
  domain. Normalise `h` so that one of its coefficients is 1. Each
  `σ ∈ Aut(K/Q(k))` acts on coefficients and fixes `w(z)`, whose coefficients lie in
  `Q[k]`. So `σ(h)` is an irreducible factor of `G` with the root `w(z)`. `G` is
  squarefree (irreducible in characteristic 0), so if `σ(h)` and `h` were not
  associates they would be coprime. Their resultant in `w`, a non-zero element of
  `K[z]`, would then vanish at `w(z)` (Section 3.2's identity), which is impossible. So
  `σ(h) = h` for every `σ`, and `h` has coefficients in the fixed field `Q(k)` (`K/Q(k)`
  is Galois in characteristic 0). Hence `h = G` up to a unit. Steps 4–5 rely on this
  too (one curve, no conjugate components).
- **Annihilation holds at every `k`.** `G(z, w(z), k) = 0` in `Q[k][[z]]`, an identity
  between power series with polynomial coefficients in `k`. So
  `G(z, w_n(z), n) = 0` for every integer `n`, where `w_n` is the series of
  `system.specialize(n)` (`series` commutes with evaluation, `system.md` §5.7). This
  needs no condition on `n`. What can change at special `n` is whether `G(·, ·, n)` is
  still the *minimal* polynomial of `w_n`, with the same degree (Section 2).

### 1.3 Normalisation

`G ∈ Z[k][z, w]`, and:

- **primitive over `Z[k]`**: the gcd of its coefficients (as polynomials in `k`) is 1.
  Integer content and content in `k` are removed. A consequence: `G(·, ·, n)` is never
  identically 0;
- **sign**: the leading coefficient in lex order `w > z > k` is positive (so
  `lc_w(G) ∈ Z[k][z]` has a positive leading coefficient in `z`, then in `k`);
- squarefree and irreducible over `Q(k)`, by construction (Section 3).

This fixes `G` uniquely: the only units left in `Q(k)[z, w]` that keep it primitive over
`Z[k]` are `±1`. **Broda 2020 eq. (17)** comes out exactly, with quotient 1 (spike):
`G = p_k(z) L^3 + k z ((k + 12) k z^2 + 4 k z + 1) L − k^2 z^2 (2z + 1)`, with `p_k`
from eq. (18). The curve uses the unknown's own symbol (`L`), not a fresh `w`.

### 1.4 The `Curve` type

```python
class Curve:
    polynomial: sympy.Poly  # G, gens (w, z, k) or (w, z); domain ZZ
    w: sympy.Symbol  # the unknown
    z: sympy.Symbol
    k: sympy.Symbol | None  # None at a fixed k, and for systems without k
    kset: KSet  # where the invariants below hold
    system: System  # the system whose w-coordinate is the branch on G

    def as_expr(self) -> sympy.Expr: ...
    def specialize(self, k: int) -> Curve: ...  # Section 2.5

    # __eq__, __hash__ (content-based), __repr__, __str__, immutable, picklable
```

**Invariants** for every `n` in `kset` (and for the only value when `k` is `None`),
with `G_n = G(·, ·, n)`:

- **annihilating**: `G_n(z, w_n(z)) = 0`;
- **no content in `z`**: the coefficients of `G_n` in `w` have no common factor in
  `Q[z]`;
- **squarefree in `w`**;
- **full `w`-degree**: `deg_w G_n = deg_w G` (`lc_w G(·, n) ≢ 0`).

For symbolic `k`, `G` is also irreducible over `Q(k)`. For a fixed `k` computed by
`eliminate`, `G` is irreducible over `Q`, i.e. it is the minimal polynomial of `w_n`.

These invariants are about `G_n` up to a unit. The integer content and the sign
convention of Section 1.3 can change at `n`: for example, `G_54` of `L_k` has integer
content 2. So `Curve.specialize(n)` and the merge test of Section 2.4 renormalise
`G_n` (Section 1.3 with `k` fixed) before comparing.

**What guarantees them**: the candidate sets `X_deg`, `X_sqf` and `X_prim` of
Section 2.3, together with recomputation at every candidate. Annihilation needs no
condition (Section 1.2). Outside those three sets, the other three invariants hold by
definition of the sets. At a candidate, either the recomputed curve equals the
renormalised `G_n` with full degree (and then `G_n` is the minimal polynomial, which has
all four properties), or the candidate gets its own piece. `X_GB` (Section 2.2) only
adds recomputation points. It is cheap and is kept, so that "eliminate then substitute"
agrees with "substitute then eliminate" outside the candidates, but no invariant
depends on it.

`kset` is redundant with the `KCases` piece that holds the curve. It stays on the curve
so that step 4 can take a single `Curve` and return a `KCases` restricted to its `kset`.
For a system without `k`, `kset = KSet.all()`. For a fixed `k = n`, `kset = KSet.of(n)`
and `system = system.specialize(n)`.

`system` is kept because steps 4–5 need the branch: which of the solutions of `G` is
`w(z)` (Pringsheim, positivity, the Puiseux branch at `ρ`). Building a `Curve` directly
from a printed `G` and a branch description is not planned in this step.

### 1.5 Functions

Mirroring `series` / `series_in_k`:

```python
def eliminate(
    system: System,
    w: sympy.Symbol,
    *,
    k: int | None = None,
    method: Literal["auto", "groebner", "resultant"] = "auto",
) -> Curve: ...


def eliminate_in_k(
    system: System,
    w: sympy.Symbol,
    *,
    method: Literal["auto", "groebner", "resultant"] = "auto",
) -> KCases[Curve]: ...
```

- `eliminate`: fixed `k` (required iff `system.k` is set; same errors as `series`).
  Equivalent to `eliminate(system.specialize(k), w)`.
- `eliminate_in_k`: symbolic `k`; requires `system.k`. **Covers all `k >= 1`**
  (`uncovered` is empty): one generic piece and one piece per exceptional `k`
  (Section 2.4).
- `method`: `"auto"` takes no elimination if the closure has one unknown, the resultant
  if it has two, and Gröbner otherwise (Section 4). `"resultant"` with three or more
  unknowns is a `ValueError`. The two explicit methods exist so that the tests can
  check them against each other.

## 2. Elimination and specialisation

### 2.1 Gröbner path

`I = (F_i − Phi_i : i ∈ U) ⊂ Q[F_U, z, k]`. Compute the reduced lex basis `𝒢` with
`sympy.groebner(..., order="lex", method="f5b")`, generators
`(others..., w, z, k)`: the other unknowns first, in system order, then `w`, `z`, and
`k` **last** (CLAUDE.md, Decisions). Lex is an elimination order, so
`E = 𝒢 ∩ Q[w, z, k]` is a basis of `I ∩ Q[w, z, k]` (Cox–Little–O'Shea, *IVA*, Ch. 3
§1, Elimination Theorem). Take `g0 ∈ E` of least degree in `w` (ties: fewest terms).
`G` is the factor of `g0` selected in Section 3.

The order of the other unknowns matters: for `L_k`, `(B, B̄, M, M̄)` takes 0.87 s and
`(M̄, M, B̄, B)` takes 7.1 s (spike, flint). System order is the default. A heuristic
(e.g. by degree) is left for later; measured cases do not need one.

**When `E` is empty** (no element of positive degree in `w`), some component of `V(I)`
of excess dimension projects onto an open set of the `(w, z)`-plane. Example (spike):
`F = z + zGF − z^2 G`, `G = z + zG^2 − z^2 G`, i.e. `(F − z)(1 − zG) = 0`,
`(G − z)(1 − zG) = 0`, whose variety contains the surface `zG = 1` (any `F`). Then
saturate by `Δ = det(∂P/∂F_U)` (Rabinowitsch: add `yΔ − 1`, `y` ordered first).
`Δ = det(I − J0) = 1` at `(z, F) = (0, c*)`, so the branch lies on a component where
`Δ ≢ 0`. Those components have dimension 1 over `Q(k)` (implicit function theorem), so
the saturated elimination ideal is non-zero. In the example it gives `F − z`.
Saturation is a fallback only: for `L_k` it did not finish in 300 s (spike).
Specialisation (Section 2.2) applies to the saturated ideal `I + (yΔ − 1)` unchanged.

### 2.2 The specialisation theorem

Write `R = Q[k]`, `x = (others, w, z)`, and view `I ⊂ R[x]`. For `n ∈ Z`,
`φ_n : R → Q` is `k ↦ n`. For `g ∈ R[x]`, `lc(g) ∈ R` is the coefficient of its
lex-leading power product in `x`.

**Theorem** (Kalkbrener 1997, *On the stability of Gröbner bases under
specializations*, J. Symbolic Comput. 24(1), 51–58, Thm 3.1; it generalises Gianni 1987,
*Properties of Gröbner bases under specializations*, EUROCAL '87, LNCS 378, 293–297).
Let `G = {g_1..g_s}` be a Gröbner basis of `I ⊆ R[x]` (`R` Noetherian), with
`φ(lc g_i) ≠ 0` exactly for `i <= r`. Then `{φ(g_1)..φ(g_r)}` is a Gröbner basis of the
specialised ideal `φ(I) Q[x]` iff every `φ(g_i)`, `i > r`, reduces to 0 modulo
`{φ(g_1)..φ(g_r)}`.

The theorem needs `𝒢` to be a Gröbner basis of `I` **over `R`** (leading monomials
`lc(g)·X` generate `lm(I)` in `R[x]`), whereas SymPy returns a basis in `Q[x, k]`. Lex
with `k` last makes the two agree. Proof sketch, to go in the docstring: take `f ∈ I`
with leading power product `X` in `x`. Its lex-leading term in `Q[x, k]` is
`LT_k(lc f)·X`. Some `g ∈ 𝒢` has `LT(g) | LT(f)`, i.e. `X_g | X` and
`LT_k(lc g) | LT_k(lc f)`. Subtracting a monomial multiple of `g` keeps `X` and lowers
`LT_k(lc f)`, or lowers `X`. By induction, `lc f ∈ (lc g : g ∈ 𝒢, X_g | X)`.

**Consequences.**

1. If `lc(g)(n) ≠ 0` for **every** `g ∈ 𝒢`, then `r = s`, the condition is vacuous, and
   `φ_n(𝒢)` is a Gröbner basis of the specialised system's ideal. By the elimination
   property, `φ_n(E)` generates its elimination ideal. So eliminating then substituting
   gives the same ideal as substituting then eliminating.
2. All of `𝒢` counts, not only `E`. `I ∩ R = 0` (the system is consistent for generic
   `k`), and `ker φ_n = (k − n)` is not an isolated prime of `0`. So Kalkbrener's
   Thm 3.2 gives no stability for free, and a vanishing `lc` outside `E` can break the
   Gröbner basis property. **At a candidate, the criterion is not used. The candidate is
   decided by recomputing and comparing curves (Section 2.4).** The criterion's answer
   is about the leading-monomial ideal. The question here is about one factor of the
   elimination ideal, and the two can differ.

   Example (spike, `kalkbrener_at`): for `L_k`, `E` has the single `lc`
   `4k^2 (2k + 27)`, but two other basis elements have `lc` `168 k (k − 54)` and
   `6 (k − 54)(k + 2)`. So `k = 54` is a candidate. Of the 20 basis elements, 18 keep
   their `lc` at 54 and 2 are dropped. **The criterion fails**: the 2 dropped elements
   are in `φ_54(I)` but do not reduce to 0 modulo the 18 kept ones. So the kept set
   generates `φ_54(I)` but is not a Gröbner basis of it. Yet the elimination ideal of
   the specialised system is still `(G(·, ·, 54))` (quotient `1/2`), the recomputed
   curve equals the renormalised `G(·, ·, 54)`, and `54` merges into the generic piece
   (Section 2.4).
3. **Candidates from the basis**: `X_GB = {n >= 1 integer : lc(g)(n) = 0 for some
   g ∈ 𝒢}`, a finite set: each `lc(g)` is a non-zero polynomial in `k`. Outside
   `X_GB`, consequence 1 applies with no criterion to check.

### 2.3 Candidates from the curve

The theorem is about ideals. Whether `G(·, ·, n)` still has the invariants of
Section 1.4 is a separate question, decided on `G` itself:

- `X_deg`: integer roots `>= 1` of `cont_z(lc_w G)` (the content in `k` of
  `lc_w G ∈ Z[k][z]`): there the degree in `w` drops;
- `X_sqf`: integer roots `>= 1` of `cont_z(disc_w G)`: there `G(·, ·, n)` is not
  squarefree. If `deg_w G = 1`, this set is empty;
- `X_prim`: the `n >= 1` where `G(·, ·, n)` gains a **content in `z`**, i.e. its
  coefficients `a_0..a_d ∈ Z[k][z]` in `w` acquire a common factor in `Q[z]`. Then
  `G(z0, w, n) = 0` identically for a root `z0` of that factor, which breaks `m_a` and
  the local analysis in step 4. Computed exactly, without sampling:
  1. If `G` has a single non-zero coefficient in `w`, then `G = w` (Section 1.1), and
     `X_prim` is empty.
  2. Let `p` be a non-zero coefficient of least `z`-degree. Let `q = Σ_j λ^j a_j` for
     `λ = 1, 2, ...`, the first `λ` with `gcd(p, q) = 1` over `Q(k)`. Such a `λ` exists:
     `gcd(a_0..a_d) = 1` over `Q(k)` (Section 1.3), so no root `ζ` of `p` is a root of
     every `a_j`. Then `Σ_j λ^j a_j(ζ)` is a non-zero polynomial in `λ`, which vanishes
     for at most `d` values of `λ`. So `gcd(p, q_λ)` is non-trivial for at most
     `d · deg_z p` values of `λ`.
  3. If `G(·, ·, n)` has a content in `z` with a root `z0`, then `p(z0, n) = 0` and
     `q(z0, n) = 0`. So `n` is a root of `Res_z(p, q) ∈ Q[k]`, which is non-zero
     because `p` and `q` are coprime of positive `z`-degree, or `n` is a root of
     `lc_z(p)` or `lc_z(q)`, where the Sylvester matrix does not specialise. If
     `deg_z p = 0`, then `p ∈ Q[k]`, and `n` must be a root of `p`.
  4. Candidates are the integer roots `>= 1` of those polynomials. For each, test
     `gcd_z(a_0(·, n), .., a_d(·, n))` exactly. `X_prim` is the set where it is
     non-constant.

  Counterexample to the rule without `X_prim` (spike):
  `F = 2zF + z − 2z^2 + kzF^2 − 2z^2 F^2`, one unknown. `G = (2z^2 − kz) F^2 +
  (1 − 2z) F + 2z^2 − z` is irreducible, `X_GB` is empty (no elimination), and
  `cont_z lc_w G = cont_z disc_w G = 1`, so no other set flags anything. But
  `G(·, ·, 1) = (2z − 1)(zF^2 − F + z)`, so `G(1/2, w, 1) = 0` identically. Steps 1–4
  give candidates `{1}`, and the gcd at `k = 1` is `2z − 1`. The recomputed curve at
  `k = 1` is `zF^2 − F + z`.

  **With `X_prim`, `deg_w G = 1` is fully certified.** `G_n = a_1 w + a_0` is linear in
  `w`, and with no content in `z` it is irreducible. So outside the candidates `G_n`
  is the minimal polynomial of `w_n`, and Noether's finite set (Section 2.5) is empty.
- `X_cont`: integer roots of the content in `k` of the unnormalised factor of `g0`
  (Section 1.3 divides it out). It is contained in `X_GB` (it divides `lc(g0)`), and
  is listed only so that the resultant path (Section 4) has the same rule.

The candidates are `X = X_GB ∪ X_deg ∪ X_sqf ∪ X_prim ∪ X_cont`, computed exactly
(`Poly.ground_roots` / `KSet.where(..., "==")`). The invariants of Section 1.4 rest on
`X_deg`, `X_sqf` and `X_prim` together with recomputation. `X_GB` and `X_cont` only add
recomputation points.

The spike's example of an exceptional `k` where the generic result fails, with
`C = 1 + z C^2` (Catalan) and `W = z + (k − 1) z C^2`, which is non-negative:
`G = z W^2 + (2kz − k − 2z^2 − 2z + 1) W + k^2 z − 2k z^2 − k z + z^3 + 2 z^2`.
`cont_z(disc_w G) = (k − 1)^2`, and one basis element has `lc = k − 1`. At `k = 1`,
`G(·, ·, 1) = z (W − z)^2` is neither squarefree nor irreducible, while `W_1 = z` and the
recomputed curve is `W − z`. A degree-drop example: `F = 1 + (k − 1) z F^2` gives
`G = (k − 1) z F^2 − F + 1`, `cont_z(lc_w G) = k − 1`, and the curve at `k = 1` is
`F − 1`.

### 2.4 Recompute, merge, and the `KCases`

For each candidate `n ∈ X`: `C_n = eliminate(system, w, k=n)` (substitute, then
eliminate over `Q`, then select the factor with the series at `k = n`). Then:

- **merge**: if `deg_w G(·, ·, n) = deg_w G` and `normalise(G(·, ·, n)) = C_n.polynomial`,
  then `n` joins the generic piece. `normalise` is Section 1.3 at fixed `k`: it removes
  the integer content and fixes the sign, which can change at `n` (Section 1.4);
- **otherwise** `n` gets its own piece `{n}` with value `C_n`.

The result is `KCases({generic: Curve(G, kset=generic), {n}: C_n, ...})`, with
`generic = KSet.all() − {separate n}`. It covers every `k >= 1`: each recomputation
succeeds (`specialize` never fails on a well-posed system, `system.md` §5.7, and
elimination at a fixed `k` always gives a curve). Nothing is dropped.

Outcomes in the spike: `L_k`, `R_k`, `S_k` and Hitchhiker's Guide `A_k` give one piece
`[1, ∞)` (`L_k`: candidate 54, merged even though Kalkbrener's criterion fails there).
The three examples of Section 2.3 (squarefreeness, degree drop, content in `z`) each give
`[2, ∞)` and `{1}`.

Cost: one fixed-`k` elimination per candidate (`L_k` at `k = 54`: 0.07 s). Candidates
are integer roots of a few polynomials in `k`, so there are few of them.

### 2.5 What is not certified: irreducibility at non-candidate `k`

At `n` in the generic piece, `G(·, ·, n)` annihilates `w_n` and is squarefree of full
degree, so the minimal polynomial of `w_n` divides it. They are equal iff
`G(·, ·, n)` is irreducible over `Q`. By Noether's irreducibility theorem
(E. Noether 1922, Math. Ann. 85; effective form: Kaltofen 1995, J. Comput. System Sci.
50(2)), since `G` is absolutely irreducible (Section 1.2), the `n` where `G(·, ·, n)`
is not absolutely irreducible form a **finite** set. (Hilbert irreducibility alone
would allow an infinite thin set.) Computing that set (Noether forms) is impractical,
and nothing in steps 4–6 needs it. Step 4 must already discard candidate singularities
that are not singularities of the branch `w(z)`, and a reducible `G(·, ·, n)` only adds
such candidates.

Decision (approved): say this in the docstrings, and make `Curve.specialize(n)` exact. It
substitutes `k = n`, factors over `Q`, selects the factor with the series at `n`
(Section 3), and returns a curve whose polynomial is the minimal polynomial of `w_n`.
That costs milliseconds for one `n`, so certification is available for any finite
range on demand. Tests and examples spot-check `k = 1..N` this way (CLAUDE.md, Verify).
If `deg_w G = 1`, the gap is empty (Section 2.3, `X_prim`).

## 3. Choosing the factor

### 3.1 Candidates

Factor `g0` over `Q` (`sympy.factor_list`; multivariate over `Q` in `(w, z, k)`, which
by Gauss's lemma is factorisation over `Q(k)` up to content). Drop the factors without
`w`: a factor in `Q[k]` is content, and a factor `h(z, k) ≠ 0` cannot vanish at
`w(z)`. Keep the distinct factors `h_1..h_r` that contain `w`. Exactly one of them, the
minimal polynomial, vanishes at `w(z)`: `g0(z, w(z)) = 0` and `Q(k)[[z]]` is a domain,
so at least one does; two distinct irreducible polynomials have no common root.

If `r = 1`, no test is needed. The implementation still checks `h_1(z, w(z)) ≡ 0`
modulo `z^5` as a guard against bugs.

### 3.2 The test, and why a finite number of terms is enough

**Bound.** Let `g` be the minimal polynomial and `h ≠ g` another candidate. They are
coprime, so `Res_w(g, h) = A g + B h` with `A, B ∈ Q(k)[z][w]` and
`Res_w(g, h) ∈ Q(k)[z] ∖ {0}` (Cox–Little–O'Shea, *IVA*, Ch. 3 §6). Substitute
`w = w(z)`. Then `B(z, w(z)) · h(z, w(z)) = Res_w(g, h)(z)`, and `B(z, w(z))` is a power
series (valuation `>= 0`). Hence

```
ord_z h(z, w(z)) <= ord_z Res_w(g, h) <= deg_z Res_w(g, h)
                 <= deg_w g · deg_z h + deg_w h · deg_z g.
```

The last bound comes from the Sylvester matrix: `deg_w h` rows with entries of
`z`-degree `<= deg_z g`, and `deg_w g` rows with entries of `z`-degree `<= deg_z h`.
`g` is not known in advance, so take

```
N = max over i ≠ j of (deg_w h_i · deg_z h_j + deg_w h_j · deg_z h_i).
```

Every wrong candidate has `h(z, w(z)) ≢ 0 mod z^(N+1)`, and `g(z, w(z)) ≡ 0` to every
order. So **exactly one candidate vanishes modulo `z^(N+1)`, and it is `g`.** This is
a proof, not a heuristic. It needs `w mod z^(N+1)`, i.e. `series(system, N)`.

**Symbolic `k` without symbolic series.** Testing in `Q[k][[z]]` needs
`series_in_k(system, N)`, which is practical only up to `N ≈ 200`. Instead, test at a
fixed `k = n0` with `series(system, N, k=n0)` (rational arithmetic, fast up to
`N = 1000`):

- `g` always passes (annihilation holds at every `n0`, Section 1.2);
- for a wrong `h`, let `v <= N` be the order of `h(z, w(z)) ∈ Q[k][[z]]` and `c(k)` its
  lowest-order non-zero `z`-coefficient, a non-zero polynomial in `k`. Substituting
  `k = n0` commutes with forming `h(z, w(z))`: `h ∈ Q[k][z, w]` and
  `w_{n0} = w(z)|_{k=n0}`. So the coefficients of `h(·, ·, n0)(z, w_{n0}(z))` below `v`
  are 0, and the one at `v` is `c(n0)`. **A wrong `h` passes at `n0` only if
  `c(n0) = 0`**, which happens for finitely many `n0`.

So if exactly one candidate passes at `n0`, it is `g`. If several pass, retry with the
next `n0` (`n0 = 2, 3, ...`). Termination: all but finitely many `n0` succeed. No
`n0` has to be excluded for degree drops. The bound `N` is a property of the generic
`h` and `g` over `Q(k)`, and evaluating at `n0` is only used through the identity
above, which holds at every `n0`. If `h(·, ·, n0)` drops degree, it may pass or fail;
passing just causes a retry.

Spike: `F = kz + zG^2`, `G = kz + zF^2` (symmetric, so `F = G`). The eliminant is
`(zF^2 − F + kz)(z^2 F^2 + zF + kz^2 + 1)`, `N = 6`, and the first factor is selected.
The resultant gives the same two factors.

### 3.3 "What if `w(z)` satisfies a factor only for some `k`?"

- **The selected factor works for every `k`.** `G(z, w(z), k) = 0` in `Q[k][[z]]`
  (Section 1.2), so there is no `k` where `w_n` leaves `G`. Dividing out the content
  does not break this, because `Q[k][[z]]` is a domain.
- **Another factor can also vanish at a particular `n`**: `h(z, w_n, n) = 0` iff
  `G(·, ·, n)` and `h(·, ·, n)` share a factor, i.e. at roots of the content in `k` of
  `Res_w(G, h)`. That does not affect `G`. Because the selection is made over `Q(k)`
  (or at a fixed `n0` with the rule above), it never picks such a coincidental factor.
- **The minimal polynomial at `n` can be a proper factor of `G(·, ·, n)`.** This happens
  at the candidates of Section 2.3 (recomputed there) and, beyond them, only at the
  finite Noether set of Section 2.5, which is not computed but available per `n`
  through `Curve.specialize`.

## 4. Resultants

**When.** If the closure has two unknowns `(w, F)`, then
`Res_F(P_1, P_2) ∈ Q[k][z, w]` eliminates `F` in one step, with no Gröbner basis. That
covers Hitchhiker's Guide `A_k = 2z + B_k`, Broda `S_k` (eliminate `R`), and the
symmetric example. With three or more unknowns, iterated resultants multiply degrees and
pile up extraneous factors, so `"auto"` uses Gröbner bases. One unknown needs no
elimination: `g0 = P_1` (Catalan, `R_k`).

**Extraneous factors.** `Res_F` also vanishes where both leading coefficients in `F`
vanish, and on the projections of other components. Example (spike):
`Res_R = −z · G_S` for `S_k`. The factor `z` has no `w` and is dropped (Section 3.1).
Factors that contain `w` go through the same test (Section 3.2).

**Specialisation.** The resultant is the Sylvester determinant, a polynomial in the
coefficients. It commutes with `k ↦ n` when neither `deg_F P_1` nor `deg_F P_2` drops,
i.e. outside the integer roots of the `k`-contents of `lc_F P_1` and `lc_F P_2`. Those
roots play the role of `X_GB`. `X_deg`, `X_sqf`, `X_cont` and the recompute-and-merge
rule are the same as in Section 2.

Spike timings: all two-unknown cases take about 1 ms by either method.

## 5. Backends

Only the Gröbner step is expensive, so only it goes behind an interface. Factorisation,
resultants, series and `KSet` stay as they are. Inputs and outputs are SymPy `Poly`s
over `QQ`, so results do not depend on the backend.

```python
class _GroebnerBackend(Protocol):
    name: str

    def groebner_lex(
        self, polys: Sequence[sympy.Poly], gens: Sequence[sympy.Symbol]
    ) -> list[sympy.Poly]: ...  # reduced lex basis, gens in the given order
```

- `_SympyBackend`: `sympy.groebner(..., order="lex", method="f5b")`. It uses flint
  integers when python-flint is installed (SymPy ground types).
- Later, not in chunk 2: `_SingularBackend` (the `Singular` binary through a text
  protocol, `std`/`slimgb` in `ordering lp`), and `_SageBackend` (when running inside
  Sage). Sage is installed on the maintainer's machine, which is useful for
  cross-checks. python-flint 0.9 has multivariate `factor` and `resultant`
  (`fmpz_mpoly`) but no Gröbner bases.
- **Chunk 2 implements only the SymPy backend**, and the interface stays private: no
  public `backend=` keyword until a second backend exists (public API small).

## 6. `KCases`

- **It becomes public in chunk 2** (`flanpy.KCases`, in `__all__`), because
  `eliminate_in_k` returns it.
- **API changes from real use here:**
  - **Add `tail`**: the piece that contains all large `k`, as `(KSet, value)`, or
    `None`. Pieces are disjoint and only one can be infinite. This is "the generic
    curve" in examples (`eliminate_in_k(ssnf, L).tail[1]`), and what step 7 takes
    limits on. Same naming as `KSet.tail_start`.
  - No other change. Construction from a dict (`{generic: c, KSet.of(1): c1}`) is
    enough, because the merging is done by `eliminate_in_k` (Section 2.4), not by
    `KCases` (merging by value equality stays out, `kset.md` §4). `map`, `restrict`
    and `combine` are not used by step 3.
  - Expected for step 4, not added now: `flat_map(f: T -> KCases[U])` (a per-piece
    analysis that itself splits on `k`, with each result restricted to its piece).
    Adding it later is backward compatible.
- Docstrings: `map`, `restrict`, `combine`, `pieces`, `covered`, `uncovered` and `tail`
  get Examples sections (EX01 applies once it is public).

## 7. Tests (`tests/test_elimination.py`)

**Quadratic first:**

- Catalan `F = 1 + zF^2`: `G = zF^2 − F + 1`, no elimination, `kset = KSet.all()`.
- Hitchhiker's Guide eqs. (7)–(9): `B = kz + 2zB^2 + 2zB`, `A = 2z + B`.
  `G = 2zA^2 − (8z^2 − 2z + 1)A + 8z^3 − 4z^2 + (k + 2)z`, and
  `disc_A G = Δ_k(z) = (4 − 8k)z^2 − 4z + 1` (eq. (9)) exactly. The root through
  `A(0) = 0` is eq. (8), `A_k = (8z^2 − 2z + 1 − √Δ_k)/(4z)`. Both methods are checked.
- Broda `R_k` (`G = 2zR^2 + (z − 1)R + (k + 1)z`, one unknown) and `S_k` (two unknowns,
  resultant and Gröbner agree; the extraneous `z` is dropped).

**Acceptance (Broda 2020):**

- `eliminate_in_k(ssnf, L)` is a single piece `[1, ∞)` whose polynomial **equals
  eq. (17)** (normalised as in Section 1.3; the quotient is 1, so the unit in `Q(k)` is
  1). The candidate set is `{54}`, merged. `eq. (18)`'s `p_k` is `lc_w G`.
- `cases[n].specialize(n) == eliminate(ssnf, L, k=n)` for `n = 1..60` (covers 54), and
  `series(ssnf, 30, k=n)` annihilates it. (`slow` for a larger range.)
- `F_k = z + B_k + B̄_k` (eq. (38)'s denominator): a cubic, no exceptional `k`
  (`cont_z disc = 16k`).

**Exceptional `k`:** the three examples of Section 2.3 (squarefreeness, degree drop,
content in `z`) each give `[2, ∞)` and `{1}`, with the values stated there. A test
also checks that the generic `G(·, ·, 1)` is **not** the curve at `k = 1` (not
squarefree / lower degree / content `2z − 1`), so the split is necessary. `X_prim` is
tested directly: candidates `{1}` and content `2z − 1` for the counterexample, and empty
for `L_k` and the other examples.

**Factor selection:** the symmetric example (`N = 6`, the right factor, the same with
both methods). A test where the fixed-`n0` test must retry: a constructed factor `h`
with `h(z, w(z))` whose leading coefficient vanishes at `n0 = 2` (built in chunk 2). The
guard check when `r = 1`.

**Saturation fallback:** the `(1 − zG)` example: `E` is empty, saturation gives
`F − z`.

**Specialisation theorem:** for `L_k` at `k = 54`, **the criterion fails** (the 2
dropped elements do not reduce to 0 modulo the 18 kept ones, Kalkbrener Thm 3.1(c)),
**and the curve merges** (the recomputed curve equals the renormalised `G(·, ·, 54)`).
Both are checked in one test of the private helpers.

**Properties (`hypothesis`):** random small non-negative systems (two or three unknowns,
degree `<= 2`, coefficients in `{0, 1, k}`), well-posed by construction (each term has
a factor `z`). Check that `G` annihilates `series` to order 30, is primitive, squarefree
and irreducible, and that `eliminate_in_k(...)[n].specialize(n) ==
eliminate(..., k=n)` for `n = 1..8`. Bounded sizes keep it fast.

**API:** errors (`w` not an unknown, `k` missing or extra, `"resultant"` with three
unknowns, bad `method`); `Curve` value semantics (equality, hash, pickling, immutability,
`repr`); zero-coordinate `w` gives `G = w`.

**Timings** (spike, single runs; flint = python-flint ground types,
Python = `SYMPY_GROUND_TYPES=python`):

| step | flint | Python |
|---|---|---|
| `L_k`, Gröbner f5b, symbolic `k` (20 basis elements) | 0.87 s | 1.43 s |
| `L_k`, Gröbner f5b, `k` fixed (9 elements) | 0.07 s | 0.09 s |
| `L_k`, other order `(M̄, M, B̄, B)` | 7.1 s | 12.0 s |
| `L_k`, saturated by `det` | > 300 s | — |
| factor `g0` (`L_k`) | 7 ms | 6 ms |
| series check (`L_k`, symbolic `k`, 5 terms) | 17 ms | 62–95 ms |
| any two-unknown case (Gröbner or resultant) | ≈ 1 ms | ≈ 1 ms |

The 1.4 s in CLAUDE.md is the pure-Python figure. `eliminate_in_k(ssnf, L)` should total
about 1 s with flint and 1.6 s without, including the recomputation at 54. A timing
test is not added (CI timing is noisy). The examples print the time.

## 8. Docs plan (chunk 2)

- **Docstrings** (NumPy style, doctested Examples, EX01): `eliminate` (Catalan; HG
  `A_k`; cites the elimination theorem and Section 3.2's bound), `eliminate_in_k`
  (`R_k`; the `k = 1` degree-drop example showing two pieces; cites Kalkbrener 1997
  Thm 3.1 and Gianni 1987), `Curve` and its members (`specialize` with the `k = 1`
  example), `KCases` and its members. The package docstring and the `index.md` warning
  list step 3 as implemented. API page entries for `eliminate`, `eliminate_in_k`,
  `Curve` and `KCases`.
- **`docs/refs.bib`** (done): Kalkbrener 1997 (doi:10.1006/jsco.1997.0113), Gianni
  (EUROCAL '87, LNCS 378, published 1989, doi:10.1007/3-540-51517-8_128), Noether
  1922 (Math. Ann. 85(1), 26–40, doi:10.1007/BF01449599; Crossref gives pages 26–40,
  the Hilbert Festschrift version is 26–33) and Kaltofen 1995 (doi:10.1006/jcss.1995.1023),
  each checked against Crossref on 2026-10-30, and Cox–Little–O'Shea *IVA*, 4th ed.
  (doi:10.1007/978-3-319-16721-3).
- **Guide page `docs/guide/elimination.md`** (executed): `eliminate` at a fixed `k` and
  for a system without `k`; `eliminate_in_k` and reading a `KCases` (`pieces`, `tail`,
  `[n]`); an exceptional `k` and why it is split; factor selection on the symmetric
  example; `method=` and the closure; `Curve.specialize` for certifying a range of `k`.
- **`examples/catalan.py`**, "Step 3: the curve": `eliminate(catalan, F)` gives
  `zF^2 − F + 1` (no elimination needed; assert), a factor check against the series.
  Status: step 3 done.
- **`examples/broda2020.py`**, "Step 3: elimination": `eliminate_in_k` for `R_k`
  (quadratic; its discriminant is the radicand of eq. (2), and eq. (2) is a root), `S_k`
  (resultant), and `L_k`: one piece `[1, ∞)`. It asserts equality with eq. (17) and that
  `lc_w` is `p_k` (eq. (18)), prints the time, and spot-checks `k = 1, 2, 10, 50, 54`
  against `eliminate(..., k=n)` and the series. Step 2's section keeps eq. (17) as
  printed (it comes first in the file), and step 3 asserts that the computed curve
  equals it. `F_k` for eq. (38). Status: step 3 done.
- **CLAUDE.md**: "Planned layout" (`_elimination.py` with `eliminate`,
  `eliminate_in_k` and the backend protocol; `_curve.py` with `Curve`; the
  `elimination/` package entry removed), a "Step 3" block under Decisions, the 1.4 s
  figure labelled as pure Python with the flint figure added, and the open item ticked.

## 9. Decisions (approved 2026-10-30)

1. API: `eliminate` / `eliminate_in_k`, and `Curve` with `polynomial`, `w`, `z`, `k`,
   `kset`, `system`, plus `method="auto"`.
2. Irreducibility at non-candidate `k` is a documented gap (Section 2.5), with exact
   per-`n` certification through `Curve.specialize`. Amendment B added `X_prim`, which
   closes the gap for `deg_w G = 1`.
3. Merge rule as in Section 2.4, with renormalisation (amendment E).
4. `KCases` is public, with `tail`.
5. The saturation fallback is implemented.

Amendments folded in: A (Section 2.2: the criterion fails at 54, and candidates are
decided by recomputation), B (`X_prim`, Section 2.3), C (what the invariants rest on,
Section 1.4), D (absolute irreducibility over the algebraic closure of `Q(k)`,
Section 1.2), E (invariants up to a unit; renormalise, Section 1.4), F (the coefficient
`c(k)`; no `n0` exclusion, Section 3.2).
