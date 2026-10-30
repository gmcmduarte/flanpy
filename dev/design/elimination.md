# Design note: `elimination`

Status: draft for approval (chunk 1), 2026-10-30. No code in `src/` yet.

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

- `G` is irreducible over `Q(k)` and in fact **absolutely irreducible** (over
  `Q̄(k)(z)`). Proof: over `Q̄(k)(z)`, `w(z)` is a root of one irreducible factor `h` of
  `G`. For `σ ∈ Gal(Q̄/Q)` acting on coefficients, `σ(h)` has the root `σ(w) = w`. `G`
  is squarefree, so `σ(h) = h` up to a unit, for every `σ`. Hence `h` has coefficients
  in `Q(k)(z)` and `h = G` up to a unit. Steps 4–5 rely on this (one curve, no
  conjugate components).
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

**Invariants** for every `n` in `kset` (and for the only value when `k` is `None`):
`G(z, w_n, n) = 0`; `G(·, ·, n)` is primitive and squarefree, and its degree in `w` is
`deg_w G` (`lc_w G(·, n) ≢ 0`). For symbolic `k`, `G` is irreducible over `Q(k)`. For a
fixed `k` computed by `eliminate`, `G` is irreducible over `Q`, i.e. it is the minimal
polynomial of `w_n`.

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
   Thm 3.2 gives no stability for free, and a vanishing `lc` outside `E` can change the
   specialised elimination ideal. Example: for `L_k`, `E` has the single `lc`
   `4k^2 (2k + 27)`, but two other basis elements have `lc` `168 k (k − 54)` and
   `6 (k − 54)(k + 2)`. So `k = 54` is a **candidate**. At `k = 54`, the criterion holds
   (the two dropped elements reduce to 0, spike), and the recomputed curve equals
   `G(·, ·, 54)`. `54` therefore ends up in the generic piece (Section 2.4).
3. **Candidates from the basis**: `X_GB = {n >= 1 integer : lc(g)(n) = 0 for some
   g ∈ 𝒢}`, a finite set: each `lc(g)` is a non-zero polynomial in `k`.

### 2.3 Candidates from the curve

The theorem is about ideals. Whether `G(·, ·, n)` still has the invariants of
Section 1.4 is a separate question, decided on `G` itself:

- `X_deg`: integer roots `>= 1` of `cont_z(lc_w G)` (the content in `k` of
  `lc_w G ∈ Z[k][z]`): there the degree in `w` drops;
- `X_sqf`: integer roots `>= 1` of `cont_z(disc_w G)`: there `G(·, ·, n)` is not
  squarefree. If `deg_w G = 1`, this set is empty;
- `X_cont`: integer roots of the content in `k` of the unnormalised factor of `g0`
  (Section 1.3 divides it out). It is contained in `X_GB` (it divides `lc(g0)`), and
  is listed only so that the resultant path (Section 4) has the same rule.

The candidates are `X = X_GB ∪ X_deg ∪ X_sqf ∪ X_cont`, computed exactly
(`Poly.ground_roots` / `KSet.where(..., "==")`).

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
  then `n` joins the generic piece;
- **otherwise** `n` gets its own piece `{n}` with value `C_n`.

The result is `KCases({generic: Curve(G, kset=generic), {n}: C_n, ...})`, with
`generic = KSet.all() − {separate n}`. It covers every `k >= 1`: each recomputation
succeeds (`specialize` never fails on a well-posed system, `system.md` §5.7, and
elimination at a fixed `k` always gives a curve). Nothing is dropped.

Outcomes in the spike: `L_k`, `R_k`, `S_k` and Hitchhiker's Guide `A_k` give one piece
`[1, ∞)` (`L_k`: candidate 54, merged). The two examples of Section 2.3 give
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

Proposal: say this in the docstrings, and make `Curve.specialize(n)` exact. It
substitutes `k = n`, factors over `Q`, selects the factor with the series at `n`
(Section 3), and returns a curve whose polynomial is the minimal polynomial of `w_n`.
That costs milliseconds for one `n`, so certification is available for any finite
range on demand. Tests and examples spot-check `k = 1..N` this way (CLAUDE.md, Verify).
See Question 2.

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
- a wrong `h` passes at `n0` only if `n0` is a root of the (non-zero) leading
  coefficient in `k` of `h(z, w(z))`. There are finitely many such roots.

So if exactly one candidate passes at `n0`, it is `g`. If several pass, retry with the
next `n0`. Use `n0 = 2, 3, ...`, but not values where some `h_i(·, ·, n0)` drops degree
in `w` or `z`, because the bound uses the generic degrees. Termination: all but
finitely many `n0` succeed.

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

**Exceptional `k`:** the two examples of Section 2.3 give `[2, ∞)` and `{1}`, with the
values stated there. A test also checks that the generic `G(·, ·, 1)` is **not** the
curve at `k = 1` (not squarefree / lower degree), so the split is necessary.

**Factor selection:** the symmetric example (`N = 6`, the right factor, the same with
both methods). A test where the fixed-`n0` test must retry: a constructed factor `h`
with `h(z, w(z))` whose leading coefficient vanishes at `n0 = 2` (built in chunk 2). The
guard check when `r = 1`.

**Saturation fallback:** the `(1 − zG)` example: `E` is empty, saturation gives
`F − z`.

**Specialisation theorem:** for `L_k` at `k = 54`, the dropped basis elements reduce to
0 (Kalkbrener Thm 3.1(c)), checked directly in a test of the private helper.

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
- **`docs/refs.bib`**: Kalkbrener 1997 (doi:10.1006/jsco.1997.0113, from the PDF
  footer), Gianni 1987 (LNCS 378), Noether 1922 and Kaltofen 1995, each checked against
  Crossref; Cox–Little–O'Shea *IVA* (4th ed.).
- **Guide page `docs/guide/elimination.md`** (executed): `eliminate` at a fixed `k` and
  for a system without `k`; `eliminate_in_k` and reading a `KCases` (`pieces`, `tail`,
  `[n]`); an exceptional `k` and why it is split; factor selection on the symmetric
  example; `method=` and the closure; `Curve.specialize` for certifying a range of `k`.
- **`examples/catalan.py`**, "Step 3: the curve": `eliminate(catalan, F)` gives
  `zF^2 − F + 1` (no elimination needed; assert), a factor check against the series.
  Status: step 3 done.
- **`examples/broda2020.py`**, "Step 3: elimination": `eliminate_in_k` for `R_k`
  (quadratic), `S_k` (resultant, extraneous `z`), and `L_k`: one piece `[1, ∞)`, assert
  equality with eq. (17), print the time, spot-check `k = 1..60` against `eliminate(...,
  k=n)`. Step 2's section then takes `G` from `eliminate_in_k` instead of typing
  eq. (17) by hand. `F_k` for eq. (38). Status: step 3 done.
- **CLAUDE.md**: "Planned layout" (`_elimination.py` with `eliminate`,
  `eliminate_in_k` and the backend protocol; `_curve.py` with `Curve`; the
  `elimination/` package entry removed), a "Step 3" block under Decisions, the 1.4 s
  figure labelled as pure Python with the flint figure added, and the open item ticked.

## 9. Questions for approval

1. **API names and shape**: `eliminate` / `eliminate_in_k` (mirroring `series` /
   `series_in_k`), `Curve` with `polynomial`, `w`, `z`, `k`, `kset`, `system`, and
   `method="auto"`. OK?
2. **Irreducibility at non-candidate `k`** (Section 2.5): document that `G(·, ·, n)` is
   certified squarefree, of full degree and annihilating, but irreducible only
   generically. Per-`n` certification is available through `Curve.specialize(n)`, and
   tests and examples spot-check ranges. The alternative, a bounded certification at
   construction (`eliminate_in_k(..., certify=range(1, N))`), would add API for little
   gain. Accept the documented gap?
3. **Merge rule** (Section 2.4): candidates whose recomputed curve equals the
   specialised generic one are folded into the generic piece, so `L_k` is one piece
   `[1, ∞)`, not `[1, 53] ∪ [55, ∞)` plus `{54}`. A degree drop always stays separate.
   OK?
4. **`KCases` public, plus `tail`** (Section 6). OK?
5. **Saturation fallback** implemented in chunk 2 (small, and it has a test case),
   rather than raising an error when the elimination ideal is empty. OK?
