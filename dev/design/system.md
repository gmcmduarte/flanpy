# Design note: `System`

Status: draft for approval (build-order step 1, chunk 1). 2026-10-29.

## 1. Purpose

`System` is the input type for the whole package. It holds the polynomial system that the
generating functions satisfy, in the form the symbolic method produces:

```
F_i = Phi_i(z, k, F_1, ..., F_m),    i = 1..m,
```

with unknowns `F_1..F_m` (formal power series in `z`), the size variable `z`, and the
parameter `k` (alphabet size, a non-negative integer). Each `Phi_i` is a polynomial. Every later stage
consumes a `System`: `series` computes coefficients from it, `elimination` turns it into
one curve `G(z, w) = 0`, and the asymptotic stages work from that curve.

`System` checks at construction that the system defines its solution (Section 5). It
does no other computation.

## 2. Sources checked

- Broda, Machiavelo, Moreira, Reis, *Guest Column: Analytic Combinatorics and
  Descriptional Complexity of Regular Languages on Average*, SIGACT News (2020).
  The grammars and equations below were read from the paper text and the rendered pages
  (the overbars in grammar (9) are lost in text extraction, so they were checked on the
  page image).
- Flajolet & Sedgewick, *Analytic Combinatorics* (CUP 2009). Page numbers are from the
  printed book: Def. I.7 (p. 33), the semantics of recursion by iteration from the empty
  class (pp. 32–33), Def. V.8 (p. 357), eq. (61) (p. 482), Def. VII.5 / Thm VII.5
  (p. 483), §VII.6.3 a-properness (p. 489), eq. (77) (p. 494).

Transcriptions were checked in a throwaway script (not committed). For `k = 1, 2` and
`n <= 7`, brute-force enumeration of the objects described in Section 3 gives the same
counts as iterating the transcribed systems: `R`, `S`, `B`, `B̄`, `M`, `M̄` and `L` all
match. The series `L_k(z)` from the transcribed `L_k` system satisfies eq. (17) modulo
`z^10`, with `k` symbolic.

## 3. The Broda 2020 systems

The size of an expression is its number of symbols, **excluding parentheses**: each
letter, each `ε`, and each operator `+`, `·`, `*`, `?` counts 1. `Σ = {σ_1..σ_k}`, with
`k = |Σ|`. No expression for `∅` is included.

### 3.1 `R_k`: all regular expressions (§2, eqs. (1)–(3))

Grammar (1):

```
r := ε | σ ∈ Σ | (r + r) | (r · r) | r*
```

System (unnumbered, p. 4, from (1)):

```
R = (k + 1) z + 2 z R^2 + z R
```

Objects: syntax trees with leaves `ε` or a letter (`k + 1` choices), binary nodes `+`
and `·`, and unary nodes `*`. `[z^n] R_k` is the number of trees with `n` nodes.
Eq. (2) gives the closed form and eq. (3) the exact coefficients.
Check values for `k = 1`, `n = 1..7`: `2, 2, 10, 26, 114, 402, 1722`.

### 3.2 `S_k`: total number of stars (§2, eqs. (4)–(5))

The bivariate series `S̄(u, z) = Σ_r u^{s(r)} z^{|r|}`, where `s(r)` is the number of
`*` in `r`, satisfies (unnumbered, p. 5):

```
S̄(u, z) = (k + 1) z + 2 z S̄(u, z)^2 + u z S̄(u, z)
```

Differentiating in `u` at `u = 1` (where `S̄(1, z) = R_k(z)`) gives the system for
`S_k = ∂_u S̄ |_{u=1}`, unnumbered on p. 6 and solved explicitly in eq. (4):

```
R = (k + 1) z + 2 z R^2 + z R
S = 4 z R S + z R + z S
```

Objects: the same trees as `R_k`. `[z^n] S_k` is the total number of `*` nodes summed
over all trees with `n` nodes. Check values for `k = 1`: `0, 2, 4, 30, 104, 570, 2412`.
Eq. (8): `[z^n] S_k / (n [z^n] R_k) → ρ_k`.

### 3.3 `L_k`: letters in expressions in strong star normal form (§3.2, eqs. (9)–(18))

`L_k` is **not in section 2**. The paper defines it in §3.2 (pp. 8–9), with the full
system given there.

Grammar (9). `s_ε` is an expression whose language contains `ε`; `s_ε̄` is one whose
language does not:

```
r   := ε | s_ε | s_ε̄
s_ε := s_ε · s_ε | s_ε + s_ε̄ | s_ε̄ + s_ε | s_ε + s_ε | s_ε̄* | s_ε̄?
s_ε̄ := σ_1 | ... | σ_k | s_ε̄ · s_ε̄ | s_ε̄ · s_ε | s_ε · s_ε̄ | s_ε̄ + s_ε̄
```

Generating functions `B_k` (for `s_ε`) and `B̄_k` (for `s_ε̄`), eqs. (10)–(11):

```
B  = 2 z B^2 + 2 z B B̄ + 2 z B̄                 (10)
B̄  = k z + 2 z B B̄ + 2 z B̄^2                   (11)
```

Marking letters with `u` gives eqs. (13)–(14), the same as (10)–(11) except that the
term `k z` becomes `k u z`. Then `M_k = ∂_u B_k |_{u=1}` and
`M̄_k = ∂_u B̄_k |_{u=1}` satisfy eqs. (15)–(16), and eq. (12) defines `L_k`:

```
M  = 4 z B M + 2 z M B̄ + 2 z B M̄ + 2 z M̄       (15)
M̄  = k z + 2 z M B̄ + 2 z B M̄ + 4 z B̄ M̄         (16)
L  = M + M̄                                       (12)
```

The `L_k` system is (10), (11), (15), (16), (12), with unknowns
`(B, B̄, M, M̄, L)`. Eliminating the other unknowns gives eq. (17),
`p_k(z) w^3 + k z ((k + 12) k z^2 + 4 k z + 1) w − k^2 z^2 (2z + 1) = 0`, with `p_k` from (18).
The denominator series for eq. (38) is `F_k = z + B_k + B̄_k` (unnumbered, p. 15).

Objects, in a form a brute-force enumerator can use. The grammar generates exactly the
expression `ε` together with all syntax trees that satisfy these conditions:

- leaves are letters (no `ε` leaf anywhere inside a larger expression);
- binary nodes are `+` and `·`; unary nodes are `*` and `?`;
- every `*` or `?` node has a child that is **not** nullable.

Nullability is defined by: a letter is not nullable; `s*` and `s?` are nullable;
`s · t` is nullable iff both `s` and `t` are; `s + t` is nullable iff `s` or `t` is.
Then `B_k` counts the nullable trees and `B̄_k` the non-nullable ones. `M_k` and `M̄_k`
sum the number of letter leaves over those trees, and `L_k` sums it over all of them.
The expression `ε` contributes `z` to `F_k` and nothing to `L_k`.
Check values (from the system, symbolic `k`): `[z^1..z^5] L_k = k, 2k, 4k^2, 24k^2, 8k^2(3k + 4)`.

## 4. Mathematical form

- **Fixed-point form (primary).** `F = Phi(z, k, F)`, with
  `Phi ∈ Q[k][z, F_1..F_m]`: polynomial in everything, rational coefficients, and **no
  denominators in `k`**. This is what the user writes and what is validated.
- **Polynomial form (for elimination).** `P_i = F_i − Phi_i ∈ Q[F_1..F_m, z, k]`, with
  `P_i(z, k, F) = 0`. This is F&S eq. (77), p. 494, specialised to the fixed-point case.
  The generators are ordered `(F_1..F_m, z, k)`, with `k` last as fixed in CLAUDE.md.
  `elimination` reorders the unknowns as it needs.

Restricting to `Q[k]` makes the validity of the system independent of `k` (Section 5.4). It
also keeps `series` coefficients in `Q[k]` (Section 9). Rational functions of `k` can be
added once `kset` exists, because they bring exceptional values of `k`. Combinatorial
inputs do not need them: `k` enters as a multiplicity (`k z`).

Constructions that are not polynomial, such as `SEQ` (`1/(1 − A)`), must be written with an
auxiliary unknown (e.g. `S = 1 + A S`). The future `symbolic` module will do this
automatically.

## 5. Validation: well-foundedness

### 5.1 What F&S give

- **Semantics** (pp. 32–33, Def. I.7): a recursive specification means the limit of the
  iteration `y^(0) = 0`, `y^(h+1) = Phi(z, y^(h))`.
- **Linear systems** (Def. V.8, p. 357): a transfer matrix `T(z)` is *proper* if `T(0)`
  is nilpotent.
- **Polynomial systems** (§VII.6.3, p. 489, condition (ii) "a-properness"): `Phi` is a
  contraction for the metric `d(u, v) = 2^(−val(u − v))`. Then the system has a unique
  solution in `C[[z]]^m`, which is the limit of the iteration above. For polynomial `Phi`
  this holds **iff `Phi(0, y)` does not depend on `y`** (our derivation: if a monomial
  with no `z` contains some `y_j`, constant vectors `y, y'` give `val(Phi(y) − Phi(y')) = 0`).

Strict a-properness is too strong for us. It rejects two systems we must accept:

- `F = z + F^2` (binary trees counted by leaves). Here `Phi(0, F) = F^2`. The system has two
  power-series solutions, `C(z)` and `1 − C(z)`, so "unique power-series solution" is
  the wrong requirement. The iteration from `0` picks out `C(z)`.
- Broda's `L = M + M̄` (eq. (12)). The `L` equation has no factor `z`.

### 5.2 The condition `System` enforces

Let `m` be the number of unknowns.

1. **Constant terms stabilise.** Iterate `c^(0) = 0`, `c^(h+1) = Phi(0, k, c^(h))`, exactly
   in `Q[k]^m`, until `c^(h+1) = c^(h) =: c*`.
2. **The Jacobian at the origin is nilpotent.** `J0 = ∂Phi/∂F (0, k, c*)` satisfies
   `J0^m = 0`, as an identity in `Q[k]`.

**Claim** (our argument, not a numbered result in F&S): under 1–2, the iteration from 0
converges in `C[[z]]^m`, and its limit `y` is the unique solution with `y(0) = c*`.

*Proof sketch.* Uniqueness: `I − J0` is invertible (unipotent), so the formal implicit
function theorem applies at `(0, c*)`. Convergence: let `e^(h) = y^(h) − y` and let `ν`
be the nilpotency index of `J0`. After step 1 is reached, `e^(h)` has zero constant
term. The coefficient of `z^n` in `e^(h+1)` is `J0` times that of `e^(h)`, plus terms that
involve only coefficients of `e^(h)` of index `< n`. By induction on `n`, coefficient `n` is
exactly 0 after at most `h_1 + n ν` steps, where `h_1` is the step at which 1 is reached.

This generalises the F&S cases. Def. V.8 is the linear case `Phi = T(z) F + b(z)`, where
`J0 = T(0)`. Strict a-properness is the case `J0 = 0` with `c* = Phi(0, 0)`.

The check is conservative. It rejects some degenerate systems whose iteration happens to
converge, e.g. `{F = z + 2G, G = 2G}`, which has `J0` not nilpotent and `G ≡ 0`. For
a-positive systems we expect it to be necessary as well, but that is **not proved here**.
Pivoteau–Salvy–Soria (JCTA 2012, "Algorithms for combinatorial structures: Well-founded
systems and Newton iterations") treat this question. That reference is from memory and
was **not checked**. Do not cite it until verified.

### 5.3 The bound on step 1

`m` iterations are **not** enough in general. `F = z + 1 − (F − 1)^2 (F − 2)/2` has
`m = 1`, but its constant terms go `0 → 2 → 1 → 1`, so they stabilise only after 2 steps,
and `J0 = 0`. For that reason step 1 iterates up to a private cap, initially `max(32, 4m)`.
If the cap is reached, construction fails with an **undecided** message. It does not say
"ill-posed". The cap is not public API; see Open questions.

### 5.4 Role of `k`

Both checks are polynomial identities in `Q[k]`. Identities survive substitution, so a
system that passes for symbolic `k` passes for **every** `k ∈ Z≥0`. No `KSet` is needed
here. If check 2 fails for symbolic `k`, the error message lists the non-negative
integers `k` for which `J0` is nilpotent. These are the common non-negative integer roots of
the coefficients of `det(λI − J0) − λ^m`. The user can substitute one of those values and
build the system again.

### 5.5 Error messages

All of these raise `flanpy.IllPosedSystemError` (a subclass of `ValueError`).

`F = F + z`:

```
flanpy.IllPosedSystemError: the system is not well-founded: the Jacobian dPhi/dF at z = 0 is not nilpotent.
    F = F + z
  At z = 0: F(0) = 0, Jacobian = [[1]].
  The dependency cycle F -> F has no factor z, so iterating F <- Phi(F) from 0 never
  fixes the coefficients of the solution.
```

A cycle can always be shown, because a matrix whose support graph is acyclic is
nilpotent.

`F = 1 + F` (undecided case):

```
flanpy.IllPosedSystemError: could not establish that the system is well-founded: the constant
terms F(0) did not stabilise under F(0) <- Phi(0, F(0)) within 32 iterations from 0.
    F = F + 1
  Last two iterates: F(0) = 31, then F(0) = 32.
```

`F = k F + z`:

```
flanpy.IllPosedSystemError: the system is not well-founded for symbolic k: the Jacobian dPhi/dF at z = 0 is not nilpotent.
    F = F*k + z
  At z = 0: F(0) = 0, Jacobian = [[k]].
  The Jacobian is nilpotent only for k in {0}; substitute such a value before building the System.
```

Malformed input raises `ValueError` or `TypeError`, not `IllPosedSystemError`. The
messages name the offending equation, e.g.:

```
ValueError: the right-hand side for F is not a polynomial in (F, z, k) with rational
coefficients: 1/(-F*z + 1). Write SEQ-like terms with an auxiliary unknown, e.g. S = 1 + z*F*S.
```

## 6. Public API

Public names: `flanpy.System` and `flanpy.IllPosedSystemError`. The implementation lives in
a private module `src/flanpy/_system.py`, re-exported from `flanpy/__init__.py`. That keeps
the import path stable if the internals move.

```python
class System:
    def __init__(
        self,
        equations: Mapping[sympy.Symbol, sympy.Expr | int],
        *,
        z: sympy.Symbol,
        k: sympy.Symbol | None = None,
    ) -> None: ...

    @property
    def unknowns(self) -> tuple[sympy.Symbol, ...]: ...        # mapping order
    @property
    def equations(self) -> Mapping[sympy.Symbol, sympy.Expr]: ...  # read-only, F_i -> Phi_i (expanded)
    @property
    def z(self) -> sympy.Symbol: ...
    @property
    def k(self) -> sympy.Symbol | None: ...

    def polynomials(self) -> tuple[sympy.Poly, ...]: ...      # F_i - Phi_i over QQ, gens (*unknowns, z, k)
    def specialize(self, k: int) -> System: ...                # k >= 0; result has k = None

    # __eq__, __hash__, __repr__ (content-based; System is immutable)
```

Example (doctest in the class docstring):

```python
>>> from sympy import symbols
>>> from flanpy import System
>>> z, k, R = symbols("z k R")
>>> System({R: (k + 1)*z + 2*z*R**2 + z*R}, z=z, k=k)   # Broda et al. 2020, from grammar (1)
System({R: 2*R**2*z + R*z + k*z + z}, z=z, k=k)
```

Input rules:

- `unknowns`, `z` and `k` are distinct `sympy.Symbol`s. Symbols with different assumptions
  (`Symbol("z")` and `Symbol("z", positive=True)`) are different symbols in SymPy, and the
  docstring will warn about this.
- `Phi_i` uses only the unknowns, `z` and `k`. Any other symbol is rejected with a message
  saying that marking variables are not supported yet.
- Coefficients are exact rationals. Floats are rejected (the "exact first" rule).
- At least one equation. `k=None` is allowed, e.g. for Catalan. A `k` that is given but
  unused is fine.
- `specialize(k)` requires an `int >= 0` and a system with `k` set. It returns a validated
  `System` without `k`.

## 7. Private, computed at construction and cached

- `_constant_terms: tuple[Expr, ...]` (`c*`), `_jacobian0: Matrix` (`J0`),
  `_nilpotency_index: int` (`ν`).
- `_strictly_proper: bool` (F&S a-proper, `Phi(0, y)` free of `y`). This is informative
  only.
- Internally the parameters are kept as a tuple `_params` (currently `(k,)` or `()`). The
  validation code treats every parameter the same way, so marking variables can join that
  tuple later without changing it (Section 8).

## 8. Room for marking variables (later; not designed here)

Cost generating functions need extra variables `u` that appear polynomially in `Phi`,
e.g. eq. (14) has `k u z`. They will be accepted through a keyword such as `markers=(u,)`,
stored in `_params`, and validated identically in them, exactly like `k`. An operation
then derives from a `System` with marker `u` the augmented system for
`(F, ∂_u F |_{u=1})`, as in eqs. (13)–(14) → (15)–(16). Nothing in Sections 4–7 depends
on there being only one parameter.

## 9. What `series` will need from `System`

`series` computes coefficients `[z^0..z^N]` of every unknown, either for symbolic `k` or
after `specialize(k)`. It needs:

- the ordered unknowns and `Phi` as sparse polynomials;
- `c*` and `J0`.

With those, coefficients can be computed one at a time by a linear solve,
`(I − J0) y_n = r_n(y_0..y_{n−1})`, or by Newton iteration. Because `I − J0` is unipotent
over `Q[k]`, its inverse has entries in `Q[k]`, so all coefficients stay in `Q[k]` (no
denominators in `k`). A naive fixed-point iteration also works, with at most `ν` passes
per new coefficient (Section 5.2). `series` gets its own design note in chunk 3.

## 10. Planned tests (`tests/test_system.py`)

Well-posed. Each test checks construction, `unknowns`, `equations`, `polynomials()` (with
the generators and domain) and the private `c*`, `J0` and strictness flag:

| system | strictly a-proper | `c*` |
|---|---|---|
| linear `F = 1 + 2 z F` | yes | `1` |
| Catalan `F = 1 + z F^2` | yes | `1` |
| Motzkin `F = 1 + z F + z F^2` | yes | `1` |
| leaves `F = z + F^2` | no | `0` |
| two equations `F = 1 + z G`, `G = 1 + z F^2` | yes | `(1, 1)` |
| Broda `R_k` (Section 3.1) | yes | `0` |
| Broda `R_k, S_k` (Section 3.2) | yes | `(0, 0)` |
| Broda `L_k` (Section 3.3, five unknowns) | no (`L = M + M̄`) | `0` |
| `F = z + 1 − (F−1)^2 (F−2)/2` (2-step stabilisation) | no | `1` |

Also:

- `specialize(k)` for `k = 0, 1, 2` on the Broda systems. The result has no `k` and
  `polynomials()` changes accordingly.
- Equality and hashing are by content; the object is immutable.
- A doctest in the docstring (shown in Section 6).

Ill-posed: each must raise `IllPosedSystemError`, and the test matches key phrases of the
message.

- `F = F + z`, `F = F` (Jacobian `[[1]]`, cycle `F -> F`)
- `F = z + F + F^2` (Jacobian `[[1]]`)
- `{F = G + z, G = F}` (cycle `F -> G -> F`)
- `F = 1 + F`, `F = 1 + F^2` (undecided: constant terms do not stabilise)
- `F = k F + z` (nilpotent only for `k in {0}`)

Malformed input (must raise `ValueError` / `TypeError`):

- non-polynomial right-hand sides: `1/(1 − z F)`, `sqrt(z)`, `exp(z)`;
- a float coefficient (`0.5*z`); a stray symbol (`u`, with the markers message);
- an unknown equal to `z` or `k`; an empty mapping; a non-`Symbol` key; `k` not a `Symbol`;
- `specialize(-1)`, `specialize(2.0)`, and `specialize` on a system without `k`.

Out of scope here: `series` coefficients and brute-force enumeration. Those tests belong to
`series` (chunk 3 onwards), which will use the objects described in Section 3 and the check
values given there.

## 11. Non-goals and open questions

- No construction from implicit polynomial equations (`P(z, F) = 0` not in fixed-point
  form). Well-foundedness cannot be checked there. Revisit if a paper needs it.
- No `Q(k)` coefficients until `kset` exists (Section 4).
- The private iteration cap for step 1 (Section 5.3): is a heuristic cap with an
  "undecided" error acceptable, or should the cap be public?
- Should the `F&S`-strict flag (`_strictly_proper`) become public later? Current proposal:
  no.
