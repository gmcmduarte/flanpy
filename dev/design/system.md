# Design note: `System`

Status: approved with changes (chunk 1); updated after reading Pivoteau–Salvy–Soria and
implemented in `src/flanpy/_system.py` (chunk 2). 2026-10-29.

## 1. Purpose

`System` is the input type for the whole package. It holds the polynomial system that the
generating functions satisfy, in the form the symbolic method produces:

```
F_i = Phi_i(z, k, F_1, ..., F_m),    i = 1..m,
```

with unknowns `F_1..F_m` (formal power series in `z`), the size variable `z`, and the
parameter `k` (alphabet size; domain: the integers `k >= 1`). Each `Phi_i` is a polynomial. Every later stage
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
- Pivoteau, Salvy, Soria, *Algorithms for combinatorial structures: Well-founded systems
  and Newton iterations*, J. Combin. Theory Ser. A 119(8) (2012) 1711–1773,
  doi:10.1016/j.jcta.2012.05.007 (checked on Crossref). **Cited throughout (here, in
  docstrings, messages and `docs/refs.bib`) with the numbering and pages of
  arXiv:1109.2688v2**, which is the text that was read; the journal pagination is not used.
  Used: Def. 3.5, Lemma 3.4, Algorithm 0-coord (p. 14), Thm 3.6 and Example 4 (p. 15),
  Prop. 4.2 (p. 16), Prop. 4.5 (p. 18), Def. 5.3, Def. 5.4, Thm 5.5 (p. 21), Algorithm
  isWellFounded and Example 8 (p. 22), Thm 5.7 (p. 23).

Transcriptions were checked with `dev/spikes/broda2020_enumerate.py`. For `k = 1, 2` and
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

Restricting to `Q[k]` makes the validity of the system independent of `k` (Section 5.7). It
also keeps `series` coefficients in `Q[k]` (Section 9). Rational functions of `k` can be
added once `kset` exists, because they bring exceptional values of `k`. Combinatorial
inputs do not need them: `k` enters as a multiplicity (`k z`).

Constructions that are not polynomial, such as `SEQ` (`1/(1 − A)`), must be written with an
auxiliary unknown (e.g. `S = 1 + A S`). The future `symbolic` module will do this
automatically.

## 5. Validation: well-posedness

### 5.1 Definitions

- **Well-posed** (the project's definition): the iteration `F^(0) = 0`,
  `F^(n+1) = Phi(z, k, F^(n))` converges in the `z`-adic topology of `Q[k][[z]]^m`. The
  limit is the solution, and it is what `series` computes. This is F&S's semantics of
  recursion (pp. 32–33, Def. I.7).
- **Domain of `k`**: the integers `k >= 1` (alphabet sizes). `specialize` accepts only
  these values.
- **Non-negative system**: all coefficients of all `Phi_i` are `>= 0` for every `k` in
  the domain. Test (sufficient, no sampling): substitute `k = 1 + t` and require every
  coefficient, as a polynomial in `(F, z, t)`, to be `>= 0`. Example:
  `F = z + (k − 1) z F^2` is non-negative. Otherwise the system is **signed**.
  All Broda 2020 systems in Section 3 are non-negative.
- **Zero coordinate**: an unknown whose iterates are all 0.
- `System` raises `IllPosedSystemError` **only when ill-posedness is proven**. Anything
  else it cannot settle raises `UndecidedSystemError` (Section 6).

### 5.2 What the literature gives

- F&S Def. V.8 (p. 357, linear systems: `T(0)` nilpotent) and §VII.6.3 a-properness
  (p. 489). A-properness amounts to "`Phi(0, F)` does not depend on `F`". That is too strict:
  it rejects `F = z + F^2` and Broda's `L = M + M̄` (eq. (12)). Note also that "unique
  power-series solution" is the wrong requirement: `F = z + F^2` has two such solutions.
- **PSS** work with species, i.e. non-negative systems. Def. 5.3 (p. 21): a system is
  *well founded* if the iteration is well defined (automatic for polynomials), converges,
  and its limit has **no zero coordinate**. Two characterisations:
  - Thm 3.6 (p. 15), for `H(0, 0) = 0`: well founded at 0 iff `∂H/∂Y(0, 0)` is nilpotent
    and `Y^[m]` has no zero coordinate.
  - Thm 5.5 (p. 21), in general: through the companion system
    `K = H − H(0, 0) + Z1 H(0, 0)` (Def. 5.4), which must be well founded at 0 and have a
    solution polynomial in `Z1` (Prop. 4.5, p. 18).

  Thm 5.7 (p. 23) gives the constant terms: `S(0) = H^m(0, 0)`, the `m`-th iterate.
  That is an **exact bound of `m` steps**. PSS say nothing about signed systems.

### 5.3 The check `System` runs (with `k` symbolic)

Let `m` be the number of unknowns.

0. **Zero coordinates**, by support propagation (PSS Lemma 3.4 and Algorithm 0-coord,
   p. 14). An unknown is marked non-zero once some monomial of its right-hand side has
   all its unknowns marked. An unknown that is never marked is 0 in every iterate, for
   any signs: each monomial of its right-hand side contains an unmarked unknown. For
   non-negative systems the converse also holds for every `k >= 2`: there is no
   cancellation, and a non-zero polynomial with non-negative coefficients in
   `t = k − 1` is positive for `t > 0`. It can vanish at `k = 1`, so specialising to
   `k = 1` may create zero coordinates (Section 5.6).
   Zero coordinates are set to 0 and dropped. This leaves a reduced system with
   `m' <= m` unknowns, and a `ZeroCoordinateWarning` names the dropped ones (Section 5.6).
1. **Constant terms.** Iterate `c^(0) = 0`, `c^(h+1) = Phi(0, k, c^(h))` on the reduced
   system. Note that `[z^0] F^(h) = c^(h)`.
   - Non-negative: compute `c^(m')` and `c^(m'+1)`. If they differ, raise
     `IllPosedSystemError` (PSS Thm 5.7). Otherwise `c* = c^(m')`.
   - Signed: iterate until stationary (`c*`). If an iterate repeats an earlier one, the
     sequence cycles, `[z^0]` never settles, and the system is **proven** ill-posed:
     raise `IllPosedSystemError`. If the private cap `max(32, 4m')` is reached, raise
     `UndecidedSystemError`.
2. **Jacobian.** `J0 = ∂Phi/∂F (0, k, c*)` on the reduced system. Test `J0^m' = 0` exactly
   in `Q[k]`.
   - Nilpotent: the system is well-posed.
   - Not nilpotent and non-negative: raise `IllPosedSystemError` (PSS Thm 5.5 with
     Thm 3.6). The message shows a cycle of the support graph of `J0`. One always
     exists, because a matrix whose support graph is acyclic is nilpotent.
   - Not nilpotent and signed: raise `UndecidedSystemError`.

### 5.4 Why this is correct

**Sufficiency (any signs).** If steps 1 and 2 pass, the iteration converges, and its
limit is the unique solution with `F(0) = c*`. For non-negative systems this follows from
PSS. For signed systems PSS do not apply, so we rely on our own argument:

*Proof sketch (signed case).* Uniqueness: `I − J0` is unipotent, hence invertible, so the
formal implicit function theorem applies at `(0, c*)`. Convergence: let
`e^(h) = F^(h) − F` and let `ν` be the nilpotency index of `J0`. Once step 1 is reached,
at some step `h_1`, `e^(h)` has zero constant term. The coefficient of `z^n` in
`e^(h+1)` is `J0` times that of `e^(h)`, plus terms that involve only coefficients of
`e^(h)` of index `< n`. By induction on `n`, coefficient `n` is exactly 0 after at most
`h_1 + n ν` steps. Zero coordinates are 0 in every iterate, so the reduction does not
change convergence.

**Necessity (non-negative systems).** Suppose the reduced system's iteration converges.
Then the reduced system is PSS-well-founded (Def. 5.3):

- it is well defined, because it is polynomial;
- it has no zero coordinate, by step 0;
- its species iteration converges. To see this, replace every positive coefficient by 1
  to get a species with the same structures. A weighted coefficient
  `[z^n] F_i^(h)` is a sum of positive weights over the structures of size `n` built
  by step `h`, so it is stationary iff that set of structures is stationary.

Then:

- Thm 5.7 gives `c* = c^(m')`, i.e. step 1 within `m'` iterations.
- Thm 5.5, condition 2, via Prop. 4.5(2), gives that `∂K/∂Y(Z1, 0, S0(Z1))` is nilpotent.
  Setting `Z1 = 1` gives `J0`, because `K(1, 0, Y) = H(0, Y)` and `S0(1) = c*`. A
  specialisation of a nilpotent matrix is nilpotent.

So a non-negative system that fails step 1 or 2 is proven ill-posed.

**Signed failures.** Only the cycle case is a proof. Example of why the other failures
prove nothing: `{F = z + 2G, G = 2G}` converges (`G ≡ 0`), yet `J0` restricted to `G` is
`[[2]]`. Step 0 catches this particular case, but in general a signed system that fails
the check may still converge.

### 5.5 Iteration counts

The signed example `F = z + 1 − (F − 1)^2 (F − 2)/2` (expanded:
`F = z + 2 − 5F/2 + 2F^2 − F^3/2`) has `m = 1`. Its constant terms go `0 → 2 → 1 → 1`, so
they need 2 steps, and `J0 = 0`. No bound like PSS Thm 5.7 holds for signed systems, so
they need the cap. Non-negative systems need no cap.

### 5.6 Zero coordinates: accept with a warning

PSS exclude zero coordinates by definition (Def. 5.3; Example 4, p. 15: `Y = Y`,
`Y = Y + ZY`, `Y = ZY`). We **accept them with a `ZeroCoordinateWarning`**, for three
reasons:

- under the project's definition their iteration converges (to 0);
- CLAUDE.md says to start with warnings;
- specialising a valid system can create them, so rejecting them would break
  "generic pass ⇒ pass for every `k`" (Section 5.7). For example,
  `{F = z + (k − 1) G, G = (k − 1) z}` has `G ≡ 0` at `k = 1`.

`series` treats them as 0.

### 5.7 Role of `k`

Every computation in steps 0–2 is done exactly in `Q[k]`: support propagation, the
iterates `c^(h)`, stationarity, and `J0^m' = 0`.

**A generic pass implies a pass for every `k`.** Evaluation at `k = k_0` is a ring
homomorphism `Q[k] → Q`, and it commutes with the polynomial operations used. So the
iterates of the specialised system are the specialised iterates. The identities
`c^(h+1) = c^(h)` and `J0^m' = 0`, and any zero coordinate, all stay true. More directly,
the whole iteration in `Q[k][[z]]` specialises, so `z`-adic convergence for symbolic `k`
gives convergence for each `k_0`. This is why `System` needs no `KSet`. Specialising can
only *add* zero coordinates, which give warnings, never errors.

**A failure is a statement about symbolic `k`.**

- Non-negative case: it holds for **every integer `k >= 2`**, and also for `k = 1`
  unless some coefficient of the system vanishes at `k = 1`. Whether one does is decided
  exactly: a non-zero coefficient `p(k)` vanishes at `k = 1` iff `p(1) = 0`. Justification:
  for a non-negative system, convergence depends only on the support. Replace the
  positive weights by 1 and you get the same structures (Section 5.4). Wherever no
  coefficient vanishes, the support is the same as for symbolic `k`, and so is the
  verdict. Since `p(1 + t)` has non-negative coefficients, `p(k) > 0` for `k >= 2`.
  Example: `F = (k − 1) F + z` is ill-posed for `k >= 2` but well-posed at `k = 1`, and
  the message says `k = 1` is not covered.
- Signed cycle: it holds for all but finitely many `k`.

The check never samples values of `k`.

### 5.8 Messages

Actual output of the implementation.

```
flanpy.IllPosedSystemError: the system is ill-posed: the iteration from 0 does not converge. At z = 0 the constant terms are F(0) = 0 and the Jacobian dPhi/dF is [[1]], which is not nilpotent: the dependency cycle F -> F has no factor z. (Non-negative system: Pivoteau, Salvy & Soria 2012, Thm 5.5 and Thm 3.6.)
  Equations:
    F = F + z

flanpy.IllPosedSystemError: the system is ill-posed: the constant terms do not stabilise, so the iteration does not converge. After 1 iteration(s) F(0) = 1, after 2: F(0) = 2. A convergent system with non-negative coefficients reaches its constant terms within 1 iteration(s) (Pivoteau, Salvy & Soria 2012, Thm 5.7).
  Equations:
    F = F + 1

flanpy.IllPosedSystemError: the system is ill-posed: the iteration from 0 does not converge. At z = 0 the constant terms are F(0) = 0 and the Jacobian dPhi/dF is [[k]], which is not nilpotent: the dependency cycle F -> F has no factor z. (Non-negative system: Pivoteau, Salvy & Soria 2012, Thm 5.5 and Thm 3.6.)
  Equations:
    F = F*k + z
  This holds for every integer k >= 1 (no coefficient vanishes there, so the support of the system does not change).

flanpy.IllPosedSystemError: the system is ill-posed: the iteration from 0 does not converge. At z = 0 the constant terms are F(0) = 0 and the Jacobian dPhi/dF is [[k - 1]], which is not nilpotent: the dependency cycle F -> F has no factor z. (Non-negative system: Pivoteau, Salvy & Soria 2012, arXiv:1109.2688v2, Thm 5.5 and Thm 3.6.)
  Equations:
    F = F*k - F + z
  This holds for every integer k >= 2. At k = 1 some coefficients vanish, so k = 1 is not covered: use specialize(1) to check it.

flanpy.IllPosedSystemError: the system is ill-posed: the constant terms cycle, so the coefficient of z^0 never settles and the iteration does not converge: F(0) = 0 -> F(0) = 1 -> F(0) = 0.
  Equations:
    F = 1 - F

flanpy.UndecidedSystemError: could not decide whether the system is well-posed: it has negative coefficients, and the nilpotency test fails. At z = 0 the constant terms are F(0) = 0 and the Jacobian dPhi/dF is [[-1]], which is not nilpotent: the dependency cycle F -> F has no factor z. For systems with negative coefficients this failure does not prove divergence.
  Equations:
    F = -F + z

flanpy.UndecidedSystemError: could not decide whether the system is well-posed: it has negative coefficients, and its constant terms did not stabilise within 32 iterations from 0 (last two: F(0) = 31, then F(0) = 32).
  Equations:
    F = -F**2*z + F + 1

flanpy.ZeroCoordinateWarning: the solution has zero coordinates: F is identically 0 (every iterate from 0 is 0) and kept as 0. Pivoteau, Salvy & Soria (2012, Def. 5.3) exclude such systems. Remove this unknown if this is not intended.
```

Malformed input raises `ValueError` or `TypeError`, naming the offending equation. For
example, a non-polynomial right-hand side such as `1/(1 − zF)` gets a hint to use an
auxiliary unknown for `SEQ`. A stray symbol gets either "marking variables are not
supported yet" or, if it has the name of an unknown, of `z` or of `k`, a hint about SymPy
assumptions.

## 6. Public API

Public names, all reporting `flanpy` as their module:

```
flanpy.System
flanpy.WellPosednessError(ValueError)    # base: the check did not pass
├── flanpy.IllPosedSystemError           # proven: the iteration does not converge
└── flanpy.UndecidedSystemError          # not decided; never a claim of ill-posedness
flanpy.ZeroCoordinateWarning(UserWarning)
```

The common base lets callers catch "rejected for any reason" with one clause, while the
two outcomes stay distinct. It subclasses `ValueError` because the constructor's argument
is rejected. The implementation lives in the private module `src/flanpy/_system.py` and
is re-exported from `flanpy`.

```python
class System:
    def __init__(
        self,
        equations: Mapping[sympy.Symbol, sympy.Expr | int],
        *,
        z: sympy.Symbol,
        k: sympy.Symbol | None = None,
    ) -> None: ...

    # In the order of the input mapping.
    @property
    def unknowns(self) -> tuple[sympy.Symbol, ...]: ...

    # Read-only, F_i -> Phi_i (expanded).
    @property
    def equations(self) -> Mapping[sympy.Symbol, sympy.Expr]: ...

    @property
    def z(self) -> sympy.Symbol: ...

    @property
    def k(self) -> sympy.Symbol | None: ...

    # F_i - Phi_i over QQ, generators (*unknowns, z, k).
    def polynomials(self) -> tuple[sympy.Poly, ...]: ...

    # k >= 1 (the domain); the result has k = None.
    def specialize(self, k: int) -> System: ...

    # __eq__, __hash__, __repr__ (content-based); immutable; picklable
```

Input rules:

- `unknowns`, `z` and `k` are distinct `sympy.Symbol`s, with distinct names (see the
  assumptions hint in Section 5.8).
- `Phi_i` uses only the unknowns, `z` and `k`. Coefficients are exact rationals; floats
  and irrational constants are rejected.
- There is at least one equation. `k=None` is allowed, and a `k` that is given but unused is fine.
- `specialize(k)` requires an `int >= 1` (not `bool`) and a system with `k` set.

## 7. Private, computed at construction

- `_active: tuple[int, ...]`: the indices of the unknowns that are not zero coordinates.
- `_constant_terms: tuple[Expr, ...]`: `c*` for all `m` unknowns, 0 for zero coordinates.
- `_jacobian0: Matrix` and `_nilpotency_index: int`: `J0` and `ν` on the **reduced**
  system, of size `m' × m'`.
- `_nonnegative: bool`.
- `_params`: currently `(k,)` or `()`. Everything is computed identically for every
  parameter, so marking variables can join later (Section 8).
- `_signed_cap(m)`: the private iteration cap. It is a function so that tests can
  monkeypatch it.

The F&S-strict flag proposed in chunk 1 was dropped, because nothing uses it.

## 8. Room for marking variables (later; not designed here)

Cost generating functions need extra variables `u` that appear polynomially in `Phi`,
e.g. eq. (14) has `k u z`. They will be accepted through a keyword such as `markers=(u,)`,
stored in `_params`, and validated identically in them, exactly like `k`. Section 5.7
applies to them unchanged. An operation then derives from a `System` with marker `u` the
augmented system for `(F, ∂_u F |_{u=1})`, as in eqs. (13)–(14) → (15)–(16).

## 9. What `series` will need from `System`

`series` computes the coefficients `[z^0..z^N]` of every unknown, either for symbolic `k`
or after `specialize(k)`. It needs the ordered unknowns, `Phi`, `_active` (zero
coordinates are 0), `c*` and `J0`.

With those, coefficients can be computed one at a time by a linear solve,
`(I − J0) y_n = r_n(y_0..y_{n−1})`, or by Newton iteration (PSS Thm 8.10). Because
`I − J0` is unipotent over `Q[k]`, all coefficients stay in `Q[k]`. A naive fixed-point
iteration also works, with at most `ν` passes per new coefficient (Section 5.4).
`series` gets its own design note in chunk 3.

## 10. Tests (`tests/test_system.py`)

Well-posed. Each test checks the fields, `polynomials()` (generators, domain and value),
`c*`, the non-negativity flag, and that `J0` is nilpotent.

| system | non-negative | `c*` |
|---|---|---|
| linear `F = 1 + 2 z F` | yes | `1` |
| Catalan `F = 1 + z F^2` | yes | `1` |
| Motzkin `F = 1 + z F + z^2 F^2` | yes | `1` |
| leaves `F = z + F^2` | yes | `0` |
| two equations `F = 1 + z G`, `G = 1 + z F^2` | yes | `(1, 1)` |
| PSS Example 8 `Y1 = 1 + z Y1`, `Y2 = 1 + Y1^2` | yes | `(1, 2)` |
| Broda `R_k` (Section 3.1) | yes | `0` |
| Broda `R_k, S_k` (Section 3.2) | yes | `(0, 0)` |
| Broda `L_k` (Section 3.3, five unknowns) | yes | `0` |
| `F = z + 1 − (F−1)^2 (F−2)/2` | no | `1` |

Zero coordinates (accepted, with a warning): `F = F`, `F = zF`, `F = F + zF` (PSS
Example 4), `{F = z + G, G = G}`, `{F = FG, G = zF}`, and
`{F = z + (k − 1) G, G = (k − 1) z}` specialised to `k = 1`. `F = z + (k − 1) z F^2` takes
the non-negative path.

Ill-posed (`IllPosedSystemError`; the test matches key phrases of the message):
`F = F + z`, `F = z + F + F^2`, `{F = G + z, G = F}`, `F = 1 + F`, `F = 1 + F^2`,
`F = kF + z` ("every integer k >= 1"), `F = (k − 1) F + z` and `F = 1 + (k − 1) F`
("every integer k >= 2"), PSS Example 8 `{Y1 = 1 + Y2 Y1, Y2 = 1}`, and the
signed cycle `F = 1 − F`.

Undecided (`UndecidedSystemError`, and not `IllPosedSystemError`):

- the two-step signed example with the cap forced to 1 (monkeypatched);
- `F = F + 1 − z F^2`, where the cap is reached;
- `F = z − F`, a signed system whose `J0` is not nilpotent.

Also:

- `specialize(k)` for `k = 1, 2, 7` on the Broda systems;
- malformed input, with 16 cases;
- `specialize` errors;
- value semantics: equality, hashing, order sensitivity, pickling and immutability;
- `repr`, and the public names;
- doctests in the `System` docstring (Catalan, Broda `R_k`, `F = F + z`).

## 11. Non-goals and open questions

- No construction from implicit polynomial equations (`P(z, F) = 0` not in fixed-point
  form). Revisit if a paper needs it.
- No `Q(k)` coefficients until `kset` exists (Section 4).
- Signed systems: a sharper decision procedure is possible but not planned. For example,
  periodicity of the order-1 coefficients when `J0` is not nilpotent would prove more
  cases ill-posed.
