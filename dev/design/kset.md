# Design note: `KSet`

Status: draft for approval (build-order step 2, chunk 1). 2026-10-30.

`k`, the alphabet size, is the only parameter, with domain the integers `k >= 1`
(`system.md` §5.7). Every condition the later steps produce is a sign or vanishing
condition on a univariate polynomial or rational function in `k` over `Q`. Examples are
leading coefficients that vanish in elimination, and discriminants and denominators in
singularity analysis, e.g. Broda 2020 eq. (18), `disc_z p_k = 32 k^2 (k − 54)^3`. So the
set of `k` where a result holds is a finite union of integer points and integer
intervals. `KSet` is that set. `KCases` (Section 4) holds a result that takes different
values on different `KSet`s.

Spike: `dev/spikes/kset_spike.py` (algorithm of Section 3, checks, timings).

## 1. `KSet`: representation

**A canonical tuple of disjoint integer intervals.** A `KSet` stores
`intervals: tuple[tuple[int, int | None], ...]`, where `(lo, hi)` means
`{lo, ..., hi}` and `hi = None` means `[lo, ∞)`. The canonical form is:

- `1 <= lo <= hi` for every interval (the universe is `k >= 1`);
- sorted, with a gap of at least one integer between neighbours (`hi_i + 1 < lo_{i+1}`).
  Adjacent or overlapping intervals are merged;
- only the last interval may be unbounded.

Every subset of `{k >= 1}` that is a finite union of intervals has exactly one canonical
form. So equality is equality of the tuples, and `hash` is `hash(intervals)`.

**Why integer intervals, not real ones.** The domain is the integers. Real intervals
would need algebraic endpoints (e.g. `17 + 7√7`, Section 6) and open/closed flags. Then
equality and canonical forms would need comparisons of algebraic numbers, and none of
that is visible at integer `k`. Real-root information is used inside `where`
(Section 3) and dropped once the integer pieces are known. Step 7 takes limits as
`k → ∞` over the integers, which only needs "contains all large `k`" (Section 2). If a
later step needs statements for real `k`, that is a separate type.

**Structural fact.** Only the last interval can be unbounded. So a `KSet` is either
finite, or contains a tail `[n0, ∞)` and has a finite complement. "Contains all
sufficiently large `k`", "is infinite" and "is cofinite" are the same property.

**Immutability and value semantics.** `__slots__ = ("_intervals",)`, no mutators, and
pickling through `__reduce__` (the intervals). Equality only with another `KSet`
(otherwise `NotImplemented`). It is hashable, so it can key a dict (as `KCases` does).

**Construction:**

- `KSet(intervals=())`: any iterable of pairs `(lo, hi)`, in any order, possibly
  overlapping or adjacent; canonicalised. Rejects `lo < 1`, `lo > hi`, `lo = None`, and
  non-integers (`operator.index`; `bool` rejected, as for `k` in `series`), with
  `TypeError` / `ValueError`. Out-of-domain values are rejected, not clipped, because
  clipping would hide a caller's bug. `KSet()` is empty.
- `KSet.all()`: `[1, ∞)`.
- `KSet.of(*ks)`: a finite set of points, e.g. the exceptional `k` that steps 3–4
  handle by instantiation.
- `KSet.where(p, rel, *, k, den=None)`: from a condition (Section 3).

**Printing** in the notation of CLAUDE.md:

| set | `str` | `repr` |
|---|---|---|
| empty | `∅` | `KSet()` |
| `{54}` | `{54}` | `KSet([(54, 54)])` |
| `[1, 53] ∪ [55, ∞)` | `[1, 53] ∪ [55, ∞)` | `KSet([(1, 53), (55, None)])` |
| `{2} ∪ {5} ∪ [7, 9]` | `{2} ∪ {5} ∪ [7, 9]` | `KSet([(2, 2), (5, 5), (7, 9)])` |

A one-point interval prints as `{n}`. Every longer interval prints as `[a, b]` (also
`[3, 4]`). `repr` evaluates back to an equal `KSet`. Because the form is canonical,
adjacent pieces never appear: the CLAUDE.md example `{2} ∪ [3,53] ∪ {54} ∪ [55, ∞)` is a
union of pieces from different conditions, and as one `KSet` it is `[2, ∞)`.

## 2. Operations

Each operation is a linear merge of the two sorted interval lists, `O(n + m)`:

| operation | spelling | note |
|---|---|---|
| intersection | `a & b` | |
| union | `a \| b` | |
| complement in `k >= 1` | `~a` | `~KSet() == KSet.all()` |
| difference | `a - b` | `a & ~b` |
| membership | `n in a` | `n` an integer (`operator.index`); `TypeError` otherwise, `bool` included; `n <= 0` gives `False` |
| emptiness | `bool(a)` | `False` iff empty (container convention) |
| subset, superset | `a <= b`, `a < b`, `a >= b`, `a > b`, `a.isdisjoint(b)` | as for `frozenset` |
| all large `k` | `a.tail_start` | least `n0` with `[n0, ∞) ⊆ a`, or `None` if `a` is finite |
| finiteness | `a.is_finite` | `a.tail_start is None` |
| finite part | `a.finite_part` | `a` below its tail: `a − [tail_start, ∞)`, or `a` itself if finite; always finite |
| iteration | `iter(a)` | ascending integers if `a` is finite; `ValueError` raised eagerly if not |
| raw form | `a.intervals` | the canonical tuple |

Step 7 uses `tail_start`: a limit as `k → ∞` is meaningful when its `KSet` is
infinite, and the result says from which `k` on the expression is valid.

No `__len__`. `list()` and `len()` call `__len__`, and a `__len__` that raises on an
infinite set is a trap. Use `len(list(a))` on a finite set.

Left out: symmetric difference (`(a − b) | (b − a)`), `min`/`max` (read them from
`intervals`), ordering by anything other than inclusion, and a LaTeX `repr`.

## 3. Construction from conditions

```python
@classmethod
def where(
    cls,
    p: sympy.Expr | sympy.Poly | int | Fraction,
    rel: Literal["==", "!=", "<", "<=", ">", ">="],
    *,
    k: sympy.Symbol,
    den: sympy.Expr | sympy.Poly | int | Fraction | None = None,
) -> KSet: ...
```

**Meaning.** `where(p, rel, k=k)` is `{n >= 1 : p(n) rel 0}`. With `den`, it is
`{n >= 1 : den(n) != 0 and p(n)/den(n) rel 0}`. The condition `den != 0` is always
implied, so the set never contains a pole.

- `p` and `den` must be polynomials in `k` with rational coefficients. A `sympy.Expr`
  is converted with `Poly(..., k, domain=QQ)`. Other symbols, non-polynomial
  expressions (`1/k`, `sqrt(k)`), float coefficients and algebraic coefficients
  (`sqrt(2)`) are rejected with `ValueError`. A zero `den` is rejected.
- **The given `den` is not cancelled against `p`.** `where(k − 1, ">", k=k, den=k − 1)`
  is `[2, ∞)`, not `[1, ∞)`. Denominators in later steps come from divisions that
  actually happened in a derivation. At a root of `den` that derivation is invalid,
  even if the reduced fraction is defined there. This is the conservative reading. For
  the lowest-terms reading, cancel first (`sympy.cancel`, then `fraction`). A rational
  `p` given as one `Expr` is rejected rather than split with `fraction`, because SymPy
  cancels identical factors when it builds an expression (`(k − 1)/(k − 1)` is `1`), so
  "the given denominator" would not be well defined. Accepting it later is a backward
  compatible extension.
- `rel` is a string, not a SymPy relational. `disc < 0` would work, but `p == 0` is a
  Python `bool` (structural equality), a silent trap.
- `k` is required, as in `System`. It also fixes the variable when `p` is a constant.

**Reduction to signs.** Where `den(n) != 0`, `sign(p(n)/den(n)) = sign(p(n)·den(n))`.
So `where(p, rel, den=q) = where(p·q, rel) − where(q, "==")`. Every relation is then a
choice of signs: `==` is `{0}`, `!=` is `{−1, 1}`, `>` is `{1}`, `>=` is `{0, 1}`, and so
on. One private routine computes all of them:

```
_sign_pieces(P) -> tuple[(lo, hi | None, s), ...]    # s in {-1, 0, 1}
```

It returns the maximal integer intervals covering `k >= 1` on which `sign(P(n)) = s`.
`where` takes the union of the pieces whose sign is in the chosen set.

**Algorithm** (all exact; no floats):

1. The zero polynomial is the single piece `(1, None, 0)`. A non-zero constant has one
   piece with its sign.
2. `Q` = the squarefree part of `P` (`P / gcd(P, P')`). It has the same real roots.
3. **Integer roots first.** The integer roots `n >= 1` of `Q` are its rational roots
   that are integers (`Poly.ground_roots()`). Divide `Q` by `k − n` for each. The
   quotient `Q'` is squarefree and has no integer root.
4. **Floors of the other roots.** Isolate the real roots of `Q'` in `[1, ∞)` with
   `Q'.intervals(inf=1)`: rational isolating intervals, from the
   Vincent–Akritas–Strzeboński continued-fraction method in exact integer arithmetic.
   Refine each `[a, b]` with `Q'.refine_root` until it contains no integer. Then
   `floor(r) = floor(a)`. This terminates because `r` is not an integer.
5. **Pieces.** Cut `{1, 2, ...}` into a singleton `{n}` at each integer root, and into
   two sides between `m` and `m + 1` at each non-integer root of floor `m`. No real root
   of `P` lies in the real span of a resulting interval. So by the intermediate value
   theorem `P` has a constant, non-zero sign there. Evaluate `P` exactly at the left end
   (Horner over `int`/`Fraction`). Merge neighbours with equal sign (roots of even
   multiplicity, roots below 1). Check: the last piece has the sign of the leading
   coefficient.

Why step 3 comes before step 4: the spike first tried "the root of `[a, b]` is the
integer `n ∈ [a, b]` where `P(n) = 0`". That is wrong. For `(k − 20)(3k − 61)`, SymPy
isolates `61/3` in `(20, 21)`, whose endpoint `20` is the *other* root. With the integer
roots removed, no step depends on whether SymPy's intervals are open or closed. Also, a
brute-force check with `lambdify` gave wrong signs: it printed a constant term as
`3896750/3`, which Python evaluates as a float. The tests use exact Horner evaluation
(Section 6).

**Cost** (spike, Apple M1, Python 3.13.16, SymPy 1.14.0 with flint ground types, single
runs):

| polynomial | degree | time | pieces |
|---|---|---|---|
| `disc_z p_k`, eq. (18) | 5 | < 1 ms | 3 |
| `lc_z res_w(G, G_w)`, `G` of eq. (17) | 13 | < 1 ms | 1 |
| `disc_z` of the squarefree part of `res_w(G, G_w)` | 41 | 3 ms | 4 |
| random, 30-bit coefficients | 50 / 100 / 200 | 6 / 24 / 205 ms | 1–2 |
| roots `1/2, 3/2, ..., (2d − 1)/2` | 30 / 60 | 30 / 201 ms | 30 / 60 |
| `(k^2 − 2) ∏_{i=1}^{40} (k − i)` | 42 | 7 ms | 2 |

Conditions from steps 3–4 are of degree tens at most, so the cost is negligible next to
the Gröbner and resultant computations that produce them. The number of exact
evaluations is the number of pieces, at most `2·deg + 1`.

**Implementation choice: SymPy only, no backend switch.** Exact real-root isolation is
available only in SymPy. python-flint 0.9 has `fmpz_poly.roots()` (integer roots,
exact) and `complex_roots()` (certified arb balls), but `real_roots()` raises
`NotImplementedError`. Using the arb balls would need extra certification for the
floors. SymPy already uses flint integers as ground types when python-flint is
installed. `_rings.py` is not involved: it provides coefficient arithmetic for
recurrences, and `KSet`'s cost is root isolation, not ring arithmetic. If very large
degrees ever appear, a flint path (integer roots by `roots()`, floors from the balls
checked by exact signs at the neighbouring integers) can be added privately behind
`_sign_pieces`. `_sign_pieces` is memoised on the coefficient tuple
(`functools.lru_cache`), because steps 3–4 often ask for `< 0`, `== 0` and `> 0` of the
same polynomial.

## 4. Piecewise results: `KCases`

Later steps return different values on different pieces. A typical case is a generic
result where a leading coefficient `lc(k) != 0`, plus instantiated results at the
exceptional `k`. `KCases[T]` (generic in the value type) holds one result per piece:

```python
class KCases(Generic[T]):
    def __init__(self, pieces: Mapping[KSet, T] | Iterable[tuple[KSet, T]]) -> None: ...
    pieces: tuple[tuple[KSet, T], ...]   # sorted by least element
    covered: KSet                        # union of the pieces
    uncovered: KSet                      # ~covered: undecided / not computed
    def __getitem__(self, k: int) -> T: ...                  # KeyError if uncovered
    def map(self, f: Callable[[T], U]) -> KCases[U]: ...
    def restrict(self, s: KSet) -> KCases[T]: ...            # pieces ∩ s
    def combine(self, other: KCases[U], f: Callable[[T, U], V]) -> KCases[V]: ...
```

**Invariants:**

- The pieces are pairwise disjoint. An overlap raises `ValueError` and names it.
- Empty pieces are dropped: they contain no `k`, so nothing is lost. This lets callers
  write `KCases({KSet.where(lc, "!=", k=k): generic, KSet.where(lc, "==", k=k): ...})`
  without special-casing an empty exceptional set.
- `covered` is exactly where the result is known. Everything else is `uncovered`.
  `uncovered` is part of the value, so it is printed and compared, and `cases[n]` raises
  `KeyError("k = 54 is not covered (undecided)")`. Nothing is dropped silently. A step
  that leaves a piece undecided also warns, under the "Rigor" rule.
- `combine` intersects pieces pairwise and keeps the non-empty intersections, with
  value `f(a, b)`. So `covered` is the intersection of the two `covered` sets. This is
  CLAUDE.md's "results combine by intersecting `KSet`s".
- Equality compares `pieces` (values with `==`). It is hashable when the values are.
- `str`: one line per piece, `[1, 53]: <value>`, then `undecided: <uncovered>` if that
  is non-empty.

Uses in steps 3–4: elimination returns `KCases[curve]` (generic `G` where the leading
coefficients and the Gröbner basis specialise, and `G` recomputed at the finitely many
exceptional `k`). Singularity returns Case I / Case II and `rho` per piece, combined
across conditions with `combine`.

**Left out**, until a step needs it: a reason attached to each undecided part; merging
pieces with equal values (structural equality of SymPy values is unreliable); arithmetic
on `KCases`; nesting; and a union of two `KCases` with disjoint covers (the constructor
over both `pieces` does it).

## 5. Public API, module, and relation to `System`

- **Module**: `src/flanpy/_kset.py`, holding `KSet`, `KCases` and `_sign_pieces`.
- **`KSet` is public now** (`flanpy.KSet`, re-exported, in `__all__`). Every later
  result carries one, and it is useful on its own (Section 7).
- **`KCases` is implemented and tested in chunk 2, but stays private** (in `_kset`, not
  re-exported) until step 3 returns one. Its API will meet real use there before it
  becomes a contract (CLAUDE.md: deprecate before removing).
- **`System` and `series` need nothing now.** A generic well-posedness pass holds for
  every `k` (`system.md` §5.7). The only `k`-dependent statement is the non-negative
  failure message ("every `k >= 2`, and `k = 1` unless ..."). It could carry a `KSet`
  attribute later, as a backward-compatible addition, but not in this step. `Q(k)`
  coefficients in `System` (`system.md` §11) would become possible with `KSet`, valid
  where the denominators do not vanish. That is not planned until a paper needs it.
- **CLAUDE.md "Planned layout"**: `_kset.py` is added next to `_rings.py`, and `KSet`
  is removed from `families/` (done in this chunk).

## 6. Planned tests (`tests/test_kset.py`)

- **Representation**: canonicalisation of unsorted, overlapping and adjacent input;
  `KSet.of`, `KSet.all()`; errors (`lo < 1`, `lo > hi`, `lo = None`, `bool`, float);
  `str`/`repr` for each row of the table in Section 1; `eval(repr(a)) == a`; equality,
  hashing, pickling, immutability.
- **Operations, hand-made**: complement of `∅` and of `[1, ∞)`; `tail_start`,
  `is_finite`, `finite_part`, iteration and its eager `ValueError`; `in` with `0`,
  negatives, `sympy.Integer`, and the `TypeError` for `bool` and `2.5`.
- **Laws, property-based with `hypothesis`**: a strategy draws `KSet`s from up to six
  random intervals with endpoints in `[1, 40]`, plus an optional unbounded tail. Each
  operation is checked against a model: a `frozenset` of the points in `[1, B]`, with
  `B` beyond all endpoints, plus a flag for "contains `B + 1, B + 2, ...`". On top of
  that: De Morgan, distributivity, absorption, `~~a == a`, `a − b == a & ~b`,
  `a <= b ⇔ a & b == a`, and "equal ⇒ equal hash". Proposal: add `hypothesis` to the
  `dev` extra. It is pure Python and dev-only, and it finds the edge cases that
  hand-made lists miss (as the random spike did twice). Register a profile in
  `tests/conftest.py` with `derandomize=True` and `deadline=None`, so CI is reproducible
  and not timing-sensitive, and add `.hypothesis/` to `.gitignore`.
- **`where`, hand-made** (all six relations each):
  - integer roots `(k − 2)(k − 5)`;
  - roots between integers `2k^2 − 7k + 4`;
  - double root at an integer `(k − 3)^2 (k − 10)`;
  - double root between integers `(2k − 7)^2`;
  - no real roots `k^2 + 1`;
  - constants `−3`, `0`;
  - close roots `(k − 1001/1000)(k − 1002/1000)`;
  - a root at 1, roots below 1;
  - a large root `k − 10^30`;
  - rational coefficients;
  - the regression `(k − 20)(3k − 61)` (an integer root at the endpoint of another
    root's isolating interval).
- **`where` with `den`**: a pole excluded for every relation; `(k − 1)/(k − 1)` gives
  `[2, ∞)` (no cancellation); a constant `den`; `den` without integer roots; a zero
  `den` rejected.
- **`where` errors**: `1/k`, `sqrt(k)`, another symbol, a float coefficient, `sqrt(2)`
  as a coefficient, a bad `rel`, `k` missing.
- **`where` against brute force (`hypothesis`)**: polynomials built from random rational
  roots in `[−20, 120]` (denominators up to 7), times an optional irreducible quadratic,
  with a random sign. Compare with exact Horner signs at `k = 1..B`, `B` = largest root
  + 2, and check that the tail sign is the sign of the leading coefficient.
- **Broda 2020 eq. (18)** (acceptance test): `disc = sympy.discriminant(p_k, z)` with
  `p_k = 4k(2k + 27)z^3 + 36kz^2 − kz − 4`. Assert `factor(disc) == 32 k^2 (k − 54)^3`.
  Then `where(disc, "<") == KSet([(1, 53)])`, `where(disc, "==") == KSet.of(54)` and
  `where(disc, ">") == KSet([(55, None)])`; together they partition `KSet.all()`. The
  root `k = 0` lies outside the domain.
- **A root between integers from the pipeline** (derived by us, not printed in the
  paper): for `G` of eq. (17), `D = disc_z` of the squarefree part of `res_w(G, G_w)`.
  It factors as `c · k^22 (k + 2)^8 (k − 54)^9 (k^2 − 34k − 54)`, with `c > 0`. Then
  `where(D, ">") == [1, 35] ∪ [55, ∞)`, `where(D, "<") == [36, 53]` and
  `where(D, "==") == {54}`. The root `17 + 7√7 ≈ 35.52` is not an integer. (Spike
  output; the test computes `D` from scratch.)
- **`KCases`**: overlap rejected; empty pieces dropped; `covered`/`uncovered`;
  `cases[n]` and the `KeyError` message; `map`, `restrict`, `combine` (cover is the
  intersection); equality, hashing, `str`.
- **Docstrings**: `tests/test_docstrings.py` picks up `KSet` and its public members
  automatically (EX01). Each gets an Examples section; no allowlist entries are
  planned.

## 7. Docs plan (chunk 2)

- **Docstrings**: `KSet` (definition, canonical form, notation; CLAUDE.md "Conditions
  on k") and every public member, with doctested Examples. `where` cites the algorithm
  (Section 3; VAS via SymPy) and shows Broda 2020 eq. (18). The `flanpy` package
  docstring and the `index.md` warning list `KSet` as implemented.
- **API page**: `.. autoclass:: flanpy.KSet` with its members in `docs/index.md`.
- **Guide page** `docs/guide/kset.md` (executed): building `KSet`s; printing; set
  operations; `where` with each relation; `den` and the no-cancellation rule;
  `tail_start` and `finite_part` for limits as `k → ∞`. `KCases` is not shown while it
  is private.
- **`examples/catalan.py`**: a short "Step 2" markdown cell. Catalan has no parameter,
  so every result holds unconditionally and step 2 adds nothing. Status cell: step 2
  "does not apply" (unchanged in substance, with a pointer to the Broda example).
- **`examples/broda2020.py`**, section "Step 2: conditions on `k`": compute
  `disc_z p_k`, assert the factored form of eq. (18), build the three `KSet`s with
  asserts and print them (`[1, 53]`, `{54}`, `[55, ∞)`). Then show the eq. (17)
  discriminant `D` above, whose sign changes between 35 and 36, as an example of a root
  between integers, marked as derived and not from the paper. Status cell: step 2 done.

## 8. Open questions for approval

1. `KCases` private until step 3 (recommended), or public now?
2. Add `hypothesis` to the `dev` extra (recommended)?
3. `where` takes polynomials only, with `den` separate and never cancelled
   (Section 3). Is the conservative reading right for your use?
4. Names: `KSet.where`, `KSet.of`, `KSet.all`, `tail_start`, `finite_part`, `KCases`.
