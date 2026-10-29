# Design note: `series`

Status: draft for approval (build-order step 1, chunk 3). 2026-10-29.

`series` computes the first coefficients `[z^0..z^N]` of every unknown of a
[`System`](system.md). These are exact, for a fixed alphabet size `k` or as polynomials
in a symbolic `k`. It is the ground truth that later stages are checked against (brute
force, Table 1 of Broda 2020), and it reproduces exact counts such as Broda 2020,
eq. (3).

Pivoteau–Salvy–Soria (PSS) are cited with the numbering and pages of arXiv:1109.2688v2,
as in `system.md`.

## 1. Public API

```python
def series(
    system: System, n: int, *, k: int | None = None
) -> Mapping[sympy.Symbol, tuple[int | Fraction, ...]]: ...


def series_in_k(
    system: System, n: int
) -> Mapping[sympy.Symbol, tuple[sympy.Poly, ...]]: ...
```

- **Result**: a read-only mapping `unknown -> (c_0, ..., c_n)`, with the coefficients of
  `z^0..z^n` (`n + 1` entries), keyed in the order of `system.unknowns`. Every unknown is
  included, including zero coordinates (all zeros) and auxiliary unknowns. `n >= 0`.
- **`series(system, n, k=...)`**: exact rational numbers, for a fixed `k`.
  - If `system.k` is set, `k` is required (an `int >= 1`), and the call means
    `series(system.specialize(k), n)`. If `system.k` is `None`, `k` must be omitted.
  - The coefficient type is `int` if the specialised `Phi` and `c*` have integer
    coefficients. Then every series coefficient is an integer, because `(I − J0)^-1` is
    an integer matrix (`J0` is nilpotent, so the inverse is `I + J0 + ... + J0^(m−1)`)
    and the recurrence in Section 3 uses only ring operations. Otherwise every
    coefficient is a `fractions.Fraction`. All coefficients of one result have the same
    type.
  - Results are plain Python numbers whatever the backend (flint types are converted at
    the boundary; this costs little next to the computation). So the result type does
    not depend on what is installed.
- **`series_in_k(system, n)`**: requires `system.k`. Each coefficient is a
  `sympy.Poly` in `k`, over `ZZ` or `QQ` by the same rule as above. Evaluating it at an
  integer `k >= 1` gives `series(system, n, k=...)` (Section 2.3).
- Two functions rather than one with a union result type, so that callers type-check
  without `isinstance`.
- Errors: `TypeError` / `ValueError` for bad `n` or `k` (negative, `bool`, a `k` given
  for a system without `k`, or missing when needed).

Public names: `flanpy.series` and `flanpy.series_in_k`, in a private module
`src/flanpy/_series.py` and re-exported. Everything else is private (Section 5).

## 2. Which solution, and why the algorithm returns it

`series` returns the **limit of the iteration** `F^(0) = 0`,
`F^(h+1) = Phi(z, k, F^(h))`, which `System` guarantees to exist (`system.md` §5).

### 2.1 Facts about every accepted system

`System` stores the indices of the zero coordinates, `c*` and `J0`, with `J0` nilpotent
on the reduced system. Then:

1. Every iterate of a zero coordinate is 0 (`system.md` §5.3, step 0). So those
   coordinates of the limit are 0. On the other coordinates, the iteration is the
   iteration of the reduced system, where `Phi` has the zero coordinates replaced by 0.
   This covers the zero-coordinate case, which PSS exclude from their definition of
   well-founded.
2. On the reduced system, the limit `Y` satisfies `Y = Phi_red(z, Y)` and `Y(0) = c*`,
   because `[z^0] F^(h) = c^(h)` stabilises at `c*`.
3. **Uniqueness.** A solution with `Y(0) = c*` is unique. Write `Y = sum y_n z^n`. For
   `n >= 1`,

   ```
   [z^n] Phi(z, Y) = J0 y_n + r_n(y_0, ..., y_{n−1}).
   ```

   A monomial `z^a F^e` with `a >= 1` only involves coefficients of index `<= n − 1`.
   A monomial with `a = 0` involves `y_n` only through terms with one factor at index `n`
   and all the others at index 0, i.e. linearly, through `∂Phi/∂F (0, c*) = J0`. Indices
   summing to `n` cannot include two `n`s. So `(I − J0) y_n = r_n`, where `I − J0` is
   invertible (unipotent). This holds for **any signs**, so it covers signed systems. It
   is also the uniqueness half of the argument in `system.md` §5.4.

So any algorithm that computes *a* solution with constant term `c*`, with zero
coordinates set to 0, returns the limit. For non-negative systems, PSS give an
independent confirmation: Thm 6.3 (p. 24) and Thm 8.10 (p. 33) show that Newton
iteration from 0 converges quadratically to the solution `S` with `S(0) = H^m(0, 0) = c*`.

### 2.2 Per algorithm

- **(b) recurrence** computes `y_n = (I − J0)^-1 r_n` for `n = 1..N`, from `y_0 = c*`.
  By fact 3, this *is* the solution.
- **(c) Newton** starts from `Y_0 = c*` and `U_0 = (I − J0)^-1`, with precision 1. Let
  `G(Y) = Y − Phi(Y)` and `E = Y_j − Y*`, with `val(E) >= p`. Every iterate has constant
  term `c*`, so `G'(Y_j) = I − J(z, Y_j)` has constant term `I − J0` and is invertible.
  Then

  ```
  Y_{j+1} − Y* = G'(Y_j)^-1 [(G'(Y_j) − G'(Y*)) E + O(E^2)],
  ```

  which has valuation `>= 2p`. With the approximate inverse `U ≡ G'(Y_j)^-1 mod z^p`, an
  extra term `(I − U G'(Y_j)) E = O(z^{2p})` appears, so the bound still holds. The
  iteration converges to the unique `Y*` with `Y*(0) = c*`, which is the limit by fact 3.
- **(a) fixed point** *is* the defining iteration. It only needs to stop when the
  first `N + 1` coefficients are final: after at most `h_1 + N ν` passes
  (`system.md` §5.4), which can be detected as two equal consecutive passes once the
  constant terms are stable.

### 2.3 Symbolic `k`

`series_in_k` runs the same recurrence over `Q[k]`. Here `c*` and `J0` are in `Q[k]`,
and `(I − J0)^-1 = I + J0 + ... + J0^(m−1)` has entries in `Q[k]`, so there is no
division by a polynomial in `k`. Evaluation at `k = k_0` is a ring homomorphism, so it
maps the symbolic recurrence onto the recurrence of `system.specialize(k_0)`, and the
two results agree coefficient by coefficient. No `KSet` is needed.

## 3. Algorithm

### 3.1 The three candidates

Notation: `m` unknowns, `P` distinct products of unknowns needed by `Phi` (the "product
nodes", e.g. `B·Bb`), `N` the order. `M(N)` is the cost of multiplying two series to
order `N`. Coefficient sizes grow linearly with the index: `[z^n]` has about
`n log2(1/ρ)` bits, where `ρ` is the radius of convergence, and `ρ` decreases with `k`
(Section 6 gives the measured sizes).

| | algorithm | cost (series / scalar operations) | notes |
|---|---|---|---|
| (a) | fixed point `Y <- Phi(Y)` | about `ν N` passes × `P` products of size `N` = `O(ν N P M(N))` | one new coefficient per `ν` passes; `O(N^3)` scalar multiplications with naive products |
| (b) | recurrence `(I − J0) y_n = r_n` | `P` convolutions of length `n` per index = `O(P N^2)` scalar multiplications | online, no series inverse; `m × m` constant matrix |
| (c) | Newton with inverse update | about `2 (P + 2m^3) M(N)` | quasi-optimal with fast `M(N)` (PSS Thm 8.15 and Cor. 8.16, p. 37); `O(N^2)` scalar multiplications per product with naive products |

Bit complexity: (b) does `O(P N^2)` multiplications of integers of `O(N)` bits, i.e.
`O(P N^2 M_Z(N))`. (c) with Kronecker substitution and FFT (flint) is `O(P M_Z(N^2))`
up to log factors, which is quasi-linear in the output size `O(m N^2)` bits. (a) is
always worse than (b) by a factor of about `N / P`.

### 3.2 Choice

**(b), the coefficient recurrence, on both backends, for fixed and symbolic `k`.**

- **Fast enough at the target size.** At `N = 1000` on the Broda systems (Section 6),
  it takes 0.1–1.3 s with flint scalars and 0.4–9.4 s in pure Python. Newton (c) with
  flint is faster, but by at most 0.4 s on these systems.
- **One code path for every case.** The same code runs over all eight coefficient
  domains (`int`, `Fraction`, `fmpz`, `fmpq`, and the four `Z[k]` / `Q[k]` types), with
  only `+` and `*`. (c) needs fast truncated series multiplication, which only flint
  provides. In pure Python, Newton is about 30× slower than (b) at `N = 1000` (projected
  from `N = 500`: 3.46 s against 0.2 s). For `Q[k][[z]]`, (c) would need Kronecker
  packing of `k`.
- **Simplest proof.** Its correctness *is* the uniqueness argument of Section 2.1, with
  no series inversion and no precision bookkeeping.
- **(a) is excluded.** With flint it takes 4.9 s at `N = 500`, against about 0.2 s for
  (b). In pure Python it takes 7.4 s at `N = 250` and scales like `N^3`.
- **When Newton would pay off.** (c) overtakes (b) for large `N`: 3.5 s against 12.4 s at
  `N = 2000`, and 12.3 s against 110 s at `N = 4000` (`L_k`, `k = 10`). No acceptance
  test needs `N > 1000`. So Newton (c) is a possible later optimisation for flint and
  fixed `k`, behind the same entry point. Section 2.2 already proves it correct, so adding
  it would not need a new argument.

## 4. Backends

The algorithm only needs a commutative ring of coefficients, with `+`, `*` and
conversions in and out. The backend chooses the scalar type:

| coefficients | pure Python (default) | python-flint (`fast` extra) |
|---|---|---|
| fixed `k`, integral | `int` | `flint.fmpz` |
| fixed `k`, rational | `fractions.Fraction` | `flint.fmpq` |
| symbolic `k`, integral | SymPy `ring("k", ZZ)` element | `flint.fmpz_poly` |
| symbolic `k`, rational | SymPy `ring("k", QQ)` element | `flint.fmpq_poly` |

- **One private interface**, `_Ring` in `src/flanpy/_rings.py`. It holds `zero`, `one`,
  `from_qq(c)` (a coefficient of `Phi`, an element of `QQ` or `QQ[k]`), and
  `to_public(x)` (to `int` / `Fraction` / `sympy.Poly`). The recurrence code is written
  once, against plain `+` and `*`. `_rings.py` is a separate module because later
  stages (elimination, resultants) will reuse it.
- **Selection.** flint is used if `import flint` succeeds, and pure Python otherwise.
  Public functions have no backend argument. The private entry point
  `_series(system, n, k, *, backend=None)` takes `"python"` or `"flint"` so that tests
  can force a path.
- **CI tests both paths.** The two `Tests` jobs differ in their install:
  - `Tests (Python 3.13)` installs `.[dev,fast]`, so flint is present and both backends
    run. Backend-parametrised tests use `pytest.importorskip("flint")` for the flint
    case.
  - `Tests (Python 3.x)` installs `.[dev]`, so flint is absent. That checks the fallback
    import path and the pure-Python backend on the newest Python, where flint wheels may
    lag.
  - Each job writes the backend in use to the step summary.

## 5. Private structure

`src/flanpy/_series.py`:

- `_compile(system, ring, k)`: specialise or keep `k`. Turn each `Phi_i` of the reduced
  system into terms `(coef, a, key)`, where `key` is a sorted tuple of unknown indices
  (the monomial). Build the product nodes: every prefix of length `>= 2` of a key, and
  for each node its left part and right factor. Compute
  `(I − J0)^-1 = I + J0 + ... + J0^(ν−1)` from `system._jacobian0` and
  `system._nilpotency_index`.
- `_recurrence(compiled, n)`: for `n' = 1..n`:
  1. each node's partial coefficient: the convolution `sum_{i=1}^{n'−1} left_i right_{n'−i}`,
     stored, plus `left_{n'}·right_0` (with `y_{n'}` taken as 0);
  2. `r_{n'}` from the terms;
  3. `y_{n'} = (I − J0)^-1 r_{n'}`;
  4. finish each node: `+ left_{n'} right_0 + left_0 right_{n'}`.

  The spike (`dev/spikes/series_spike.py`, `recurrence`) is this code without the
  checks.
- `_series(...)`: the shared entry point behind `series` and `series_in_k`. It
  re-inserts the zero coordinates as zeros and converts the result to the public types.

It uses only the private attributes of `System` that `system.md` §7 documents
(`_active`, `_constant_terms`, `_jacobian0`, `_nilpotency_index`).

## 6. Measurements

Source: `dev/spikes/series_spike.py`, run on an Apple M1 (16 GB), macOS 27.0.1,
Python 3.13.16, python-flint 0.9.0, SymPy 1.14.0. SymPy picks its ground types
automatically and uses flint when python-flint is installed, which it was. So the
"Python `ZZ[k]`" rows below used SymPy's sparse polynomials over flint integers.
Re-timing with `SYMPY_GROUND_TYPES=python` (pure Python, no gmpy2) changes them by
under 15% (e.g. `L_k`, `N = 200`: 3.2 s against 3.8 s). The fixed-`k` Python rows use
plain `int` and do not depend on SymPy's ground types.
Timings are single runs, in seconds. The spike first cross-checks that every algorithm and
backend agree, that the results match the enumerator values, and that symbolic `k`
evaluated at `k ∈ {1, 2, 3, 7}` matches fixed `k`. All checks pass.

**Fixed `k`, `N = 1000`:**

| system | `k` | (b) Python | (b) flint | (c) flint | bits of the last `[z^1000]` |
|---|---|---|---|---|---|
| `R_k` | 2 | 0.43 | 0.10 | 0.02 | 2545 |
| `R_k` | 10 | 0.72 | 0.12 | 0.03 | 3361 |
| `R_k` | 50 | 1.25 | 0.17 | 0.04 | 4393 |
| `R_k, S_k` | 2 | 0.88 | 0.21 | 0.07 | 2553 |
| `R_k, S_k` | 10 | 1.48 | 0.28 | 0.09 | 3368 |
| `R_k, S_k` | 50 | 2.54 | 0.37 | 0.11 | 4398 |
| `L_k` (5 unknowns) | 2 | 2.95 | 0.69 | 0.45 | 2378 |
| `L_k` | 10 | 5.35 | 0.87 | 0.64 | 3378 |
| `L_k` | 50 | 9.42 | 1.26 | 0.85 | 4436 |

Converting a flint result (5 × 1001 coefficients) to Python `int`s takes 2.4 ms.

**Slower variants and larger `N`** (`L_k`, `k = 10`):

| `N` | (a) Python | (a) flint | (c) Python | (b) Python | (b) flint | (c) flint |
|---|---|---|---|---|---|---|
| 125 | 0.54 | 0.08 | 0.06 | | | |
| 250 | 7.42 | 0.60 | 0.37 | | | |
| 500 | | 4.89 | 3.46 | | | |
| 1000 | | | | 5.41 | 0.91 | 0.64 |
| 2000 | | | | 87.4 | 12.4 | 3.51 |
| 4000 | | | | | 110 | 12.3 |

**Symbolic `k`, recurrence (b):**

| system | `N` = 50 | 100 | 200 | 400 | 1000 |
|---|---|---|---|---|---|
| `R_k`, Python `ZZ[k]` | 0.04 | 0.32 | 4.3 | | |
| `R_k`, flint `fmpz_poly` | < 0.01 | 0.02 | 0.25 | 4.6 | 176 |
| `R_k, S_k`, Python | 0.05 | 0.72 | 8.9 | | |
| `R_k, S_k`, flint | < 0.01 | 0.04 | 0.50 | 9.4 | |
| `L_k`, Python | 0.04 | 0.34 | 3.8 | | |
| `L_k`, flint | < 0.01 | 0.08 | 1.3 | 29 | |

**Realistic limit for symbolic `k`:** `N ≈ 200` with either backend (a few seconds), and
`N ≈ 400` with flint (5–30 s). Cost grows by about 12–20× per doubling of `N`: the
coefficient of `z^n` is a polynomial of degree about `n` in `k` with `O(n)`-bit
coefficients, and (b) does `O(N^2)` products of such polynomials. Symbolic `k` is meant for
small `n` (shapes in `k`, spot checks of closed forms). For `N = 1000`, use fixed `k`.
The docstring of `series_in_k` will say so.

## 7. Planned tests (`tests/test_series.py`)

Unless stated, each test runs on both backends (`backend` parametrised; flint skipped
if it is not installed).

- **Hand-made systems with known coefficients:**
  - linear `F = 1 + 2zF` (`2^n`);
  - Fibonacci `F = 1 + zF + z^2 F`;
  - rational coefficients `F = 1 + zF/2` (`Fraction(1, 2^n)`, checks the `Fraction`
    result type);
  - Catalan `F = 1 + zF^2` (`binomial(2n, n)/(n + 1)`, `n <= 200`);
  - Motzkin `F = 1 + zF + z^2 F^2` (three-term recurrence, `n <= 200`);
  - a 2-equation system `{F = 1 + zG^2, G = 1 + zF^2}` (both are Catalan);
  - zero coordinates `{F = z + G, G = G}` (warns at construction; `F = z`, `G = 0`);
  - signed `F = z − zF` (`(−1)^(n+1)`), and the two-step signed system
    `F = z + 1 − (F − 1)^2 (F − 2)/2` (`c* = 1`; residual check below).
- **Residual check for every system above:** `Phi(z, Y) − Y ≡ 0 mod z^(n+1)` for the
  returned `Y`, computed independently with SymPy, `n <= 30`.
- **Broda 2020 against brute force:** `R_k`, `S_k`, `L_k` (and the auxiliary `B`, `B̄`,
  `M`, `M̄`) for `n <= 7` and `k ∈ {1, 2, 3}`, against the enumerator. The enumerator
  moves from `dev/spikes/broda2020_enumerate.py` to a test helper, `tests/_enumerate.py`,
  with typed functions and no `main`; the spike file stays as it is.
- **Broda 2020 against the paper's closed forms:** `[z^n] R_k` from eq. (3) and
  `[z^n] S_k` from eq. (5), for `n <= 200` and `k ∈ {1, 2, 10, 50}`.
  - Both were checked in the spike session: eq. (3) matches exactly. Eq. (5) matches for
    even `n`, but **for odd `n` the paper's prefactor `(n − 1)/2^(n+2)` is off by a
    factor 2; the correct one is `(n − 1)/2^(n+1)`.**
  - Check by hand for `k = 1`, `n = 3`: the only trees with stars are `ε**` and `a**`,
    with 2 stars each, so `[z^3] S_1 = 4` (the enumerator agrees). The printed formula
    gives `(2/32)(2 + 30) = 2`.
  - The test uses the corrected prefactor and cites this as an erratum to eq. (5).
- **Symbolic `k`:** `series_in_k(system, 60)` evaluated at `k = 1..10` equals
  `series(system, 60, k=j)`, for `R_k`, `S_k` and `L_k`. Also the closed form
  `[z^5] L_k = 8k^2 (3k + 4)`.
- **API:** result types (all `int`, all `Fraction`, `sympy.Poly` over `ZZ`/`QQ` in `k`);
  read-only mapping; key order; the length `n + 1`; `n = 0`; error cases (`n < 0`,
  `bool`, `k` missing or extra, `k < 1`).
- **N = 1000** (marked `slow`):
  - `R_k` and `S_k` at `k = 50` against eqs. (3) and (5);
  - `L_k` at `k ∈ {2, 10, 50}`: both backends agree, and eq. (17) holds:
    `G(z, L_k(z)) ≡ 0 mod z^1001`, checked with exact polynomial arithmetic;
  - a generous time budget (5× the measured pure-Python time) to catch complexity
    regressions without being flaky.

**Slow marker.** Register `slow` in `pyproject.toml` (`--strict-markers` is on), and add
`-m "not slow"` to the default `addopts`, so a local `pytest` stays fast. CI adds a
**separate `Slow tests` job on every push and PR** that runs `pytest -m slow` on both
backends (two matrix entries, with and without `fast`). Measured cost is under a minute
in total, so a nightly schedule is not needed.

## 8. Docs plan (implemented in chunk 5)

- **API page.** Add `flanpy.series` and `flanpy.series_in_k` to the API section of
  `docs/index.md`, with NumPy docstrings that cite `system.md`'s definition of the
  solution, PSS Thm 6.3 / 8.10 and Broda 2020 eq. (3). Doctests: Catalan up to `z^6`,
  and `series_in_k` for `R_k` up to `z^3`.
- **Worked example.** A MyST-NB text notebook, `docs/examples/series.md`, executed in
  every docs build (`nb_execution_mode = "force"` and `nb_execution_raise_on_error`
  are already set). It shows:
  1. Catalan numbers: `System({F: 1 + z F^2})`, then `series(..., 10)`, compared with
     `binomial(2n, n)/(n + 1)` by an `assert` in a cell, so a mismatch fails the build.
  2. `R_k` for `k = 2`: the first 10 coefficients, compared with Broda 2020 eq. (3) and
     with the enumerator count for `n <= 7`. Then `series_in_k` for the first
     coefficients as polynomials in `k`.

  The page is linked from `index.md` in a toctree, with a citation to Broda 2020.
