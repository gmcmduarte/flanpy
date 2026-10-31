# CLAUDE.md — flanpy

Open-source Python package implementing the analytic-combinatorics toolchain used by our research group (Broda, Machiavelo, Moreira, Reis, and Guilherme), so that papers can ship a runnable script that reproduces their results, instead of Mathematica files.

This file holds the project's context, decisions and base rules. Build-order steps 1 (`System`, `series`), 2 (`KSet`), 3 (`elimination`), 4 (`singularity`), 5 (`newton`) and 6 (`transfer`) are implemented; the later steps are not. Update this file when decisions change.

## What the package does

Pipeline for counting problems whose generating functions are algebraic, possibly indexed by a parameter `k` (the alphabet size, an integer `k >= 1`):

1. **Symbolic method**: grammar -> polynomial system for generating functions; cost/multivariate series via the derivative in `u` at `u = 1`.
2. **Elimination**: Gröbner basis (Buchberger/F5B) and resultants, to get one algebraic equation `G(z, w) = 0`.
3. **Singularity analysis**: dominant singularity `rho` and value `a`, via `res_w(G, dG/dw)`; minimal polynomials `m_rho`, `m_a`; Case I (`w(rho) = a` finite) vs Case II (`w -> infinity`, use the reciprocal polynomial); uniqueness of the dominant singularity (Pringsheim).
4. **Local expansion**: which partial derivatives of `G` vanish at `(rho, a)` (resultant + gcd tests); Newton polygon -> exponent `alpha`; Puiseux expansion; solve for the leading constant `b` (or `c`).
5. **Transfer**: `[z^n] w(z) ~ -b/Gamma(-alpha) * rho^-n * n^(-alpha-1)` (Case I), `~ (1/c)/Gamma(alpha) * rho^-n * n^(alpha-1)` (Case II). Includes the square-root special case (quadratic `G`, closed form).
6. **Families in k**: `rho_k`, `a_k`, `b_k`, `c_k` as Puiseux expansions in `1/k`, plus limits as `k -> infinity`.

Reference papers (in Guilherme's Zotero library):
- Broda, Machiavelo, Moreira, Reis, *Analytic Combinatorics and Descriptional Complexity of Regular Languages on Average* (SIGACT News, 2020) — the full machinery for implicit curves.
- Broda, Machiavelo, Moreira, Reis, *A Hitchhiker's Guide to descriptional complexity through analytic combinatorics* (TCS 528, 2014) — simpler curves with square-root singularities and cost generating functions.
- Flajolet & Sedgewick, *Analytic Combinatorics* (2008) — the general theory (Thm VI.1 and Cor. VI.1, Pringsheim IV.6, algebraic coefficient asymptotics).
- Pivoteau, Salvy, Soria, *Algorithms for combinatorial structures: Well-founded systems and Newton iterations* (JCTA 119, 2012; arXiv:1109.2688) — well-founded systems and Newton iteration; used by step 1 (cited with arXiv v2 numbering).
- Pivoteau, Salvy, *Effective Asymptotics of Combinatorial Systems* (arXiv:2508.20008, 2025) — relevant to steps 4–6 (singularity, Newton–Puiseux, transfer); read for steps 4 and 5 (§§3–8; cited with arXiv v1 numbering).

## Decisions

- **Name**: `flanpy` (PyPI distribution name and import name: `import flanpy`).
- **Name history**: `pyflan` was rejected by PyPI on 2026-10-29 as too similar to `pyfian`; check any future name against look-alikes (l/i/1, o/0, separators) before using it.
- **Repository**: `git@github.com:gmcmduarte/flanpy.git` (GitHub, public; remote `origin`).
- **PyPI**: https://pypi.org/project/flanpy/ (0.0.1.dev0 released 2026-10-29, tag `v0.0.1.dev0`).
- **Docs**: https://flanpy.readthedocs.io/ (Read the Docs, builds `latest` on every push to `main` and on pull requests).
- **Author**: Guilherme Duarte (for `pyproject.toml`, `CITATION.cff` and licence headers). Add the other group members later if/when the group decides.
- **Python**: `requires-python >= 3.13`; CI also on the newest release.
- **Licence**: GPL-3.0-or-later (compatible with SymPy BSD, python-flint MIT, Sage GPL). Confirmed by the supervisors and the institution. Add an SPDX header to every source file.
- **Audience**: mainly our group, but external use is encouraged, so: stable public API, documentation, and proper packaging from the start.
- **Backend**: SymPy by default (pure Python, pip-installable). Optional Sage/Singular backend for heavy Gröbner computations. `python-flint` may be used for fast polynomial arithmetic when installed.
- **Gröbner computations**: use `sympy.groebner(..., order='lex', method='f5b')` with `k` as an ordinary polynomial variable ordered **last**. Measured on eq. (17) of Broda 2020: F5B 1.4 s with pure-Python ground types, 0.87 s with python-flint ground types; default Buchberger 109 s; `domain=QQ.frac_field(k)` did not finish in 250 s. Do not use the fraction-field domain for the elimination step.
- **k appears only in coefficients** of the curves (never in degrees or exponents) in the current framework.
- **Conditions on k**: `k` is the only parameter, so every condition is a sign/vanishing condition on univariate polynomials in `k`, i.e. a finite union of integer points and intervals, e.g. `{2} U [3,53] U {54} U [55, inf)`. Each result carries the set `KSet` of `k` where it is valid (exact real-root isolation); results combine by intersecting `KSet`s. Exceptional `k` (roots of leading coefficients, denominators, discriminants) are collected and handled by instantiation. If a case cannot be decided, say so explicitly; never guess.
- **Finding `alpha` and Puiseux expansions (validated by a spike, see `dev/spikes/np_spike.py`)**. *Superseded by the Step 5 block below* (kept for the history): the gcd test is one rule of a zero-test cascade, the points `(j-1, i)` are Broda's shifted columns, and "minus the minimum slope" holds only for a single edge (`newton.md` §2.1-2.2):
  - `alpha` needs no algebraic-number arithmetic. As in Broda 2020 (eqs. (21)-(27)), take the derivatives `d^i_z d^j_w G` at `(rho, a)`, decide which are non-zero with `gcd(derivative(z), m_rho(z))` over `Q(k)` (gcd = 1 -> non-zero; zero polynomial -> zero; otherwise inconclusive), then take the lower convex hull of the points `(j-1, i)`. Group support points by x and keep the minimum y before computing the hull. Reproduced `alpha = 1/2` for eq. (32).
  - Puiseux expansions in `1/k` (e.g. `rho_k`) use the standard Newton-polygon recursion: substitute `x = s^q`, `y = s^p (c + y1)`, solve the characteristic polynomial for `c`, recurse. With SymPy radicals this reproduced eq. (35) exactly (coefficients in `Q(sqrt 2)`) and `c_k ~ 4 k^(-1/2)` from eqs. (34)/(36).
  - Characteristic polynomials whose roots are not expressible in radicals: handled since step 5 by explicit real number fields `Q(θ)` (`_field.py`, behind the `_Field` protocol), not `sympy.roots`; see Step 5 below.
- **Exactness**: `rho`, `a`, `alpha` are exact algebraic data (minimal polynomial + isolating interval). Floats (`mpmath`) are only for evaluation and for cross-checking.
- **Rigor**: start with checks that emit warnings (e.g. uniqueness of the dominant singularity); certified proofs can come later.

### Step 1: `System` and `series` (design notes: `dev/design/system.md`, `dev/design/series.md`)

- **Public API**: `flanpy.System`; `flanpy.series(system, n, *, k=None)` (fixed `k`; `int`/`Fraction` coefficients of `z^0..z^n`, `n` included); `flanpy.series_in_k(system, n)` (`sympy.Poly` in `k`); exceptions `WellPosednessError` (a `ValueError`) with subclasses `IllPosedSystemError` (proven) and `UndecidedSystemError`; warning `ZeroCoordinateWarning`. Results are plain Python/SymPy objects whatever the backend.
- **Domain of k**: integers `k >= 1`. `System.specialize` and `series(..., k=...)` reject other values with the same errors. A system is non-negative if it is so on that domain, tested by substituting `k = 1 + t` and requiring all coefficients in `(F, z, t)` to be `>= 0` (sufficient, no sampling).
- **Well-posed** means: the iteration `F^(0) = 0`, `F^(n+1) = Phi(z, k, F^(n))` converges `z`-adically, for symbolic `k`; its limit is the solution `series` computes. Criterion (checked with `k` symbolic, never by sampling): drop zero coordinates (support propagation), the constant terms stabilise, the Jacobian at `z = 0` is nilpotent. For non-negative systems this is a decision procedure (Pivoteau–Salvy–Soria 2012, Thm 3.6, 5.5, 5.7, **arXiv:1109.2688v2 numbering**), with an exact bound of `m` steps for the constant terms; a failure is proven ill-posed for every `k >= 2`, and for `k = 1` unless a coefficient vanishes there. Signed systems: a passing check proves well-posedness (own argument, `system.md` §5.4); a failure is "undecided" except when the constant terms cycle (proven ill-posed); a private cap bounds the iteration. A pass for symbolic `k` implies a pass for every `k`, so `System` needs no `KSet`.
- **Zero coordinates** (unknowns identically 0) are accepted with a `ZeroCoordinateWarning`, kept as 0 (PSS exclude them; specialising can create them).
- **`series` algorithm**: the recurrence `(I − J0) y_n = r_n` from the stabilised constant terms (`series.md` §2–3), `O(P N^2)` coefficient operations. One code path over a private coefficient-ring interface (`_rings.py`): python-flint (`fmpz`, `fmpq`, `fmpz_poly`, `fmpq_poly`) when installed, otherwise pure Python (`int`, `Fraction`, SymPy `ZZ[k]`/`QQ[k]`). Measured at `N = 1000` (Apple M1): `L_k`, `k = 50`, 1.3 s with flint, 9.4 s pure Python. Symbolic `k` is practical up to `N ≈ 200` (≈ 400 with flint). Newton iteration (PSS Thm 8.10) is faster from `N ≈ 2000` and can be added later behind the same interface.
- **Erratum, Broda 2020 eq. (5)** (pending confirmation with the authors): for odd `n` the prefactor should be `(n − 1)/2^(n+1)`, not `(n − 1)/2^(n+2)` (e.g. `k = 1`, `n = 3`: 4 stars, the printed formula gives 2). The tests use the corrected form and cite the erratum.

### Step 2: `KSet` (design note: `dev/design/kset.md`)

- **Public API**: `flanpy.KSet`, an immutable set of integers `k >= 1` stored as canonical sorted, disjoint, non-adjacent integer intervals (last one possibly `[n0, ∞)`); printed as `[1, 53] ∪ {54} ∪ [55, ∞)`. Set operators as for `frozenset` (`~` is the complement in `k >= 1`), `tail_start` (least `n0` with `[n0, ∞)` inside, for limits `k -> ∞`), `is_finite`, `finite_part`; iteration only when finite.
- **Construction**: `KSet(intervals)`, `KSet.of(*ks)`, `KSet.all()`, and `KSet.where(p, rel, *, k, den=None)` for `p(k) rel 0` with `p` in `Q[k]`. Exact: integer roots first, then VAS isolation (`sympy.Poly.intervals`) and refinement for the floors of the other roots, then exact signs at integers. SymPy only (python-flint 0.9 has no exact real-root isolation); memoised per polynomial.
- **Denominators**: `den` is passed separately and never cancelled against `p`: its roots are always excluded (they record a division in the derivation). Rational `Expr`s are rejected. Cancel first for the lowest-terms reading.
- **`KCases`** (piecewise results: disjoint `KSet` pieces -> values, with `uncovered` explicit and `KeyError` there): implemented and tested, private in `flanpy._kset` until step 3 returns one.
- `System` and `series` do not use `KSet` (a generic well-posedness pass holds for every `k`).
- **`KCases`** became public in step 3, with `tail` (the piece containing all large `k`).

### Step 3: `elimination` (design note: `dev/design/elimination.md`)

- **Public API**: `flanpy.eliminate(system, w, *, k=None, method="auto") -> Curve` (fixed `k`), `flanpy.eliminate_in_k(system, w, *, method="auto") -> KCases[Curve]` (symbolic `k`, covering every `k >= 1`), `flanpy.Curve` (`polynomial`, `w`, `z`, `k`, `kset`, `system`, `as_expr`, `specialize`). Curves are created only by these functions.
- **`G`** is the minimal polynomial of the series `w(z)` over `Q(k)(z)`, normalised: `Z[k]` coefficients, primitive over `Z[k]`, positive leading coefficient in lex `w > z > k`. Broda 2020 eq. (17) comes out exactly (unit 1). `G` is absolutely irreducible.
- **Method**: only the unknowns `w` depends on take part, with zero coordinates set to 0; one unknown means no elimination; two use a resultant; otherwise lex F5B with the other unknowns first (system order) and `k` last. If the elimination ideal has no element in `w`, the ideal is saturated by the Jacobian determinant. That is a fallback only (did not finish in 300 s for `L_k`). Variable order matters (`L_k`: 0.87 s vs 7.1 s reversed).
- **Factor selection is exact**: `N = max_{i≠j} (deg_w h_i deg_z h_j + deg_w h_j deg_z h_i)` terms separate the factors (`Res_w(g, h) = A g + B h`). For symbolic `k` the test runs at `k = 2, 3, ...` (a single passing factor is proven right), and falls back to series in `Q[k]`.
- **Exceptional `k`**: the candidates `X_GB` (leading coefficients of **all** Gröbner basis elements; Kalkbrener 1997 Thm 3.1, generalising Gianni 1987), `X_cont`, `X_deg` (degree drop in `w`), `X_sqf` (not squarefree) and `X_prim` (content in `z` at `n`, computed exactly via `Res_z`) are recomputed at `k = n`. A candidate merges back into the generic piece when the renormalised `G(·, ·, n)` equals the recomputed curve with full degree. For `L_k`, `k = 54` is a candidate (Kalkbrener's criterion **fails** there), yet it merges: one piece `[1, ∞)`.
- **Documented gap**: on the generic piece, irreducibility of `G(·, ·, n)` is not certified (finitely many failures, Noether 1922). `Curve.specialize(n)` certifies one `n`. There is no gap when `deg_w G = 1`.
- **Backends**: private `_GroebnerBackend` protocol with the SymPy backend only; no public `backend=` until a second backend exists (Singular / Sage planned; Sage is installed on the maintainer's machine).

### Step 4: `singularity` (design note: `dev/design/singularity.md`)

- **Public API**: `flanpy.AlgebraicReal(polynomial, index)` (minimal polynomial over `Q` + index among the real roots, normalised silently; exact `==`, `<`, `sign`, `evalf`, `as_expr`); `flanpy.KRoot(polynomial, positive_index, *, k)` (the `positive_index`-th **positive** root of `f(x, k)`, `f` irreducible over `Q(k)`; `at(n)` exact; not `index`, which counts real roots); `flanpy.singularity(curve, *, k=None) -> Singularity[AlgebraicReal]`; `flanpy.singularity_in_k(Curve | KCases[Curve]) -> KCases[Singularity]` (pieces cover each input curve's `kset`; `KCases.flat_map` for a `KCases` input); `flanpy.Singularity` (generic in its root type: `curve`, `rho`, `case` `"I"`/`"II"`, `a` (`None` in Case II), `period`, `unique` (`True`/`False`/`None`), `k`, `kset`, `specialize(n)`); `UndecidedSingularityError` (a `ValueError`), `SingularityWarning`. `m_ρ`, `m_a` are `rho.polynomial`, `a.polynomial`.
- **Scope**: non-negative closures only (the unknowns `w` depends on; `k = 1 + t` test for symbolic `k`, sufficient, not necessary); a signed closure raises `UndecidedSingularityError` listing the candidates. A polynomial series raises `ValueError`.
- **`ρ`**: Pringsheim (FS Thm IV.6); candidates = roots of `lc_w G · disc_w G` (FS Lemma VII.4); a candidate need not be a singularity (`L_k`: the positive root of `q_k` lies below `ρ_k` for every `k`). `ρ` is located by binary search with **certified rational comparisons** on the closure: `q < ρ` via a rational super-solution `H(q, u) <= u`; `q > ρ` via a sound lower iterate (rounded-down Newton and Kleene steps, clamped) with `Λ(J) >= 1` (PS 2025 Prop. 3.11, Thm 4.1, Lemmas 3.5, 3.6, 6.7, B.5). Soundness and termination (with rounding) are in the Notes of `singularity`.
- **Case I/II and `a`**: exact window test at a rational `x0 < ρ` (separators of the roots of `Res_z(m_ρ, G)`, certified enclosure of `w(x0)`); Case I gives `a` exactly, checked against Broda eq. (19).
- **Uniqueness** (§5.3, §5.5): `period > 1` -> `False` (PS §7.1 periods); exact circle test proves no other candidate on `|z| = ρ` -> `True`; **every recursive strongly connected component of the closure has period `p_C = 1`** -> `True` (purely combinatorial, no need to know which components are critical; nonlinear `p_C` = coordinate period, PS Thm 6.2; linear `p_C` = **cycle period** of `B = ∂H_C/∂Y_C`, the gcd of closed-walk exponents; the coordinate period is not enough for linear components, PS Example 23); otherwise `None` + `SingularityWarning`. Proof: induction over the components (Wielandt PS Lemma 6.7, daffodil Lemma 6.5), in the Notes of `singularity`; it relies on the BBY periods being exact (no cancellation). Step 6 must refuse `unique != True`. Still `None`: a periodic component anywhere, e.g. `W = z^2 + T` (truth `False`), PS Example 23 (truth `False`).
- **Symbolic `k`**: on a piece, `ρ_k` = `KRoot(f, j)` with `f` a factor of the candidates, Case I `a_k` = `KRoot(g, j')` with `g` a factor of `Res_z(f, G)`; case, period, `unique` constant. Pieces come from the real roots `>= 1` of the **projection set** (`P1`, then `P2(f)` for the factor of `ρ_k`; §6.2), exact via `KSet`'s breakpoints; one sample per interval; every integer root instantiated; equal descriptions merged; nothing dropped. The proof (continuity of `ρ_k`, constancy of the case and `a_k`) is in the Notes of `singularity_in_k`.
- **`L_k` result**: one piece `[1, ∞)`: `ρ_k = KRoot(p_k, 0)`, Case II, period 1, `unique`, for every `k >= 1` (same for `F_k`, Case I). At `k = 108`, `p_108 = 4(27z + 1)(972z^2 − 1)` puts `−ρ_108` on the circle; the component rule proves it regular (`{B, B̄}` nonlinear and `{M, M̄}` linear, both period 1; `{B, B̄}` is itself critical, `ρ_B = ρ_k`). Cuts at 35.52 (`17 + 7√7`), 42.x, 54, 58.x, 108, 112.x, 146.x; `k = 54` is a cut and merges (`p_54 = 2(45z − 2)(18z + 1)^2`, `m_ρ = 45z − 2`). Times: `singularity_in_k(L_k)` 0.9 s after elimination; `L_k`, `k = 1..60`, fixed `k`, 9.6 s (flint).
- **python-flint 0.9**: `fmpz_poly.complex_roots()` returns certified Arb balls (fast); `fmpz_poly.real_roots()` raises `NotImplementedError`, so exact real isolation stays with SymPy. Flint is at most an optional accelerator for the numerical step of the circle test (not used yet).

### Step 5: `newton` (design note: `dev/design/newton.md`)

- **Public API**: `flanpy.local_expansion(singularity, *, k=None) -> LocalExpansion[AlgebraicReal]`; `flanpy.local_expansion_in_k(Singularity[KRoot] | KCases[Singularity]) -> KCases[LocalExpansion]` (pieces cover each input piece's `kset`, `uncovered` propagated); `flanpy.LocalExpansion` (`singularity`, `alpha`, `ramification` κ, `b` (Case I) / `c` (Case II), `terms(n)` (fixed `k` only: `n` consecutive exponents in `s = 1 - z/ρ`, step `1/κ`, from 0 (I) or `-α` (II), exact, cached), `specialize`, `k`, `kset`). Broda's conventions: `w = (analytic) - b s^α + ...` (eq. (20)), `1/w = c s^α + ...` (eq. (31)); in Case I `α` is the first non-integer exponent with a non-zero coefficient.
- **Coefficient field**: `_field.py`, real number fields `Q(θ)` (`θ` an `AlgebraicReal`): arithmetic modulo `m_θ` (flint `fmpq_poly` or SymPy `QQ[x]`, by the `_rings` backend selection), exact zero test and sign, enclosures, `to_real` (squarefree part of the characteristic polynomial), real roots over `Q(θ)` by Trager norms (Trager 1976; Loos 1983), tower flattened. Only real fields are needed: every Puiseux coefficient of `w` is real (Lemma R). SymPy radicals are not used (heuristic zero test; `ρ` itself is often non-radical: the S5 tree `T = z(1 + T^2 + T^5)`). Pitfall: to decide `x = target` refine only `x`'s enclosure.
- **Local problem**: Case I `G(ρ(1 - s), a + Y)`, Case II the reciprocal polynomial (eq. (29)); `a ∈ Q(ρ)` from `gcd(G(ρ, ·), G_w(ρ, ·))` (in every example). Zero tests of the hull entries by a cascade (zero polynomial; remainder mod `m_ρ`; Broda's gcd test; the field). Polygon on the columns `j <= r` only (`r` = multiplicity of the point).
- **Which branch is `w`**: the real branches (real roots of each edge's characteristic polynomial); the **rank test** is always on (step 4's window data: rank of `w(x0)` among the real roots of `G(x0, ·)` in its window = rank of `w` among the branches sorted as `s → 0⁺`); the sign rule (`b > 0`, `c > 0`, by monotonicity) and the window count are assertions.
- **PS 2025 Thm 8.5** re-checked for our scope (polynomial systems, non-negative rational coefficients): its proof uses no integrality; the regular nonlinear case, implicit in the printed proof, is filled in the Notes of `local_expansion`. So Case I has `α = 1/κ`, `κ = 2^m >= 2`, `b > 0`: asserted (`RuntimeError`). Cube roots or `(3/2)`-type singularities cannot occur in Case I (tested at the private curve level); Case II allows `α >= 1` (poles `α = 1`, `(1 - 4z)^{-3/2}`).
- **Symbolic `k`**: the constant is a `KRoot` **relative to `ρ_k`** (`KRoot(Φ, j, k=k, over=rho_k)`: the `j`-th distinct positive root of `Φ(y, ρ_k, k)`; `Φ` need not be irreducible; reduced modulo `f` over `Q(k)`). An absolute `KRoot` of the norm would split `L_k` at 54/55 (a conjugate of `ρ_k` adds a positive root). `a_k = -s_10/s_11 (ρ_k)` (first subresultant). Cut set: step 4's `P1 ∪ P2(f)` and `P5` (`Res_z(f, s_11)`, `Res_z(f, lc_w G)`, `Res_z(f, q_ij)` on the generic hull, `Res_z(f, disc Φ_E)`); proposition and proof in the Notes of `local_expansion_in_k`. Gaps (`NotImplementedError` naming the piece; fixed `k` works): `a_k` not a rational function of `ρ_k`, a first-level tie.
- **Results**: Catalan `b = 2`; Motzkin = FS Example VII.18's printed terms; HG `A_k`, Broda `R_k` closed forms for `k = 1..20`; `L_k`: one piece `[1, ∞)`, Case II, `α = 1/2`, `c_k` = eq. (34) exactly (`c_54 = √15/7`), `at(n)` = fixed `k` for `k = 1..60`, 108; `F_k`: one piece, `b_k > 0`, `b_k/√k → 1` (`b_54 = 2√15`); `K = kU/4 + K^2`: pieces `{1}`, `{2}` (`α = 1/4`), `[3, ∞)`, cut at 2 by the hull vertex `(2, 0)`. At `n = 1000` the transfer of the exact terms is within `1e-8..4e-6` (`L_k` 5 terms `8.9e-11`). Times: `local_expansion` 3–30 ms after step 4; `local_expansion_in_k` `L_k` 0.68 s, `F_k` 0.83 s (flint and pure Python alike).

### Step 6: `transfer` (design note: `dev/design/transfer.md`)

- **Public API (fixed `k`, chunk 2)**: `flanpy.asymptotics(expansion, *, k=None) -> Asymptotics[AlgebraicReal]` (`k=n` on a symbolic piece specializes it); `flanpy.Asymptotics` (`expansion`, `rho`, `exponent` β, `leading` C, `terms(count)`, `error_exponent(count)`, `evalf(n, *, count=1, dps=15)`, `as_expr(n, *, count=1)`, `k`, `kset`); `flanpy.GammaConstant` (`algebraic()` = the `AlgebraicReal` part, `gamma` = pairs `(ν, m)` for `Π Γ(ν)^m`, `evalf` = the full value, `as_expr`). `[z^n] w = ρ^{-n} Σ_j D_j n^{-1-p_j} + O(ρ^{-n} n^e)`, `p_j = e_0 + j/κ`, `e_0 = α` (I) / `-α` (II).
- **Public API (chunk 3)**: `flanpy.ratio(A, B) -> Ratio` (`numerator`, `denominator`, `exponent` = `β_A - β_B`, `limit` = `C_A/C_B` exact, `terms(count)`, `evalf(n, *, count=1, dps=15)`, `specialize`, `k`, `kset`); `flanpy.asymptotics_in_k(LocalExpansion[KRoot] | KCases) -> KCases[Asymptotics[KRoot]]` (one piece per step 5 piece, no new cuts, `uncovered` kept); `Asymptotics.specialize`; `GammaConstant.rational`, `.factors`, `.at(n)`, `*`, `/` (also by `int`/`Fraction`). Ratios in `k`: `cases_a.combine(cases_b, ratio)`; `KCases.combine` now calls `f` only on non-empty intersections (it dropped empty pieces anyway; needed because pieces with no common `k` may have different `ρ_k`).
- **Ratios**: same `ρ` by exact equality (`AlgebraicReal`, or structural `KRoot`), else `ValueError` (the factor `(ρ_B/ρ_A)^n` is not represented); also refused: fixed `k` vs symbolic, different `k` (fixed), no common `k` (symbolic). `q_j` by series division in one number field (`_common_field`: real roots of `m_θ2` over `K1`, step 5's Trager machinery). `q_j` is one `Γ` monomial when every non-zero `B_i` has `B_0`'s `Γ` class; otherwise `terms` raises `NotImplementedError` from that index (a sum of `Γ` monomials, e.g. a pole plus a square root in the denominator), and `evalf` (numeric series division) still works.
- **Symbolic `k` constants**: the algebraic part is the monomial `rational · Π factors` (`KRoot`s with integer exponents; `1/c_k` is `(c_k, -1)`, no new `KRoot`): Case I `α b_k/Γ(1 - α)`, Case II `r c_k^{-1}/Γ(ν)`. `algebraic()`, `evalf`, `as_expr` raise `ValueError` for symbolic `k`; `at(n)` evaluates the monomial exactly (`_monomial`: products in one number field). At a fixed `k`, `factors` is `()` (rational `A`, `rational = A`) or `((A, 1),)` (`rational = 1`). Further terms for symbolic `k` are out of scope (step 5 has no Hensel lifting over `Q(k)(ρ_k, b_k)`).
- **Γ key** (for `==`, `hash`): `Γ(1/2) = π^{1/2}`, `Γ(ν)` with `ν > 1/2` by reflection `π/(sin(πν)Γ(1 - ν))`, `sin(πν)` exact (`_sin_pi`: `cos(2πa/N)`, index from the exact order of its conjugates `cos(2πa'/N)`), moved into the algebraic part; so `Γ(1/4)Γ(3/4) == √2 π`. Decides equality for `κ <= 4` (known algebraic independence of `π`, `Γ(1/4)`).
- **Results**: Broda Thm 3.2 / eq. (38) / Table 1: `[z^n] L_k / (n [z^n] F_k) → 2/(b_k c_k)`, exact limits `0.417256345028`, `0.443252417412`, `0.465746500119` (`k = 2, 10, 50`; cubic minimal polynomials), `7/15` at 54, `0.474216576937` at 108 (agree with review's Richardson extrapolation to 12 digits); `n^{-1}` coefficients `0.421941, 0.450121, 0.473185, 0.474074, 0.481125`; at `n = 1000` the limit plus the `1/n` term is within `2e-8`. Table 1's simulation (`0.42, 0.44, 0.47`) fits the exact limits better than eq. (38)'s approximation at `k = 2` (`0.375`). Eq. (8): `[z^n] S_k / (n [z^n] R_k) = ρ_k (1 + 1/(2n) + ...)` exactly (`k = 1..10`). Symbolic: `L_k`, `F_k`, `R_k`, `A_k` one piece; `L_k/F_k` = `2 b_k^{-1} c_k^{-1}` on `[1, ∞)`, `at(n)` = fixed `k` for `k = 1..60`, 108; `K = kU/4 + K^2`: `{1}`, `{2}` (`Γ(3/4)`), `[3, ∞)`.
- **Naming**: a number of terms is `count`, never `n` (`n` is the coefficient index), and counts the `D_j` **including zeros** (Catalan `count=5` = 3 non-zero terms). Step 5's `LocalExpansion.terms(count)` was renamed likewise (pre-0.1.0, no deprecation).
- **Refusals**: `unique is False` -> `ValueError` naming the period (FS Thm VI.5 needed; periodic `w = z^r V(z^p)` is the cheap route later, transfer.md §1.2); `unique is None` -> `ValueError` "undecided". No new exception class.
- **Method**: each Puiseux term `d s^x` transfers by FS Thm VI.1, `e_l` exact in `Q[a]` from Stirling's series (DLMF 5.11.8, Bernoulli polynomials, recurrence `l e_l = Σ j s_j e_{l-j}`, cached); `D_j` collects one class of `x` mod 1, so `D_j = A_j/Γ(ν_j)` with `1/Γ(-x) = r/Γ(ν)` by the recurrence; `A_j` computed in step 5's branch field (`LocalExpansion._field_terms`), exact. `D_j = 0` whenever `p_j` is a non-negative integer (analytic terms; poles terminate); `error_exponent` skips those, never accidental zeros; `None` when `κ = 1` (remainder `O((ρ + ε)^{-n})`). Case I needs `count + 1` Puiseux terms, Case II `count`.
- **Γ form**: stored as FS/Broda print it, `Γ(ν)`, `0 < ν < 1` (`α = 1/4`: `1/(8Γ(3/4))`); the reflection key is only for `==` and `hash` (below). `as_expr`: radicals for degree <= 2, `CRootOf` above. `O`-terms only (explicit bounds later, transfer.md §4).
- **Numerics**: `evalf` is the `n`-expansion (as accurate as step 5's Γ-ratio helper at equal Puiseux depth, and what the papers print; the helper is gone). `ρ` at `dps + len(str(n)) + 10` digits. At `n = 1000` with 1/2/3 non-zero terms: Catalan `1.1e-3/1.1e-6/1.1e-9`; `F_k` (`k` in {2, 10, 50, 54, 108}) `<= 8.0e-4/8.4e-7/8.6e-10`; `L_k` `<= 2.4e-4/3.9e-8/3.6e-11`.

## Planned layout

Flat private modules (`_name.py`), re-exported from `flanpy`, as in steps 1–5; a
subpackage only where a design note argues for one.

```
src/flanpy/
  _system.py     System (F = Phi(z, k, F)) and its well-posedness check; re-exported from flanpy
  _series.py     series / series_in_k: coefficients of a System; re-exported from flanpy
  _rings.py      private coefficient rings: pure-Python and python-flint backends
  _kset.py       KSet (sets of k >= 1 from sign conditions) and KCases (piecewise
                 results); both re-exported from flanpy (design: dev/design/kset.md)
  _curve.py      Curve (the curve G(z, w, k) = 0 of one coordinate), normalisation and
                 the exact factor selection; re-exported from flanpy
  _elimination.py  eliminate / eliminate_in_k, resultants, Groebner backend protocol
                 (SymPy only), exceptional k (design: dev/design/elimination.md)
  _doctests.py   method_doctests: methods of public classes for a module's __test__
  _algebraic.py  AlgebraicReal (exact real algebraic numbers: minimal polynomial + index,
                 cached isolating interval), KRoot (a positive root of f(x, k), or of
                 Φ(y, ρ_k, k) relative to a KRoot ρ_k), root
                 isolation, the circle test; re-exported from flanpy
                 (design: dev/design/singularity.md)
  _singularity.py  Singularity, singularity / singularity_in_k: certificates, window
                 test (Case I/II, a), period, uniqueness, projection set in k
  _field.py      real number fields Q(θ) (exact arithmetic, signs, to AlgebraicReal),
                 real roots over Q(θ) by Trager norms (design: dev/design/newton.md §4)
  _newton.py     LocalExpansion, local_expansion / local_expansion_in_k: local problem,
                 Newton polygon, real branches, rank test, Hensel lifting, cut set in k
                 (design: dev/design/newton.md); re-exported from flanpy
  _transfer.py   GammaConstant, Asymptotics, asymptotics, asymptotics_in_k, Ratio, ratio:
                 transfer of the local expansion (FS Thm VI.1/VI.4/VII.8, Broda Thm 3.2,
                 HG Prop. 4), e_l polynomials, Γ classes and the reflection key, ratios
                 in one number field, monomials in KRoots for symbolic k
                 (design: dev/design/transfer.md); re-exported from flanpy
  _families.py   expansions in 1/k
  _symbolic.py   grammar -> equations, bivariate cost series (d/du at u=1)
  _automata.py   thin regex/NFA front-end (cost vectors c_empty..c_star); optional
tests/
docs/            index, API reference, guide/ (one executed user-guide page per module)
examples/        running examples, jupytext percent scripts (`python examples/<name>.py`)
                 that are also executed docs pages (docs/examples -> ../examples):
  catalan.py     no parameter: F = 1 + z F^2, to rho = 1/4, alpha = 1/2 and the
                 transfer [z^n] F ~ 4^n n^(-3/2) / sqrt(pi) (step 6)
  broda2020.py   with k: Broda 2020 R_k, S_k, L_k, to eqs. (35)-(38) and Table 1 (step 7)
```

### Build order

1 `system` + `series` → 2 `kset` → 3 `elimination` → 4 `singularity` (+ exact algebraic reals) → 5 `newton` → 6 `transfer` → 7 `families` → 8 `symbolic` → 9 `automata` (optional).

- Release 0.1.0 after step 6 (first end-to-end run: equations → asymptotics for fixed `k`). Set up PyPI trusted publishing before that release.
- Each module: first a quadratic / square-root example (Hitchhiker's Guide style), then the Broda 2020 case as the real acceptance test.

## Acceptance tests from the papers

These are the ground truth; the package is not "done" for a feature until its test passes.

- Broda 2020: eq. (17)-(18) (cubic for `L_k`, `p_k(z)`, discriminant of `p_k` is `32 k^2 (k-54)^3`); eq. (32) gives `alpha = 1/2`; eq. (35)-(38) (`rho_k`, `c_k`, `b_k`, limit as `k -> infinity`); Table 1 (ssnf, `k` in {2, 10, 50}, `n = 1000`).
- Broda 2020 section 2: `R_k(z)`, `S_k(z)`, eq. (2)-(8), `[z^n] S_k / (n [z^n] R_k) -> rho_k`.
- Hitchhiker's Guide: square-root singularity, `Delta_k`, `rho_k`, cost generating functions; Table 3 (averages vs `k` for the Thompson, SSS and epsilon-follow constructions).

## Working rules

- **Verify, don't assert.** Run the code and the tests before saying something works. Every symbolic result for general `k` is spot-checked by instantiating `k = 1..N`, exactly or with `mpmath`.
- **Exact first.** No floating point inside the algebraic pipeline. Numerics only at the edges and in tests.
- **Be explicit about conditions.** A result valid only for some `k` must say which (`KSet`); a case that is undecided must be reported, not silently dropped.
- **Public API is a contract.** Keep it small and typed; anything not documented is private (leading underscore). Deprecate before removing. Use semantic versioning.
- **Docs and examples with every feature**: docstrings (state the theorem or paper equation implemented). Every build-order step extends both running examples (`examples/catalan.py`, `examples/broda2020.py`) with a section that reproduces that step's results (asserts against the paper's numbers) and updates their Status cell.
- **Examples for the public API**: every public, non-trivial function, method and property has a self-contained, doctested NumPy `Examples` section (own imports and `symbols(...)`; reuse Catalan `F = 1 + z F^2` and Broda 2020 `R_k`; show the output; exceptions as tracebacks with the real message). Enforced in CI by `tests/test_docstrings.py` (numpydoc EX01 over `flanpy.__all__`, with an allowlist of trivial members and the reason for each).
- **User guide**: each module gets a short, executed user-guide page in `docs/guide/` (API usage; the paper reproductions stay in the running examples).
- **Cite sources in code**: reference the paper and equation number next to each algorithm (e.g. "Broda et al. 2020, eq. (27)").
- **Small, tested steps.** One module at a time, with its acceptance test from the list above.
- **Commit hygiene**: clear messages, one logical change per commit, no generated artifacts or large data files in the repo.
- **CI after every push**: check CI (`gh pr checks`) after every push, and only report when it is green or the failure is explained.
- **Never merge PRs**: stop when CI is green and the PR is ready for review; Guilherme merges.
- **Reports**: compact, at most ~40 lines, one line per item, point to design notes instead of repeating them, full detail only for PROBLEMS and QUESTIONS.

## Tooling

All configuration lives in `pyproject.toml` (build backend `hatchling`, `src/` layout). Version is static in `pyproject.toml`; `flanpy.__version__` reads it from the installed metadata.

- **Extras**: `fast` (python-flint), `dev` (pytest, hypothesis, ruff, mypy, numpydoc, build, twine), `docs` (sphinx, shibuya, myst-parser, myst-nb, jupytext, sphinxcontrib-bibtex).
- **Types**: `mypy --strict` (chosen over pyright: Python-native, no Node dependency). Tests are type-checked but may omit annotations.
- **Lint/format**: `ruff` with pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, RUF and pydocstyle (NumPy convention). `dev/spikes/` is excluded (throwaway code).
- **Tests**: `pytest` with `--strict-markers --strict-config --doctest-modules`; collects `tests/` and the docstrings in `src/flanpy`; warnings are errors. Tests marked `slow` (N = 1000) are deselected by default (`-m "not slow"` in `addopts`); run them with `pytest -m slow`. Test helpers (e.g. `tests/_enumerate.py`, the brute-force enumerator for Broda 2020) are importable because `tests` is on pytest's `pythonpath`. Property-based tests use `hypothesis` with a derandomized profile (`tests/conftest.py`). Doctests of a private module import its names from that module, not from `flanpy`: when pytest (importlib mode) collects a module alone, it executes it a second time, and the re-exported classes then differ. Public classes set `__module__ = "flanpy"`. pytest then collects their class docstrings and properties through `src/flanpy/__init__.py`. It misses their methods: doctest matches a function to the module whose dict is its `__globals__` (the private module), where the class is skipped. So each private module sets `__test__ = method_doctests(<classes>)` (`_doctests.py`). That lists only the functions, classmethods and staticmethods, so no docstring runs twice. To check, compare `pytest --collect-only -q src` before and after a change: no item may be lost.
- **SPDX**: every `.py` file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- **CI**: `.github/workflows/ci.yml` runs ruff, mypy, pytest and `python -m build` on Python 3.13 with the `fast` extra (flint backend) and on `3.x` without it (pure-Python backend; each job checks which backend it got), a `Slow tests` job (`pytest -m slow`, both entries) on every push and PR, plus a docs job (`sphinx-build -W` for HTML, executing the MyST-NB examples, then the `doctest` builder). Read the Docs config: `.readthedocs.yaml` (`fail_on_warning: true`).
- **Repo files**: `README.md`, `CITATION.cff`, `CONTRIBUTING.md`, `docs/refs.bib` (DOIs checked against Crossref; arXiv-only entries against the arXiv API).

Commands (from the repo root, in a Python 3.13 venv):

```
pip install -e ".[dev,docs]"         # add ,fast for python-flint
pytest                               # tests + doctests (slow tests deselected)
pytest -m slow                       # N = 1000 tests
ruff check . && ruff format --check .
mypy                                 # strict, src + tests
sphinx-build -W --keep-going docs docs/_build/html
sphinx-build -W --keep-going -b doctest docs docs/_build/doctest
python -m build                      # sdist + wheel into dist/
python -m twine check dist/*         # release metadata check
```

## Releasing

Manual for now (until trusted publishing is set up). From a clean `main` that CI has passed:

1. Build: `rm -rf dist && .venv/bin/python -m build`.
2. Check: `.venv/bin/python -m twine check dist/*`.
3. Upload: `.venv/bin/python -m twine upload dist/*`. Credentials come from the `[pypi]` section of the maintainer's `~/.pypirc` (username `__token__`, project-scoped token); never read or print that file. Use `python -m twine`, not the `twine` script, which can carry a stale interpreter path.
4. Tag the uploaded commit: `git tag -a v<version> <commit> -m "flanpy <version>"` and `git push origin v<version>`.
5. Bump the version in `pyproject.toml` and `CITATION.cff` to the next `.devN` in a separate commit.

### Documentation (decided)

- **Sphinx** with the **Shibuya** theme. Not MkDocs Material: it entered maintenance mode in Nov 2025 (security fixes only until Nov 2026), and MkDocs has a weaker cross-reference ecosystem.
- **MyST-Parser**: write pages in Markdown. **MyST-NB**: executed notebooks, run as part of the docs build; the running examples are jupytext percent scripts read through `nb_custom_formats` (`jupytext` is in the `docs` extra).
- **MathJax** for equations, with numbered equations that can be cited.
- **`autodoc`** with NumPy-style docstrings for the API reference (`napoleon`).
- **`intersphinx`** to link to SymPy / Python / mpmath docs.
- **`sphinxcontrib-bibtex`** to cite the papers from a `.bib` file.
- **`doctest`** (and executed notebooks) so the docs fail to build if a paper's number stops reproducing.
- Hosting: Read the Docs (decided 2026-10-29; the repository is public), configured by `.readthedocs.yaml`. CI also builds the docs and runs the Sphinx doctest builder. HTML is the primary output.
- Docs grow with the code: every feature lands, in the same change, with its docstring (naming the theorem or paper equation implemented) and doctested `Examples` section, its user-guide page in `docs/guide/`, and a new section in both running examples.

## Open items

- [x] Tooling: pytest, ruff, mypy (strict) set up. Shibuya 2026.7.12 builds cleanly with Sphinx 9.2.0 on Python 3.13 (`-W`, no warnings); tests and mypy also pass on Python 3.15.
- [x] Read the Docs: project `flanpy` imported 2026-10-29; https://flanpy.readthedocs.io/ (pull-request builds enabled).
- [x] PyPI: name reserved with `0.0.1.dev0` on 2026-10-29; https://pypi.org/project/flanpy/.
- [x] PyPI token: project-scoped token for `flanpy` in `~/.pypirc` `[pypi]` (done by Guilherme, 2026-10-29).
- [x] Build-order step 1 (`system` + `series`): done 2026-10-29 (PR #1).
- [x] Build-order step 2 (`kset`): done 2026-10-30 (PR #4).
- [x] Build-order step 3 (`elimination`): done 2026-10-30 (PR #5).
- [x] Build-order step 4 (`singularity`): done 2026-10-30 (PR #6).
- [x] Build-order step 5 (`newton`): done 2026-10-31 (PR #7).
- [x] Build-order step 6 (`transfer`): done 2026-10-31 (PR #8).
- Step 7: `2/(b_k c_k) ∈ Q(ρ_k)` at every sampled `k` (it is `-ρ'(1)/ρ(1)` for the letter-marking parameter), so eq. (38) may need only `ρ_k`'s expansion (eq. (35)): reduce the limit to `A(ρ_k, k)`, `A ∈ Q(k)[z]`, by `A^2 ≡ 4 D_b D_c/(N_b N_c) (mod f)` (transfer.md §7.3).
- Broda 2020 §3.4 prints `res_w(G, ∂G/∂w)` as a rational function; it differs from the computed resultant `k^3 z^3 p_k^2 q_k^2`; the conclusion stands; to confirm with the authors (not an erratum).
- Broda 2020 §3.4, uniqueness for `k > 54`: as extracted, the argument writes `µ_k = 1/(k(2k + 27)ρ_k^2) > 0`; the product of the roots gives `−1/(k(2k + 27)ρ_k^2) < 0`, so there is no contradiction, and `−ρ_k` is a root exactly at `k = 108`. The conclusion stands (design note §5.5). To check against the printed text, then raise with the authors.
- Pivoteau–Salvy 2025, Algorithm Irr-DS (§7.3), nonlinear branch drops the dominant singularities of critical inputs (a period-1 component fed by a period-2 critical input is singular at `−1`; design note §5.5, test `test_critical_input_is_singular_at_minus_one`). To check against the printed text, then raise with the authors (low priority).
- Confirm the Broda 2020 eq. (5) erratum with the authors (see Decisions, step 1).
- FS Note VI.3 (p. 383), as extracted, lacks a sign: `e_k = Σ_ℓ λ_{k,ℓ} (−1)^ℓ (α − 1)…(α − ℓ)` matches eq. (13) (`λ_{1,1} = −1`, `λ_{1,2} = 1/2`; confirmed in review). Our reading, not an erratum; check the printed page (low priority; the code uses the Bernoulli recurrence).
- Broda 2020 eqs. (37)-(38) use -b_k relative to eq. (20)'s b (b_k > 0 by monotonicity); the limit 1/2 stands; check the printed eqs. (20), (37), (38), then raise with the authors.
- PyPI trusted publishing (GitHub Actions OIDC release workflow): not set up yet; do it before the first real release.
- Remaining technical risk: Case I curves with larger systems, where the Groebner step may be slow. Non-radical roots are handled by the number fields of step 5. The Groebner step, `alpha` detection and the Puiseux expansion in `1/k` are all validated on the Broda 2020 examples.
