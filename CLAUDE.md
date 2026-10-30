# CLAUDE.md — flanpy

Open-source Python package implementing the analytic-combinatorics toolchain used by our research group (Broda, Machiavelo, Moreira, Reis, and Guilherme), so that papers can ship a runnable script that reproduces their results, instead of Mathematica files.

This file holds the project's context, decisions and base rules. Build-order steps 1 (`System`, `series`) and 2 (`KSet`) are implemented; the later steps are not. Update this file when decisions change.

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
- Pivoteau, Salvy, *Effective Asymptotics of Combinatorial Systems* (arXiv:2508.20008, 2025) — relevant to steps 4–6 (singularity, Newton–Puiseux, transfer); not read yet.

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
- **Gröbner computations**: use `sympy.groebner(..., order='lex', method='f5b')` with `k` as an ordinary polynomial variable ordered **last**. Measured on eq. (17) of Broda 2020: F5B 1.4 s; default Buchberger 109 s; `domain=QQ.frac_field(k)` did not finish in 250 s. Do not use the fraction-field domain for the elimination step.
- **k appears only in coefficients** of the curves (never in degrees or exponents) in the current framework.
- **Conditions on k**: `k` is the only parameter, so every condition is a sign/vanishing condition on univariate polynomials in `k`, i.e. a finite union of integer points and intervals, e.g. `{2} U [3,53] U {54} U [55, inf)`. Each result carries the set `KSet` of `k` where it is valid (exact real-root isolation); results combine by intersecting `KSet`s. Exceptional `k` (roots of leading coefficients, denominators, discriminants) are collected and handled by instantiation. If a case cannot be decided, say so explicitly; never guess.
- **Finding `alpha` and Puiseux expansions (validated by a spike, see `dev/spikes/np_spike.py`)**:
  - `alpha` needs no algebraic-number arithmetic. As in Broda 2020 (eqs. (21)-(27)), take the derivatives `d^i_z d^j_w G` at `(rho, a)`, decide which are non-zero with `gcd(derivative(z), m_rho(z))` over `Q(k)` (gcd = 1 -> non-zero; zero polynomial -> zero; otherwise inconclusive), then take the lower convex hull of the points `(j-1, i)`. Group support points by x and keep the minimum y before computing the hull. Reproduced `alpha = 1/2` for eq. (32).
  - Puiseux expansions in `1/k` (e.g. `rho_k`) use the standard Newton-polygon recursion: substitute `x = s^q`, `y = s^p (c + y1)`, solve the characteristic polynomial for `c`, recurse. With SymPy radicals this reproduced eq. (35) exactly (coefficients in `Q(sqrt 2)`) and `c_k ~ 4 k^(-1/2)` from eqs. (34)/(36).
  - Not yet tested: characteristic polynomials whose roots are not expressible in radicals. Those need an explicit number-field representation (minimal polynomial + root isolation) instead of `sympy.roots`. Keep the coefficient field behind an interface so it can be swapped.
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

## Planned layout

```
src/flanpy/
  _system.py     System (F = Phi(z, k, F)) and its well-posedness check; re-exported from flanpy
  _series.py     series / series_in_k: coefficients of a System; re-exported from flanpy
  _rings.py      private coefficient rings: pure-Python and python-flint backends
  _kset.py       KSet (sets of k >= 1 from sign conditions) and KCases (piecewise
                 results); KSet re-exported from flanpy (design: dev/design/kset.md)
  symbolic/      grammar -> equations, bivariate cost series (d/du at u=1)
  elimination/   Groebner and resultants, backend interface
  singularity/   m_rho, m_a, Case I/II, dominant-singularity checks
  newton/        Newton polygon, Newton-Puiseux
  transfer/      Thm 3.1 / 3.2, square-root case
  families/      expansions in 1/k
  automata/      thin regex/NFA front-end (cost vectors c_empty..c_star); optional
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
- **Reports**: compact, at most ~40 lines, one line per item, point to design notes instead of repeating them, full detail only for PROBLEMS and QUESTIONS.

## Tooling

All configuration lives in `pyproject.toml` (build backend `hatchling`, `src/` layout). Version is static in `pyproject.toml`; `flanpy.__version__` reads it from the installed metadata.

- **Extras**: `fast` (python-flint), `dev` (pytest, hypothesis, ruff, mypy, numpydoc, build, twine), `docs` (sphinx, shibuya, myst-parser, myst-nb, jupytext, sphinxcontrib-bibtex).
- **Types**: `mypy --strict` (chosen over pyright: Python-native, no Node dependency). Tests are type-checked but may omit annotations.
- **Lint/format**: `ruff` with pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, RUF and pydocstyle (NumPy convention). `dev/spikes/` is excluded (throwaway code).
- **Tests**: `pytest` with `--strict-markers --strict-config --doctest-modules`; collects `tests/` and the docstrings in `src/flanpy`; warnings are errors. Tests marked `slow` (N = 1000) are deselected by default (`-m "not slow"` in `addopts`); run them with `pytest -m slow`. Test helpers (e.g. `tests/_enumerate.py`, the brute-force enumerator for Broda 2020) are importable because `tests` is on pytest's `pythonpath`. Property-based tests use `hypothesis` with a derandomized profile (`tests/conftest.py`). Doctests of a private module import its names from that module, not from `flanpy`: when pytest (importlib mode) collects a module alone, it executes it a second time, and the re-exported classes then differ.
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
- Confirm the Broda 2020 eq. (5) erratum with the authors (see Decisions, step 1).
- PyPI trusted publishing (GitHub Actions OIDC release workflow): not set up yet; do it before the first real release.
- Remaining technical risk: roots of characteristic polynomials that are not expressible in radicals (see Decisions), and Case I curves with larger systems, where the Groebner step may be slow. The Groebner step, `alpha` detection and the Puiseux expansion in `1/k` are all validated on the Broda 2020 examples.
