# CLAUDE.md — pyflan

Open-source Python package implementing the analytic-combinatorics toolchain used by our research group (Broda, Machiavelo, Moreira, Reis, and Guilherme), so that papers can ship a runnable script that reproduces their results, instead of Mathematica files.

This file holds the project's context, decisions and base rules. The package scaffolding exists (see Tooling); no algorithm is implemented yet. Update this file when decisions change.

## What the package does

Pipeline for counting problems whose generating functions are algebraic, possibly indexed by a parameter `k` (the alphabet size, a non-negative integer):

1. **Symbolic method**: grammar -> polynomial system for generating functions; cost/multivariate series via the derivative in `u` at `u = 1`.
2. **Elimination**: Gröbner basis (Buchberger/F5B) and resultants, to get one algebraic equation `G(z, w) = 0`.
3. **Singularity analysis**: dominant singularity `rho` and value `a`, via `res_w(G, dG/dw)`; minimal polynomials `m_rho`, `m_a`; Case I (`w(rho) = a` finite) vs Case II (`w -> infinity`, use the reciprocal polynomial); uniqueness of the dominant singularity (Pringsheim).
4. **Local expansion**: which partial derivatives of `G` vanish at `(rho, a)` (resultant + gcd tests); Newton polygon -> exponent `alpha`; Puiseux expansion; solve for the leading constant `b` (or `c`).
5. **Transfer**: `[z^n] w(z) ~ -b/Gamma(-alpha) * rho^-n * n^(-alpha-1)` (Case I), `~ (1/c)/Gamma(alpha) * rho^-n * n^(alpha-1)` (Case II). Includes the square-root special case (quadratic `G`, closed form).
6. **Families in k**: `rho_k`, `a_k`, `b_k`, `c_k` as Puiseux expansions in `1/k`, plus limits as `k -> infinity`.

Reference papers (uploaded to the Claude project):
- Broda, Machiavelo, Moreira, Reis, *Analytic Combinatorics and Descriptional Complexity of Regular Languages on Average* (SIGACT News, 2020) — the full machinery for implicit curves.
- Broda, Machiavelo, Moreira, Reis, *A Hitchhiker's Guide to descriptional complexity through analytic combinatorics* (TCS 528, 2014) — simpler curves with square-root singularities and cost generating functions.
- Flajolet & Sedgewick, *Analytic Combinatorics* (2008) — the general theory (Thm VI.1 and Cor. VI.1, Pringsheim IV.6, algebraic coefficient asymptotics).

## Decisions

- **Name**: `pyflan` (PyPI distribution name and import name: `import pyflan`).
- **Repository**: `git@github.com:gmcmduarte/pyflan.git` (GitHub, created; remote `origin`).
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

## Planned layout

```
src/pyflan/
  symbolic/      grammar -> equations, bivariate cost series (d/du at u=1)
  elimination/   Groebner and resultants, backend interface
  singularity/   m_rho, m_a, Case I/II, dominant-singularity checks
  newton/        Newton polygon, Newton-Puiseux
  transfer/      Thm 3.1 / 3.2, square-root case
  families/      expansions in 1/k, KSet
  automata/      thin regex/NFA front-end (cost vectors c_empty..c_star); optional
tests/
docs/
examples/        one runnable script or notebook per paper
```

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
- **Docs and examples with every feature**: docstrings (state the theorem or paper equation implemented), plus a worked example that reproduces the paper's number.
- **Cite sources in code**: reference the paper and equation number next to each algorithm (e.g. "Broda et al. 2020, eq. (27)").
- **Small, tested steps.** One module at a time, with its acceptance test from the list above.
- **Commit hygiene**: clear messages, one logical change per commit, no generated artifacts or large data files in the repo.

## Tooling

All configuration lives in `pyproject.toml` (build backend `hatchling`, `src/` layout). Version is static in `pyproject.toml`; `pyflan.__version__` reads it from the installed metadata.

- **Extras**: `fast` (python-flint), `dev` (pytest, ruff, mypy, build), `docs` (sphinx, shibuya, myst-parser, myst-nb, sphinxcontrib-bibtex).
- **Types**: `mypy --strict` (chosen over pyright: Python-native, no Node dependency). Tests are type-checked but may omit annotations.
- **Lint/format**: `ruff` with pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify, RUF and pydocstyle (NumPy convention). `dev/spikes/` is excluded (throwaway code).
- **Tests**: `pytest` with `--strict-markers --strict-config --doctest-modules`; collects `tests/` and the docstrings in `src/pyflan`; warnings are errors.
- **SPDX**: every `.py` file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- **CI**: `.github/workflows/ci.yml` runs ruff, mypy, pytest and `python -m build` on Python 3.13 and `3.x`, plus a docs job (`sphinx-build -W`). Read the Docs config: `.readthedocs.yaml` (`fail_on_warning: true`).
- **Repo files**: `README.md`, `CITATION.cff`, `CONTRIBUTING.md`, `docs/refs.bib` (DOIs checked against Crossref).

Commands (from the repo root, in a Python 3.13 venv):

```
pip install -e ".[dev,docs]"         # add ,fast for python-flint
pytest                               # tests + doctests
ruff check . && ruff format --check .
mypy                                 # strict, src + tests
sphinx-build -W --keep-going docs docs/_build/html
python -m build                      # sdist + wheel into dist/
```

### Documentation (decided)

- **Sphinx** with the **Shibuya** theme. Not MkDocs Material: it entered maintenance mode in Nov 2025 (security fixes only until Nov 2026), and MkDocs has a weaker cross-reference ecosystem.
- **MyST-Parser**: write pages in Markdown. **MyST-NB**: one executed notebook per paper, run as part of the docs build.
- **MathJax** for equations, with numbered equations that can be cited.
- **`autodoc`** with NumPy-style docstrings for the API reference (`napoleon`).
- **`intersphinx`** to link to SymPy / Python / mpmath docs.
- **`sphinxcontrib-bibtex`** to cite the papers from a `.bib` file.
- **`doctest`** (and executed notebooks) so the docs fail to build if a paper's number stops reproducing.
- Hosting: Read the Docs or GitHub Pages, built by CI. HTML is the primary output.
- Docs grow with the code: every feature lands with its docstring (naming the theorem or paper equation implemented) and a worked example in the same change.

## Open items

- [x] Tooling: pytest, ruff, mypy (strict) set up. Shibuya 2026.7.12 builds cleanly with Sphinx 9.2.0 on Python 3.13 (`-W`, no warnings); tests and mypy also pass on Python 3.15.
- Docs hosting: `.readthedocs.yaml` exists, but the Read the Docs project is not connected yet (or use GitHub Pages instead).
- `pyflan` was free on PyPI on 2026-10-29; reserve it (e.g. by publishing an early release) when convenient.
- Remaining technical risk: roots of characteristic polynomials that are not expressible in radicals (see Decisions), and Case I curves with larger systems, where the Groebner step may be slow. The Groebner step, `alpha` detection and the Puiseux expansion in `1/k` are all validated on the Broda 2020 examples.
