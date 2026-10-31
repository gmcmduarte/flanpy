# flanpy

Analytic combinatorics for algebraic generating functions, as used in the study of
the average descriptional complexity of regular languages. flanpy implements the
toolchain of Broda, Machiavelo, Moreira and Reis: polynomial system → elimination
(Gröbner bases, resultants) → dominant singularity → Newton–Puiseux expansion →
coefficient asymptotics, including families indexed by the alphabet size `k`. Every
step is exact (algebraic numbers, sets of `k` from exact root isolation); floating
point is used only for evaluation. The goal is that papers can ship a runnable script
that reproduces their results.

## Status

Alpha (0.1.0): the API may still change.

- **Implemented (steps 1–6)**: from a polynomial system to the asymptotics of the
  coefficients at a fixed `k`, with all terms of the expansion, exact constants and
  ratios; for symbolic `k`, the same pipeline up to the leading constants, with the
  set of `k` where each result holds.
- **Planned (steps 7–9)**: expansions of the constants in `1/k` and limits as
  `k → ∞`; grammars to systems, and cost generating functions; a regular-expression
  front end.

## Installation

Requires Python 3.13 or newer.

```console
pip install "flanpy[fast]"      # recommended: python-flint for fast polynomial arithmetic
pip install flanpy              # pure Python (SymPy only)
```

## Example

The Catalan numbers, `F = 1 + z F^2`, from the system to the asymptotics of
`[z^n] F`:

```pycon
>>> from sympy import symbols
>>> from flanpy import System, asymptotics, eliminate, local_expansion, series, singularity
>>> z, F, n = symbols("z F n")
>>> catalan = System({F: 1 + z * F**2}, z=z)
>>> s = singularity(eliminate(catalan, F))  # curve G(z, F) = 0, then rho = 1/4
>>> s.rho, s.case
(AlgebraicReal(4*z - 1, 0), 'I')
>>> a = asymptotics(local_expansion(s))  # Puiseux expansion at rho, then transfer
>>> print(a)
4**n/(sqrt(pi)*n**(3/2)) (1 + O(n^-1))
>>> a.as_expr(n, count=3)  # count counts the terms, zeros included: two non-zero ones
4**n*(1/(sqrt(pi)*n**(3/2)) - 9/(8*sqrt(pi)*n**(5/2)))
>>> exact = series(catalan, 1000)[F][1000]
>>> for count in (1, 3):
...     print(count, f"{float(abs(a.evalf(1000, count=count) / exact - 1)):.1e}")
1 1.1e-03
3 1.1e-06
```

The same calls with `k` symbolic (`eliminate_in_k`, `singularity_in_k`, ...) return
piecewise results over sets of `k`.

## Documentation

- Documentation: <https://flanpy.readthedocs.io/>
- User guide, one page per step:
  [system](https://flanpy.readthedocs.io/en/stable/guide/system.html),
  [kset](https://flanpy.readthedocs.io/en/stable/guide/kset.html),
  [elimination](https://flanpy.readthedocs.io/en/stable/guide/elimination.html),
  [singularity](https://flanpy.readthedocs.io/en/stable/guide/singularity.html),
  [newton](https://flanpy.readthedocs.io/en/stable/guide/newton.html),
  [transfer](https://flanpy.readthedocs.io/en/stable/guide/transfer.html).
- Running examples (scripts in [`examples/`](https://github.com/gmcmduarte/flanpy/tree/main/examples), also executed docs pages):
  [Catalan](https://flanpy.readthedocs.io/en/stable/examples/catalan.html) (no
  parameter) and [Broda et al. 2020](https://flanpy.readthedocs.io/en/stable/examples/broda2020.html)
  (families in `k`, up to eq. (38) and Table 1).
- Changes: [CHANGELOG.md](https://github.com/gmcmduarte/flanpy/blob/main/CHANGELOG.md).
- Development (from a clone, tests, docs): [CONTRIBUTING.md](https://github.com/gmcmduarte/flanpy/blob/main/CONTRIBUTING.md).

## Citing

If you use flanpy in research, please cite it; the metadata is in
[CITATION.cff](https://github.com/gmcmduarte/flanpy/blob/main/CITATION.cff) (GitHub's "Cite this repository" button gives BibTeX
and APA).

## Licence

GPL-3.0-or-later. See [LICENSE](https://github.com/gmcmduarte/flanpy/blob/main/LICENSE).
