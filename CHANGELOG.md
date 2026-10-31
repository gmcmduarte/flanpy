# Changelog

All notable changes to flanpy are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and flanpy follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0.0, minor
releases may still change the public API; every such change is listed here.

## [Unreleased]

## [0.1.0] - 2026-10-31

First usable release: build-order steps 1–6, from a polynomial system to the
asymptotics of the coefficients at a fixed alphabet size `k`, and for symbolic `k`
up to the leading constants, with the set of `k` where each result holds. Exact
throughout (algebraic numbers, number fields, exact root isolation); floating point
only in `evalf`. The running examples reproduce Catalan and Broda et al. 2020 (up to
eq. (38) and Table 1).

### Added

The public API, `flanpy.__all__`, by build-order step:

- **Step 1, systems and series**: `System` (a polynomial system `F = Φ(z, k, F)` and
  its well-posedness check), `series` (coefficients at a fixed `k`), `series_in_k`
  (coefficients as polynomials in `k`); `WellPosednessError` and its subclasses
  `IllPosedSystemError` and `UndecidedSystemError`; `ZeroCoordinateWarning`.
- **Step 2, sets of `k`**: `KSet` (sets of integers `k >= 1` from sign conditions on
  polynomials in `k`) and `KCases` (piecewise results over disjoint `KSet`s).
- **Step 3, elimination**: `eliminate` (fixed `k`), `eliminate_in_k` (symbolic `k`),
  `Curve` (the minimal polynomial `G(z, w)` of one unknown).
- **Step 4, dominant singularity**: `AlgebraicReal` (exact real algebraic numbers),
  `KRoot` (a positive root of a polynomial with coefficients in `Q(k)`),
  `singularity`, `singularity_in_k`, `Singularity` (`ρ`, Case I/II, `a = w(ρ)`,
  period, uniqueness); `UndecidedSingularityError`, `SingularityWarning`.
- **Step 5, local expansion**: `local_expansion`, `local_expansion_in_k`,
  `LocalExpansion` (exponent `α`, constants `b` / `c`, exact Puiseux terms).
- **Step 6, transfer**: `asymptotics`, `asymptotics_in_k`, `Asymptotics` (the
  expansion of `[z^n] w` in powers of `n`, exact constants, `evalf`), `GammaConstant`
  (an algebraic number times a product of `Γ` values), `ratio` and `Ratio` (ratios
  of two expansions with the same `ρ`, e.g. Broda et al. 2020 Thm 3.2).
- `__version__`.
- Optional extra `fast` (python-flint) for faster polynomial arithmetic.
- Releases are published to PyPI from tags by trusted publishing.

### Changed

Pre-release changes since 0.0.1.dev0 that readers of the earlier (step 5) online
documentation may trip on:

- `LocalExpansion.terms(n)` is now `LocalExpansion.terms(count)`: a number of terms
  is called `count` everywhere (`n` is the coefficient index). Positional calls are
  unaffected.
- `KCases.combine(other, f)` calls `f` only on non-empty intersections of pieces
  (empty ones were dropped anyway). `f` may now assume that its two values share
  some `k`, e.g. `cases_a.combine(cases_b, ratio)` for pieces with different `ρ_k`.

## [0.0.1.dev0] - 2026-10-29

### Added

- Package skeleton, published to reserve the name `flanpy` on PyPI. No
  functionality.

[Unreleased]: https://github.com/gmcmduarte/flanpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gmcmduarte/flanpy/compare/v0.0.1.dev0...v0.1.0
[0.0.1.dev0]: https://github.com/gmcmduarte/flanpy/releases/tag/v0.0.1.dev0
