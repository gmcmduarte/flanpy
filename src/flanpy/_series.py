# SPDX-License-Identifier: GPL-3.0-or-later
"""Power-series coefficients of a :class:`~flanpy.System`.

The design is recorded in ``dev/design/series.md``. Pivoteau, Salvy & Soria (2012)
are cited with the numbering of arXiv:1109.2688v2.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from operator import mul
from types import MappingProxyType
from typing import Any

import sympy as sp

from flanpy._rings import BackendName, Domain, _Ring, backend
from flanpy._system import System

__all__ = ["series", "series_in_k"]


def series(
    system: System, n: int, *, k: int | None = None
) -> Mapping[sp.Symbol, tuple[int | Fraction, ...]]:
    """Return the coefficients of ``z^0, ..., z^n`` of every unknown, for a fixed ``k``.

    Parameters
    ----------
    system : System
        The system. Its solution is the limit of the iteration from 0 (see the
        Notes of :class:`~flanpy.System`).
    n : int
        The last index: the result holds the coefficients of indices ``0..n``
        **inclusive**, i.e. ``n + 1`` of them. For example, ``n = 1000`` gives
        ``[z^1000]`` as the last entry.
    k : int, optional
        The alphabet size, an integer ``k >= 1``. Required if ``system.k`` is set;
        must be omitted otherwise. Use :func:`series_in_k` for symbolic ``k``.

    Returns
    -------
    Mapping[sympy.Symbol, tuple[int | Fraction, ...]]
        Read-only mapping from each unknown (in the order of ``system.unknowns``)
        to ``([z^0] F, ..., [z^n] F)``. Zero coordinates give all zeros. The
        coefficients are ``int`` when the system has integer coefficients and
        integer constant terms (then every coefficient is an integer), and
        :class:`fractions.Fraction` otherwise. All entries have the same type,
        whatever the backend.

    Raises
    ------
    TypeError
        If ``system`` is not a :class:`~flanpy.System`, or ``n`` or ``k`` is not an
        ``int``.
    ValueError
        If ``n < 0``, ``k < 1``, or ``k`` is missing or given when it should not be.

    Warns
    -----
    ZeroCoordinateWarning
        If some unknowns are identically zero at this ``k``. ``k`` is validated
        exactly as by :meth:`System.specialize`.

    Notes
    -----
    The coefficients are computed by the recurrence ``(I - J0) y_n = r_n``, where
    ``J0`` is the (nilpotent) Jacobian of the system at ``z = 0`` and ``r_n``
    depends only on ``y_0, ..., y_{n-1}``. It starts from the stabilised constant
    terms ``y_0 = c*``. A solution with constant term ``c*`` is unique, so this is
    the limit of the iteration, for any signs (series.md, section 2). For
    non-negative systems this agrees with Pivoteau, Salvy & Soria (2012), Thm 6.3
    and Thm 8.10. The cost is ``O(P n^2)`` coefficient multiplications, with ``P``
    the number of distinct products of unknowns in the system.

    python-flint is used when it is installed (``pip install flanpy[fast]``);
    otherwise the computation runs in pure Python. The results are the same.

    Examples
    --------
    Catalan numbers; ``n = 6`` gives the 7 coefficients of ``z^0..z^6``:

    >>> from sympy import symbols
    >>> from flanpy import System, series
    >>> z, k, F, R = symbols("z k F R")
    >>> catalan = System({F: 1 + z * F**2}, z=z)
    >>> series(catalan, 6)[F]
    (1, 1, 2, 5, 14, 42, 132)

    The number of regular expressions of each size over 2 letters
    (Broda et al. 2020, grammar (1) and eq. (3)):

    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> series(rk, 5, k=2)[R]
    (0, 3, 3, 21, 57, 327)
    """
    _check_system_and_n(system, n)
    if system.k is None:
        if k is not None:
            raise ValueError("the system has no parameter k; omit the k argument")
        return _series(system, n, None)
    if k is None:
        raise ValueError(
            "the system has a parameter k: pass k=<int >= 1>, or use "
            "series_in_k for symbolic k"
        )
    # Validate k exactly like System.specialize (same errors, and the same
    # ZeroCoordinateWarning for zero coordinates that appear only at this k).
    system.specialize(k)
    return _series(system, n, k)


def series_in_k(system: System, n: int) -> Mapping[sp.Symbol, tuple[sp.Poly, ...]]:
    """Return the coefficients of ``z^0, ..., z^n`` as polynomials in ``k``.

    Parameters
    ----------
    system : System
        A system with a parameter ``k`` (``system.k`` is not ``None``).
    n : int
        The last index: indices ``0..n`` **inclusive**, ``n + 1`` coefficients.

    Returns
    -------
    Mapping[sympy.Symbol, tuple[sympy.Poly, ...]]
        Read-only mapping from each unknown to its coefficients, each a
        :class:`sympy.Poly` in ``k`` over ``ZZ`` (integer system) or ``QQ``.

    Raises
    ------
    TypeError
        If ``system`` is not a :class:`~flanpy.System` or ``n`` is not an ``int``.
    ValueError
        If ``n < 0`` or the system has no parameter ``k``.

    Notes
    -----
    The recurrence of :func:`series` runs over ``Q[k]``. No division by a
    polynomial in ``k`` occurs, because ``(I - J0)^-1 = I + J0 + ... + J0^(m-1)``.
    Evaluating the result at an integer ``k >= 1`` gives exactly
    ``series(system, n, k=...)`` (series.md, section 2.3).

    The coefficient of ``z^n`` has degree about ``n`` in ``k``, so the cost grows
    quickly with ``n``. Measured on an Apple M1: about 4 s for the five-unknown
    ``L_k`` system of Broda et al. (2020) at ``n = 200``, and 30 s at ``n = 400``
    with python-flint. For large ``n``, use :func:`series` with a fixed ``k``.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy import System, series_in_k
    >>> z, k, R = symbols("z k R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> [c.as_expr() for c in series_in_k(rk, 4)[R]]
    [0, k + 1, k + 1, 2*k**2 + 5*k + 3, 6*k**2 + 13*k + 7]
    """
    _check_system_and_n(system, n)
    if system.k is None:
        raise ValueError("series_in_k needs a system with a parameter k; use series")
    return _series(system, n, None)


def _check_system_and_n(system: object, n: object) -> None:
    if not isinstance(system, System):
        raise TypeError(f"expected a flanpy.System, got {type(system).__name__}")
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {n!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")


# -- compilation ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Compiled:
    """The reduced system, ready for the recurrence (series.md, section 5)."""

    ring: _Ring
    m: int  # number of active unknowns
    # Per equation: (coefficient, power of z, key); key = sorted unknown indices.
    terms: tuple[tuple[tuple[Any, int, tuple[int, ...]], ...], ...]
    nodes: tuple[tuple[int, ...], ...]  # product keys of length >= 2, by length
    constants: tuple[Any, ...]  # c*
    inverse: tuple[tuple[tuple[int, Any], ...], ...]  # sparse rows of (I - J0)^-1


def _in_k(expr: Any, k: sp.Symbol | None) -> list[Fraction]:
    """Coefficients (lowest first) of ``expr`` as a polynomial in ``k``."""
    if k is None:
        r = sp.Rational(expr)
        return [Fraction(int(r.p), int(r.q))]
    poly = sp.Poly(expr, k, domain=sp.QQ)
    return [Fraction(int(c.p), int(c.q)) for c in reversed(poly.all_coeffs())]


def _compile(
    system: System, value: int | None, backend_name: BackendName | None
) -> _Compiled:
    """Specialise ``k`` to ``value`` (or keep it symbolic) and build the recurrence."""
    k = system.k
    sub: dict[Any, Any] = {} if k is None or value is None else {k: sp.Integer(value)}
    var = k if (k is not None and value is None) else None  # symbolic variable
    active = system._active
    syms = [system.unknowns[i] for i in active]
    zero = {f: sp.Integer(0) for i, f in enumerate(system.unknowns) if i not in active}

    raw_terms = []
    for i in active:
        phi = sp.expand(system._rhs[i].xreplace(zero).xreplace(sub))
        eq = []
        if phi != 0:
            for monom, coef in sp.Poly(phi, *syms, system.z).terms():
                key = tuple(j for j in range(len(syms)) for _ in range(monom[j]))
                eq.append((_in_k(coef, var), monom[-1], key))
        raw_terms.append(eq)
    raw_constants = [
        _in_k(sp.expand(system._constant_terms[i].xreplace(sub)), var) for i in active
    ]
    jac = system._jacobian0
    raw_jac = [
        [_in_k(sp.expand(jac[i, j].xreplace(sub)), var) for j in range(len(syms))]
        for i in range(len(syms))
    ]

    every = [c for eq in raw_terms for cs, _, _ in eq for c in cs]
    every += [c for cs in raw_constants for c in cs]
    every += [c for row in raw_jac for cs in row for c in cs]
    integral = all(c.denominator == 1 for c in every)
    domain: Domain
    if var is None:
        domain = "ZZ" if integral else "QQ"
    else:
        domain = "ZZ[k]" if integral else "QQ[k]"
    ring = backend(backend_name).ring(domain, var)

    terms = tuple(
        tuple((ring.from_coeffs(cs), a, key) for cs, a, key in eq) for eq in raw_terms
    )
    keys = {
        key[:length]
        for eq in terms
        for _, _, key in eq
        for length in range(2, len(key) + 1)
    }
    nodes = tuple(sorted(keys, key=lambda t: (len(t), t)))
    constants = tuple(ring.from_coeffs(cs) for cs in raw_constants)
    j0 = [[ring.from_coeffs(cs) for cs in row] for row in raw_jac]
    return _Compiled(
        ring,
        len(syms),
        terms,
        nodes,
        constants,
        _unipotent_inverse(j0, ring, system._nilpotency_index),
    )


def _unipotent_inverse(
    j0: Sequence[Sequence[Any]], ring: _Ring, index: int
) -> tuple[tuple[tuple[int, Any], ...], ...]:
    """Return the sparse rows of ``(I - J0)^-1 = I + J0 + ... + J0^(index-1)``.

    ``index`` is the nilpotency index of ``J0`` (``J0^index = 0``).
    """
    m = len(j0)
    ident = [[ring.one if i == j else ring.zero for j in range(m)] for i in range(m)]
    total, power = ident, ident
    for _ in range(1, index):
        power = [
            [
                sum((power[i][t] * j0[t][j] for t in range(m)), ring.zero)
                for j in range(m)
            ]
            for i in range(m)
        ]
        total = [[total[i][j] + power[i][j] for j in range(m)] for i in range(m)]
    return tuple(
        tuple((j, total[i][j]) for j in range(m) if total[i][j] != ring.zero)
        for i in range(m)
    )


# -- the recurrence --------------------------------------------------------------------


def _recurrence(c: _Compiled, n_max: int) -> list[list[Any]]:
    """Coefficients ``0..n_max`` of the active unknowns (series.md, sections 2 and 5).

    For each index ``n``: (1) each product node gets its coefficient ``n`` with
    ``y_n`` taken as 0 ("partial"); (2) ``r_n`` is the coefficient ``n`` of ``Phi``
    without the terms linear in ``y_n``; (3) ``y_n = (I - J0)^-1 r_n``; (4) each
    node's coefficient ``n`` is completed with the terms in ``y_n``.
    """
    zero = c.ring.zero
    y: list[list[Any]] = [[c0] for c0 in c.constants]
    node: dict[tuple[int, ...], list[Any]] = {}

    def series_of(key: tuple[int, ...]) -> list[Any]:
        return y[key[0]] if len(key) == 1 else node[key]

    for t in c.nodes:
        node[t] = [series_of(t[:-1])[0] * y[t[-1]][0]]
    for n in range(1, n_max + 1):
        partial: dict[tuple[int, ...], Any] = {}
        sums: dict[tuple[int, ...], Any] = {}
        for t in c.nodes:
            left, right = series_of(t[:-1]), y[t[-1]]
            s = sum(map(mul, left[1:n], right[n - 1 : 0 : -1]), zero)
            sums[t] = s
            left_n = partial[t[:-1]] if len(t) > 2 else zero
            partial[t] = s + left_n * right[0]
        r = []
        for eq in c.terms:
            acc = zero
            for coef, a, key in eq:
                if not key:
                    if a == n:
                        acc += coef
                elif a == 0:
                    if len(key) > 1:  # degree-1 terms with a = 0 are in J0
                        acc += coef * partial[key]
                elif a <= n:
                    acc += coef * series_of(key)[n - a]
            r.append(acc)
        for i in range(c.m):
            y[i].append(sum((v * r[j] for j, v in c.inverse[i]), zero))
        for t in c.nodes:
            left, right = series_of(t[:-1]), y[t[-1]]
            node[t].append(sums[t] + left[n] * right[0] + left[0] * right[n])
    return y


def _series(
    system: System,
    n: int,
    k: int | None,
    *,
    backend_name: BackendName | None = None,
) -> Mapping[sp.Symbol, tuple[Any, ...]]:
    """Shared entry point of :func:`series` and :func:`series_in_k` (no input checks).

    ``k=None`` means symbolic ``k`` if the system has one. ``backend_name`` forces
    ``"python"`` or ``"flint"`` (used by the tests).
    """
    compiled = _compile(system, k, backend_name)
    values = _recurrence(compiled, n)
    to_public = compiled.ring.to_public
    zeros = (to_public(compiled.ring.zero),) * (n + 1)
    out: dict[sp.Symbol, tuple[Any, ...]] = {}
    position = {i: p for p, i in enumerate(system._active)}
    for i, f in enumerate(system.unknowns):
        out[f] = tuple(map(to_public, values[position[i]])) if i in position else zeros
    return MappingProxyType(out)
