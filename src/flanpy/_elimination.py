# SPDX-License-Identifier: GPL-3.0-or-later
"""Elimination: from a :class:`~flanpy.System` to one curve ``G(z, w, k) = 0``.

The design is recorded in ``dev/design/elimination.md``: the eliminant (sections 2.1
and 4), the factor that the series satisfies (section 3), and the exceptional ``k``
(sections 2.2-2.4).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import sympy as sp

from flanpy._curve import (
    Curve,
    _gcd_in_k,
    _normalise,
    _select_factor,
    _w_factors,
    _w_free_factors,
)
from flanpy._kset import KCases, KSet
from flanpy._system import System, ZeroCoordinateWarning

__all__ = ["eliminate", "eliminate_in_k"]

Method = Literal["auto", "groebner", "resultant"]
_METHODS = ("auto", "groebner", "resultant")


# -- Groebner backends (elimination.md, section 5) ---------------------------------


class _GroebnerBackend(Protocol):
    """Computes reduced lex Groebner bases; inputs and outputs are SymPy ``Poly``s."""

    name: str

    def groebner_lex(
        self, polys: Sequence[sp.Poly], gens: Sequence[sp.Symbol]
    ) -> list[sp.Poly]:
        """Return the reduced lex basis of ``polys``, generators in the given order."""
        ...


class _SympyBackend:
    """SymPy's F5B (CLAUDE.md, Decisions: 1.4 s for Broda 2020 eq. (17))."""

    name = "sympy"

    def groebner_lex(
        self, polys: Sequence[sp.Poly], gens: Sequence[sp.Symbol]
    ) -> list[sp.Poly]:
        basis = sp.groebner(
            [p.as_expr() for p in polys], *gens, order="lex", method="f5b"
        )
        return [sp.Poly(g, *gens, domain=sp.QQ) for g in basis.exprs]


_BACKEND: _GroebnerBackend = _SympyBackend()


# -- public functions --------------------------------------------------------------


def eliminate(
    system: System, w: sp.Symbol, *, k: int | None = None, method: Method = "auto"
) -> Curve:
    """Return the curve ``G(z, w) = 0`` of the coordinate ``w``, for a fixed ``k``.

    ``G`` is the minimal polynomial of the series ``w(z)`` over ``Q(z)``, normalised
    (see :class:`~flanpy.Curve`). It is computed from the polynomial form of the
    system (Flajolet & Sedgewick 2009, eq. (VII.77)): the other unknowns are
    eliminated, and the factor of the eliminant that the series satisfies is kept.

    Parameters
    ----------
    system : System
        The system.
    w : sympy.Symbol
        One of ``system.unknowns``.
    k : int, optional
        The alphabet size, an integer ``k >= 1``. Required if ``system.k`` is set;
        must be omitted otherwise. Use :func:`eliminate_in_k` for symbolic ``k``.
    method : {"auto", "groebner", "resultant"}, optional
        How to eliminate. ``"auto"`` (default) uses no elimination when ``w``
        depends on no other unknown, a resultant when it depends on one, and a
        lex Gröbner basis otherwise. ``"resultant"`` needs at most one other
        unknown.

    Returns
    -------
    Curve
        The curve, with ``kset = {k}`` (or ``KSet.all()`` for a system without
        parameter) and the system specialised at ``k``.

    Raises
    ------
    TypeError
        If ``system`` is not a :class:`~flanpy.System` or ``k`` is not an ``int``.
    ValueError
        If ``w`` is not an unknown, ``k < 1``, ``k`` is missing or given when it
        should not be, or ``method`` is unknown or not applicable.

    Warns
    -----
    ZeroCoordinateWarning
        If some unknowns are identically zero at this ``k`` (as
        :meth:`System.specialize`).

    See Also
    --------
    eliminate_in_k : the same for symbolic ``k``, with the exceptional ``k``.

    Notes
    -----
    Only the unknowns that ``w`` depends on take part, and zero coordinates are
    replaced by 0. With other unknowns, the eliminant is an element of least degree
    in ``w`` of ``I ∩ Q[w, z]``, ``I`` the ideal of the system, from a lex Gröbner
    basis (Cox, Little & O'Shea, *Ideals, Varieties, and Algorithms*, Ch. 3 §1), or
    the resultant in the other unknown. If the elimination ideal has no element in
    ``w`` (a component of excess dimension), the ideal is first saturated by the
    Jacobian determinant of the system, which is 1 at ``z = 0``.

    The factor is chosen exactly: with ``N`` the maximum over pairs of factors of
    ``deg_w h_i deg_z h_j + deg_w h_j deg_z h_i``, only the minimal polynomial
    vanishes at the series modulo ``z^(N+1)`` (elimination.md, section 3.2, from
    ``Res_w(g, h) = A g + B h``).

    Examples
    --------
    Catalan numbers: ``F`` needs no elimination.

    >>> from sympy import symbols
    >>> from flanpy import System, eliminate
    >>> z, k, F, A, B = symbols("z k F A B")
    >>> print(eliminate(System({F: 1 + z * F**2}, z=z), F))
    F**2*z - F + 1 = 0

    Broda et al. (2014), eqs. (7)-(8): regular expressions ``A_k = 2 z + B_k``, here
    over 1 letter. ``B`` is eliminated with a resultant:

    >>> hg = System({B: k * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}, z=z, k=k)
    >>> print(eliminate(hg, A, k=1))
    2*A**2*z - 8*A*z**2 + 2*A*z - A + 8*z**3 - 4*z**2 + 3*z = 0
    """
    _check(system, w, method)
    if system.k is None:
        if k is not None:
            raise ValueError("the system has no parameter k; omit the k argument")
        return _eliminate_fixed(system, w, method, KSet.all())
    if k is None:
        raise ValueError(
            "the system has a parameter k: pass k=<int >= 1>, or use "
            "eliminate_in_k for symbolic k"
        )
    return _eliminate_fixed(system.specialize(k), w, method, KSet.of(k))


def eliminate_in_k(
    system: System, w: sp.Symbol, *, method: Method = "auto"
) -> KCases[Curve]:
    r"""Return the curve of the coordinate ``w`` for symbolic ``k``, piece by piece.

    The generic curve ``G(z, w, k)`` is the normalised minimal polynomial of the
    series ``w(z)`` over ``Q(k)(z)``. It is valid on a :class:`~flanpy.KSet` that
    contains all but finitely many ``k``. Each remaining ``k`` gets its own piece,
    with the curve recomputed at that ``k``. The pieces cover every ``k >= 1``.

    Parameters
    ----------
    system : System
        A system with a parameter ``k``.
    w : sympy.Symbol
        One of ``system.unknowns``.
    method : {"auto", "groebner", "resultant"}, optional
        As for :func:`eliminate`.

    Returns
    -------
    KCases[Curve]
        One generic piece, and one piece ``{n}`` per exceptional ``n`` with
        ``eliminate(system, w, k=n)``. Nothing is uncovered.

    Raises
    ------
    TypeError
        If ``system`` is not a :class:`~flanpy.System`.
    ValueError
        If the system has no parameter, ``w`` is not an unknown, or ``method`` is
        unknown or not applicable.

    See Also
    --------
    eliminate : the curve at one fixed ``k``.
    Curve.specialize : certify the generic curve at one ``k``.

    Notes
    -----
    The candidates for exceptional ``k`` are the integers ``n >= 1`` where

    - the leading coefficient (a polynomial in ``k``) of some element of the lex
      Gröbner basis vanishes, or, for a resultant, the leading coefficient in the
      eliminated unknown drops degree; outside these, the basis specialises to a
      Gröbner basis of the specialised system (Kalkbrener 1997, Thm 3.1, which
      generalises Gianni 1987);
    - the eliminant vanishes identically (its content in ``k``);
    - ``G(·, ·, n)`` drops degree in ``w``, is not squarefree, or gains a content
      in ``z`` (a common factor of its coefficients in ``w``).

    All are found exactly (integer roots of polynomials in ``k``; no sampling). At
    each candidate the curve is recomputed with ``k = n`` substituted. If it equals
    the renormalised ``G(·, ·, n)`` with the same degree, ``n`` stays in the generic
    piece; otherwise it gets its own piece (elimination.md, sections 2.2-2.4).

    The invariants of :class:`~flanpy.Curve` then hold on every piece. On the generic
    piece, ``G(·, ·, n)`` is irreducible for all but finitely many ``n`` (Noether's
    irreducibility theorem), but those ``n`` are not computed; use
    :meth:`Curve.specialize` to certify a given ``n``.

    References
    ----------
    .. [1] P. Gianni, "Properties of Gröbner bases under specializations", EUROCAL
       '87, LNCS 378, Springer (1989) 293-297.
    .. [2] M. Kalkbrener, "On the stability of Gröbner bases under
       specializations", *J. Symbolic Comput.* 24(1) (1997) 51-58.

    Examples
    --------
    All regular expressions over ``k`` letters (Broda et al. 2020, grammar (1)): one
    curve for every ``k``.

    >>> from sympy import symbols
    >>> from flanpy import System, eliminate_in_k
    >>> z, k, F, R = symbols("z k F R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> print(eliminate_in_k(rk, R))
    [1, ∞): 2*R**2*z + R*z - R + k*z + z = 0

    A degree drop: at ``k = 1`` the series is ``F = 1``.

    >>> cases = eliminate_in_k(System({F: 1 + (k - 1) * z * F**2}, z=z, k=k), F)
    >>> print(cases)
    {1}: F - 1 = 0
    [2, ∞): F**2*k*z - F**2*z - F + 1 = 0
    >>> cases.tail[1].polynomial.degree(F), cases[1].polynomial.degree(F)
    (2, 1)
    """
    _check(system, w, method)
    if system.k is None:
        raise ValueError(
            "eliminate_in_k needs a system with a parameter k; use eliminate"
        )
    k, z = system.k, system.z
    eliminant = _eliminant(system, w, method)
    if eliminant is None:  # w is a zero coordinate
        generic_curve = Curve._make(
            _normalise(w, w, z, k), w=w, z=z, k=k, kset=KSet.all(), system=system
        )
        return KCases({KSet.all(): generic_curve})
    g = _normalise(_select_factor(_w_factors(eliminant.g0, w), system, w), w, z, k)
    separate: dict[int, Curve] = {}
    for n in _candidates(g, eliminant, w, z, k):
        with warnings.catch_warnings():
            # A zero coordinate at one k is reflected in that k's curve.
            warnings.simplefilter("ignore", ZeroCoordinateWarning)
            curve_n = eliminate(system, w, k=n, method=method)
        g_n = sp.Poly(g.as_expr().xreplace({k: sp.Integer(n)}), w, z, domain=sp.QQ)
        if g_n.degree(w) == g.degree(w) and _normalise(g_n, w, z, None) == (
            curve_n.polynomial
        ):
            continue  # the generic curve is right at n (merge)
        separate[n] = curve_n
    generic = KSet.all() - KSet.of(*separate)
    pieces: dict[KSet, Curve] = {
        generic: Curve._make(g, w=w, z=z, k=k, kset=generic, system=system)
    }
    pieces.update({KSet.of(n): c for n, c in separate.items()})
    return KCases(pieces)


# -- argument checks ---------------------------------------------------------------


def _check(system: object, w: object, method: object) -> None:
    if not isinstance(system, System):
        raise TypeError(f"expected a flanpy.System, got {type(system).__name__}")
    if w not in system.unknowns:
        names = ", ".join(sp.sstr(u) for u in system.unknowns)
        raise ValueError(f"{w!r} is not an unknown of the system (unknowns: {names})")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")


# -- the eliminant (elimination.md, sections 2.1 and 4) ----------------------------


@dataclass(frozen=True)
class _Eliminant:
    """A non-zero element ``g0`` of ``I ∩ Q[w, z, k]`` of positive degree in ``w``.

    ``conditions`` are polynomials in ``k`` (non-zero) whose integer roots are the
    candidates of the elimination step: leading coefficients of the Gröbner basis
    elements (``X_GB``), or of the two polynomials in the resultant.
    """

    g0: Any
    conditions: tuple[Any, ...]


def _closure(system: System, w: Any) -> tuple[list[Any], dict[Any, Any]] | None:
    """Return the unknowns ``w`` depends on and their equations (zero coordinates 0).

    Returns ``None`` if ``w`` itself is a zero coordinate.
    """
    unknowns = system.unknowns
    zero = {unknowns[i]: sp.Integer(0) for i in range(len(unknowns))}
    for i in system._active:
        del zero[unknowns[i]]
    if w in zero:
        return None
    equations = {
        u: sp.expand(phi.xreplace(zero))
        for u, phi in system.equations.items()
        if u not in zero
    }
    needed, todo = {w}, [w]
    while todo:
        for v in equations[todo.pop()].free_symbols & equations.keys():
            if v not in needed:
                needed.add(v)
                todo.append(v)
    closure = [u for u in unknowns if u in needed]
    return closure, {u: equations[u] for u in closure}


def _eliminant(system: System, w: Any, method: Method) -> _Eliminant | None:
    found = _closure(system, w)
    if found is None:
        return None
    closure, equations = found
    polys = [u - phi for u, phi in equations.items()]
    if len(closure) == 1:
        return _Eliminant(polys[0], ())
    if method == "resultant" or (method == "auto" and len(closure) == 2):
        if len(closure) > 2:
            raise ValueError(
                f"method='resultant' needs at most one unknown besides {w} after "
                f"keeping only the unknowns it depends on, got "
                f"{len(closure) - 1}; use 'groebner' or 'auto'"
            )
        result = _resultant(polys, closure, w, system.z, system.k)
        if result is not None:
            return result
    return _groebner(polys, closure, w, system.z, system.k)


def _resultant(
    polys: list[Any], closure: list[Any], w: Any, z: Any, k: Any | None
) -> _Eliminant | None:
    """``Res_F(P_1, P_2)`` for the other unknown ``F`` (section 4); ``None`` if 0."""
    (other,) = [u for u in closure if u != w]
    g0 = sp.resultant(polys[0], polys[1], other)
    if sp.expand(g0) == 0:
        return None  # a common factor in F: let the Groebner path saturate
    conditions: tuple[Any, ...] = ()
    if k is not None:
        conditions = tuple(_k_content(sp.Poly(p, other).LC(), [w, z], k) for p in polys)
    return _Eliminant(g0, conditions)


def _groebner(
    polys: list[Any], closure: list[Any], w: Any, z: Any, k: Any | None
) -> _Eliminant:
    """Eliminate with a lex Groebner basis: other unknowns first, ``k`` last (2.1)."""
    others = [u for u in closure if u != w]
    main = [*others, w, z]
    params = [] if k is None else [k]
    gens = [*main, *params]
    basis = _BACKEND.groebner_lex([sp.Poly(p, *gens) for p in polys], gens)
    in_w = [g for g in basis if not g.as_expr().has(*others) and g.degree(w) > 0]
    if not in_w:
        # A component of excess dimension: saturate by the Jacobian determinant,
        # which is 1 at (0, c*), with the Rabinowitsch variable y first.
        y = sp.Dummy("y")
        delta = sp.Matrix(polys).jacobian(closure).det()
        main = [y, *main]
        gens = [*main, *params]
        basis = _BACKEND.groebner_lex(
            [sp.Poly(p, *gens) for p in [*polys, y * delta - 1]], gens
        )
        in_w = [g for g in basis if not g.as_expr().has(y, *others) and g.degree(w) > 0]
        if not in_w:
            raise AssertionError(
                "the saturated elimination ideal has no element in w; please report "
                "this as a bug"
            )
    g0 = min(in_w, key=lambda g: (g.degree(w), len(g.terms()))).as_expr()
    conditions: tuple[Any, ...] = ()
    if k is not None:
        conditions = tuple(
            sp.Poly(g.as_expr(), *main, domain=sp.QQ[k]).LC(order="lex") for g in basis
        )
    return _Eliminant(g0, conditions)


# -- exceptional k (elimination.md, sections 2.2-2.3) ------------------------------


def _k_content(expr: Any, gens: Sequence[Any], k: Any) -> Any:
    """Return the gcd in ``Q[k]`` of the coefficients of ``expr`` in ``gens``."""
    return _gcd_in_k(sp.Poly(expr, *gens).coeffs(), k)


def _zeros(expr: Any, k: Any) -> KSet:
    """Return the integers ``n >= 1`` where the non-zero ``expr(k)`` vanishes."""
    return KSet.where(sp.expand(expr), "==", k=k)


def _candidates(g: sp.Poly, eliminant: _Eliminant, w: Any, z: Any, k: Any) -> KSet:
    """``X_GB ∪ X_cont ∪ X_deg ∪ X_sqf ∪ X_prim`` (section 2.3)."""
    out = KSet()
    for condition in eliminant.conditions:
        out |= _zeros(condition, k)  # X_GB, or the resultant's degree drops
    for f in _w_free_factors(eliminant.g0, w):
        if not f.has(z):
            out |= _zeros(f, k)  # X_cont
    expr = g.as_expr()
    out |= _zeros(_k_content(sp.Poly(expr, w).LC(), [z], k), k)  # X_deg
    if g.degree(w) > 1:
        out |= _zeros(_k_content(sp.discriminant(expr, w), [z], k), k)  # X_sqf
    return out | _content_in_z(g, w, z, k)  # X_prim


def _content_in_z(g: sp.Poly, w: Any, z: Any, k: Any) -> KSet:
    """``X_prim``: the ``n >= 1`` where ``G(·, ·, n)`` has a content in ``z``.

    Exact (elimination.md, section 2.3): ``p``, a coefficient of least ``z``-degree,
    and ``q = sum_j lambda^j a_j``, coprime over ``Q(k)``; a common root of the
    specialised coefficients is a common root of ``p`` and ``q`` at ``n``, so ``n``
    is a root of ``Res_z(p, q)`` or of a leading coefficient in ``z``. Each
    candidate is tested with a gcd at ``k = n``.
    """
    a = [c for c in sp.Poly(g.as_expr(), w).all_coeffs() if c != 0]
    if len(a) < 2:
        return KSet()
    p = min(a, key=lambda c: sp.degree(c, z))
    lam = 1
    while True:
        q = sp.expand(sum(lam**j * c for j, c in enumerate(a)))
        if q != 0 and sp.degree(sp.gcd(p, q), z) == 0:
            break
        lam += 1
    if sp.degree(p, z) == 0:
        candidates = _zeros(p, k)
    else:
        candidates = (
            _zeros(sp.resultant(p, q, z), k)
            | _zeros(sp.Poly(p, z).LC(), k)
            | _zeros(sp.Poly(q, z).LC(), k)
        )
    hits = []
    for n in candidates:
        at_n = [sp.expand(c.xreplace({k: sp.Integer(n)})) for c in a]
        if sp.degree(sp.gcd_list(at_n), z) > 0:
            hits.append(n)
    return KSet.of(*hits)


# -- fixed k -----------------------------------------------------------------------


def _eliminate_fixed(system: System, w: Any, method: Method, kset: KSet) -> Curve:
    """Return the curve of ``w`` for a system without parameter."""
    z = system.z
    eliminant = _eliminant(system, w, method)
    factor = (
        w
        if eliminant is None
        else _select_factor(_w_factors(eliminant.g0, w), system, w)
    )
    return Curve._make(
        _normalise(factor, w, z, None), w=w, z=z, k=None, kset=kset, system=system
    )
