# SPDX-License-Identifier: GPL-3.0-or-later
"""Algebraic curves ``G(z, w, k) = 0`` satisfied by one coordinate of a system.

The design is recorded in ``dev/design/elimination.md`` (sections 1 and 3).
:class:`Curve` is public; it is created by :func:`~flanpy.eliminate` and
:func:`~flanpy.eliminate_in_k`. This module also holds the operations on a single
polynomial that elimination and :meth:`Curve.specialize` share: normalisation
(section 1.3) and the choice of the factor that the series satisfies (section 3).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, NoReturn

import sympy as sp

from flanpy._kset import KSet
from flanpy._series import series, series_in_k
from flanpy._system import System, ZeroCoordinateWarning

__all__ = ["Curve"]

# Values of k tried, in order, by the fixed-k factor test before falling back to
# series in Q[k] (elimination.md, section 3.2). Private, so tests can change it.
_TRIAL_KS: tuple[int, ...] = tuple(range(2, 34))

# The guard order when only one factor contains w (elimination.md, section 3.1).
_GUARD_ORDER = 4


class Curve:
    """An algebraic curve ``G(z, w) = 0`` through the series ``w(z)`` of a system.

    ``G`` is the minimal polynomial of the coordinate ``w(z)`` of the solution of
    :attr:`system`, over ``Q(k)(z)`` (or ``Q(z)`` at a fixed ``k``), with
    denominators cleared and normalised: integer coefficients, primitive over
    ``Z[k]``, and a positive leading coefficient in lex order ``w > z > k``. This
    fixes ``G`` up to nothing: Broda et al. (2020) eq. (17) comes out exactly.

    Curves are returned by :func:`~flanpy.eliminate` and
    :func:`~flanpy.eliminate_in_k`; they are not built directly.

    Notes
    -----
    For every ``n`` in :attr:`kset`, ``G_n = G(z, w, n)`` is

    - *annihilating*: ``G_n(z, w_n(z)) = 0``, with ``w_n`` the series at ``k = n``;
    - *without content in z*: its coefficients in ``w`` have no common factor;
    - *squarefree* in ``w``;
    - of *full degree* in ``w``: ``deg_w G_n = deg_w G``.

    For symbolic ``k``, ``G`` is irreducible over ``Q(k)``, in fact absolutely
    irreducible (elimination.md, section 1.2). At a fixed ``k`` computed by
    :func:`~flanpy.eliminate`, ``G`` is irreducible over ``Q``: it is the minimal
    polynomial of ``w_n``. For symbolic ``k``, irreducibility of each ``G_n`` is not
    certified (it fails for at most finitely many ``n``, by Noether's
    irreducibility theorem); :meth:`specialize` certifies any given ``n``. If
    ``deg_w G = 1`` there is no gap: each ``G_n`` is irreducible.

    Equality and hashing compare all the attributes. A curve is immutable and can
    be pickled.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy import System, eliminate
    >>> z, F = symbols("z F")
    >>> curve = eliminate(System({F: 1 + z * F**2}, z=z), F)
    >>> curve
    Curve(F**2*z - F + 1, w=F, z=z, kset=KSet([(1, None)]))
    >>> print(curve)
    F**2*z - F + 1 = 0
    """

    __slots__ = ("_k", "_kset", "_polynomial", "_system", "_w", "_z")

    _polynomial: Any
    _w: Any
    _z: Any
    _k: Any
    _kset: KSet
    _system: System

    def __init__(self) -> None:
        raise TypeError(
            "Curve objects are created by flanpy.eliminate and flanpy.eliminate_in_k"
        )

    @classmethod
    def _make(
        cls,
        polynomial: sp.Poly,
        *,
        w: sp.Symbol,
        z: sp.Symbol,
        k: sp.Symbol | None,
        kset: KSet,
        system: System,
    ) -> Curve:
        curve = object.__new__(cls)
        _set = object.__setattr__
        _set(curve, "_polynomial", polynomial)
        _set(curve, "_w", w)
        _set(curve, "_z", z)
        _set(curve, "_k", k)
        _set(curve, "_kset", kset)
        _set(curve, "_system", system)
        return curve

    # -- public interface ----------------------------------------------------------

    @property
    def polynomial(self) -> sp.Poly:
        """``G`` as a :class:`sympy.Poly` over ``ZZ``, generators ``(w, z, k)``.

        ``k`` is omitted at a fixed ``k`` and for systems without a parameter.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate
        >>> z, F = symbols("z F")
        >>> eliminate(System({F: 1 + z * F**2}, z=z), F).polynomial
        Poly(F**2*z - F + 1, F, z, domain='ZZ')
        """
        return self._polynomial

    @property
    def w(self) -> sp.Symbol:
        """The unknown whose series lies on the curve."""
        return self._w

    @property
    def z(self) -> sp.Symbol:
        """The size variable."""
        return self._z

    @property
    def k(self) -> sp.Symbol | None:
        """The parameter, or ``None`` (fixed ``k``, or a system without one)."""
        return self._k

    @property
    def kset(self) -> KSet:
        """The ``k`` where the invariants hold (see Notes of :class:`Curve`).

        ``KSet.all()`` for a system without parameter, ``{n}`` at a fixed ``k = n``,
        and the piece of :func:`~flanpy.eliminate_in_k` that holds the curve.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> print(eliminate(rk, R, k=2).kset)
        {2}
        """
        return self._kset

    @property
    def system(self) -> System:
        """The system whose coordinate :attr:`w` is the branch of interest on ``G``.

        At a fixed ``k = n`` this is ``system.specialize(n)``. Later steps use it to
        tell which solution of ``G = 0`` is the generating function.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> eliminate(rk, R, k=2).system
        System({R: 2*R**2*z + R*z + 3*z}, z=z)
        """
        return self._system

    def as_expr(self) -> sp.Expr:
        """``G`` as a SymPy expression.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate
        >>> z, F = symbols("z F")
        >>> eliminate(System({F: 1 + z * F**2}, z=z), F).as_expr()
        F**2*z - F + 1
        """
        return self._polynomial.as_expr()

    def specialize(self, k: int) -> Curve:
        """Return the curve at a fixed ``k``, certified to be the minimal polynomial.

        Substitutes ``k``, factors over ``Q``, and keeps the factor that the series
        at ``k`` satisfies (elimination.md, sections 2.5 and 3). The result equals
        ``eliminate(system, w, k=k)`` for the original system.

        Parameters
        ----------
        k : int
            A value in :attr:`kset`.

        Returns
        -------
        Curve
            The curve at ``k``, with ``kset = {k}`` and no parameter.

        Raises
        ------
        ValueError
            If the curve has no parameter, or ``k`` is not in :attr:`kset`.
        TypeError
            If ``k`` is not an ``int``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> eliminate_in_k(rk, R)[2].specialize(2)
        Curve(2*R**2*z + R*z - R + 3*z, w=R, z=z, kset=KSet([(2, 2)]))
        """
        if self._k is None:
            raise ValueError("the curve has no parameter k to specialize")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an int, got {k!r} of type {type(k).__name__}")
        if k not in self._kset:
            raise ValueError(f"k = {k} is not in the curve's kset {self._kset}")
        system = self._system.specialize(k)
        g = sp.Poly(
            self._polynomial.as_expr().xreplace({self._k: sp.Integer(k)}),
            self._w,
            self._z,
            domain=sp.QQ,
        )
        factor = _select_factor(_w_factors(g.as_expr(), self._w), system, self._w)
        return Curve._make(
            _normalise(factor, self._w, self._z, None),
            w=self._w,
            z=self._z,
            k=None,
            kset=KSet.of(k),
            system=system,
        )

    # -- value semantics -----------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (self._polynomial, self._w, self._z, self._k, self._kset, self._system)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Curve):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"Curve({sp.sstr(self.as_expr())}, w={sp.sstr(self._w)}, "
            f"z={sp.sstr(self._z)}"
            + ("" if self._k is None else f", k={sp.sstr(self._k)}")
            + f", kset={self._kset!r})"
        )

    def __str__(self) -> str:
        return f"{sp.sstr(self.as_expr())} = 0"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore, tuple(getattr(self, name) for name in Curve.__slots__))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Curve is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Curve is immutable")


def _restore(
    k: Any, kset: KSet, polynomial: Any, system: System, w: Any, z: Any
) -> Curve:
    """Rebuild a pickled :class:`Curve` (arguments in ``__slots__`` order)."""
    return Curve._make(polynomial, w=w, z=z, k=k, kset=kset, system=system)


Curve.__module__ = "flanpy"


# -- normalisation (elimination.md, section 1.3) -----------------------------------


def _normalise(g: Any, w: Any, z: Any, k: Any | None) -> sp.Poly:
    """``g`` over ``ZZ``, primitive over ``Z[k]``, positive lex-leading coefficient.

    ``g`` is an expression or a ``Poly`` in ``w``, ``z`` (and ``k``) with rational
    coefficients. Generators of the result: ``(w, z, k)``, or ``(w, z)``.
    """
    gens = (w, z) if k is None else (w, z, k)
    expr = g.as_expr() if isinstance(g, sp.Poly) else g
    p = sp.Poly(expr, *gens, domain=sp.QQ)
    _, p = p.clear_denoms(convert=True)
    _, p = p.primitive()
    if k is not None:
        content = sp.Poly(
            _gcd_in_k(sp.Poly(p.as_expr(), w, z).coeffs(), k), *gens, domain=sp.ZZ
        )
        p, rem = p.div(content)
        assert rem.is_zero, "content does not divide the polynomial"
        _, p = p.primitive()
    if p.LC(order="lex") < 0:
        p = -p
    return p


def _gcd_in_k(coefficients: Sequence[Any], k: Any) -> Any:
    """Return the gcd in ``Q[k]`` of the given polynomials in ``k`` (monic, or 1)."""
    polys = [sp.Poly(c, k, domain=sp.QQ) for c in coefficients]
    return functools.reduce(lambda a, b: a.gcd(b), polys).as_expr()


# -- choosing the factor (elimination.md, section 3) -------------------------------


def _w_factors(g: Any, w: Any) -> list[Any]:
    """Return the distinct irreducible factors over ``Q`` of ``g`` containing ``w``."""
    _, factors = sp.factor_list(sp.expand(g))
    return [f for f, _ in factors if f.has(w)]


def _w_free_factors(g: Any, w: Any) -> list[Any]:
    """Return the irreducible factors over ``Q`` of ``g`` without ``w``."""
    _, factors = sp.factor_list(sp.expand(g))
    return [f for f, _ in factors if not f.has(w)]


def _order_bound(factors: Sequence[Any], w: Any, z: Any) -> int:
    """``N`` such that only the minimal polynomial vanishes modulo ``z^(N+1)``.

    ``ord_z h(z, w(z)) <= deg_z Res_w(g, h) <= deg_w g deg_z h + deg_w h deg_z g``
    for the minimal polynomial ``g`` and any other factor ``h``
    (elimination.md, section 3.2). ``g`` is not known in advance, so take the
    maximum over all pairs; at least the guard order.
    """
    degrees = [(int(sp.degree(f, w)), int(sp.degree(f, z))) for f in factors]
    bound = _GUARD_ORDER
    for i, (wi, zi) in enumerate(degrees):
        for wj, zj in degrees[i + 1 :]:
            bound = max(bound, wi * zj + wj * zi)
    return bound


def _truncated_value(
    coefficients: Sequence[Sequence[Any]],
    series_w: Sequence[Any],
    n: int,
    zero: Any,
) -> list[Any]:
    """``h(z, w(z)) mod z^(n+1)``, by Horner's rule in ``w``.

    ``coefficients[j]`` is the dense list (in ``z``, from ``z^0``) of the
    coefficient of ``w^j`` in ``h``; ``series_w`` holds ``[z^0..z^n] w``.
    """
    acc = [zero] * (n + 1)
    for a in reversed(coefficients):
        product = [zero] * (n + 1)
        for i, x in enumerate(acc):
            if x != 0:
                for j in range(n + 1 - i):
                    product[i + j] = product[i + j] + x * series_w[j]
        for i, c in enumerate(a[: n + 1]):
            product[i] = product[i] + c
        acc = product
    return acc


def _dense(h: Any, w: Any, z: Any, convert: Callable[[Any], Any]) -> list[list[Any]]:
    """Coefficients of ``h`` in ``w`` (ascending), each dense in ``z``."""
    p = sp.Poly(h, w, z)
    out = [[] for _ in range(p.degree(w) + 1)]  # type: list[list[Any]]
    for (j, i), c in p.terms():
        row = out[j]
        row.extend([None] * (i + 1 - len(row)))
        row[i] = convert(c)
    zero = convert(sp.Integer(0))
    return [[zero if c is None else c for c in row] for row in out]


def _fraction(c: Any) -> Fraction:
    r = sp.Rational(c)
    return Fraction(int(r.p), int(r.q))


def _vanishing_at(
    factors: Sequence[Any], system: System, w: Any, n: int, k_value: int | None
) -> list[Any]:
    """Return the factors ``h`` with ``h(z, w(z)) = 0 mod z^(n+1)`` at fixed ``k``."""
    z = system.z
    with warnings.catch_warnings():
        # Zero coordinates that appear only at this k do not matter here.
        warnings.simplefilter("ignore", ZeroCoordinateWarning)
        values = series(system, n, k=k_value)[w]
    out = []
    for h in factors:
        hk = h if k_value is None else h.xreplace({system.k: sp.Integer(k_value)})
        value = _truncated_value(_dense(hk, w, z, _fraction), values, n, Fraction(0))
        if all(c == 0 for c in value):
            out.append(h)
    return out


def _vanishing_in_k(
    factors: Sequence[Any], system: System, w: Any, n: int
) -> list[Any]:
    """As :func:`_vanishing_at`, for symbolic ``k`` (coefficients in ``Q[k]``)."""
    z = system.z
    values = [c.as_expr() for c in series_in_k(system, n)[w]]
    out = []
    for h in factors:
        value = _truncated_value(_dense(h, w, z, lambda c: c), values, n, sp.Integer(0))
        if all(sp.expand(c) == 0 for c in value):
            out.append(h)
    return out


def _select_factor(factors: Sequence[Any], system: System, w: Any) -> Any:
    """Return the factor that the series ``w(z)`` of ``system`` satisfies.

    Exactly one factor vanishes modulo ``z^(N+1)``, ``N`` from :func:`_order_bound`
    (elimination.md, section 3.2). For symbolic ``k`` the test runs at fixed
    ``k = n0``: the minimal polynomial always passes, and another factor passes
    only if ``n0`` is a root of the lowest non-zero coefficient of ``h(z, w(z))``,
    so a single passing factor is the answer; otherwise the next ``n0`` is tried,
    and as a last resort the test runs in ``Q[k]``.
    """
    if not factors:
        raise AssertionError("no factor contains w")
    n = _order_bound(factors, w, system.z)
    if system.k is None:
        passing = _vanishing_at(factors, system, w, n, None)
    else:
        for n0 in _TRIAL_KS:
            passing = _vanishing_at(factors, system, w, n, n0)
            if len(passing) == 1:
                return passing[0]
            if not passing:
                break
        else:
            passing = _vanishing_in_k(factors, system, w, n)
    if len(passing) != 1:
        raise AssertionError(
            f"{len(passing)} factors vanish at the series of {w} (expected exactly "
            "one); please report this as a bug"
        )
    return passing[0]


# Curve reports "flanpy" as its module; list it so that pytest runs its examples.
__test__ = {"Curve": Curve}
