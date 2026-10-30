# SPDX-License-Identifier: GPL-3.0-or-later
"""Exact real algebraic numbers, and the root computations of step 4.

The design is recorded in ``dev/design/singularity.md`` (sections 2 and 5.1).
:class:`AlgebraicReal` is public. The module-level functions are private helpers
for the singularity analysis: the real and positive roots of a polynomial over
``Q``, rational separators between sorted roots, minimal polynomials of ``θ^2``,
``-θ`` and ``1/θ``, and the circle test (which other candidates lie on
``|z| = ρ``).

Everything is exact (SymPy polynomials over ``ZZ``/``QQ``, real-root isolation,
rational intervals). Floating point appears only in :meth:`AlgebraicReal.evalf`
and in the numerical *identification* step of :func:`_others_on_circle`, after an
exact test has decided that a root on the circle exists.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, NamedTuple, NoReturn

import mpmath
import sympy as sp

from flanpy._doctests import method_doctests

__all__ = ["AlgebraicReal"]

_Interval = tuple[Fraction, Fraction]


def _fraction(c: Any) -> Fraction:
    """Return a SymPy rational (or int) as a :class:`Fraction`."""
    r = sp.Rational(c)
    return Fraction(int(r.p), int(r.q))


def _rational(q: Fraction) -> sp.Rational:
    return sp.Rational(q.numerator, q.denominator)


def _as_poly(p: object, what: str) -> sp.Poly:
    """Return ``p`` as a univariate polynomial with rational coefficients over ``QQ``.

    ``p`` is a :class:`sympy.Poly` with one generator or an expression with at most
    one free symbol (a constant becomes a polynomial in a dummy symbol).
    """
    if isinstance(p, bool):
        raise TypeError(f"{what} must be a polynomial, got {p!r}")
    if isinstance(p, int | Fraction):
        p = _rational(Fraction(p))
    if isinstance(p, sp.Poly):
        if len(p.gens) != 1:
            raise ValueError(f"{what} must be univariate, got generators {p.gens}")
        poly = p
    else:
        try:
            expr = sp.sympify(p)
        except (sp.SympifyError, TypeError):
            raise TypeError(f"{what} must be a polynomial, got {p!r}") from None
        free = sorted(expr.free_symbols, key=str)
        if len(free) > 1:
            raise ValueError(f"{what} must be univariate, got symbols {free}")
        gen = free[0] if free else sp.Dummy("x")
        try:
            poly = sp.Poly(expr, gen)
        except sp.PolynomialError:
            raise ValueError(f"{what} must be a polynomial, got {expr}") from None
    try:
        return poly.set_domain(sp.QQ)
    except (sp.CoercionFailed, sp.polys.polyerrors.OperationNotSupported):
        raise ValueError(
            f"{what} must have rational coefficients, got {poly.as_expr()}"
        ) from None


def _normalised(p: sp.Poly) -> sp.Poly:
    """``p`` over ``ZZ``, primitive, with a positive leading coefficient."""
    _, q = p.clear_denoms(convert=True)
    _, q = q.primitive()
    return -q if q.LC() < 0 else q


class AlgebraicReal:
    """A real algebraic number: a minimal polynomial and the index of a real root.

    The number is the ``index``-th real root, in increasing order from ``0``, of
    ``polynomial``, which must be irreducible over ``Q``. It is stored normalised,
    as the minimal polynomial with integer coefficients, primitive and with a
    positive leading coefficient: rational coefficients, a sign or a content are
    removed silently, since they do not change the roots. This is the same
    indexing as :class:`sympy.CRootOf`. The pair is canonical, so equality and
    hashing are exact. An isolating interval with rational endpoints
    is kept as a cache and narrowed when needed (singularity.md, section 2.1).

    Parameters
    ----------
    polynomial : sympy.Poly or sympy.Expr
        An irreducible univariate polynomial with rational coefficients. The name
        of its variable is kept for printing but does not affect equality.
    index : int
        Which real root, ``0 <= index < number of real roots``.

    Raises
    ------
    TypeError
        If ``index`` is not an integer, or ``polynomial`` is not a polynomial.
    ValueError
        If ``polynomial`` is zero, constant, not univariate with rational
        coefficients, or reducible over ``Q``, or has no real root of that index.

    See Also
    --------
    sympy.CRootOf : the same number as a SymPy expression (:meth:`as_expr`).

    Notes
    -----
    Comparisons (``==``, ``<``, ``<=``, ``>``, ``>=``) with another
    ``AlgebraicReal``, an ``int`` or a :class:`~fractions.Fraction` are exact: two
    different numbers are told apart by refining their intervals, and equality is
    decided on the canonical pair. A rational number (degree 1) is equal to, and
    hashes like, the equal ``Fraction``. Floats are not compared (that would not
    be exact). ``float(x)`` gives the nearest double.

    An ``AlgebraicReal`` is immutable from the outside and can be pickled. The
    cached interval is invisible to equality and hashing; operations such as
    comparisons may narrow it.

    Examples
    --------
    The radius of convergence of Broda et al. (2020) ``R_k`` at ``k = 2``, the
    positive root of ``Δ_2(z) = 1 - 2z - 23z^2`` (section 3.1):

    >>> from fractions import Fraction
    >>> from sympy import symbols
    >>> from flanpy import AlgebraicReal
    >>> z = symbols("z")
    >>> rho = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
    >>> rho
    AlgebraicReal(23*z**2 + 2*z - 1, 1)
    >>> print(rho)
    0.169520847198537 (root 1 of 23*z**2 + 2*z - 1)
    >>> Fraction(1, 6) < rho < Fraction(1, 5)
    True
    >>> rho.sign(1 - 2 * z - 23 * z**2)  # exactly zero: Δ_2(ρ) = 0
    0

    Catalan's radius ``1/4`` is rational:

    >>> AlgebraicReal(4 * z - 1, 0) == Fraction(1, 4)
    True
    >>> AlgebraicReal(1 / 8 - z / 2, 0)  # normalised: the same root
    AlgebraicReal(4*z - 1, 0)
    >>> AlgebraicReal(z**2 - 1, 0)
    Traceback (most recent call last):
        ...
    ValueError: z**2 - 1 is not irreducible over Q: factors z - 1, z + 1
    """

    __slots__ = ("_coeffs", "_index", "_interval", "_poly")

    _poly: sp.Poly  # over ZZ, normalised and irreducible
    _coeffs: tuple[int, ...]  # its coefficients, highest degree first
    _index: int
    _interval: _Interval  # the cache (narrows; invisible to == and hash)

    def __init__(self, polynomial: sp.Poly | sp.Expr, index: int) -> None:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"index must be an int, got {index!r}")
        poly = _as_poly(polynomial, "polynomial")
        if poly.is_zero:
            raise ValueError("polynomial must be non-zero")
        if poly.degree() < 1:
            raise ValueError(f"polynomial must have degree >= 1, got {poly.as_expr()}")
        # Rational coefficients, sign and content do not change the roots.
        zz = _normalised(poly)
        expr = zz.as_expr()
        _, factors = zz.factor_list()
        if len(factors) != 1 or factors[0][1] != 1:
            names = ", ".join(sp.sstr(f.as_expr()) for f, _ in factors)
            raise ValueError(f"{expr} is not irreducible over Q: factors {names}")
        intervals = _intervals(zz)
        if not 0 <= index < len(intervals):
            raise ValueError(
                f"{expr} has {len(intervals)} real root(s); index {index} is out "
                "of range"
            )
        self._init(zz, index, intervals[index])

    def _init(self, poly: sp.Poly, index: int, interval: _Interval) -> None:
        _set = object.__setattr__
        _set(self, "_poly", poly)
        _set(self, "_coeffs", tuple(int(c) for c in poly.all_coeffs()))
        _set(self, "_index", index)
        _set(self, "_interval", interval)

    @classmethod
    def _make(cls, poly: sp.Poly, index: int, interval: _Interval) -> AlgebraicReal:
        """Wrap an already validated ``(poly, index)`` (no checks)."""
        new = object.__new__(cls)
        new._init(poly, index, interval)
        return new

    # -- public interface ----------------------------------------------------------

    @property
    def polynomial(self) -> sp.Poly:
        """The minimal polynomial, a :class:`sympy.Poly` over ``ZZ``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import AlgebraicReal
        >>> z = symbols("z")
        >>> AlgebraicReal(z**2 - 2, 1).polynomial
        Poly(z**2 - 2, z, domain='ZZ')
        """
        return self._poly

    @property
    def index(self) -> int:
        """Which real root of :attr:`polynomial`, in increasing order from ``0``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import AlgebraicReal
        >>> z = symbols("z")
        >>> sqrt2 = AlgebraicReal(z**2 - 2, 1)
        >>> sqrt2.index, sqrt2 > 0
        (1, True)
        """
        return self._index

    @property
    def interval(self) -> tuple[Fraction, Fraction]:
        """The current isolating interval ``(lo, hi)``, rational endpoints.

        The number lies in ``[lo, hi]`` and is the only root of :attr:`polynomial`
        there. The interval only ever narrows (:meth:`refine`, comparisons).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import AlgebraicReal
        >>> z = symbols("z")
        >>> AlgebraicReal(4 * z - 1, 0).interval  # rational: exact
        (Fraction(1, 4), Fraction(1, 4))
        >>> lo, hi = AlgebraicReal(z**2 - 2, 1).interval
        >>> lo**2 <= 2 <= hi**2
        True
        """
        return self._interval

    def refine(self, eps: int | Fraction) -> AlgebraicReal:
        """Narrow the isolating interval to width at most ``eps``; return ``self``.

        Parameters
        ----------
        eps : int or Fraction
            The target width, positive.

        Returns
        -------
        AlgebraicReal
            ``self``, whose :attr:`interval` is now at most ``eps`` wide.

        Raises
        ------
        ValueError
            If ``eps <= 0``.

        Examples
        --------
        >>> from fractions import Fraction
        >>> from sympy import symbols
        >>> from flanpy._algebraic import AlgebraicReal
        >>> z = symbols("z")
        >>> lo, hi = AlgebraicReal(z**2 - 2, 1).refine(Fraction(1, 10**6)).interval
        >>> hi - lo <= Fraction(1, 10**6) and lo**2 < 2 < hi**2
        True
        """
        if isinstance(eps, bool) or not isinstance(eps, int | Fraction):
            raise TypeError(f"eps must be an int or a Fraction, got {eps!r}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self._refine(Fraction(eps))
        return self

    def sign(self, p: sp.Poly | sp.Expr | int | Fraction) -> int:
        """Return the exact sign of ``p`` at this number: ``-1``, ``0`` or ``1``.

        ``p(θ) = 0`` exactly when the minimal polynomial divides ``p``; otherwise
        the interval is narrowed until ``p`` has no root in it, and ``p`` is
        evaluated at an endpoint.

        Parameters
        ----------
        p : sympy.Poly, sympy.Expr, int or Fraction
            A univariate polynomial with rational coefficients (its variable is
            ignored), or a constant.

        Returns
        -------
        int
            The sign of ``p(θ)``.

        Raises
        ------
        ValueError
            If ``p`` is not a univariate polynomial with rational coefficients.

        Examples
        --------
        Broda et al. (2020) eq. (18) at ``k = 54``: ``p_54`` has the rational
        positive root ``2/45`` and a double root ``-1/18``.

        >>> from sympy import symbols
        >>> from flanpy._algebraic import AlgebraicReal
        >>> z = symbols("z")
        >>> p54 = 4 * 54 * (2 * 54 + 27) * z**3 + 36 * 54 * z**2 - 54 * z - 4
        >>> AlgebraicReal(45 * z - 2, 0).sign(p54)
        0
        >>> sqrt2 = AlgebraicReal(z**2 - 2, 1)
        >>> sqrt2.sign(z - 1), sqrt2.sign(10**8 * z - 141421357)
        (1, -1)
        """
        return self._sign(_as_poly(p, "p"))

    def evalf(self, dps: int = 15) -> mpmath.mpf:
        """Return the number as an :class:`mpmath.mpf` with ``dps`` correct digits.

        Parameters
        ----------
        dps : int, default 15
            Decimal digits of relative accuracy.

        Returns
        -------
        mpmath.mpf
            A value within a relative error ``10^-dps`` (``0`` exactly for zero).
            mpmath prints at its global precision; use ``mpmath.nstr(value, dps)``
            to see all the digits.

        Examples
        --------
        >>> import mpmath
        >>> from sympy import symbols
        >>> from flanpy._algebraic import AlgebraicReal
        >>> z = symbols("z")
        >>> mpmath.nstr(AlgebraicReal(z**2 - 2, 1).evalf(40), 40)
        '1.41421356237309504880168872420969807857'
        >>> float(AlgebraicReal(4 * z - 1, 0))
        0.25
        """
        if isinstance(dps, bool) or not isinstance(dps, int) or dps < 1:
            raise ValueError(f"dps must be a positive int, got {dps!r}")
        lo, hi = self._interval
        if lo == hi:
            mid = lo
        else:
            while True:
                lo, hi = self._interval
                tolerance = Fraction(1, 10 ** (dps + 3)) * min(abs(lo), abs(hi))
                if (lo > 0 or hi < 0) and hi - lo <= tolerance:
                    break
                self._refine((hi - lo) / 1024)
            mid = (lo + hi) / 2
        with mpmath.workdps(dps + 5):
            value = mpmath.mpf(mid.numerator) / mid.denominator
        with mpmath.workdps(dps):
            return +value

    def as_expr(self) -> sp.Expr:
        """Return the same number as a SymPy expression (:class:`sympy.CRootOf`).

        Returns
        -------
        sympy.Expr
            ``CRootOf(polynomial, index)``; SymPy turns a rational root into a
            :class:`sympy.Rational`.

        Examples
        --------
        >>> from sympy import CRootOf, symbols
        >>> from flanpy._algebraic import AlgebraicReal
        >>> z = symbols("z")
        >>> AlgebraicReal(23 * z**2 + 2 * z - 1, 1).as_expr()
        CRootOf(23*z**2 + 2*z - 1, 1)
        >>> AlgebraicReal(4 * z - 1, 0).as_expr()
        1/4
        """
        return sp.CRootOf(self._poly.as_expr(), self._index)

    # -- exact arithmetic on the interval ------------------------------------------

    def _refine(self, eps: Fraction) -> None:
        lo, hi = self._interval
        if hi - lo <= eps:
            return
        if self._poly.degree() == 1:  # never: degree 1 is exact from the start
            return
        s, t = self._poly.refine_root(_rational(lo), _rational(hi), eps=_rational(eps))
        object.__setattr__(self, "_interval", (_fraction(s), _fraction(t)))

    def _shrink(self) -> None:
        """Narrow the interval by a factor of at least 4 (no-op if exact)."""
        lo, hi = self._interval
        if lo != hi:
            self._refine((hi - lo) / 4)

    def _sign(self, p: sp.Poly) -> int:
        if p.is_zero:
            return 0
        if p.degree() <= 0:
            c = p.LC()
            return 1 if c > 0 else -1
        m = sp.Poly(self._poly.as_expr(), self._poly.gen, domain=sp.QQ)
        x = self._poly.gen
        q = sp.Poly(p.as_expr().subs(p.gen, x), x, domain=sp.QQ)
        if q.rem(m).is_zero:
            return 0
        lo, hi = self._interval
        while lo != hi and q.count_roots(_rational(lo), _rational(hi)) > 0:
            self._shrink()
            lo, hi = self._interval
        v = q.eval(_rational(lo))
        return 1 if v > 0 else -1

    def _compare(self, other: object) -> int | None:
        """``sign(self - other)``, or ``None`` if ``other`` is not comparable."""
        if isinstance(other, AlgebraicReal):
            if self._key() == other._key():
                return 0
            while True:  # distinct numbers: refine until the intervals separate
                (a, b), (c, d) = self._interval, other._interval
                if b < c:
                    return -1
                if d < a:
                    return 1
                self._shrink()
                other._shrink()
        if isinstance(other, bool) or not isinstance(other, int | Fraction):
            return None
        q = Fraction(other)
        x = self._poly.gen
        return self._sign(sp.Poly(q.denominator * x - q.numerator, x, domain=sp.QQ))

    # -- value semantics -----------------------------------------------------------

    def _key(self) -> tuple[tuple[int, ...], int]:
        return (self._coeffs, self._index)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AlgebraicReal):
            return self._key() == other._key()
        c = self._compare(other)
        return NotImplemented if c is None else c == 0

    def __hash__(self) -> int:
        if len(self._coeffs) == 2:  # rational: hash like the equal Fraction
            a, b = self._coeffs
            return hash(Fraction(-b, a))
        return hash(self._key())

    def _ordered(self, other: object) -> int:
        c = self._compare(other)
        if c is None:
            raise TypeError(
                f"cannot compare AlgebraicReal with {type(other).__name__} exactly"
            )
        return c

    def __lt__(self, other: object) -> bool:
        return self._ordered(other) < 0

    def __le__(self, other: object) -> bool:
        return self._ordered(other) <= 0

    def __gt__(self, other: object) -> bool:
        return self._ordered(other) > 0

    def __ge__(self, other: object) -> bool:
        return self._ordered(other) >= 0

    def __float__(self) -> float:
        return float(self.evalf(20))

    def __repr__(self) -> str:
        return f"AlgebraicReal({sp.sstr(self._poly.as_expr())}, {self._index})"

    def __str__(self) -> str:
        if len(self._coeffs) == 2:  # rational: exact
            a, b = self._coeffs
            return str(sp.sstr(sp.Rational(-b, a)))
        value = mpmath.nstr(self.evalf(20), 15)
        return f"{value} (root {self._index} of {sp.sstr(self._poly.as_expr())})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore, (self._poly, self._index, self._interval))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("AlgebraicReal is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("AlgebraicReal is immutable")


def _restore(poly: sp.Poly, index: int, interval: _Interval) -> AlgebraicReal:
    """Rebuild a pickled :class:`AlgebraicReal`."""
    return AlgebraicReal._make(poly, index, interval)


AlgebraicReal.__module__ = "flanpy"


# -- roots of polynomials over Q ---------------------------------------------------


def _intervals(p: sp.Poly) -> list[_Interval]:
    """Isolating intervals of the real roots of the irreducible ``p``, increasing.

    A rational root (degree 1) gets its exact value.
    """
    if p.degree() == 1:
        a, b = (_fraction(c) for c in p.all_coeffs())
        return [(-b / a, -b / a)]
    return [(_fraction(s), _fraction(t)) for (s, t), _ in p.intervals()]


def _separate(roots: Sequence[AlgebraicReal]) -> None:
    """Refine sorted distinct roots until consecutive intervals are disjoint."""
    for r, s in itertools.pairwise(roots):
        while not r.interval[1] < s.interval[0]:
            r._shrink()
            s._shrink()


def _real_roots(f: sp.Poly | sp.Expr) -> list[AlgebraicReal]:
    """Return the distinct real roots of ``f != 0`` in ``Q[x]``, increasing.

    Each root carries its own minimal polynomial (a factor of ``f``).

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy._algebraic import _real_roots
    >>> z = symbols("z")
    >>> _real_roots((z**2 - 2) * (3 * z - 1) ** 2 * (z**2 + 1))
    [AlgebraicReal(z**2 - 2, 0), AlgebraicReal(3*z - 1, 0), AlgebraicReal(z**2 - 2, 1)]
    """
    poly = _as_poly(f, "f")
    if poly.is_zero:
        raise ValueError("f must be non-zero")
    roots: list[AlgebraicReal] = []
    if poly.degree() < 1:
        return roots
    _, factors = poly.factor_list()
    for factor, _ in factors:
        q = _normalised(factor)
        for i, interval in enumerate(_intervals(q)):
            roots.append(AlgebraicReal._make(q, i, interval))
    # Sort: intervals of different factors may overlap; refine until they do not.
    while True:
        roots.sort(key=lambda r: r.interval[0])
        overlapping = [
            (r, s)
            for r, s in itertools.pairwise(roots)
            if not r.interval[1] < s.interval[0]
        ]
        if not overlapping:
            return roots
        for r, s in overlapping:
            r._shrink()
            s._shrink()


def _positive_roots(f: sp.Poly | sp.Expr) -> list[AlgebraicReal]:
    """Return the distinct positive roots of ``f != 0`` in ``Q[x]``, increasing.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy._algebraic import _positive_roots
    >>> z = symbols("z")
    >>> _positive_roots(z * (23 * z**2 + 2 * z - 1))
    [AlgebraicReal(23*z**2 + 2*z - 1, 1)]
    """
    out = []
    for r in _real_roots(f):
        if r.sign(r.polynomial.gen) > 0:
            out.append(r)
    return out


def _separators(roots: Sequence[AlgebraicReal]) -> list[Fraction]:
    """Rationals ``q_0 < r_1 < q_1 < ... < r_s < q_s`` around sorted distinct roots.

    No ``q_j`` is a root. With no roots, ``[0]``.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy._algebraic import _real_roots, _separators
    >>> z = symbols("z")
    >>> roots = _real_roots(z**2 - 2)
    >>> qs = _separators(roots)
    >>> qs[0] < roots[0] < qs[1] < roots[1] < qs[2]
    True
    """
    if not roots:
        return [Fraction(0)]
    _separate(roots)
    qs = [roots[0].interval[0] - 1]
    for r, s in itertools.pairwise(roots):
        qs.append((r.interval[1] + s.interval[0]) / 2)
    qs.append(roots[-1].interval[1] + 1)
    return qs


# -- minimal polynomials of simple transforms --------------------------------------


def _identify(
    g: sp.Poly, theta: AlgebraicReal, image: Callable[[_Interval], _Interval]
) -> AlgebraicReal:
    """Return the real root of ``g`` that is the image of ``theta``.

    ``image`` maps any isolating interval of ``θ`` to an interval that contains the
    target, and shrinks with it. The target is a root of ``g``, so once exactly one
    real root of ``g`` meets the image, that root is the target.
    """
    candidates = _real_roots(g)
    while True:
        lo, hi = image(theta.interval)
        meeting = [
            r for r in candidates if not (r.interval[1] < lo or hi < r.interval[0])
        ]
        if len(meeting) == 1:
            return meeting[0]
        theta._shrink()
        for r in meeting:
            r._shrink()


def _square(theta: AlgebraicReal) -> AlgebraicReal:
    """``θ^2`` as an :class:`AlgebraicReal` (minimal polynomial by a resultant).

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy._algebraic import AlgebraicReal, _square
    >>> z = symbols("z")
    >>> _square(AlgebraicReal(z**2 - 2, 0))
    AlgebraicReal(z - 2, 0)
    >>> _square(AlgebraicReal(z**3 - 2, 0))
    AlgebraicReal(z**3 - 4, 0)
    """
    x = theta.polynomial.gen
    y = sp.Dummy("y")
    g = sp.Poly(sp.resultant(theta.polynomial.as_expr(), y - x**2, x).subs(y, x), x)

    def image(i: _Interval) -> _Interval:
        lo, hi = i
        if lo >= 0:
            return (lo * lo, hi * hi)
        if hi <= 0:
            return (hi * hi, lo * lo)
        return (Fraction(0), max(lo * lo, hi * hi))

    return _identify(g, theta, image)


def _negative(theta: AlgebraicReal) -> AlgebraicReal:
    """``-θ`` as an :class:`AlgebraicReal`.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy._algebraic import AlgebraicReal, _negative
    >>> z = symbols("z")
    >>> _negative(AlgebraicReal(z**3 - 2, 0))
    AlgebraicReal(z**3 + 2, 0)
    """
    m = theta.polynomial
    x = m.gen
    g = _normalised(sp.Poly(m.as_expr().subs(x, -x), x, domain=sp.QQ))
    n = len(_intervals(g))
    lo, hi = theta.interval
    return AlgebraicReal._make(g, n - 1 - theta.index, (-hi, -lo))


def _reciprocal(theta: AlgebraicReal) -> AlgebraicReal:
    """``1/θ`` as an :class:`AlgebraicReal` (``θ != 0``).

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy._algebraic import AlgebraicReal, _reciprocal
    >>> z = symbols("z")
    >>> _reciprocal(AlgebraicReal(23 * z**2 + 2 * z - 1, 1))
    AlgebraicReal(z**2 - 2*z - 23, 1)
    """
    m = theta.polynomial
    x = m.gen
    if m.as_expr() == x:
        raise ZeroDivisionError("1/θ with θ = 0")
    d = m.degree()
    reversed_m = sp.expand(x**d * m.as_expr().subs(x, 1 / x))
    g = _normalised(sp.Poly(reversed_m, x, domain=sp.QQ))
    while theta.interval[0] <= 0 <= theta.interval[1]:
        theta._shrink()

    def image(i: _Interval) -> _Interval:
        lo, hi = i
        return (1 / hi, 1 / lo)

    return _identify(g, theta, image)


# -- the circle test (singularity.md, section 5.1) ---------------------------------


class _OnCircle(NamedTuple):
    """The other candidates on ``|z| = ρ``, and how that was established.

    ``proved_none`` is ``True`` only when the exact tests exclude every root
    ``σ != ρ`` with ``|σ| = ρ`` (then ``roots`` is empty). Otherwise ``roots`` holds
    what is on the circle: ``-ρ`` (found exactly) and conjugate pairs (identified
    numerically, after an exact test). ``roots`` can then be empty, when the pairs
    with product ``ρ^2`` are not conjugate: that is a numerical conclusion, and the
    caller treats it as undecided (singularity.md, section 5.3).
    """

    proved_none: bool
    roots: list[complex]


def _others_on_circle(
    candidates: sp.Poly | sp.Expr, rho: AlgebraicReal, *, dps: int = 30
) -> _OnCircle:
    """Return the roots ``σ != ρ`` of ``candidates`` with ``|σ| = ρ`` (``ρ > 0``).

    Factor by factor ``f`` of ``candidates`` (the factor ``z`` is skipped: ``0`` is
    never on the circle, and ``P_z`` is degenerate):

    - a real root ``σ = -ρ``: ``f(-ρ) = 0``, an exact sign test;
    - a non-real root has ``σσ̄ = ρ^2``, so ``ρ^2`` is a product ``σ_i σ_j``
      (``i != j``) of two roots of ``f``. Those products are the roots of
      ``Q_f = P_f / G_f``, with ``P_f(y) = Res_x(f(x), x^d f(y/x))`` (all ordered
      pairs) and ``G_f(y) = Res_x(f(x), y - x^2)`` (the pairs ``i = j``). The
      number ``c`` of ordered pairs with product exactly ``ρ^2`` is the
      multiplicity in ``Q_f`` of ``μ``, the minimal polynomial of ``ρ^2``
      (exact: ``μ`` is irreducible, so all its roots have the same
      multiplicity). ``c = 0`` excludes the factor.

    If no factor has ``-ρ`` and every ``c`` is ``0``, the result is a proof that no
    other candidate lies on the circle (``proved_none``). Otherwise the ``c / 2``
    unordered pairs with product ``ρ^2`` are identified numerically: the roots of
    ``f`` are computed with mpmath at increasing precision until exactly ``c / 2``
    pair products lie within ``10^-(dps/2)`` of ``ρ^2`` and all others are at least
    a thousand times farther. Those pairs exist (``c`` is exact) and the others are
    at a positive distance from ``ρ^2``, so this terminates. The conjugate pairs
    among them are on the circle.

    That ``c > 0`` whenever ``gcd(Q_f, m_{ρ²}) != 1``, with
    ``m_{ρ²} = Res_x(m_ρ(x), y - x^2)``: the gcd means that some square ``ρ'^2``
    of a conjugate ``ρ'`` of ``ρ`` is a product ``σ_i σ_j``. An automorphism of the
    splitting field with ``ρ' ↦ ρ`` maps ``σ_i, σ_j`` to two distinct roots of
    ``f`` with product ``ρ^2``. So a pair with product exactly ``ρ^2`` exists and
    the numerical step never looks for something that is not there.

    Examples
    --------
    The curve ``zT^2 - T + z`` of ``T = z + zT^2`` (period 2): ``-1/2`` is on the
    circle of ``ρ = 1/2``.

    >>> from sympy import symbols
    >>> from flanpy._algebraic import AlgebraicReal, _others_on_circle
    >>> z = symbols("z")
    >>> _others_on_circle(z * (1 - 4 * z**2), AlgebraicReal(2 * z - 1, 0))
    _OnCircle(proved_none=False, roots=[(-0.5+0j)])
    >>> _others_on_circle((z - 1) * (z**2 - z + 1), AlgebraicReal(z - 1, 0)).roots
    [(0.5-0.8660254037844386j), (0.5+0.8660254037844386j)]
    >>> _others_on_circle(z * (4 * z - 1) * (z + 1), AlgebraicReal(4 * z - 1, 0))
    _OnCircle(proved_none=True, roots=[])
    """
    poly = _as_poly(candidates, "candidates")
    if rho <= 0:
        raise ValueError(f"rho must be positive, got {rho}")
    x = poly.gen
    y = sp.Dummy("y")
    square = _square(rho).polynomial  # μ, the minimal polynomial of ρ^2
    mu = sp.Poly(square.as_expr().subs(square.gen, y), y, domain=sp.QQ)
    proved_none = True
    found: list[complex] = []
    _, factors = poly.factor_list()
    for factor, _ in factors:
        f = _normalised(factor)
        if f.degree() < 1 or f.as_expr() == x:
            continue
        if rho.sign(sp.Poly(f.as_expr().subs(x, -x), x, domain=sp.QQ)) == 0:
            proved_none = False
            found.append(complex(-rho.evalf(dps)))
        if f.degree() < 2:
            continue
        d = f.degree()
        fe = f.as_expr()
        products = sp.Poly(
            sp.resultant(fe, sp.expand(x**d * fe.subs(x, y / x)), x), y, domain=sp.QQ
        )
        squares = sp.Poly(sp.resultant(fe, y - x**2, x), y, domain=sp.QQ)
        pairs, remainder = products.div(squares)
        assert remainder.is_zero
        c = 0
        while True:
            quotient, remainder = pairs.div(mu)
            if not remainder.is_zero:
                break
            pairs, c = quotient, c + 1
        if c:
            proved_none = False
            found.extend(_conjugate_pairs_on_circle(f, rho, c // 2, dps))
    found.sort(key=lambda s: (math.atan2(s.imag, s.real), s.real))
    return _OnCircle(proved_none, found)


def _conjugate_pairs_on_circle(
    f: sp.Poly, rho: AlgebraicReal, count: int, dps: int
) -> list[complex]:
    """Return the non-real roots of ``f`` with ``σσ̄ = ρ^2`` (numerical).

    ``count`` unordered pairs of distinct roots have product exactly ``ρ^2``.
    """
    coeffs = [int(c) for c in f.all_coeffs()]
    while True:
        with mpmath.workdps(dps):
            target = rho.evalf(dps) ** 2
            roots = mpmath.polyroots(coeffs, maxsteps=400, extraprec=4 * dps)
            tol = mpmath.mpf(10) ** (-(dps // 2))
            gaps = sorted(
                (abs(roots[i] * roots[j] - target), i, j)
                for i, j in itertools.combinations(range(len(roots)), 2)
            )
            near = gaps[:count]
            rest = gaps[count:]
            if near[-1][0] < tol and (not rest or rest[0][0] > 1000 * tol):
                out = []
                for _, i, j in near:
                    a, b = roots[i], roots[j]
                    if abs(mpmath.im(a)) > tol and abs(a - mpmath.conj(b)) < tol:
                        out.extend([complex(a), complex(b)])
                return out
        dps *= 2
        if dps > 4000:  # pragma: no cover - the loop terminates (see the caller)
            raise RuntimeError("could not identify the roots on the circle")


# Methods of the public class, which __init__.py's doctests miss (_doctests.py).
__test__ = method_doctests(AlgebraicReal)
