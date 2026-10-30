# SPDX-License-Identifier: GPL-3.0-or-later
"""Sets of alphabet sizes ``k >= 1``, and results that depend on ``k`` piecewise.

The design is recorded in ``dev/design/kset.md``. Both classes are public;
:class:`KCases` became public with step 3, which returns one
(``dev/design/elimination.md``, section 6).
"""

from __future__ import annotations

import bisect
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping
from fractions import Fraction
from functools import lru_cache
from typing import Any, Literal, NoReturn, SupportsIndex

import sympy as sp

__all__ = ["KCases", "KSet"]

Relation = Literal["==", "!=", "<", "<=", ">", ">="]

# The signs of p(k) for which each relation "p(k) rel 0" holds.
_SIGNS: dict[str, frozenset[int]] = {
    "==": frozenset({0}),
    "!=": frozenset({-1, 1}),
    "<": frozenset({-1}),
    "<=": frozenset({-1, 0}),
    ">": frozenset({1}),
    ">=": frozenset({0, 1}),
}

_Interval = tuple[int, int | None]
_Piece = tuple[int, int | None, int]


def _integer(value: object, what: str) -> int:
    """``value`` as an ``int`` (``operator.index``); ``bool`` is rejected."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{what} must be an integer, got {value!r}")
    return operator.index(value)


def _canonical(intervals: Iterable[_Interval]) -> tuple[_Interval, ...]:
    """Sort and merge overlapping or adjacent intervals (kset.md, section 1)."""
    out: list[_Interval] = []
    for lo, hi in sorted(intervals, key=lambda iv: iv[0]):
        if out:
            last_lo, last_hi = out[-1]
            if last_hi is None:
                break
            if lo <= last_hi + 1:
                out[-1] = (last_lo, None if hi is None else max(hi, last_hi))
                continue
        out.append((lo, hi))
    return tuple(out)


def _complement(intervals: tuple[_Interval, ...]) -> tuple[_Interval, ...]:
    out: list[_Interval] = []
    start = 1
    for lo, hi in intervals:
        if lo > start:
            out.append((start, lo - 1))
        if hi is None:
            return tuple(out)
        start = hi + 1
    out.append((start, None))
    return tuple(out)


def _intersection(
    a: tuple[_Interval, ...], b: tuple[_Interval, ...]
) -> tuple[_Interval, ...]:
    out: list[_Interval] = []
    i = j = 0
    while i < len(a) and j < len(b):
        (alo, ahi), (blo, bhi) = a[i], b[j]
        lo = max(alo, blo)
        hi = bhi if ahi is None else ahi if bhi is None else min(ahi, bhi)
        if hi is None or lo <= hi:
            out.append((lo, hi))
        # Advance the interval that ends first.
        if ahi is not None and (bhi is None or ahi < bhi):
            i += 1
        else:
            j += 1
    return _canonical(out)


class KSet:
    """A set of alphabet sizes: a finite union of integer intervals in ``k >= 1``.

    Every condition on ``k`` that the pipeline produces is a sign or vanishing
    condition on a polynomial in ``k`` over ``Q``, so the set of ``k`` where a result
    holds is a finite union of integer points and integer intervals, possibly
    unbounded above. Build one from such a condition with :meth:`where`.

    Parameters
    ----------
    intervals : iterable of (int, int or None), optional
        Pairs ``(lo, hi)`` meaning ``{lo, ..., hi}``; ``hi = None`` means
        ``[lo, ∞)``. Any order; overlapping and adjacent pairs are merged. Empty by
        default.

    Raises
    ------
    TypeError
        If an interval is not a pair of integers (``bool`` is rejected), or ``lo``
        is ``None``.
    ValueError
        If ``lo < 1`` (the domain is ``k >= 1``) or ``lo > hi``.

    See Also
    --------
    where : the set where a polynomial condition holds.

    Notes
    -----
    The set is stored in a canonical form, :attr:`intervals`: sorted, disjoint
    intervals with at least one integer between neighbours, only the last one
    possibly unbounded. Equality and hashing compare that form. A ``KSet`` is
    immutable and either finite or contains a tail ``[n0, ∞)`` (then its complement
    is finite), see :attr:`tail_start`.

    Operators, as for :class:`frozenset`: ``a & b``, ``a | b``, ``a - b``,
    ``~a`` (complement in ``k >= 1``), ``n in a`` (``n`` an integer; ``n <= 0`` is
    never a member), ``bool(a)`` (non-empty), and inclusion ``a <= b``,
    ``a < b``, ``a >= b``, ``a > b``. A finite set iterates over its elements in
    increasing order; iterating over an infinite set raises ``ValueError``.

    Examples
    --------
    >>> from flanpy import KSet
    >>> a = KSet([(55, None), (1, 53)])
    >>> print(a)
    [1, 53] ∪ [55, ∞)
    >>> a
    KSet([(1, 53), (55, None)])
    >>> print(~a)
    {54}
    >>> print(a & KSet([(50, 60)]))
    [50, 53] ∪ [55, 60]
    >>> 54 in a, 0 in a
    (False, False)
    >>> list(KSet([(2, 4), (7, 7)]))
    [2, 3, 4, 7]
    >>> print(KSet())
    ∅
    """

    __slots__ = ("_intervals",)
    _intervals: tuple[_Interval, ...]

    def __init__(self, intervals: Iterable[tuple[int, int | None]] = ()) -> None:
        parsed: list[_Interval] = []
        for item in intervals:
            try:
                lo, hi = item
            except (TypeError, ValueError):
                raise TypeError(
                    f"an interval is a pair (lo, hi), got {item!r}"
                ) from None
            if lo is None:
                raise TypeError("lo must be an integer; only hi can be None (∞)")
            lo = _integer(lo, "lo")
            hi = None if hi is None else _integer(hi, "hi")
            if lo < 1:
                raise ValueError(
                    f"the domain is k >= 1, but the interval ({lo}, {hi}) starts "
                    "below 1"
                )
            if hi is not None and hi < lo:
                raise ValueError(f"empty interval ({lo}, {hi}): lo > hi")
            parsed.append((lo, hi))
        object.__setattr__(self, "_intervals", _canonical(parsed))

    @classmethod
    def _make(cls, intervals: tuple[_Interval, ...]) -> KSet:
        """Wrap intervals already in canonical form (no validation)."""
        new = object.__new__(cls)
        object.__setattr__(new, "_intervals", intervals)
        return new

    # -- construction ----------------------------------------------------------

    @classmethod
    def all(cls) -> KSet:
        """Return the whole domain ``[1, ∞)``.

        Returns
        -------
        KSet
            Every alphabet size ``k >= 1``.

        Examples
        --------
        >>> from flanpy import KSet
        >>> print(KSet.all())
        [1, ∞)
        >>> KSet.all() == ~KSet()
        True
        """
        return cls._make(((1, None),))

    @classmethod
    def of(cls, *ks: int) -> KSet:
        """Return the finite set of the given alphabet sizes.

        Parameters
        ----------
        *ks : int
            Integers ``>= 1``, in any order, repetitions allowed.

        Returns
        -------
        KSet
            The set ``{ks}`` (empty if no argument is given).

        Raises
        ------
        TypeError
            If an argument is not an integer.
        ValueError
            If an argument is ``< 1``.

        Examples
        --------
        >>> from flanpy import KSet
        >>> print(KSet.of(54, 2))
        {2} ∪ {54}
        >>> print(KSet.of(2, 3, 4, 3))
        [2, 4]
        """
        return cls((n, n) for n in ks)

    @classmethod
    def where(
        cls,
        p: sp.Expr | sp.Poly | int | Fraction,
        rel: Relation,
        *,
        k: sp.Symbol,
        den: sp.Expr | sp.Poly | int | Fraction | None = None,
    ) -> KSet:
        """Return the set of ``k >= 1`` where ``p(k) rel 0`` (or ``p(k)/den(k) rel 0``).

        Parameters
        ----------
        p : sympy.Expr, sympy.Poly, int or Fraction
            A polynomial in ``k`` with rational coefficients.
        rel : {"==", "!=", "<", "<=", ">", ">="}
            The relation to 0.
        k : sympy.Symbol
            The variable of ``p`` (required, also when ``p`` is constant).
        den : sympy.Expr, sympy.Poly, int or Fraction, optional
            A non-zero polynomial denominator. The result is the set of ``k`` where
            ``den(k) != 0`` and ``p(k)/den(k) rel 0``. ``den`` is **not** cancelled
            against ``p``: a root of ``den`` is never in the result, even where the
            fraction in lowest terms is defined (kset.md, section 3). Cancel first
            (:func:`sympy.cancel`) for the lowest-terms reading.

        Returns
        -------
        KSet
            ``{n >= 1 : den(n) != 0 and p(n)/den(n) rel 0}``, with ``den = 1`` if
            omitted.

        Raises
        ------
        TypeError
            If ``k`` is not a :class:`sympy.Symbol`, or ``p`` or ``den`` has an
            unsupported type.
        ValueError
            If ``rel`` is not one of the six relations; if ``p`` or ``den`` is not a
            polynomial in ``k`` alone with rational coefficients (``1/k``,
            ``sqrt(k)``, a rational expression, another symbol, a float or an
            algebraic coefficient); or if ``den`` is zero.

        Notes
        -----
        Exact throughout; no floating point. With ``Q`` the squarefree part of ``p``
        (``p·den`` if ``den`` is given):

        1. the integer roots ``n >= 1`` of ``Q`` are found exactly (its rational
           roots) and divided out;
        2. the remaining real roots in ``[1, ∞)`` are isolated with
           :meth:`sympy.Poly.intervals` (Vincent-Akritas-Strzeboński continued
           fractions; Akritas & Strzeboński 2005) and refined with
           :meth:`sympy.Poly.refine_root` until no isolating interval contains an
           integer, which gives the floor of each root;
        3. between consecutive breakpoints the sign is constant (intermediate value
           theorem), and is found by evaluating ``p`` exactly at one integer.

        Since ``den(n) != 0`` implies ``sign(p(n)/den(n)) = sign(p(n)·den(n))``,
        a denominator only multiplies the polynomial and removes the roots of
        ``den``.

        Examples
        --------
        The discriminant of ``p_k`` in Broda et al. (2020), eq. (18), is
        ``32 k^2 (k - 54)^3``; for ``k >= 1`` (the root ``k = 0`` is outside the
        domain):

        >>> from sympy import discriminant, factor, symbols
        >>> from flanpy import KSet
        >>> z, k = symbols("z k")
        >>> p_k = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
        >>> disc = discriminant(p_k, z)
        >>> factor(disc)
        32*k**2*(k - 54)**3
        >>> print(KSet.where(disc, "<", k=k))
        [1, 53]
        >>> print(KSet.where(disc, "==", k=k))
        {54}
        >>> print(KSet.where(disc, ">", k=k))
        [55, ∞)

        Roots between integers, and a denominator that is not cancelled:

        >>> print(KSet.where(2 * k**2 - 7 * k + 4, "<=", k=k))
        [1, 2]
        >>> print(KSet.where(k - 1, ">", k=k, den=k - 1))
        [2, ∞)
        >>> KSet.where(1 / k, ">", k=k)
        Traceback (most recent call last):
        ...
        ValueError: p is not a polynomial in k over Q: 1/k; use den= for a denominator
        """
        if rel not in _SIGNS:
            raise ValueError(
                f"rel must be one of {', '.join(map(repr, _SIGNS))}; got {rel!r}"
            )
        if not isinstance(k, sp.Symbol):
            raise TypeError(f"k must be a sympy.Symbol, got {k!r}")
        num = _poly(p, k, "p")
        if den is None:
            return cls._from_pieces(_sign_pieces(_key(num)), _SIGNS[rel])
        d = _poly(den, k, "den")
        if d.is_zero:
            raise ValueError("den is the zero polynomial")
        poles = cls._from_pieces(_sign_pieces(_key(d)), _SIGNS["=="])
        return cls._from_pieces(_sign_pieces(_key(num * d)), _SIGNS[rel]) - poles

    @classmethod
    def _from_pieces(cls, pieces: tuple[_Piece, ...], signs: frozenset[int]) -> KSet:
        return cls._make(_canonical((lo, hi) for lo, hi, s in pieces if s in signs))

    # -- properties ------------------------------------------------------------

    @property
    def intervals(self) -> tuple[tuple[int, int | None], ...]:
        """The canonical form: sorted, disjoint, non-adjacent intervals.

        Each ``(lo, hi)`` means ``{lo, ..., hi}``, and ``hi = None`` means
        ``[lo, ∞)`` (only possible for the last interval).

        Examples
        --------
        >>> from flanpy import KSet
        >>> KSet([(5, None), (1, 2), (3, 3)]).intervals
        ((1, 3), (5, None))
        """
        return self._intervals

    @property
    def tail_start(self) -> int | None:
        """The least ``n0`` with ``[n0, ∞)`` contained in the set; ``None`` if finite.

        A ``KSet`` contains all sufficiently large ``k`` exactly when it is
        infinite. A limit as ``k → ∞`` is meaningful for a result whose set is
        infinite; ``tail_start`` says from which ``k`` on it holds.

        Examples
        --------
        >>> from flanpy import KSet
        >>> KSet([(1, 53), (55, None)]).tail_start
        55
        >>> KSet.of(54).tail_start is None
        True
        """
        if self._intervals and self._intervals[-1][1] is None:
            return self._intervals[-1][0]
        return None

    @property
    def is_finite(self) -> bool:
        """Whether the set is finite (equivalently, has no tail ``[n0, ∞)``).

        Examples
        --------
        >>> from flanpy import KSet
        >>> KSet.of(2, 54).is_finite, KSet.all().is_finite
        (True, False)
        """
        return self.tail_start is None

    @property
    def finite_part(self) -> KSet:
        """The elements below the tail, or the whole set if it is finite.

        Always finite: ``a == a.finite_part | KSet([(a.tail_start, None)])`` for an
        infinite ``a``.

        Examples
        --------
        >>> from flanpy import KSet
        >>> print(KSet([(1, 3), (10, None)]).finite_part)
        [1, 3]
        >>> print(KSet.of(2, 54).finite_part)
        {2} ∪ {54}
        """
        if self.tail_start is None:
            return self
        return KSet._make(self._intervals[:-1])

    def isdisjoint(self, other: KSet) -> bool:
        """Whether the two sets have no element in common.

        Parameters
        ----------
        other : KSet
            The other set.

        Returns
        -------
        bool
            ``True`` iff ``self & other`` is empty.

        Examples
        --------
        >>> from flanpy import KSet
        >>> KSet([(1, 53)]).isdisjoint(KSet([(54, None)]))
        True
        """
        if not isinstance(other, KSet):
            raise TypeError(f"expected a KSet, got {type(other).__name__}")
        return not _intersection(self._intervals, other._intervals)

    # -- set operations --------------------------------------------------------

    def __and__(self, other: object) -> KSet:
        if not isinstance(other, KSet):
            return NotImplemented
        return KSet._make(_intersection(self._intervals, other._intervals))

    def __or__(self, other: object) -> KSet:
        if not isinstance(other, KSet):
            return NotImplemented
        return KSet._make(_canonical(self._intervals + other._intervals))

    def __sub__(self, other: object) -> KSet:
        if not isinstance(other, KSet):
            return NotImplemented
        return KSet._make(_intersection(self._intervals, _complement(other._intervals)))

    def __invert__(self) -> KSet:
        return KSet._make(_complement(self._intervals))

    def __le__(self, other: object) -> bool:
        if not isinstance(other, KSet):
            return NotImplemented
        return _intersection(self._intervals, other._intervals) == self._intervals

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, KSet):
            return NotImplemented
        return self <= other and self != other

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, KSet):
            return NotImplemented
        return other <= self

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, KSet):
            return NotImplemented
        return other < self

    def __contains__(self, n: object) -> bool:
        m = _integer(n, "a member of a KSet")
        i = bisect.bisect_right(self._intervals, m, key=lambda iv: iv[0]) - 1
        if i < 0:
            return False
        hi = self._intervals[i][1]
        return hi is None or m <= hi

    def __bool__(self) -> bool:
        return bool(self._intervals)

    def __iter__(self) -> Iterator[int]:
        # Not a generator function: the error must come from iter(), not next().
        if not self.is_finite:
            raise ValueError(f"cannot iterate over the infinite set {self}")
        return (
            n for lo, hi in self._intervals if hi is not None for n in range(lo, hi + 1)
        )

    # -- value semantics -------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KSet):
            return NotImplemented
        return self._intervals == other._intervals

    def __hash__(self) -> int:
        return hash(self._intervals)

    def __repr__(self) -> str:
        if not self._intervals:
            return "KSet()"
        return f"KSet({list(self._intervals)!r})"

    def __str__(self) -> str:
        if not self._intervals:
            return "∅"
        return " ∪ ".join(
            f"[{lo}, ∞)" if hi is None else f"{{{lo}}}" if lo == hi else f"[{lo}, {hi}]"
            for lo, hi in self._intervals
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (KSet, (self._intervals,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("KSet is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("KSet is immutable")


# Public names report flanpy as their module (tracebacks, reprs, pickling).
KSet.__module__ = "flanpy"


# -- conditions: sign of a polynomial at the integers k >= 1 -------------------


def _poly(p: object, k: sp.Symbol, what: str) -> sp.Poly:
    """``p`` as a ``Poly`` in ``k`` over ``QQ``, or an error (kset.md, section 3)."""
    if isinstance(p, bool):
        raise TypeError(f"{what} must be a polynomial in k, got {p!r}")
    if isinstance(p, int | Fraction):
        return sp.Poly(sp.Rational(p.numerator, p.denominator), k, domain=sp.QQ)
    bad = ValueError(
        f"{what} is not a polynomial in {k} over Q: {p}"
        + ("; use den= for a denominator" if what == "p" else "")
    )
    if isinstance(p, sp.Poly):
        if p.gens != (k,):
            raise ValueError(
                f"{what} must be a polynomial in {k} alone, got generators {p.gens}"
            )
        poly = p
    elif isinstance(p, sp.Expr):
        others = p.free_symbols - {k}
        if others:
            names = ", ".join(sorted(map(str, others)))
            raise ValueError(f"{what} must be a polynomial in {k} alone; has {names}")
        try:
            poly = sp.Poly(p, k)
        except sp.PolynomialError:
            raise bad from None
    else:
        raise TypeError(
            f"{what} must be a sympy expression or Poly, an int or a Fraction, "
            f"got {type(p).__name__}"
        )
    if poly.domain not in (sp.ZZ, sp.QQ):
        raise bad
    return poly.set_domain(sp.QQ)


def _key(poly: sp.Poly) -> tuple[Fraction, ...]:
    """Coefficients, highest degree first, as a hashable tuple of ``Fraction``."""
    return tuple(Fraction(int(c.p), int(c.q)) for c in poly.all_coeffs())


@lru_cache(maxsize=1024)
def _sign_pieces(coeffs: tuple[Fraction, ...]) -> tuple[_Piece, ...]:
    """Maximal integer intervals of ``k >= 1`` where ``p`` has constant sign.

    ``coeffs`` are the coefficients of ``p`` in ``Q[k]``, highest degree first.
    Returns ``(lo, hi, s)`` triples covering ``[1, ∞)`` in order, with ``hi = None``
    for the last one and ``s`` the sign of ``p`` on ``{lo, ..., hi}``. Algorithm:
    kset.md, section 3 (integer roots first, then the floors of the other real roots
    by VAS isolation, then one exact evaluation per piece). Memoised, because callers
    often ask for several relations of the same polynomial.
    """
    # Clear denominators: an integer polynomial with the same signs.
    scale = math.lcm(*(c.denominator for c in coeffs))
    ints = [int(c * scale) for c in coeffs]
    if not any(ints):
        return ((1, None, 0),)

    def sign(n: int) -> int:
        v = 0
        for c in ints:
            v = v * n + c
        return (v > 0) - (v < 0)

    k = sp.Dummy("k")
    q = sp.Poly(ints, k, domain=sp.ZZ).sqf_part()
    # 1. Integer roots >= 1, exactly. Dividing them out leaves a squarefree q with
    # no integer root, so step 2 can refine until no interval holds an integer; no
    # reliance on whether SymPy's isolating intervals are open or closed (for
    # (k - 20)(3k - 61) it isolates 61/3 in (20, 21), and 20 is the other root).
    integer_roots = sorted(int(r) for r in q.ground_roots() if r.is_integer and r >= 1)
    for n in integer_roots:
        q = q.exquo(sp.Poly(k - n, k, domain=sp.ZZ))
    # 2. Floors of the other real roots >= 1 (exact VAS isolation, then refinement).
    floors = []
    for (a, b), _ in q.intervals(inf=1):
        while math.ceil(Fraction(a.p, a.q)) <= math.floor(Fraction(b.p, b.q)):
            a, b = q.refine_root(a, b, eps=(b - a) / 4)
        floors.append(math.floor(Fraction(a.p, a.q)))
    # 3. Cut [1, ∞) between c and c + 1 for each cut c: on both sides of an integer
    # root n (c = n - 1, n), and at the floor m of any other root (c = m). No real
    # root lies in the span of a resulting interval, so the sign is constant on it.
    cuts = sorted(
        {c for n in integer_roots for c in (n - 1, n) if c >= 1} | set(floors)
    )
    starts = [1, *(c + 1 for c in cuts)]
    spans: list[_Interval] = [
        (lo, hi) for lo, hi in zip(starts, [*cuts, None], strict=True)
    ]
    pieces: list[_Piece] = []
    for lo, hi in spans:
        s = sign(lo)
        if pieces and pieces[-1][2] == s:
            pieces[-1] = (pieces[-1][0], hi, s)
        else:
            pieces.append((lo, hi, s))
    lc = ints[0]
    assert pieces[-1][2] == (lc > 0) - (lc < 0), "sign at infinity"
    return tuple(pieces)


# -- piecewise results ----------------------------------------------------------


class KCases[T]:
    """A result that takes one value on each of several disjoint :class:`KSet` s.

    The pieces are pairwise disjoint; their union is :attr:`covered`, where the
    result is known.
    Everything else, :attr:`uncovered`, is undecided or not computed, and is part
    of the value: it is printed and compared, and looking up a ``k`` there raises
    ``KeyError``. Empty pieces are dropped (they hold no ``k``).

    Parameters
    ----------
    pieces : mapping or iterable of (KSet, value)
        The pieces and their values.

    Raises
    ------
    TypeError
        If a key is not a :class:`KSet`.
    ValueError
        If two pieces overlap.

    Notes
    -----
    Equal values on different pieces are not merged (structural equality of SymPy
    values is unreliable); a function that returns a ``KCases`` merges where it can
    prove equality, as :func:`~flanpy.eliminate_in_k` does. Looking up ``k`` with
    ``cases[k]`` returns the value of the piece that contains it.

    Examples
    --------
    >>> from flanpy._kset import KCases, KSet
    >>> cases = KCases({KSet([(1, 53)]): "Case I", KSet.of(54): "Case II"})
    >>> cases[10], cases[54]
    ('Case I', 'Case II')
    >>> print(cases)
    [1, 53]: Case I
    {54}: Case II
    undecided: [55, ∞)
    >>> cases[55]
    Traceback (most recent call last):
        ...
    KeyError: 'k = 55 is not covered (undecided)'
    """

    __slots__ = ("_covered", "_pieces")
    _pieces: tuple[tuple[KSet, T], ...]
    _covered: KSet

    def __init__(self, pieces: Mapping[KSet, T] | Iterable[tuple[KSet, T]]) -> None:
        items = list(pieces.items()) if isinstance(pieces, Mapping) else list(pieces)
        covered = KSet()
        kept: list[tuple[KSet, T]] = []
        for s, value in items:
            if not isinstance(s, KSet):
                raise TypeError(f"pieces are keyed by KSet, got {type(s).__name__}")
            if not s:
                continue
            overlap = covered & s
            if overlap:
                raise ValueError(f"pieces overlap on {overlap}")
            covered |= s
            kept.append((s, value))
        kept.sort(key=lambda piece: piece[0].intervals[0][0])
        object.__setattr__(self, "_pieces", tuple(kept))
        object.__setattr__(self, "_covered", covered)

    @property
    def pieces(self) -> tuple[tuple[KSet, T], ...]:
        """The non-empty pieces and their values, sorted by least element.

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> KCases({KSet.of(54): "b", KSet([(1, 53)]): "a"}).pieces
        ((KSet([(1, 53)]), 'a'), (KSet([(54, 54)]), 'b'))
        """
        return self._pieces

    @property
    def covered(self) -> KSet:
        """Where the result is known: the union of the pieces.

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> print(KCases({KSet.of(54): "b", KSet([(1, 53)]): "a"}).covered)
        [1, 54]
        """
        return self._covered

    @property
    def uncovered(self) -> KSet:
        """Where the result is undecided or not computed: ``~covered``.

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> print(KCases({KSet.of(54): "b", KSet([(1, 53)]): "a"}).uncovered)
        [55, ∞)
        """
        return ~self._covered

    @property
    def tail(self) -> tuple[KSet, T] | None:
        """The piece that contains all large ``k``, or ``None`` if there is none.

        At most one piece is infinite, because the pieces are disjoint. This is the
        generic piece of a result with finitely many exceptional ``k``, and the one
        on which limits as ``k -> ∞`` are taken (compare :attr:`KSet.tail_start`).

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> cases = KCases({KSet.of(1): "special", KSet([(2, None)]): "generic"})
        >>> cases.tail
        (KSet([(2, None)]), 'generic')
        >>> KCases({KSet.of(1): "special"}).tail is None
        True
        """
        for s, value in self._pieces:
            if not s.is_finite:
                return (s, value)
        return None

    def __getitem__(self, k: int) -> T:
        n = _integer(k, "k")
        for s, value in self._pieces:
            if n in s:
                return value
        if n < 1:
            raise KeyError(f"k = {n} is outside the domain k >= 1")
        raise KeyError(f"k = {n} is not covered (undecided)")

    def map[U](self, f: Callable[[T], U]) -> KCases[U]:
        """Apply ``f`` to every value; the pieces are unchanged.

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> print(KCases({KSet.of(1): 2, KSet([(2, None)]): 3}).map(lambda v: v * 10))
        {1}: 20
        [2, ∞): 30
        """
        return KCases((s, f(value)) for s, value in self._pieces)

    def restrict(self, s: KSet) -> KCases[T]:
        """Intersect every piece with ``s``; ``uncovered`` grows by ``~s``.

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> cases = KCases({KSet.of(1): "a", KSet([(2, None)]): "b"})
        >>> print(cases.restrict(KSet([(2, 10)])))
        [2, 10]: b
        undecided: {1} ∪ [11, ∞)
        """
        return KCases((piece & s, value) for piece, value in self._pieces)

    def combine[U, V](self, other: KCases[U], f: Callable[[T, U], V]) -> KCases[V]:
        """Pairwise intersections of the pieces, with value ``f(a, b)``.

        The result covers ``self.covered & other.covered``.

        Examples
        --------
        >>> from flanpy._kset import KCases, KSet
        >>> a = KCases({KSet([(1, 9)]): 1, KSet([(10, None)]): 2})
        >>> b = KCases({KSet([(1, 4)]): 10, KSet([(5, None)]): 20})
        >>> print(a.combine(b, lambda u, v: u + v))
        [1, 4]: 11
        [5, 9]: 21
        [10, ∞): 22
        """
        return KCases(
            (s & t, f(a, b)) for s, a in self._pieces for t, b in other._pieces
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KCases):
            return NotImplemented
        return self._pieces == other._pieces

    def __hash__(self) -> int:
        return hash(self._pieces)

    def __repr__(self) -> str:
        return f"KCases({list(self._pieces)!r})"

    def __str__(self) -> str:
        lines = [f"{s}: {value}" for s, value in self._pieces]
        if self.uncovered:
            lines.append(f"undecided: {self.uncovered}")
        return "\n".join(lines)

    def __reduce__(self) -> tuple[Any, ...]:
        return (KCases, (self._pieces,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("KCases is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("KCases is immutable")


KCases.__module__ = "flanpy"


# The public classes report "flanpy" as their module, so the doctest finder would
# skip them; list them here so that pytest --doctest-modules runs their examples.
__test__ = {"KSet": KSet, "KCases": KCases}
