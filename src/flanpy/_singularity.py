# SPDX-License-Identifier: GPL-3.0-or-later
"""The dominant singularity of the series on a curve (build-order step 4).

The design is recorded in ``dev/design/singularity.md``. :class:`Singularity`,
:func:`singularity`, :class:`UndecidedSingularityError` and
:class:`SingularityWarning` are public. This chunk covers a fixed ``k`` and
non-negative closures; the description in ``k`` (``singularity_in_k``) comes next.

References: Flajolet and Sedgewick, *Analytic Combinatorics* (2009), "FS";
Pivoteau and Salvy, *Effective Asymptotics of Combinatorial Systems*
(arXiv:2508.20008v1, 2025), "PS"; Broda, Machiavelo, Moreira and Reis (2020), "Broda".
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal, NoReturn

import mpmath
import sympy as sp

from flanpy._algebraic import (
    AlgebraicReal,
    _others_on_circle,
    _positive_roots,
    _real_roots,
    _separate,
    _separators,
)
from flanpy._curve import Curve
from flanpy._doctests import method_doctests
from flanpy._elimination import _closure
from flanpy._kset import KSet

__all__ = [
    "Singularity",
    "SingularityWarning",
    "UndecidedSingularityError",
    "singularity",
]

Case = Literal["I", "II"]

# Rounds of the certified comparison before q is moved within its gap; round r
# works at 64 * 2^r bits (singularity.md, section 3.3). Private, for the tests.
_ROUNDS = 7
# Rational points tried per gap (the first is the midpoint).
_TRIES = 6


class UndecidedSingularityError(ValueError):
    """The dominant singularity cannot be decided: the closure is signed.

    Raised by :func:`singularity` when the closure of the unknown (the unknowns it
    depends on) has a negative coefficient. Every coefficient being non-negative
    (with ``k = 1 + t`` for symbolic ``k``) is sufficient for the series to have
    non-negative coefficients, which Pringsheim's theorem needs, but not necessary.
    Signed systems are not supported yet. The message lists the candidates.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy import System, eliminate, singularity
    >>> z, F = symbols("z F")
    >>> singularity(eliminate(System({F: 1 - z * F**2}, z=z), F))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    flanpy.UndecidedSingularityError: cannot decide ... real ones: -1/4, 0.
    """


class SingularityWarning(UserWarning):
    """Whether ``ρ`` is the only dominant singularity is undecided.

    Emitted by :func:`singularity` when other candidates lie on the circle
    ``|z| = ρ`` (or may, by a numerical identification) and neither the period
    nor Pivoteau-Salvy (2025) Thm 6.3 decides; ``unique`` is then ``None``.

    Examples
    --------
    ``W = z^2 + T`` with ``T = z + zT^2``: ``W`` is aperiodic but, like ``T``,
    singular at ``±1/2``.

    >>> import warnings
    >>> from sympy import symbols
    >>> from flanpy import System, eliminate, singularity
    >>> z, T, W = symbols("z T W")
    >>> system = System({T: z + z * T**2, W: z**2 + T}, z=z)
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     result = singularity(eliminate(system, W))
    >>> result.unique is None, caught[0].category.__name__
    (True, 'SingularityWarning')
    """


class Singularity:
    """The dominant singularity of the series ``w(z)`` on a curve, at a fixed ``k``.

    Returned by :func:`singularity`; not built directly. The fields follow
    Broda et al. (2020), section 3.3:

    - :attr:`rho`: the radius of convergence ``ρ`` of ``w``, its singularity on
      ``(0, ∞)`` (Pringsheim), exact; ``rho.polynomial`` is ``m_ρ``;
    - :attr:`case`: ``"I"`` if ``w(x) → a < ∞`` as ``x → ρ⁻``, ``"II"`` if
      ``w(x) → ∞``;
    - :attr:`a`: ``a = w(ρ)`` in Case I, exact (``a.polynomial`` is ``m_a``);
      ``None`` in Case II;
    - :attr:`period`: the largest ``p`` with ``w(z) = z^r V(z^p)`` (FS Def. IV.5);
    - :attr:`unique`: whether ``ρ`` is the only singularity on ``|z| = ρ``
      (``None`` when undecided, with a :class:`SingularityWarning`).

    Notes
    -----
    Equality and hashing compare all the fields. A ``Singularity`` is immutable and
    can be pickled.

    Examples
    --------
    >>> from sympy import symbols
    >>> from flanpy import System, eliminate, singularity
    >>> z, F = symbols("z F")
    >>> s = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
    >>> s  # doctest: +NORMALIZE_WHITESPACE
    Singularity(rho=AlgebraicReal(4*z - 1, 0), case='I', a=AlgebraicReal(F - 2, 0),
                period=1, unique=True)
    >>> print(s)
    ρ = 1/4, Case I, a = 2, period 1, unique
    """

    __slots__ = ("_a", "_case", "_curve", "_k", "_kset", "_period", "_rho", "_unique")

    _curve: Curve
    _rho: AlgebraicReal
    _case: Case
    _a: AlgebraicReal | None
    _period: int
    _unique: bool | None
    _k: Any
    _kset: KSet

    def __init__(self) -> None:
        raise TypeError("Singularity objects are created by flanpy.singularity")

    @classmethod
    def _make(
        cls,
        curve: Curve,
        rho: AlgebraicReal,
        case: Case,
        a: AlgebraicReal | None,
        period: int,
        unique: bool | None,
    ) -> Singularity:
        new = object.__new__(cls)
        _set = object.__setattr__
        _set(new, "_curve", curve)
        _set(new, "_rho", rho)
        _set(new, "_case", case)
        _set(new, "_a", a)
        _set(new, "_period", period)
        _set(new, "_unique", unique)
        _set(new, "_k", curve.k)
        _set(new, "_kset", curve.kset)
        return new

    # -- public interface ----------------------------------------------------------

    @property
    def curve(self) -> Curve:
        """The curve at the fixed ``k``, whose branch is analysed.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, F = symbols("z F")
        >>> print(singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)).curve)
        F**2*z - F + 1 = 0
        """
        return self._curve

    @property
    def rho(self) -> AlgebraicReal:
        """The radius of convergence ``ρ``, exact; ``rho.polynomial`` is ``m_ρ``.

        Examples
        --------
        Broda et al. (2020) ``R_k`` at ``k = 2``: the positive root of
        ``Δ_2 = 1 - 2z - 23z^2`` (section 3.1).

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> singularity(eliminate(rk, R, k=2)).rho
        AlgebraicReal(23*z**2 + 2*z - 1, 1)
        """
        return self._rho

    @property
    def case(self) -> Case:
        """``"I"`` if ``w`` has a finite limit at ``ρ``, ``"II"`` if it tends to ``∞``.

        Examples
        --------
        Broda et al. (2020) ``L_k`` (eq. (17)) is in Case II:

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, k, B, Bb, M, Mb, L = symbols("z k B Bb M Mb L")
        >>> ssnf = System(
        ...     {
        ...         B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        ...         Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        ...         M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
        ...         Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
        ...         L: M + Mb,
        ...     },
        ...     z=z,
        ...     k=k,
        ... )
        >>> s = singularity(eliminate(ssnf, L, k=2))
        >>> s.case, s.rho.polynomial.as_expr()  # p_2, eq. (18)
        ('II', 124*z**3 + 36*z**2 - z - 2)
        """
        return self._case

    @property
    def a(self) -> AlgebraicReal | None:
        """``w(ρ)`` in Case I, exact (``a.polynomial`` is ``m_a``); ``None`` in Case II.

        Examples
        --------
        Hitchhiker's Guide (2014) eq. (8) at ``k = 2``: ``ρ = 1/6`` and
        ``A(ρ) = (8ρ^2 - 2ρ + 1)/(4ρ) = 4/3``.

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, B, A = symbols("z B A")
        >>> hg = System({B: 2 * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}, z=z)
        >>> s = singularity(eliminate(hg, A))
        >>> s.rho, s.a
        (AlgebraicReal(6*z - 1, 0), AlgebraicReal(3*A - 4, 0))
        """
        return self._a

    @property
    def period(self) -> int:
        """The period ``p`` of ``w``: ``w(z) = z^r V(z^p)`` with ``p`` largest.

        Examples
        --------
        ``T = z + zT^2`` (binary trees by nodes) has only odd sizes:

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, T = symbols("z T")
        >>> singularity(eliminate(System({T: z + z * T**2}, z=z), T)).period
        2
        """
        return self._period

    @property
    def unique(self) -> bool | None:
        """Whether ``ρ`` is the only singularity of ``w`` on ``|z| = ρ``.

        ``True`` is proved (no other candidate on the circle; or Pivoteau-Salvy
        2025 Thm 6.3), ``False`` is proved (``period > 1``: every ``ρ e^{2iπj/p}``
        is a singularity), and ``None`` is undecided.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, F, T = symbols("z F T")
        >>> singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)).unique
        True
        >>> singularity(eliminate(System({T: z + z * T**2}, z=z), T)).unique
        False
        """
        return self._unique

    @property
    def k(self) -> Any:
        """The parameter: ``None`` at a fixed ``k``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, F = symbols("z F")
        >>> singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)).k is None
        True
        """
        return self._k

    @property
    def kset(self) -> KSet:
        """The ``k`` this result is for: the curve's :attr:`~flanpy.Curve.kset`.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, singularity
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> singularity(eliminate(rk, R, k=2)).kset
        KSet([(2, 2)])
        """
        return self._kset

    # -- value semantics -----------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (
            self._curve,
            self._rho,
            self._case,
            self._a,
            self._period,
            self._unique,
            self._k,
            self._kset,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Singularity):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"Singularity(rho={self._rho!r}, case={self._case!r}, a={self._a!r}, "
            f"period={self._period}, unique={self._unique})"
        )

    def __str__(self) -> str:
        a = f"a = {self._a}" if self._a is not None else "w → ∞"
        unique = {True: "unique", False: "not unique", None: "uniqueness undecided"}
        return (
            f"ρ = {self._rho}, Case {self._case}, {a}, period {self._period}, "
            f"{unique[self._unique]}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _restore,
            (self._curve, self._rho, self._case, self._a, self._period, self._unique),
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Singularity is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Singularity is immutable")


def _restore(
    curve: Curve,
    rho: AlgebraicReal,
    case: Case,
    a: AlgebraicReal | None,
    period: int,
    unique: bool | None,
) -> Singularity:
    """Rebuild a pickled :class:`Singularity`."""
    return Singularity._make(curve, rho, case, a, period, unique)


Singularity.__module__ = "flanpy"
UndecidedSingularityError.__module__ = "flanpy"
SingularityWarning.__module__ = "flanpy"


# -- the public function -----------------------------------------------------------


def singularity(curve: Curve, *, k: int | None = None) -> Singularity:
    """Return the dominant singularity of the series on ``curve``, at a fixed ``k``.

    The branch is the series ``w(z)`` of ``curve.system``. Its closure (the unknowns
    ``w`` depends on) must have non-negative coefficients, so that Pringsheim's
    theorem applies (FS Thm IV.6): the radius of convergence ``ρ`` is a
    singularity, hence a positive root of ``lc_w G · disc_w G`` (FS Lemma VII.4,
    eq. (78)). The candidates are found exactly; a candidate need not be a
    singularity of the branch (Broda's ``L_k`` has one below ``ρ_k``), so ``ρ`` is
    located among them by comparisons ``q < ρ`` / ``q > ρ`` with rationals ``q``
    between consecutive candidates, each certified in exact rational arithmetic
    (see Notes). Case I / II and ``a`` come from an exact window test at a rational
    point below ``ρ``; the period from the system (PS 2025 §7.1); uniqueness from an
    exact circle test (singularity.md, sections 3-5).

    Parameters
    ----------
    curve : Curve
        A curve from :func:`~flanpy.eliminate` or :func:`~flanpy.eliminate_in_k`.
    k : int, optional
        Required when the curve has the parameter ``k``: the analysis is done on
        ``curve.specialize(k)`` (exact minimal polynomial at ``k``). Must be
        omitted otherwise.

    Returns
    -------
    Singularity
        ``ρ``, Case I or II, ``a``, the period and uniqueness.

    Raises
    ------
    TypeError
        If ``curve`` is not a :class:`~flanpy.Curve`.
    ValueError
        If ``k`` is missing or extra, or not in the curve's ``kset``; or if the
        series is a polynomial (it has no singularity).
    UndecidedSingularityError
        If the closure has a negative coefficient (signed systems are not
        supported yet).

    Warns
    -----
    SingularityWarning
        When uniqueness on ``|z| = ρ`` is undecided (``unique`` is ``None``).

    See Also
    --------
    Singularity : the fields of the result.

    Notes
    -----
    **The certificates** (singularity.md, section 3.3). Let ``Y = H(z, Y)`` be the
    closure of ``w`` with zero coordinates removed, ``J = ∂H/∂Y``, ``Λ`` the
    spectral radius. The closure is well founded (PS Def. 3.1) and its radius is
    the radius of ``w``. For a rational ``q > 0`` that is not a candidate:

    - ``q < ρ`` is certified by a rational ``u >= 0`` with ``H(q, u) <= u``: the
      Kleene iterates stay below ``u``, so ``Y(q) <= u`` (monotone convergence;
      cf. PS Prop. 3.11) and ``q <= ρ``, hence ``q < ρ``;
    - ``q > ρ`` is certified by a lower iterate ``ℓ`` (below) with
      ``Λ(J(q, ℓ)) >= 1``: if ``q < ρ`` then ``ℓ <= Y(q)`` and
      ``Λ(J(q, ℓ)) <= Λ(J(q, Y(q))) < 1`` (PS Lemma B.5 per irreducible
      component, Lemma 3.6; monotonicity, PS Lemma 6.7).

    The lower iterates start at ``ℓ = 0`` and take
    ``ℓ ← max(⌊N(ℓ)⌋, ⌊H(q, ℓ)⌋, ℓ)`` componentwise, where ``N`` is the Newton
    step (only while ``Λ(J(ℓ)) < 1``) and ``⌊·⌋`` rounds down to the current
    precision. They are non-negative and non-decreasing. *Soundness:* if
    ``q < ρ`` and ``ℓ <= Y(q)``, then ``H(q, ℓ) <= Y(q)`` (monotonicity) and
    ``Y - N(ℓ) = (I - J)^{-1} [H(Y) - H(ℓ) - J(ℓ)(Y - ℓ)] >= 0`` (convexity on
    ordered pairs, PS Lemma 3.5, and ``(I - J)^{-1} >= 0``); rounding down and the
    maximum keep the bound. *Termination:* the precision doubles each round and
    the iterates continue from where they were. If ``q < ρ`` they increase to a
    limit ``L <= Y(q)`` with ``H(q, L) <= L`` (the Kleene component, as the
    rounding error vanishes), so ``L = Y(q)``; then ``u = ℓ + δ (I - J)^{-1} 1``
    rounded up passes for some ``δ`` once ``ℓ`` is close enough, because
    ``Λ(J(q, Y(q))) < 1`` gives ``H(q, Y + δv) = Y + δv - δ + O(δ^2)``. If
    ``q > ρ`` and ``Λ`` stays below 1, a bounded limit ``L`` would satisfy
    ``H(q, L) <= L``, a non-negative super-solution, which forces ``q <= ρ``; so
    some coordinate is unbounded. In the first strongly connected component
    (dependency order) with an unbounded coordinate, if the component is
    nonlinear, an entry of ``J`` on a cycle grows without bound and so does
    ``Λ`` (contradiction); if it is linear, ``J`` restricted to it is ``B(q, U)``
    with ``U`` the bounded earlier coordinates, and unbounded values need
    ``Λ(B(q, U)) = 1`` in the limit, which happens only when ``q`` is the radius
    of that component (PS Thm 6.4, ``ρ_C``). There are finitely many such ``q``,
    so after a budget of rounds the comparison moves ``q`` to another rational in
    the same gap between candidates (where the answer is the same).

    Examples
    --------
    Catalan, ``F = 1 + zF^2``: ``ρ = 1/4``, ``F(1/4) = 2`` (FS Note VII.36).

    >>> from sympy import symbols
    >>> from flanpy import System, eliminate, singularity
    >>> z, F = symbols("z F")
    >>> s = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
    >>> s.rho, s.case, s.a, s.unique
    (AlgebraicReal(4*z - 1, 0), 'I', AlgebraicReal(F - 2, 0), True)

    A symbolic curve needs ``k``:

    >>> from flanpy import eliminate_in_k
    >>> k, R = symbols("k R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> curve = eliminate_in_k(rk, R).tail[1]
    >>> singularity(curve, k=2).rho == singularity(eliminate(rk, R, k=2)).rho
    True
    """
    if not isinstance(curve, Curve):
        raise TypeError(f"expected a flanpy.Curve, got {type(curve).__name__}")
    if curve.k is not None:
        if k is None:
            raise ValueError(
                "the curve has the parameter k: give k=n to analyse it at k = n"
            )
        curve = curve.specialize(k)
    elif k is not None:
        raise ValueError("the curve has no parameter k; omit the k argument")
    return _analyse(curve)


# -- the analysis at a fixed k -----------------------------------------------------


def _analyse(curve: Curve) -> Singularity:
    w, z = curve.w, curve.z
    g = sp.Poly(curve.polynomial.as_expr(), w, z, domain=sp.QQ)
    candidates = _candidates(g)
    found = _closure(curve.system, w)
    if found is None:
        raise ValueError(f"{w} is identically 0: it has no singularity")
    closure = _Closure(z, found[0], found[1])
    if closure.negative is not None:
        raise UndecidedSingularityError(_signed_message(w, closure, candidates))
    positive = _positive_roots(candidates)
    if not positive:
        # A non-negative series that is not a polynomial has a finite radius, a
        # positive candidate; an entire algebraic function is a polynomial.
        raise ValueError(f"the series {w} is a polynomial: it has no singularity")
    index = _locate(closure, positive)
    rho = positive[index]
    previous = positive[index - 1].interval[1] if index else Fraction(0)
    case, a = _window_test(g, closure, rho, previous)
    period = _period(closure, closure.unknowns.index(w))
    unique = _unique(candidates, rho, period, closure, w)
    return Singularity._make(curve, rho, case, a, period, unique)


def _candidates(g: sp.Poly) -> sp.Poly:
    """Return ``lc_w G · disc_w G`` as a polynomial in ``z`` (FS eq. (78), (P1))."""
    w, z = g.gens
    lc = sp.Poly(g.as_expr(), w).LC()
    product = lc * sp.discriminant(g.as_expr(), w) if g.degree(w) > 1 else lc
    return sp.Poly(sp.expand(product), z, domain=sp.QQ)


def _signed_message(w: Any, closure: _Closure, candidates: sp.Poly) -> str:
    assert closure.negative is not None
    u, term, c = closure.negative
    real = ", ".join(
        sp.sstr(r.as_expr())
        if r.polynomial.degree() == 1
        else mpmath.nstr(r.evalf(), 6)
        for r in _real_roots(candidates)
    )
    return (
        f"cannot decide the dominant singularity of {w}: its closure has the "
        f"negative coefficient {c} (of {sp.sstr(term)} in the equation of {u}). "
        "Non-negative coefficients (with k = 1 + t for symbolic k) are sufficient "
        "for a series with non-negative coefficients, not necessary; signed systems "
        "are not supported yet. The candidates are the roots of "
        f"{sp.sstr(sp.factor(candidates.as_expr()))}"
        + (f", real ones: {real}." if real else ", none of them real.")
    )


# -- the closure as exact polynomial maps -------------------------------------------

_Term = tuple[Fraction, int, tuple[int, ...]]  # coefficient, power of z, exponents


class _Closure:
    """``Y = H(z, Y)``, the closure of ``w`` at a fixed ``k``, for exact evaluation."""

    def __init__(self, z: Any, unknowns: Sequence[Any], equations: dict[Any, Any]):
        self.unknowns = list(unknowns)
        self.m = len(self.unknowns)
        self.terms: list[list[_Term]] = []
        self.negative: tuple[Any, Any, Fraction] | None = None
        for u in self.unknowns:
            poly = sp.Poly(equations[u], z, *self.unknowns, domain=sp.QQ)
            row = []
            for monomial, c in poly.terms():
                coefficient = Fraction(int(c.numerator), int(c.denominator))
                if coefficient < 0 and self.negative is None:
                    term = sp.Mul(
                        z ** monomial[0],
                        *(
                            y**e
                            for y, e in zip(self.unknowns, monomial[1:], strict=True)
                        ),
                    )
                    self.negative = (u, term, coefficient)
                row.append((coefficient, monomial[0], tuple(monomial[1:])))
            self.terms.append(row)
        self.jacobian_terms = [
            [
                [
                    (c * e[j], ez, (*e[:j], e[j] - 1, *e[j + 1 :]))
                    for c, ez, e in row
                    if e[j]
                ]
                for j in range(self.m)
            ]
            for row in self.terms
        ]

    @staticmethod
    def _evaluate(terms: list[_Term], x: Fraction, y: Sequence[Fraction]) -> Fraction:
        total = Fraction(0)
        for c, ez, e in terms:
            v = c * x**ez
            for yi, ei in zip(y, e, strict=True):
                if ei:
                    v *= yi**ei
            total += v
        return total

    def h(self, x: Fraction, y: Sequence[Fraction]) -> list[Fraction]:
        return [self._evaluate(row, x, y) for row in self.terms]

    def jacobian(self, x: Fraction, y: Sequence[Fraction]) -> list[list[Fraction]]:
        return [[self._evaluate(t, x, y) for t in row] for row in self.jacobian_terms]

    def graph(self) -> list[set[int]]:
        """Edges ``i -> j`` when ``H_i`` depends on ``Y_j``."""
        return [
            {j for _, _, e in row for j in range(self.m) if e[j]} for row in self.terms
        ]

    def nonlinear(self) -> bool:
        return any(sum(e) >= 2 for row in self.terms for _, _, e in row)


# -- exact linear algebra ------------------------------------------------------------


def _solve(j: list[list[Fraction]], r: Sequence[Fraction]) -> list[Fraction]:
    """Solve ``(I - J) d = r`` exactly (``I - J`` invertible)."""
    m = len(r)
    a = [[Fraction(i == c) - j[i][c] for c in range(m)] + [r[i]] for i in range(m)]
    for c in range(m):
        p = next(i for i in range(c, m) if a[i][c] != 0)
        a[c], a[p] = a[p], a[c]
        for i in range(m):
            if i != c and a[i][c] != 0:
                f = a[i][c] / a[c][c]
                a[i] = [x - f * y for x, y in zip(a[i], a[c], strict=True)]
    return [a[i][m] / a[i][i] for i in range(m)]


def _spectral_radius_at_least_one(j: list[list[Fraction]]) -> bool:
    """``Λ(J) >= 1`` for a non-negative rational ``J``, exactly.

    By Perron-Frobenius, ``Λ`` is the largest real eigenvalue of ``J``, so
    ``Λ >= 1`` iff the characteristic polynomial (Faddeev-LeVerrier, exact) has a
    root in ``[1, ∞)`` (Sturm count).
    """
    m = len(j)
    coefficients = [Fraction(1)]  # of λ^m, λ^(m-1), ...
    previous = [[Fraction(0)] * m for _ in range(m)]
    for step in range(1, m + 1):
        c = coefficients[-1]
        current = [
            [
                sum((j[r][t] * previous[t][s] for t in range(m)), Fraction(0))
                + (c if r == s else 0)
                for s in range(m)
            ]
            for r in range(m)
        ]
        trace = sum(
            (
                sum((j[r][t] * current[t][r] for t in range(m)), Fraction(0))
                for r in range(m)
            ),
            Fraction(0),
        )
        coefficients.append(-trace / step)
        previous = current
    lam = sp.Dummy("lam")
    chi = sp.Poly([sp.Rational(c.numerator, c.denominator) for c in coefficients], lam)
    return bool(chi.count_roots(1, None) > 0)


def _down(x: Fraction, bits: int) -> Fraction:
    return Fraction(math.floor(x * 2**bits), 2**bits)


def _up(x: Fraction, bits: int) -> Fraction:
    return Fraction(math.ceil(x * 2**bits), 2**bits)


# -- the certified comparison (singularity.md, section 3.3) -------------------------


@dataclass(frozen=True)
class _Below:
    """``q < ρ``, with ``lower <= Y(q) <= upper``."""

    lower: list[Fraction]
    upper: list[Fraction]


@dataclass(frozen=True)
class _Above:
    """``q > ρ``: ``Λ(J(q, lower)) >= 1`` for a sound lower iterate."""

    lower: list[Fraction]


def _compare(
    closure: _Closure, q: Fraction, *, first_round: int = 0
) -> _Below | _Above | None:
    """Certify ``q < ρ`` or ``q > ρ``, for a rational ``q`` that is not a candidate.

    Returns ``None`` if the budget runs out (then ``q`` may be one of the finitely
    many bad points, and the caller moves it).

    The argument (soundness, rounding, termination) is in the Notes of
    :func:`singularity`.
    """
    lower = [Fraction(0)] * closure.m
    for r in range(_ROUNDS):
        bits = 64 * 2**r
        steps = 8 + 4 * r
        for _ in range(steps):
            jac = closure.jacobian(q, lower)
            if _spectral_radius_at_least_one(jac):
                return _Above(lower)
            image = closure.h(q, lower)
            step = _solve(jac, [a - b for a, b in zip(image, lower, strict=True)])
            lower = [
                max(_down(old + d, bits), _down(kleene, bits), old)
                for old, d, kleene in zip(lower, step, image, strict=True)
            ]
        if r < first_round:
            continue
        jac = closure.jacobian(q, lower)
        if _spectral_radius_at_least_one(jac):
            return _Above(lower)
        upper = _upper(closure, q, lower, jac, bits)
        if upper is not None:
            return _Below(lower, upper)
    return None


def _upper(
    closure: _Closure,
    q: Fraction,
    approx: list[Fraction],
    jac: list[list[Fraction]],
    bits: int,
) -> list[Fraction] | None:
    """Return a rational ``u >= 0`` with ``H(q, u) <= u`` near ``approx``, or ``None``.

    The smallest ``δ`` is tried first, for the tightest enclosure.
    """
    v = _solve(jac, [Fraction(1)] * closure.m)  # (I - J)^-1 1 > 0
    for e in range(bits // 2, 1, -1):
        delta = Fraction(1, 2**e)
        u = [_up(a + delta * b, bits) for a, b in zip(approx, v, strict=True)]
        if all(x >= 0 for x in u) and all(
            hx <= x for hx, x in zip(closure.h(q, u), u, strict=True)
        ):
            return u
    return None


def _points(lo: Fraction, hi: Fraction) -> Iterator[Fraction]:
    """Rationals in the open interval ``(lo, hi)``: the midpoint, then others."""
    for i in range(_TRIES):
        yield lo + (hi - lo) * Fraction(2 * i + 1, 2 * (i + 1))


def _decide(closure: _Closure, lo: Fraction, hi: Fraction) -> _Below | _Above:
    """Compare ``ρ`` with a rational of ``(lo, hi)``, which holds no candidate."""
    for q in _points(lo, hi):
        result = _compare(closure, q)
        if result is not None:
            return result
    raise RuntimeError(  # pragma: no cover - finitely many bad points (Notes)
        f"no certificate for any rational tried in ({lo}, {hi})"
    )


def _locate(closure: _Closure, positive: list[AlgebraicReal]) -> int:
    """Return the index of ``ρ`` among the positive candidates (binary search)."""
    _separate(positive)
    gaps = [
        (r.interval[1], s.interval[0]) for r, s in itertools.pairwise(positive)
    ]  # gaps[i] lies between positive[i] and positive[i + 1]
    lo, hi = 0, len(positive) - 1  # ρ = positive[i] for some lo <= i <= hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if isinstance(_decide(closure, *gaps[mid - 1]), _Below):
            lo = mid  # ρ > the gap below positive[mid]
        else:
            hi = mid - 1
    return lo


# -- Case I / II and a: the window test (singularity.md, section 4.1) ---------------


def _window_test(
    g: sp.Poly, closure: _Closure, rho: AlgebraicReal, previous: Fraction
) -> tuple[Case, AlgebraicReal | None]:
    w, z = g.gens
    m_rho = rho.polynomial.as_expr().subs(rho.polynomial.gen, z)
    norm = sp.Poly(sp.resultant(m_rho, g.as_expr(), z), w, domain=sp.QQ)
    roots = _real_roots(norm) if not norm.is_zero else []
    separators = _separators(roots)
    # x0 in (previous, ρ) above every root of G(·, q_j) below ρ: on [x0, ρ) no real
    # sheet crosses a separator and none comes from infinity.
    bound = previous
    for q in separators:
        column = sp.Poly(
            g.as_expr().subs(w, sp.Rational(q.numerator, q.denominator)), z
        )
        if column.is_zero or column.degree() < 1:
            continue
        for r in _real_roots(column):
            if r < rho:  # never equal: q is not a root of G(ρ, ·)
                while not r.interval[1] < rho.interval[0]:
                    r._shrink()
                    rho._shrink()
                bound = max(bound, r.interval[1])
    while not bound < rho.interval[0]:
        rho._shrink()
    for x0 in _points(bound, rho.interval[0]):
        result = _compare(closure, x0)
        if result is not None:
            break
    else:  # pragma: no cover - finitely many bad points
        raise RuntimeError("no certified enclosure below rho")
    if not isinstance(result, _Below):  # pragma: no cover - x0 < ρ by the search
        raise RuntimeError(f"the comparison contradicts {x0} < ρ")
    i = closure.unknowns.index(w)
    fiber = sp.Poly(g.as_expr().subs(z, sp.Rational(x0.numerator, x0.denominator)), w)
    round_ = 0
    while True:
        lo, hi = result.lower[i], result.upper[i]
        count = fiber.count_roots(_q(lo), _q(hi))
        if count < 1:  # pragma: no cover - w(x0) is a root of G(x0, ·)
            raise RuntimeError(f"the enclosure of w({x0}) holds no root of G({x0}, ·)")
        if count == 1:
            break
        round_ += 1  # tighten the enclosure: more precision
        tighter = _compare(closure, x0, first_round=round_)
        if tighter is None:  # pragma: no cover - the enclosure shrinks to w(x0)
            raise RuntimeError(
                f"the enclosure of w({x0}) holds {count} roots of G({x0}, ·) "
                "after the last round"
            )
        if not isinstance(tighter, _Below):  # pragma: no cover - x0 < ρ is proved
            raise RuntimeError(f"the comparison contradicts {x0} < ρ")
        result = tighter
    windows = [None, *separators, None]
    for j in range(len(windows) - 1):
        a, b = windows[j], windows[j + 1]
        left = lo if a is None else max(lo, a)
        right = hi if b is None else min(hi, b)
        if left <= right and fiber.count_roots(_q(left), _q(right)) == 1:
            break
    else:  # pragma: no cover - w(x0) is no separator: G(x0, q_j) != 0
        raise RuntimeError(f"no window between the separators holds w({x0})")
    if j == 0:  # pragma: no cover - a non-negative series does not tend to -oo
        raise RuntimeError(f"w({x0}) lies below every root of N")
    if j == len(separators):
        return "II", None
    a_value = roots[j - 1]
    _check_branch_point(g, rho, a_value)
    return "I", a_value


def _q(x: Fraction) -> sp.Rational:
    return sp.Rational(x.numerator, x.denominator)


def _check_branch_point(g: sp.Poly, rho: AlgebraicReal, a: AlgebraicReal) -> None:
    """Check Broda eq. (19), ``G(ρ, a) = ∂_w G(ρ, a) = 0`` (internal consistency).

    Otherwise the implicit function theorem would make the branch analytic at
    ``ρ``, against Pringsheim. Checked numerically to 40 digits.
    """
    w, z = g.gens
    with mpmath.workdps(50):
        point = {z: sp.Float(rho.evalf(50), 50), w: sp.Float(a.evalf(50), 50)}
        scale = 1 + sum(abs(c) for c in g.coeffs()) * (1 + abs(point[w])) ** g.degree(w)
        for p in (g.as_expr(), sp.diff(g.as_expr(), w)):
            value = abs(sp.N(p.subs(point), 50))
            if not value < scale * sp.Float(10) ** -35:  # pragma: no cover
                raise RuntimeError(f"(ρ, a) = ({rho}, {a}) is not a branch point")


# -- period (singularity.md, section 5.2) -------------------------------------------


def _valuations(closure: _Closure) -> list[int]:
    """Return the valuations of the coordinates.

    They are the least fixed point of the min-plus system (no cancellation:
    non-negative coefficients; no zero coordinates).
    """
    inf = math.inf
    val: list[float] = [inf] * closure.m
    while True:
        new = [
            min(
                (
                    ez + sum(e_j * v for e_j, v in zip(e, val, strict=True) if e_j)
                    for _, ez, e in row
                ),
                default=inf,
            )
            for row in closure.terms
        ]
        if new == val:
            assert all(v < inf for v in val)
            return [int(v) for v in val]
        val = new


def _period(closure: _Closure, target: int) -> int:
    """Return the period of coordinate ``target`` (PS 2025 §7.1, Algorithm Periods).

    The rules are Bell-Burris-Yeats's: ``Π(sum) = gcd(Π's, valuation differences)``
    and ``Π(product) = gcd(Π's)``, iterated to a fixed point from ``0``. They are
    exact because no coefficient cancels (non-negative coefficients).
    """
    val = _valuations(closure)
    per = [0] * closure.m
    while True:
        new = []
        for i, row in enumerate(closure.terms):
            g = 0
            for _, ez, e in row:
                v = ez + sum(e_j * val[j] for j, e_j in enumerate(e))
                factors = [per[j] for j, e_j in enumerate(e) if e_j]
                g = math.gcd(g, *factors, v - val[i])
            new.append(g)
        if new == per:
            return per[target]
        per = new


# -- uniqueness (singularity.md, section 5.3) ----------------------------------------


def _irreducible(closure: _Closure) -> bool:
    """Tell whether the closure is irreducible (PS Def. 3.2): strongly connected."""
    graph = closure.graph()
    if not any(graph):
        return False
    for start in range(closure.m):
        seen, todo = {start}, [start]
        while todo:
            for j in graph[todo.pop()]:
                if j not in seen:
                    seen.add(j)
                    todo.append(j)
        if len(seen) != closure.m:
            return False
    return True


def _unique(
    candidates: sp.Poly, rho: AlgebraicReal, period: int, closure: _Closure, w: Any
) -> bool | None:
    if period > 1:
        return False  # every ρ e^{2iπj/p} is a singularity (Pringsheim for V)
    circle = _others_on_circle(candidates, rho)
    if circle.proved_none:
        return True  # every singularity is a candidate (FS Lemma VII.4)
    if _irreducible(closure) and closure.nonlinear():
        return True  # PS 2025 Thm 6.3, period 1
    others = ", ".join(f"{s:.6g}" for s in circle.roots) or "none found numerically"
    warnings.warn(
        SingularityWarning(
            f"uniqueness of the dominant singularity of {w} is undecided: other "
            f"candidates on |z| = ρ: {others}; the closure is not irreducible and "
            "nonlinear, so Pivoteau-Salvy (2025) Thm 6.3 does not apply"
        ),
        stacklevel=4,
    )
    return None


# Methods of the public class, which __init__.py's doctests miss (_doctests.py).
__test__ = method_doctests(Singularity)
