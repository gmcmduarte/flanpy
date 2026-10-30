# SPDX-License-Identifier: GPL-3.0-or-later
"""The dominant singularity of the series on a curve (build-order step 4).

The design is recorded in ``dev/design/singularity.md``. :class:`Singularity`,
:func:`singularity`, :func:`singularity_in_k`, :class:`UndecidedSingularityError` and
:class:`SingularityWarning` are public. Non-negative closures only.

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
    KRoot,
    _others_on_circle,
    _positive_roots,
    _primitive_in_x,
    _real_roots,
    _separate,
    _separators,
    _square,
)
from flanpy._curve import Curve
from flanpy._doctests import method_doctests
from flanpy._elimination import _closure
from flanpy._kset import KCases, KSet, _breakpoints, _Interval, _key

__all__ = [
    "Singularity",
    "SingularityWarning",
    "UndecidedSingularityError",
    "singularity",
    "singularity_in_k",
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

    Emitted by :func:`singularity` and :func:`singularity_in_k` when other
    candidates lie on the circle ``|z| = ρ`` (or may, by a numerical
    identification) and neither the period of ``w`` nor the component rule (every
    recursive component of the closure has period 1; Notes of :func:`singularity`)
    decides; ``unique`` is then ``None``.

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


class Singularity[X: (AlgebraicReal, KRoot)]:
    """The dominant singularity of the series ``w(z)`` on a curve.

    Returned by :func:`singularity` (a fixed ``k``) and :func:`singularity_in_k`
    (one per piece of ``k``); not built directly. The fields follow Broda et al.
    (2020), section 3.3:

    - :attr:`rho`: the radius of convergence ``ρ`` of ``w``, its singularity on
      ``(0, ∞)`` (Pringsheim), exact; ``rho.polynomial`` is ``m_ρ`` at a fixed
      ``k``;
    - :attr:`case`: ``"I"`` if ``w(x) → a < ∞`` as ``x → ρ⁻``, ``"II"`` if
      ``w(x) → ∞``;
    - :attr:`a`: ``a = w(ρ)`` in Case I, exact (``a.polynomial`` is ``m_a`` at a
      fixed ``k``); ``None`` in Case II;
    - :attr:`period`: the largest ``p`` with ``w(z) = z^r V(z^p)`` (FS Def. IV.5);
    - :attr:`unique`: whether ``ρ`` is the only singularity on ``|z| = ρ``
      (``None`` when undecided, with a :class:`SingularityWarning`).

    At a fixed ``k``, ``rho`` and ``a`` are :class:`~flanpy.AlgebraicReal`. On a
    piece of :func:`singularity_in_k` they are :class:`~flanpy.KRoot`, valid at
    every ``k`` of :attr:`kset`; :meth:`specialize` gives the fixed-``k`` result.

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
    _rho: X
    _case: Case
    _a: X | None
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
        rho: X,
        case: Case,
        a: X | None,
        period: int,
        unique: bool | None,
        kset: KSet | None = None,
    ) -> Singularity[X]:
        new = object.__new__(cls)
        _set = object.__setattr__
        _set(new, "_curve", curve)
        _set(new, "_rho", rho)
        _set(new, "_case", case)
        _set(new, "_a", a)
        _set(new, "_period", period)
        _set(new, "_unique", unique)
        _set(new, "_k", curve.k)
        _set(new, "_kset", curve.kset if kset is None else kset)
        return new

    # -- public interface ----------------------------------------------------------

    @property
    def curve(self) -> Curve:
        """The curve whose branch is analysed (symbolic on a piece of ``k``).

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
    def rho(self) -> X:
        """The radius of convergence ``ρ``, exact; ``rho.polynomial`` is ``m_ρ``.

        A :class:`~flanpy.KRoot` on a piece of :func:`singularity_in_k`.

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
    def a(self) -> X | None:
        """``w(ρ)`` in Case I, exact (``a.polynomial`` is ``m_a``); ``None`` in Case II.

        A :class:`~flanpy.KRoot` on a piece of :func:`singularity_in_k`.

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

        ``True`` is proved (no other candidate on the circle; or every recursive
        component of the closure has period 1, see Notes of :func:`singularity`),
        ``False`` is proved (``period > 1``: every ``ρ e^{2iπj/p}`` is a
        singularity), and ``None`` is undecided.

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
        """The parameter: ``None`` at a fixed ``k``, else the curve's ``k``.

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
        """The ``k`` this result is for.

        At a fixed ``k``, the curve's :attr:`~flanpy.Curve.kset`; on a piece of
        :func:`singularity_in_k`, the piece.

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

    def specialize(self, k: int) -> Singularity[AlgebraicReal]:
        """Return the result at a fixed ``k``, with exact minimal polynomials.

        ``rho`` and ``a`` become :class:`~flanpy.AlgebraicReal` (``KRoot.at``),
        so ``rho.polynomial`` is ``m_ρ`` at ``k``; the curve becomes
        ``curve.specialize(k)``. The case, the period and ``unique`` are those of
        the piece. On a piece where ``unique`` is ``None``, ``singularity(curve,
        k=k)`` may still decide one ``k`` (singularity.md, section 6.4).

        Parameters
        ----------
        k : int
            A value in :attr:`kset`.

        Returns
        -------
        Singularity
            The result at ``k``, as :func:`singularity` gives it.

        Raises
        ------
        TypeError
            If ``k`` is not an ``int``.
        ValueError
            If the result is already at a fixed ``k``, or ``k`` is not in
            :attr:`kset`.

        Examples
        --------
        Broda et al. (2020) ``R_k``: ``ρ_2`` is the positive root of
        ``Δ_2 = 1 - 2z - 23z^2``.

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate_in_k, singularity_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> s = singularity_in_k(eliminate_in_k(rk, R))[2]
        >>> s.rho
        KRoot(8*k*z**2 + 7*z**2 + 2*z - 1, 0, k=k)
        >>> s.specialize(2).rho
        AlgebraicReal(23*z**2 + 2*z - 1, 1)
        """
        if self._k is None:
            raise ValueError("the result is at a fixed k: nothing to specialize")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an int, got {k!r} of type {type(k).__name__}")
        if k not in self._kset:
            raise ValueError(f"k = {k} is not in the result's kset {self._kset}")
        assert isinstance(self._rho, KRoot)
        a = self._a.at(k) if isinstance(self._a, KRoot) else None
        return Singularity._make(
            self._curve.specialize(k),
            self._rho.at(k),
            self._case,
            a,
            self._period,
            self._unique,
        )

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
            (
                self._curve,
                self._rho,
                self._case,
                self._a,
                self._period,
                self._unique,
                self._kset,
            ),
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Singularity is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Singularity is immutable")


def _restore[X: (AlgebraicReal, KRoot)](
    curve: Curve,
    rho: X,
    case: Case,
    a: X | None,
    period: int,
    unique: bool | None,
    kset: KSet,
) -> Singularity[X]:
    """Rebuild a pickled :class:`Singularity`."""
    return Singularity._make(curve, rho, case, a, period, unique, kset)


Singularity.__module__ = "flanpy"
UndecidedSingularityError.__module__ = "flanpy"
SingularityWarning.__module__ = "flanpy"


# -- the public function -----------------------------------------------------------


def singularity(curve: Curve, *, k: int | None = None) -> Singularity[AlgebraicReal]:
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

    **Uniqueness** (singularity.md, sections 5.3 and 5.5). ``unique`` is ``False``
    if the period of ``w`` exceeds 1; ``True`` if an exact circle test proves that
    no other candidate lies on ``|z| = ρ`` (FS Lemma VII.4), or if every recursive
    strongly connected component ``C`` of the closure has period ``p_C = 1``; and
    ``None`` otherwise (with a :class:`SingularityWarning`). For a nonlinear ``C``,
    ``p_C`` is the common period of its coordinates (PS Thm 6.2); for a linear
    ``C`` (``Y_C = A + B Y_C``) it is the cycle period, the gcd of the exponents of
    the closed walks in the graph of ``B`` (the period of the diagonal of
    ``(I - B)^{-1}``, PS Thm 6.4). Both come from the support by the
    Bell-Burris-Yeats rules (PS §7.1), which are exact because no coefficient
    cancels; a computed ``1`` is therefore the true period.

    *Proof of the rule.* Let ``σ = ρ e^{iθ} != ρ``. By induction over the
    components in dependency order, every coordinate is analytic at ``σ``. Each
    coordinate has radius ``>= ρ`` (no cancellation), so only components of
    radius ``ρ`` need an argument; a nonrecursive coordinate is a polynomial in
    ``z`` and earlier ones. For a recursive ``C`` of radius ``ρ``, the earlier
    solutions ``S`` are analytic at ``σ`` (induction), so
    ``Ĥ(z, Y) = H_C(z, S(z), Y)``, an irreducible system (PS Lemma 3.6), is
    polynomial in ``Y`` with coefficients analytic at ``σ``. This is what the
    proofs of PS Thm 6.3 and 6.4 need, also when an input is itself critical (as
    ``{B, B̄}`` for Broda's ``L_k``). *Linear:* for ``r < ρ``,
    ``Λ(B(r)) < 1``, and since each edge of ``C`` lies on a cycle the entries stay
    bounded, so ``|B_ij(σ)| <= B_ij(ρ) < ∞``. If ``1`` is an eigenvalue of
    ``B(σ)``, Wielandt (PS Lemma 6.7) gives ``B(σ) = D B(ρ) D^{-1}``,
    ``D = diag(e^{iφ_j})``, so ``e^{inθ} = e^{i(φ_i - φ_j)}`` for every ``n`` in the
    support of ``B_ij``. Around a closed walk this telescopes, so
    ``θ ∈ (2π/p_C)Z``, and ``σ = ρ``. Otherwise ``Y_C = (I - B)^{-1} A`` is analytic
    at ``σ``. *Nonlinear:* ``Y_C`` and ``J = ∂Ĥ/∂Y`` stay bounded as ``z → ρ``
    (an unbounded coordinate would make an entry of ``J`` on a cycle, and ``Λ``,
    unbounded, against ``Λ < 1`` below ``ρ``, PS Lemma B.5). Assume ``Y_C`` is
    singular at ``σ``. Then ``I - J(σ)`` is singular (otherwise the implicit
    function theorem continues ``Y_C``), so ``1 <= Λ(J(σ)) <= Λ(J(ρ)) <= 1``, and
    Wielandt (PS Lemmas 6.7, 6.8) forces ``|J_ij(σ)| = J_ij(ρ)`` for every entry.
    An entry with a positive term containing some ``Y_k`` (``C`` is nonlinear) then
    forces ``|Y_k(σ)| = Y_k(ρ)``: every term, and every factor of it, is bounded by
    its value at ``ρ``. The daffodil lemma (PS Lemma 6.5) gives
    ``θ ∈ (2π/Π(Y_k))Z = (2π/p_C)Z``, so ``σ = ρ``: a contradiction. ∎

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
                "the curve has the parameter k: give k=n to analyse it at k = n, "
                "or use flanpy.singularity_in_k for every k"
            )
        curve = curve.specialize(k)
    elif k is not None:
        raise ValueError("the curve has no parameter k; omit the k argument")
    return _analyse(curve)


def singularity_in_k(curves: Curve | KCases[Curve]) -> KCases[Singularity[Any]]:
    """Return the dominant singularity for symbolic ``k``, piece by piece.

    On each piece, ``ρ_k`` is described as the ``j``-th positive root of an
    irreducible factor ``f(z, k)`` of the candidates, and in Case I ``a_k`` as the
    ``j'``-th positive root of an irreducible factor ``g(w, k)`` (both
    :class:`~flanpy.KRoot`); the case, the period and ``unique`` are constant on the
    piece. The pieces come from the real roots of a finite set of polynomials in
    ``k`` (the projection set; see Notes), one exact fixed-``k`` analysis
    (:func:`singularity`) per interval between them, and one at each integer root.
    Pieces with the same description are merged.

    Parameters
    ----------
    curves : Curve or KCases[Curve]
        A curve with the parameter ``k``, or the result of
        :func:`~flanpy.eliminate_in_k` (handled piece by piece with
        :meth:`KCases.flat_map <flanpy.KCases.flat_map>`; a piece with a fixed-``k``
        curve gets :func:`singularity` of that curve).

    Returns
    -------
    KCases[Singularity]
        Pieces covering the ``kset`` of every input curve; nothing is dropped.

    Raises
    ------
    TypeError
        If ``curves`` is not a :class:`~flanpy.Curve` or a ``KCases`` of them.
    ValueError
        If a curve has no parameter ``k`` (use :func:`singularity`), or the series
        is a polynomial at some ``k`` (restrict the input to exclude it).
    UndecidedSingularityError
        If the closure has a coefficient that is negative for some ``k >= 1``
        (tested with ``k = 1 + t``, as in :class:`~flanpy.System`).

    Warns
    -----
    SingularityWarning
        For the pieces where uniqueness on ``|z| = ρ_k`` is undecided.

    See Also
    --------
    singularity : one fixed ``k``.
    Singularity.specialize : the result of a piece at one ``k``.

    Notes
    -----
    **Projection set** (singularity.md, section 6.2). Let ``R = lc_w G · disc_w G``
    (``lc_w G`` if ``deg_w G = 1``), ``R̃`` the product of its distinct irreducible
    factors in ``Q[z, k]`` of positive degree in ``z``, and ``R̃'`` the same without
    the factor ``z``. ``P1`` holds ``lc_z R̃``, ``disc_z R̃``, ``R̃'(0, k)``, the
    factors in ``k`` alone of ``lc_w G`` and ``disc_w G``, and the coefficients in
    ``k`` of the closure of ``w`` (its support). For a factor ``f`` of ``R̃'``, let
    ``Ñ`` be the product of the distinct irreducible factors of positive degree in
    ``w`` of ``Res_z(f, G)``, and ``Ñ'`` the same without a factor ``w``. ``P2(f)``
    holds the factors in ``k`` alone of ``Res_z(f, G)``, ``lc_w Ñ``, ``disc_w Ñ``,
    ``Ñ'(0, k)``, and ``E(f)``: the resultants ``Res_z(f, f'(-z))`` and
    ``Res_y(μ, Q_f')`` that are not identically zero, for the factors ``f'`` of
    ``R̃'``, the irreducible factors ``μ`` of ``Res_z(f, y - z^2)`` and the
    polynomials ``Q_f'`` of the circle test (section 5.1; with the denominator
    left by the division).

    **Algorithm.** The real roots ``>= 1`` of ``P1`` cut ``[1, ∞)`` into open
    intervals. An interval holding an integer of the curve's ``kset`` is sampled
    at one such integer, which gives ``f``; the real roots of ``P2(f)`` cut it
    again, and every subinterval is sampled once. The integer roots are
    instantiated: ``singularity(curve, k=n)``. An instantiated ``n`` joins a
    description that holds exactly at ``n`` (``KRoot.at(n)`` equal to ``ρ_n``,
    the rest equal), or keeps its own, with the minimal polynomials at ``n``. All
    root computations are exact (as for :meth:`KSet.where <flanpy.KSet.where>`).

    **Proposition.** On an open interval ``I ⊂ [1, ∞)`` where no polynomial of
    ``P1`` vanishes, ``f`` and ``j`` are constant; where moreover no polynomial of
    ``P2(f)`` vanishes, so are the case, ``g``, ``j'``, the period and ``unique``.

    *Proof.* (1) For real ``k ∈ I`` the closure has non-negative coefficients
    (polynomials in ``t = k - 1`` with non-negative coefficients) and a constant
    support. It is well posed (a pass for symbolic ``k`` holds at every ``k``,
    system.md §5.7), so ``w(z, k) = Σ w_n(k) z^n`` with ``w_n`` non-negative and
    non-decreasing on ``[1, ∞)``; which ``w_n`` vanish depends only on the support
    (no cancellation), so ``w(·, k)`` is not a polynomial (as at the sample) and
    ``ρ_k < ∞``. ``G(z, w(z, k), k) = 0`` holds in ``Q[k][[z]]``, hence at ``k``;
    the contents of ``lc_w G`` and ``disc_w G`` do not vanish, so ``G(·, ·, k)``
    has full degree in ``w`` and is squarefree, and every singularity of
    ``w(·, k)`` is a root of ``R̃(·, k)`` (FS Lemma VII.4). By Pringsheim (FS
    Thm IV.6), ``ρ_k`` is one, positive.

    (2) The roots of ``R̃(·, k)`` are simple (``disc_z``), constant in number
    (``lc_z``) and never ``0`` except for the factor ``z`` (``R̃'(0, k)``). So the
    positive ones are continuous functions ``σ_1 < ... < σ_s`` on ``I``; each is
    a root of one factor and keeps its position among that factor's positive
    roots.

    (3) ``k ↦ ρ_k`` is continuous on ``I``. If ``q < ρ_{k0}``, then
    ``u = Y(q, k0) + δ (I - J)^{-1} 1`` satisfies ``H(q, u, k0) < u`` strictly
    for small ``δ > 0`` (Notes of :func:`singularity`); ``H`` is polynomial in
    ``k``, so ``H(q, u, k) <= u`` near ``k0``, and the Kleene iterates stay below
    ``u``: ``q <= ρ_k``. If ``q > ρ_{k0}``, the Kleene iterates
    ``ℓ_N(k) = H^N(q, 0, k)`` are unbounded at ``k0`` (a bounded limit would be a
    super-solution, so ``q <= ρ_{k0}``). In the first strongly connected
    component ``C`` with an unbounded coordinate, ``Λ(J(q, ℓ_N(k0), k0)) -> ∞`` if
    ``C`` is nonlinear; if ``C`` is linear, its block of ``J`` tends to
    ``B(q, U)``, ``U`` the finite limits of the earlier coordinates, with
    ``Λ(B(q, U)) >= 1``. Every cycle of ``B`` has an entry that vanishes at
    ``z = 0`` (``J(0, Y(0))`` is nilpotent: well-posedness) and increases strictly
    with ``q``, so ``Λ(B)`` increases strictly with ``q`` (Perron-Frobenius) and
    equals 1 for at most one ``q`` per component (its radius ``ρ_C``, PS
    Thm 6.4). So for all but finitely many ``q > ρ_{k0}`` some ``N`` has
    ``Λ(J(q, ℓ_N(k0), k0)) > 1``; ``Λ`` is continuous, so the same holds near
    ``k0``, where it forces ``q >= ρ_k``: if ``q < ρ_k``, then ``ℓ_N(k) <= Y(q, k)``
    and ``Λ(J(q, ℓ_N(k), k)) <= Λ(J(q, Y(q, k), k)) < 1`` (PS Lemma 6.7, B.5,
    3.6). Hence ``ρ`` is lower and upper semicontinuous.

    (4) A continuous function on the connected ``I`` with values among the
    distinct continuous ``σ_1(k), ..., σ_s(k)`` is one ``σ_i``: ``f`` and ``j`` are
    constant.

    (5) The case and ``a_k``. A finite limit ``w(ρ_k⁻, k)`` is a root of
    ``G(ρ_k, ·, k)``, hence of ``Res_z(f, G)(·, k)``, hence of ``Ñ(·, k)`` (the
    content in ``k`` does not vanish). Fix ``k0 ∈ I``, rationals
    ``q_0 < τ_1 < q_1 < ... < τ_t < q_t`` around the real roots ``τ_i`` of
    ``Ñ(·, k0)``, chosen with ``G(·, q_i, k0) ≢ 0``, and ``x0 < ρ_{k0}`` above
    every root of these polynomials below ``ρ_{k0}``. They are not zero at
    ``ρ_{k0}`` either (``q_i`` is no root of ``Res_z(f, G)(·, k0)``). (a) Near
    ``k0`` the ``τ_i(k)`` stay simple and continuous, none appears or escapes
    (``disc_w Ñ``, ``lc_w Ñ``), so the same ``q_i`` separate them. (b)
    ``G(x, q_i, k0) != 0`` on the compact ``[x0, ρ_{k0}]``, so
    ``G(x, q_i, k) != 0`` on ``[x0, ρ_k]`` for ``k`` near ``k0`` (item 3). (c)
    ``w(x0, k)`` is continuous near ``k0``: ``w_n(k) <= w_n(k0 + ε)`` and
    ``Σ w_n(k0 + ε) x0^n < ∞`` since ``x0 < ρ_{k0 + ε}`` (M-test). So for ``k``
    near ``k0``, ``w(·, k)`` stays on ``[x0, ρ_k)`` in the window of ``w(x0, k0)``
    and its limit at ``ρ_k`` is no ``q_i``: it is ``∞`` (top window, which holds no
    root) or the root ``τ_i(k)`` of that window. The window is locally constant,
    hence constant on ``I``: so is the case, and in Case I ``a_k = τ_i(k)``. This
    root belongs to one factor ``g`` (roots of distinct factors never meet), is
    positive, and keeps its position ``j'`` among the positive roots of ``g``
    (``Ñ'(0, k) != 0``).

    (6) The period depends only on the support of the closure (PS §7.1), so it is
    constant, and so are the component periods ``p_C``. On ``I``, ``unique`` is
    decided by the table of section 5.5 with an
    exact circle test in ``k``: let ``μ`` be the factor of ``Res_z(f, y - z^2)``
    that vanishes at ``ρ_k^2`` (the same on ``I``: two factors share ``ρ_k^2`` only
    where ``-ρ_k`` is a root of ``f``). If neither ``Res_z(f, f'(-z))`` nor
    ``Res_y(μ, Q_f')`` vanishes identically for any ``f'``, no other candidate
    lies on ``|z| = ρ_k`` for any ``k ∈ I``: such a candidate ``σ`` is a root of
    some ``f'``, and ``σ = -ρ_k`` makes the first resultant vanish at ``k``, a
    non-real ``σ`` makes ``σσ̄ = ρ_k^2`` a common root of ``μ(·, k)`` and
    ``Q_f'(·, k)`` (leading coefficients do not vanish: ``lc_z f``). So
    ``unique`` is ``True`` (FS Lemma VII.4). Otherwise the period (``False``) or
    the component rule (``True``; Notes of :func:`singularity`, which hold at
    every real ``k`` of ``I``: they use only the support) decides, or ``unique`` is
    ``None``. ∎

    Merging needs no argument: each piece's description is proved on each of its
    parts. Sources as for :func:`singularity`; Perron-Frobenius in the form of PS
    Lemma B.5.

    Examples
    --------
    Broda et al. (2020) ``R_k``: one piece, ``ρ_k`` the positive root of
    ``Δ_k = 1 - 2z - (7 + 8k) z^2`` (section 3.1), and
    ``a_k = R_k(ρ_k) = sqrt((k + 1)/2)``.

    >>> from sympy import symbols
    >>> from flanpy import System, eliminate_in_k, singularity_in_k
    >>> z, k, R = symbols("z k R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> cases = singularity_in_k(eliminate_in_k(rk, R))
    >>> ((piece, s),) = cases.pieces
    >>> print(piece)
    [1, ∞)
    >>> s.rho, s.case, s.a
    (KRoot(8*k*z**2 + 7*z**2 + 2*z - 1, 0, k=k), 'I', KRoot(2*R**2 - k - 1, 0, k=k))
    >>> s.specialize(2).rho
    AlgebraicReal(23*z**2 + 2*z - 1, 1)
    """
    if isinstance(curves, KCases):
        # The values carry their piece (an input piece may be smaller than its
        # curve's kset).
        return KCases(
            (
                part,
                Singularity._make(
                    s.curve, s.rho, s.case, s.a, s.period, s.unique, part
                ),
            )
            for part, s in curves.flat_map(_in_k).pieces
        )
    if not isinstance(curves, Curve):
        raise TypeError(
            f"expected a flanpy.Curve or KCases of them, got {type(curves).__name__}"
        )
    return _in_k(curves)


def _in_k(curve: object) -> KCases[Singularity[Any]]:
    if not isinstance(curve, Curve):
        raise TypeError(f"expected a flanpy.Curve, got {type(curve).__name__}")
    if curve.k is None:
        if curve.kset == KSet.all():
            raise ValueError("the curve has no parameter k; use flanpy.singularity")
        # A piece of eliminate_in_k with a curve at one fixed k.
        try:
            return KCases({curve.kset: singularity(curve)})
        except UndecidedSingularityError:  # pragma: no cover - not at a fixed k >= 1
            raise
        except ValueError as error:
            raise ValueError(
                f"at k in {curve.kset}: {error}; restrict the input to exclude it"
            ) from error
    return _Family(curve).run()


# -- symbolic k (singularity.md, section 6) -------------------------------------------


@dataclass(frozen=True)
class _Description:
    """What is constant on a piece (singularity.md, section 6.1)."""

    rho: KRoot
    case: Case
    a: KRoot | None
    period: int
    unique: bool | None


@dataclass(frozen=True)
class _Second:
    """``P2(f)`` and what the description on a subinterval needs."""

    polys: tuple[Any, ...]  # in k
    g_factors: tuple[sp.Poly, ...]  # of Ñ, gens (w, k)
    mu_factors: tuple[sp.Poly, ...]  # of Res_z(f, y - z^2), gens (y, k)
    always_pairs: frozenset[int]  # μ with Res_y(μ, Q_f') ≡ 0 for some f'
    always_minus: bool  # Res_z(f, f'(-z)) ≡ 0 for some f'


def _split(expr: Any, x: Any, k: Any) -> tuple[list[sp.Poly], list[Any]]:
    """Distinct irreducible factors of positive degree in ``x``, and those in ``k``.

    The first are normalised polynomials with gens ``(x, k)``; the second are
    expressions in ``k``.
    """
    with_x: list[sp.Poly] = []
    in_k: list[Any] = []
    for factor, _ in sp.factor_list(sp.expand(expr), x, k)[1]:
        poly = sp.Poly(factor, x, k, domain=sp.QQ)
        if poly.degree(x) > 0:
            normal = _primitive_in_x(poly)
            if normal not in with_x:
                with_x.append(normal)
        elif poly.degree(k) > 0:
            in_k.append(factor)
    return with_x, in_k


def _at(poly: sp.Poly, n: int) -> sp.Poly:
    """``poly(x, n)`` for ``poly`` with gens ``(x, k)``, over ``QQ``."""
    x, k = poly.gens
    return sp.Poly(poly.as_expr().xreplace({k: sp.Integer(n)}), x, domain=sp.QQ)


def _layout(polys: Sequence[Any], k: Any) -> tuple[set[int], list[_Interval]]:
    """Integer roots ``>= 1`` of the polynomials, and the integer spans between.

    A span ``(lo, hi)`` holds the integers of an open interval between consecutive
    real roots (``hi = None``: up to ``∞``); exact (``KSet.where``'s breakpoints).
    """
    roots: set[int] = set()
    events: list[tuple[int, int]] = []  # (n, 0): root n; (m, 1): a root in (m, m+1)
    for p in polys:
        poly = sp.Poly(p, k, domain=sp.QQ)
        if poly.degree() < 1:
            continue
        integer_roots, floors = _breakpoints(_key(poly))
        roots.update(integer_roots)
        events.extend((m, 1) for m in floors)
    events.extend((n, 0) for n in roots)
    spans: list[_Interval] = []
    start = 1
    for position, kind in sorted(events):
        hi = position - 1 if kind == 0 else position
        if start <= hi:
            spans.append((start, hi))
        start = max(start, position + 1)
    spans.append((start, None))
    return roots, spans


def _inside(span: _Interval, outer: _Interval) -> bool:
    lo, hi = span
    a, b = outer
    return a <= lo and (b is None or (hi is not None and hi <= b))


class _Family:
    """The analysis of a curve with symbolic ``k`` (singularity.md, section 6)."""

    def __init__(self, curve: Curve) -> None:
        self.curve = curve
        w, z, k = curve.w, curve.z, curve.k
        self.w, self.z, self.k = w, z, k
        self.g = curve.polynomial.as_expr()
        found = _closure(curve.system, w)
        if found is None:
            raise ValueError(f"{w} is identically 0: it has no singularity")
        unknowns, equations = found
        lc = sp.Poly(self.g, w).LC()
        candidates = (
            lc * sp.discriminant(self.g, w) if sp.Poly(self.g, w).degree() > 1 else lc
        )
        self._check_signs(unknowns, equations, candidates)
        factors, contents = _split(candidates, z, k)
        self.factors = [f for f in factors if f.as_expr() != z]  # R̃'
        whole = sp.Mul(*(f.as_expr() for f in factors))  # R̃
        support = [
            c
            for u in unknowns
            for c in sp.Poly(equations[u], z, *unknowns).coeffs()
            if c.has(k)
        ]
        self.first: list[Any] = [
            *contents,
            sp.Poly(whole, z).LC(),
            sp.discriminant(whole, z) if sp.Poly(whole, z).degree() > 1 else 1,
            sp.Mul(*(f.as_expr() for f in self.factors)).subs(z, 0),
            *support,
        ]
        self._samples: dict[int, tuple[Singularity[AlgebraicReal], _Closure]] = {}
        self._second: dict[sp.Poly, _Second] = {}

    def _check_signs(
        self, unknowns: Sequence[Any], equations: dict[Any, Any], candidates: Any
    ) -> None:
        """Raise unless the closure is non-negative for ``k >= 1`` (``k = 1 + t``)."""
        t = sp.Dummy("t")
        for u in unknowns:
            poly = sp.Poly(equations[u], self.z, *unknowns)
            for monomial, c in poly.terms():
                shifted = sp.Poly(sp.expand(c.subs(self.k, 1 + t)), t, domain=sp.QQ)
                if any(x < 0 for x in shifted.coeffs()):
                    term = sp.Mul(
                        *(
                            v**e
                            for v, e in zip((self.z, *unknowns), monomial, strict=True)
                        )
                    )
                    raise UndecidedSingularityError(
                        f"cannot decide the dominant singularity of {self.w} for "
                        f"symbolic k: in the equation of {u}, the coefficient "
                        f"{sp.sstr(c)} of {sp.sstr(term)} has a negative coefficient "
                        f"as a polynomial in t = {self.k} - 1. Non-negative "
                        "coefficients in t are sufficient for a series with "
                        "non-negative coefficients at every k >= 1, not necessary; "
                        "signed systems are not supported yet. The candidates are "
                        f"the roots of {sp.sstr(sp.factor(candidates))}."
                    )

    # -- samples -----------------------------------------------------------------

    def sample(self, n: int) -> tuple[Singularity[AlgebraicReal], _Closure]:
        """Return the exact analysis at ``k = n`` (warnings are the caller's)."""
        if n not in self._samples:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SingularityWarning)
                try:
                    self._samples[n] = _analysis(self.curve.specialize(n))
                except UndecidedSingularityError:  # pragma: no cover - k = 1 + t
                    raise
                except ValueError as error:
                    raise ValueError(
                        f"at k = {n}: {error}; restrict the input to exclude it"
                    ) from error
        return self._samples[n]

    def factor_of(self, rho: AlgebraicReal, n: int) -> sp.Poly:
        """Return the factor ``f`` of ``R̃'`` with ``f(ρ_n, n) = 0``."""
        for f in self.factors:
            at_n = _at(f, n)
            if not at_n.is_zero and rho.sign(at_n) == 0:
                return f
        raise RuntimeError(  # pragma: no cover - ρ_n is a candidate
            f"ρ = {rho} is a root of no factor of the candidates at k = {n}"
        )

    # -- P2(f) -------------------------------------------------------------------

    def second(self, f: sp.Poly) -> _Second:
        if f not in self._second:
            self._second[f] = self._compute_second(f)
        return self._second[f]

    def _compute_second(self, f: sp.Poly) -> _Second:
        w, z, k = self.w, self.z, self.k
        fe = f.as_expr()
        polys: list[Any] = []
        g_factors, contents = _split(sp.resultant(fe, self.g, z), w, k)
        polys += contents
        n_tilde = sp.Mul(*(g.as_expr() for g in g_factors))
        polys.append(sp.Poly(n_tilde, w).LC())
        if sp.Poly(n_tilde, w).degree() > 1:
            polys.append(sp.discriminant(n_tilde, w))
        polys.append(
            sp.Mul(*(g.as_expr() for g in g_factors if g.as_expr() != w)).subs(w, 0)
        )
        # E(f): the circle test with k symbolic (section 6.4).
        y = sp.Dummy("y")
        mu_factors, contents = _split(sp.resultant(fe, y - z**2, z), y, k)
        polys += contents
        always_pairs: set[int] = set()
        always_minus = False
        for other in self.factors:
            oe = other.as_expr()
            minus = sp.expand(sp.resultant(fe, oe.subs(z, -z), z))
            if minus == 0:
                always_minus = True
            else:
                polys.append(minus)
            d = other.degree(z)
            if d < 2:
                continue
            products = sp.resultant(oe, sp.expand(z**d * oe.subs(z, y / z)), z)
            squares = sp.resultant(oe, y - z**2, z)
            numerator, denominator = sp.fraction(sp.cancel(products / squares))
            if denominator.has(y):  # pragma: no cover - squares divides products
                raise RuntimeError("the pair products are not a polynomial")
            polys.append(denominator)
            for i, mu in enumerate(mu_factors):
                r = sp.expand(sp.resultant(mu.as_expr(), numerator, y))
                if r == 0:
                    always_pairs.add(i)
                else:
                    polys.append(r)
        return _Second(
            tuple(polys),
            tuple(g_factors),
            tuple(mu_factors),
            frozenset(always_pairs),
            always_minus,
        )

    # -- descriptions --------------------------------------------------------------

    def describe(self, n: int, f: sp.Poly, second: _Second) -> _Description:
        """Return the description at a sample ``n`` of a subinterval (section 6.1)."""
        s, closure = self.sample(n)
        rho = KRoot._make(f, _positive_roots(_at(f, n)).index(s.rho))
        a = None
        if s.a is not None:
            g = next(
                g
                for g in second.g_factors
                if not _at(g, n).is_zero and s.a.sign(_at(g, n)) == 0
            )
            a = KRoot._make(g, _positive_roots(_at(g, n)).index(s.a))
        square = _square(s.rho)
        mu = next(
            i
            for i, m in enumerate(second.mu_factors)
            if not _at(m, n).is_zero and square.sign(_at(m, n)) == 0
        )
        proved_none = mu not in second.always_pairs and not second.always_minus
        unique = _rule(s.period, proved_none, closure)
        return _Description(rho, s.case, a, s.period, unique)

    def exceptional(self, n: int, known: Sequence[_Description]) -> _Description:
        """Return the description at an integer root ``n``.

        A known description if it holds exactly at ``n``. Otherwise ``ρ`` and ``a``
        are a known ``KRoot`` that holds exactly at ``n``, or the minimal
        polynomial at ``n`` (constant in ``k``).
        """
        s, _ = self.sample(n)
        for d in known:
            if (d.case, d.period, d.unique) != (s.case, s.period, s.unique):
                continue
            if not _holds(d.rho, n, s.rho):
                continue
            if (d.a is None) != (s.a is None):
                continue
            if d.a is not None and s.a is not None and not _holds(d.a, n, s.a):
                continue
            return d
        rho = next(
            (d.rho for d in known if _holds(d.rho, n, s.rho)),
            _constant_root(s.rho, self.k),
        )
        a = None
        if s.a is not None:
            a = next(
                (d.a for d in known if d.a is not None and _holds(d.a, n, s.a)),
                _constant_root(s.a, self.k),
            )
        return _Description(rho, s.case, a, s.period, s.unique)

    # -- the pieces ------------------------------------------------------------------

    def run(self) -> KCases[Singularity[KRoot]]:
        kset = self.curve.kset
        intervals: list[tuple[KSet, _Description]] = []
        points1, spans1 = _layout(self.first, self.k)
        points = {n for n in points1 if n in kset}
        for span in spans1:
            inside = KSet([span]) & kset
            if not inside:
                continue
            n0 = inside.intervals[0][0]
            rho0 = self.sample(n0)[0].rho
            f = self.factor_of(rho0, n0)
            second = self.second(f)
            points2, spans2 = _layout([*self.first, *second.polys], self.k)
            for sub in spans2:
                part = KSet([sub]) & kset
                if part and _inside(sub, span):
                    n = n0 if n0 in part else part.intervals[0][0]
                    intervals.append((part, self.describe(n, f, second)))
            points |= {n for n in points2 if n in inside}
        known = list(dict.fromkeys(d for _, d in intervals))
        pieces = intervals + [
            (KSet.of(n), self.exceptional(n, known)) for n in sorted(points)
        ]
        merged: dict[_Description, KSet] = {}
        for part, d in pieces:
            merged[d] = merged.get(d, KSet()) | part
        undecided = [part for d, part in merged.items() if d.unique is None]
        if undecided:
            where = " ∪ ".join(str(part) for part in undecided)
            warnings.warn(
                SingularityWarning(
                    f"uniqueness of the dominant singularity of {self.w} is "
                    f"undecided for k in {where}"
                ),
                stacklevel=4,
            )
        return KCases(
            (
                part,
                Singularity._make(
                    self.curve, d.rho, d.case, d.a, d.period, d.unique, part
                ),
            )
            for d, part in merged.items()
        )


def _holds(root: KRoot, n: int, x: AlgebraicReal) -> bool:
    """Tell whether ``root.at(n)`` exists and equals ``x`` (exact)."""
    try:
        return root.at(n) == x
    except ValueError:
        return False


def _constant_root(x: AlgebraicReal, k: Any) -> KRoot:
    """``x`` as a :class:`KRoot` of its minimal polynomial (constant in ``k``)."""
    m = x.polynomial
    poly = sp.Poly(m.as_expr(), m.gen, k, domain=sp.ZZ)
    return KRoot._make(poly, _positive_roots(m).index(x))


# -- the analysis at a fixed k -----------------------------------------------------


def _analyse(curve: Curve) -> Singularity[AlgebraicReal]:
    return _analysis(curve)[0]


def _analysis(curve: Curve) -> tuple[Singularity[AlgebraicReal], _Closure]:
    """Return the analysis at a fixed ``k``, and the closure it used."""
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
    return Singularity._make(curve, rho, case, a, period, unique), closure


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


@dataclass(frozen=True)
class _Window:
    """The data of the window test, also used by step 5 (newton.md, section 3.3).

    ``x0 < ρ`` is rational with no candidate in ``[x0, ρ)``; ``separators`` separate
    the real roots of ``N = Res_z(m_ρ, G)``; ``w``'s window is
    ``(left, right)`` (``None`` for ``∓∞``): ``(q_{j-1}, q_j)`` with ``a`` in it in
    Case I, the top window ``(q_s, ∞)`` in Case II; ``[lower, upper]`` encloses
    ``w(x0)`` and holds exactly one root of ``G(x0, ·)``.
    """

    case: Case
    a: AlgebraicReal | None
    x0: Fraction
    separators: tuple[Fraction, ...]
    left: Fraction | None
    right: Fraction | None
    lower: Fraction
    upper: Fraction


def _window_test(
    g: sp.Poly, closure: _Closure, rho: AlgebraicReal, previous: Fraction
) -> tuple[Case, AlgebraicReal | None]:
    """Case I / II and ``a`` (singularity.md, section 4.1)."""
    window = _window(g, closure, rho, previous)
    return window.case, window.a


def _window(
    g: sp.Poly, closure: _Closure, rho: AlgebraicReal, previous: Fraction
) -> _Window:
    """Run the window test and return its data (singularity.md, section 4.1)."""
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
    data = (x0, tuple(separators), windows[j], windows[j + 1], lo, hi)
    if j == len(separators):
        return _Window("II", None, *data)
    a_value = roots[j - 1]
    _check_branch_point(g, rho, a_value)
    return _Window("I", a_value, *data)


def _window_of(s: Singularity[AlgebraicReal]) -> _Window:
    """Recompute the window data of a fixed-``k`` singularity (for step 5).

    The closure, the candidates and the candidate below ``ρ`` are recomputed as in
    :func:`_analysis`; ``ρ`` is found among the positive candidates by exact
    equality, so the result agrees with the analysis that produced ``s``.
    """
    curve = s.curve
    w, z = curve.w, curve.z
    g = sp.Poly(curve.polynomial.as_expr(), w, z, domain=sp.QQ)
    found = _closure(curve.system, w)
    assert found is not None  # s exists, so w is not identically 0
    closure = _Closure(z, found[0], found[1])
    positive = _positive_roots(_candidates(g))
    index = positive.index(s.rho)
    previous = positive[index - 1].interval[1] if index else Fraction(0)
    return _window(g, closure, s.rho, previous)


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


def _periods(closure: _Closure) -> list[int]:
    """Return the periods of all coordinates (PS 2025 §7.1, Algorithm Periods).

    The rules are Bell-Burris-Yeats's: ``Π(sum) = gcd(Π's, valuation differences)``
    and ``Π(product) = gcd(Π's)``, iterated to a fixed point from ``0``. They are
    exact, not just bounds, because no coefficient cancels (non-negative
    coefficients: the support of a sum is the union of the supports, that of a
    product their sumset; BBY Prop. 2.9). Section 5.5 relies on this exactness.
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
            return per
        per = new


def _period(closure: _Closure, target: int) -> int:
    """Return the period of coordinate ``target`` (see :func:`_periods`)."""
    return _periods(closure)[target]


# -- uniqueness (singularity.md, sections 5.3 and 5.5) -------------------------------


def _components(closure: _Closure) -> list[list[int]]:
    """Return the strongly connected components, each after those it depends on.

    Tarjan's algorithm on the dependency graph (``i -> j`` when ``H_i`` depends on
    ``Y_j``); it emits a component once everything reachable from it is emitted.
    """
    graph = [sorted(edges) for edges in closure.graph()]
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    on_stack: set[int] = set()
    out: list[list[int]] = []
    for root in range(closure.m):
        if root in index:
            continue
        work = [(root, 0)]  # (vertex, position in its edge list), iteratively
        while work:
            v, position = work.pop()
            if position == 0:
                index[v] = low[v] = len(index)
                stack.append(v)
                on_stack.add(v)
            for p in range(position, len(graph[v])):
                u = graph[v][p]
                if u not in index:
                    work.append((v, p + 1))
                    work.append((u, 0))
                    break
                if u in on_stack:
                    low[v] = min(low[v], index[u])
            else:
                if low[v] == index[v]:
                    component = []
                    while True:
                        u = stack.pop()
                        on_stack.discard(u)
                        component.append(u)
                        if u == v:
                            break
                    out.append(sorted(component))
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[v])
    return out


def _cycle_period(
    closure: _Closure, component: Sequence[int], val: Sequence[int], per: Sequence[int]
) -> int:
    """Return the cycle period of a linear component (singularity.md, section 5.5).

    The gcd of the exponents of the closed walks in the graph of
    ``B = ∂H_C/∂Y_C``, where an edge ``i -> j`` may use any exponent of the support
    of ``B_ij``, a polynomial in ``z`` and earlier coordinates with valuation
    ``v_ij`` and period ``π_ij`` (exact BBY rules). With potentials ``d`` along a
    spanning tree (``d_j = d_i + v_ij``), it is ``gcd(d_i + v_ij - d_j, π_ij)``
    over the edges. It is the period of the diagonal of ``(I - B)^{-1}`` (PS 2025
    Thm 6.4, Algorithm Irr-DS), and can exceed the coordinate period (PS Example
    23: coordinates of period 1, cycle period 5).
    """
    weight: dict[tuple[int, int], int] = {}
    period: dict[tuple[int, int], int] = {}
    for i in component:
        for j in component:
            terms = closure.jacobian_terms[i][j]
            if not terms:
                continue
            exponents = [
                ez + sum(e_u * val[u] for u, e_u in enumerate(e)) for _, ez, e in terms
            ]
            v = min(exponents)
            weight[i, j] = v
            period[i, j] = math.gcd(
                *(x - v for x in exponents),
                *(per[u] for _, _, e in terms for u, e_u in enumerate(e) if e_u),
            )
    d = {component[0]: 0}
    todo = [component[0]]
    while todo:  # a spanning tree (the component is strongly connected)
        i = todo.pop()
        for (a, b), v in weight.items():
            if a == i and b not in d:
                d[b] = d[a] + v
                todo.append(b)
    return math.gcd(
        *(d[a] + v - d[b] for (a, b), v in weight.items()), *period.values()
    )


def _component_periods(closure: _Closure) -> list[tuple[list[int], bool, int]]:
    """Return ``(component, nonlinear, p_C)`` for each recursive component.

    ``p_C`` is the common period of the coordinates for a nonlinear component (PS
    Thm 6.2) and the cycle period for a linear one (singularity.md, section 5.5).
    """
    val = _valuations(closure)
    per = _periods(closure)
    graph = closure.graph()
    out = []
    for component in _components(closure):
        inside = set(component)
        if not any(graph[i] & inside for i in component):
            continue  # nonrecursive: a polynomial in z and earlier coordinates
        nonlinear = any(
            sum(e[j] for j in component) >= 2
            for i in component
            for _, _, e in closure.terms[i]
        )
        if nonlinear:
            p = per[component[0]]
        else:
            p = _cycle_period(closure, component, val, per)
        out.append((component, nonlinear, p))
    return out


def _rule(period: int, proved_none: bool, closure: _Closure) -> bool | None:
    """Decide ``unique`` by the table of singularity.md, section 5.5."""
    if period > 1:
        return False  # every ρ e^{2iπj/p} is a singularity (Pringsheim for V)
    if proved_none:
        return True  # every singularity is a candidate (FS Lemma VII.4)
    if all(p == 1 for _, _, p in _component_periods(closure)):
        return True  # section 5.5: every recursive component has period 1
    return None


def _unique(
    candidates: sp.Poly, rho: AlgebraicReal, period: int, closure: _Closure, w: Any
) -> bool | None:
    if period > 1:
        return False
    circle = _others_on_circle(candidates, rho)
    unique = _rule(period, circle.proved_none, closure)
    if unique is not None:
        return unique
    others = ", ".join(f"{s:.6g}" for s in circle.roots) or "none found numerically"
    periodic = "; ".join(
        f"{{{', '.join(str(closure.unknowns[i]) for i in component)}}} has period {p}"
        for component, _, p in _component_periods(closure)
        if p != 1
    )
    warnings.warn(
        SingularityWarning(
            f"uniqueness of the dominant singularity of {w} is undecided: other "
            f"candidates on |z| = ρ: {others}; a recursive component of the "
            f"closure is periodic ({periodic}), so the rule of singularity.md, "
            "section 5.5 does not apply"
        ),
        stacklevel=4,
    )
    return None


# Methods of the public class, which __init__.py's doctests miss (_doctests.py).
__test__ = method_doctests(Singularity)
