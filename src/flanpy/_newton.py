# SPDX-License-Identifier: GPL-3.0-or-later
"""The local expansion of the branch at its dominant singularity (build-order step 5).

The design is recorded in ``dev/design/newton.md``. :class:`LocalExpansion` and
:func:`local_expansion` are public (fixed ``k``; symbolic ``k`` comes in chunk 4).

References: Flajolet and Sedgewick, *Analytic Combinatorics* (2009), "FS";
Broda, Machiavelo, Moreira and Reis (2020), "Broda"; Pivoteau and Salvy, *Effective
Asymptotics of Combinatorial Systems* (arXiv:2508.20008v1, 2025), "PS"; Trager (1976)
and Loos (1983) for the number fields (:mod:`flanpy._field`).
"""

from __future__ import annotations

import functools
import itertools
import math
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from fractions import Fraction
from typing import Any, NoReturn

import sympy as sp

from flanpy._algebraic import AlgebraicReal, KRoot, _intervals, _normalised, _rational
from flanpy._doctests import method_doctests
from flanpy._field import (
    _divmod,
    _Element,
    _gcd,
    _identity,
    _monic,
    _NumberField,
    _trim,
)
from flanpy._kset import KSet
from flanpy._singularity import Case, Singularity, _Window, _window_of

__all__ = ["LocalExpansion", "local_expansion"]

# Private hook (newton.md, section 2.2): which rule decided each hull entry of the
# top-level Newton polygon. Keys: "zero polynomial", "remainder mod m_rho",
# "gcd test", "field". Tests reset and read it.
_STATS: Counter[str] = Counter()

# A bound on the terms scanned for the first non-integer exponent in Case I. In our
# scope the first term is already singular (PS Thm 8.5; Notes of local_expansion).
_SCAN = 1000

_Poly = dict[tuple[int, int], _Element]  # (power of the variable, power of Y)


# -- the public type -------------------------------------------------------------------


class LocalExpansion[X: (AlgebraicReal, KRoot)]:
    """The local expansion of ``w(z)`` at its dominant singularity ``ρ``.

    Returned by :func:`local_expansion`; not built directly. With ``s = 1 - z/ρ``
    and ``t = s^{1/κ}`` (the positive root for ``z < ρ``), Newton-Puiseux (FS
    Thm VII.7) gives near ``ρ``, in Broda et al. (2020)'s notation (eqs. (20),
    (31)):

    - Case I: ``w = a + d_1 t + d_2 t^2 + ...``; :attr:`alpha` is the least
      exponent ``n/κ`` that is not an integer with ``d_n != 0``, and
      ``b = -d_{ακ}``, so ``w = (analytic) - b s^α + ...``;
    - Case II: ``1/w = c s^α + ...``, so ``w = (1/c) s^{-α} + ...``.

    Fields: :attr:`alpha`, :attr:`ramification` (``κ``), :attr:`b` (Case I) or
    :attr:`c` (Case II), exact; :meth:`terms` gives the coefficients ``d_n``.

    Notes
    -----
    Equality and hashing compare the singularity, ``alpha``, ``ramification``,
    ``b`` and ``c``. A ``LocalExpansion`` is immutable and can be pickled (the
    cached terms are recomputed after unpickling).

    Examples
    --------
    Catalan, ``F = 1 + zF^2``: ``F = 2/(1 + √s)`` with ``s = 1 - 4z``, so
    ``α = 1/2`` and ``b = 2`` (FS eq. (81)).

    >>> from sympy import symbols
    >>> from flanpy import System, eliminate, local_expansion, singularity
    >>> z, F = symbols("z F")
    >>> e = local_expansion(singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)))
    >>> e  # doctest: +NORMALIZE_WHITESPACE
    LocalExpansion(alpha=Fraction(1, 2), ramification=2, b=AlgebraicReal(y - 2, 0),
                   c=None)
    >>> print(e)
    α = 1/2, κ = 2, Case I, b = 2
    """

    __slots__ = (
        "_alpha",
        "_b",
        "_c",
        "_ramification",
        "_singularity",
        "_state",
        "_terms",
    )

    _singularity: Singularity[X]
    _alpha: Fraction
    _ramification: int
    _b: X | None
    _c: X | None
    _state: _State | None
    _terms: list[AlgebraicReal]

    def __init__(self) -> None:
        raise TypeError("LocalExpansion objects are created by flanpy.local_expansion")

    @classmethod
    def _make(
        cls,
        singularity: Singularity[X],
        alpha: Fraction,
        ramification: int,
        b: X | None,
        c: X | None,
        state: _State | None,
    ) -> LocalExpansion[X]:
        new = object.__new__(cls)
        _set = object.__setattr__
        _set(new, "_singularity", singularity)
        _set(new, "_alpha", alpha)
        _set(new, "_ramification", ramification)
        _set(new, "_b", b)
        _set(new, "_c", c)
        _set(new, "_state", state)
        _set(new, "_terms", [])
        return new

    # -- public interface ----------------------------------------------------------

    @property
    def singularity(self) -> Singularity[X]:
        """The dominant singularity the expansion is at (step 4).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, F = symbols("z F")
        >>> s = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
        >>> local_expansion(s).singularity is s
        True
        """
        return self._singularity

    @property
    def alpha(self) -> Fraction:
        """Broda's ``α``: ``w = (analytic) - b s^α + ...`` (I), ``w ~ s^{-α}/c`` (II).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, F = symbols("z F")
        >>> s = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
        >>> local_expansion(s).alpha
        Fraction(1, 2)
        """
        return self._alpha

    @property
    def ramification(self) -> int:
        """``κ``: the expansion is in powers of ``t = (1 - z/ρ)^{1/κ}`` (FS Thm VII.7).

        Examples
        --------
        ``K = U/2 + K^2`` with ``U = z + U^2`` is ``K = (1 - (1 - 4z)^{1/4})/2``:

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, U, K = symbols("z U K")
        >>> system = System({U: z + U**2, K: U / 2 + K**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(system, K)))
        >>> e.ramification, e.alpha
        (4, Fraction(1, 4))
        """
        return self._ramification

    @property
    def b(self) -> X | None:
        """Case I: ``b > 0`` with ``w = (analytic) - b s^α + ...`` (Broda eq. (20)).

        ``None`` in Case II.

        Examples
        --------
        Motzkin, ``F = z + zF + zF^2``: ``F = 1 - √3 s^{1/2} + ...`` with
        ``s = 1 - 3z`` (FS Example VII.18):

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, F = symbols("z F")
        >>> motzkin = System({F: z + z * F + z * F**2}, z=z)
        >>> local_expansion(singularity(eliminate(motzkin, F))).b
        AlgebraicReal(y**2 - 3, 1)
        """
        return self._b

    @property
    def c(self) -> X | None:
        """Case II: ``c > 0`` with ``1/w = c s^α + ...`` (Broda eq. (31)).

        ``None`` in Case I.

        Examples
        --------
        ``Y = 1 + 2zY`` is ``1/(1 - 2z) = 1/s``: a pole, ``α = 1``, ``c = 1``.

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, Y = symbols("z Y")
        >>> e = local_expansion(
        ...     singularity(eliminate(System({Y: 1 + 2 * z * Y}, z=z), Y))
        ... )
        >>> e.alpha, e.c
        (Fraction(1, 1), AlgebraicReal(y - 1, 0))
        """
        return self._c

    @property
    def k(self) -> Any:
        """The parameter: ``None`` at a fixed ``k`` (forwarded from the singularity).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, F = symbols("z F")
        >>> s = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
        >>> local_expansion(s).k is None
        True
        """
        return self._singularity.k

    @property
    def kset(self) -> KSet:
        """The ``k`` this result is for (forwarded from the singularity).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> local_expansion(singularity(eliminate(rk, R, k=2))).kset
        KSet([(2, 2)])
        """
        return self._singularity.kset

    def terms(self, n: int) -> list[tuple[Fraction, AlgebraicReal]]:
        """Return the first ``n`` terms ``(exponent, coefficient)`` of ``w``.

        ``w`` is expanded in powers of ``t = s^{1/κ}``, ``s = 1 - z/ρ``, with
        consecutive exponents (zeros included): from ``0`` (Case I, the first
        coefficient is ``a``) or from ``-α`` (Case II). The coefficients are exact
        and cached.

        Parameters
        ----------
        n : int
            How many terms, ``n >= 0``.

        Returns
        -------
        list of (Fraction, AlgebraicReal)
            The exponents in ``s`` and the coefficients.

        Raises
        ------
        TypeError
            If ``n`` is not an ``int``.
        ValueError
            If ``n < 0``, or the expansion is for symbolic ``k``.

        Examples
        --------
        Catalan, ``F = 2/(1 + √s)`` with ``s = 1 - 4z``:

        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, F = symbols("z F")
        >>> e = local_expansion(
        ...     singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
        ... )
        >>> [(str(x), str(d)) for x, d in e.terms(4)]
        [('0', '2'), ('1/2', '-2'), ('1', '2'), ('3/2', '-2')]
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {n!r}")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if self._singularity.k is not None:  # pragma: no cover - chunk 4
            raise ValueError("terms need a fixed k: use specialize(k)")
        if len(self._terms) < n:
            if self._state is None:  # after unpickling
                fixed = self._singularity
                assert isinstance(fixed.rho, AlgebraicReal)
                object.__setattr__(self, "_state", _analyse(fixed)[1])
            assert self._state is not None
            values = self._state.terms(n)
            for d in values[len(self._terms) : n]:
                self._terms.append(d.to_real())
        start, kappa = self._state_start(), self._ramification
        return [(start + Fraction(i, kappa), d) for i, d in enumerate(self._terms[:n])]

    def _state_start(self) -> Fraction:
        return Fraction(0) if self._singularity.case == "I" else -self._alpha

    def specialize(self, k: int) -> LocalExpansion[AlgebraicReal]:
        """Return the expansion at a fixed ``k`` (for results on a piece of ``k``).

        Pieces of ``k`` come with ``local_expansion_in_k`` (build-order step 5,
        chunk 4); a fixed-``k`` expansion has nothing to specialize.

        Parameters
        ----------
        k : int
            A value in :attr:`kset`.

        Returns
        -------
        LocalExpansion
            The expansion at ``k``.

        Raises
        ------
        ValueError
            If the expansion is already at a fixed ``k``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, eliminate, local_expansion, singularity
        >>> z, F = symbols("z F")
        >>> s = singularity(eliminate(System({F: 1 + z * F**2}, z=z), F))
        >>> local_expansion(s).specialize(2)
        Traceback (most recent call last):
            ...
        ValueError: the expansion is at a fixed k: nothing to specialize
        """
        if self._singularity.k is None:
            raise ValueError("the expansion is at a fixed k: nothing to specialize")
        return local_expansion(self._singularity, k=k)  # pragma: no cover - chunk 4

    # -- value semantics -----------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (self._singularity, self._alpha, self._ramification, self._b, self._c)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LocalExpansion):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return (
            f"LocalExpansion(alpha={self._alpha!r}, ramification={self._ramification}, "
            f"b={self._b!r}, c={self._c!r})"
        )

    def __str__(self) -> str:
        constant = f"b = {self._b}" if self._b is not None else f"c = {self._c}"
        return (
            f"α = {self._alpha}, κ = {self._ramification}, "
            f"Case {self._singularity.case}, {constant}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _restore,
            (self._singularity, self._alpha, self._ramification, self._b, self._c),
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("LocalExpansion is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("LocalExpansion is immutable")


def _restore[X: (AlgebraicReal, KRoot)](
    singularity: Singularity[X],
    alpha: Fraction,
    ramification: int,
    b: X | None,
    c: X | None,
) -> LocalExpansion[X]:
    """Rebuild a pickled :class:`LocalExpansion` (terms recomputed on demand)."""
    return LocalExpansion._make(singularity, alpha, ramification, b, c, None)


LocalExpansion.__module__ = "flanpy"


# -- the public function -------------------------------------------------------------


def local_expansion(
    singularity: Singularity[Any], *, k: int | None = None
) -> LocalExpansion[AlgebraicReal]:
    """Return the local expansion of ``w`` at its dominant singularity, fixed ``k``.

    The branch ``w(z)`` of ``singularity.curve`` is expanded at ``ρ`` by
    Newton-Puiseux (FS Thm VII.7; Broda et al. 2020, eqs. (20)-(31)):

    1. **The local problem.** Case I: ``Q(s, Y) = G(ρ(1 - s), a + Y)``; Case II:
       ``Q(s, V) = H(ρ(1 - s), V)`` with ``H(z, v) = v^m G(z, 1/v)`` the reciprocal
       polynomial (Broda eq. (29)). The coefficients
       ``q_ij = (-ρ)^i/(i! j!) ∂^i_z ∂^j_w G(ρ, a)`` (Broda eq. (21)) are exact, in
       ``Q(ρ)`` when ``a ∈ Q(ρ)`` (always in Case II; ``a`` from
       ``gcd(G(ρ, ·), G_w(ρ, ·))`` over ``Q(ρ)``), else in ``Q(ρ, a)``.
    2. **The Newton polygon** of ``Q`` (FS pp. 499-500; Broda eqs. (23)-(26)):
       only the columns ``j <= r``, ``r`` the multiplicity of the point; which
       entries vanish is decided exactly (a zero polynomial; a remainder modulo
       ``m_ρ``; Broda's gcd test; the field). Each edge of slope ``-γ`` carries the
       branches ``Y ~ c s^γ``, ``c`` a root of its characteristic polynomial.
    3. **The real branches and which one is ``w``** (newton.md, section 3): see
       Notes.
    4. **The terms.** A simple root ``c`` is lifted by undetermined coefficients
       (FS p. 500); a multiple root restarts the polygon. In Case II ``w = 1/V`` is
       inverted as a series. The constant ``b`` or ``c`` comes from a binomial edge
       without building a new field when possible (``c^d = u``, newton.md 2.4).

    Parameters
    ----------
    singularity : Singularity
        From :func:`~flanpy.singularity` (a fixed ``k``), or a piece of
        :func:`~flanpy.singularity_in_k` together with ``k``.
    k : int, optional
        Required for a singularity of symbolic ``k`` (it is specialized first);
        must be omitted otherwise.

    Returns
    -------
    LocalExpansion
        ``alpha``, ``ramification``, ``b`` or ``c``, and the terms.

    Raises
    ------
    TypeError
        If ``singularity`` is not a :class:`~flanpy.Singularity`.
    ValueError
        If ``k`` is missing or extra, or not in the singularity's ``kset``.

    See Also
    --------
    LocalExpansion : the fields of the result.

    Notes
    -----
    **Lemma R** (newton.md, section 3.1). Every coefficient ``d_n`` is real. FS
    Thm VII.7 holds in a disc around ``ρ`` slit along ``[ρ, ρ + ε)``; on
    ``(ρ - ε, ρ)``, ``t = s^{1/κ} > 0`` and ``t ↦ Σ d_n t^n = w(ρ(1 - t^κ))`` is a
    convergent Laurent series that is real on an interval, so its coefficients
    are real. Hence ``w``'s leading coefficient is a *real* root of some
    characteristic polynomial, and each real root gives one real branch over
    ``(ρ - ε, ρ)``. All the arithmetic is in real number fields.

    **The sign rule** (section 3.2). ``w`` has non-negative coefficients and is not
    a polynomial, so it increases strictly on ``[0, ρ)``. In Case I ``w < a``
    there, so ``Y = w - a < 0`` and the leading coefficient is negative; in Case II
    ``V = 1/w > 0``. This is asserted (a violation is a bug, ``RuntimeError``).

    **The rank test** (section 3.3; always on). Step 4 gives a rational
    ``x0 < ρ`` with no candidate in ``[x0, ρ)``, separators of the real roots of
    ``N = Res_z(m_ρ, G)``, the window ``W`` of ``w`` (``a ∈ W`` in Case I, the top
    window in Case II) and a certified enclosure of ``w(x0)``. Let ``n_W`` be the
    number of real roots of ``G(x0, ·)`` in ``W`` and ``r_w`` the number below
    ``w(x0)`` (Sturm counts). *Claim 1:* on ``[x0, ρ)`` the roots in ``W`` are
    ``n_W`` continuous functions ``y_1 < ... < y_{n_W}`` and ``w = y_{r_w + 1}``:
    no candidate means simple roots and ``lc_w G != 0``, so roots move
    continuously without meeting, and ``G(x, q_j) != 0`` on ``[x0, ρ]`` keeps them
    inside ``W``. *Claim 2:* as ``x → ρ⁻`` every ``y_i`` tends to ``a`` (Case I; the
    only root of ``G(ρ, ·)`` in ``W``) or to ``+∞`` (Case II; none above the top
    separator), and conversely every real branch at ``(ρ, a)`` (Case II: with
    ``V > 0``) lies in ``W`` near ``ρ``; distinct branches are distinct functions
    (``G`` squarefree). So ``w`` is the branch of rank ``r_w`` when the real
    branches are sorted by their value as ``s → 0⁺``: by the sign of the first
    non-zero term of their difference (exact comparisons, deeper terms on a tie).
    ``n_W`` equal to the number of real branches is asserted.

    **PS 2025, Thm 8.5, for our scope.** The closure of ``w`` (zero coordinates
    removed) is a well-founded polynomial system with non-negative rational
    coefficients. The proof of PS Thm 8.5 (an induction over the strongly
    connected components) uses only sums, products, linear components and
    PS Prop. 8.2, whose proof (PS Appendix D) needs non-negative Taylor
    coefficients, irreducibility and a polynomial dependence on the inputs, never
    integer coefficients; a positive constant factor keeps a negative coefficient
    negative. Its induction hypothesis is that every non-constant coordinate with a
    finite limit has an expansion in ``R{Z^{1/r}}``, ``r = 2^m``, with a *negative*
    coefficient of ``Z^{1/r}``. The printed proof leaves one case implicit, the
    regular nonlinear one (``Λ_H(ρ) < 1``, PS Prop. 8.2 first case): there
    ``Ỹ'(0) = (I - J(ρ))^{-1} W`` with ``(I - J(ρ))^{-1} > 0`` entrywise (``J``
    irreducible, non-negative, ``Λ < 1``) and ``W >= 0``, ``W != 0`` (PS
    eq. (35)), so the coefficient is negative too. For ``w`` in Case I this gives
    ``w = a - b t + ...`` with ``b > 0`` and ``κ = r = 2^m >= 2``, so
    ``α = 1/κ``. This is asserted (``RuntimeError``, a bug if it fails); the
    algorithm itself does not depend on it.

    Examples
    --------
    Catalan, ``F = 1 + zF^2``, at ``ρ = 1/4``: ``F = 2 - 2 s^{1/2} + 2s - ...``
    with ``s = 1 - 4z``.

    >>> from sympy import symbols
    >>> from flanpy import System, eliminate, local_expansion, singularity
    >>> z, F = symbols("z F")
    >>> e = local_expansion(singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)))
    >>> e.alpha, e.ramification, e.b, e.c
    (Fraction(1, 2), 2, AlgebraicReal(y - 2, 0), None)

    A singularity of symbolic ``k`` needs ``k``:

    >>> from flanpy import eliminate_in_k, singularity_in_k
    >>> k, R = symbols("k R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> piece = singularity_in_k(eliminate_in_k(rk, R))[2]
    >>> local_expansion(piece, k=2).b == local_expansion(piece.specialize(2)).b
    True
    """
    if not isinstance(singularity, Singularity):
        raise TypeError(
            f"expected a flanpy.Singularity, got {type(singularity).__name__}"
        )
    if singularity.k is not None:
        if k is None:
            raise ValueError(
                "the singularity has the parameter k: give k=n to expand it at k = n"
            )
        singularity = singularity.specialize(k)
    elif k is not None:
        raise ValueError("the singularity has no parameter k; omit the k argument")
    return _analyse(singularity)[0]


# -- the analysis ----------------------------------------------------------------------


def _analyse(
    s: Singularity[AlgebraicReal],
) -> tuple[LocalExpansion[AlgebraicReal], _State]:
    """Run the fixed-``k`` analysis: the expansion and its state (for the terms)."""
    curve = s.curve
    w, z = curve.w, curve.z
    g = sp.Poly(curve.polynomial.as_expr(), w, z, domain=sp.QQ)
    rho = s.rho
    case = s.case
    window = _window_of(s)
    if window.case != case:  # pragma: no cover - the same analysis as step 4
        raise RuntimeError("the window test disagrees with the singularity")
    problem = _LocalProblem.build(g, rho, case, s.a)
    branches = problem.branches()
    chosen = _choose(branches, window, g, case)
    state = _State(problem, chosen)
    if case == "I":
        alpha, b = state.case_one_constant()
        if b <= 0:  # pragma: no cover - proved, newton.md section 3.2
            raise RuntimeError(f"b = {b} is not positive")
        _check_ps_8_5(alpha, chosen.kappa)
        return LocalExpansion._make(s, alpha, chosen.kappa, b, None, state), state
    c = chosen.leading
    if c <= 0:  # pragma: no cover - the candidates have V > 0
        raise RuntimeError(f"c = {c} is not positive")
    return LocalExpansion._make(s, chosen.gamma, chosen.kappa, None, c, state), state


def _check_ps_8_5(alpha: Fraction, kappa: int) -> None:
    """Case I: ``α = 1/κ`` with ``κ = 2^m >= 2`` (PS Thm 8.5; Notes above)."""
    if not (kappa >= 2 and kappa & (kappa - 1) == 0 and alpha == Fraction(1, kappa)):
        raise RuntimeError(  # pragma: no cover - PS Thm 8.5
            f"alpha = {alpha}, kappa = {kappa}: PS 2025 Thm 8.5 predicts 1/2^m"
        )


@dataclass
class _LocalProblem:
    """``Q(s, Y)`` over ``K`` (newton.md, section 2.1) and its exact zero tests."""

    g: sp.Poly  # G(w, z) over QQ
    rho: AlgebraicReal
    case: Case
    field: _NumberField  # K
    rho_k: _Element
    a_k: _Element  # 0 in Case II
    in_rho: bool  # K = Q(ρ)
    poly: dict[tuple[int, int], Fraction]  # G (Case I) or H (Case II): (dw, dz) -> c
    q: _Poly  # q_ij, (i, j) = (power of s, power of Y)
    max_i: int

    @classmethod
    def build(
        cls, g: sp.Poly, rho: AlgebraicReal, case: Case, a: AlgebraicReal | None
    ) -> _LocalProblem:
        w, _ = g.gens
        terms = {(dw, dz): Fraction(int(c.p), int(c.q)) for (dw, dz), c in g.terms()}
        k_rho = _NumberField(rho)
        if case == "II":
            m = g.degree(w)
            poly = {(m - dw, dz): c for (dw, dz), c in terms.items()}
            field_, rho_k, a_k, in_rho = k_rho, k_rho.generator, k_rho(0), True
        else:
            assert a is not None
            poly = terms
            field_, rho_k, a_k, in_rho = _field_of_a(terms, k_rho, a)
        q = _taylor(poly, field_, rho_k, a_k)
        max_i = max(dz for _, dz in poly)
        return cls(g, rho, case, field_, rho_k, a_k, in_rho, poly, q, max_i)

    def nonzero(self, i: int, j: int) -> bool:
        """Exact ``q_ij != 0``, with the rule that decided it (the cascade, 2.2)."""
        value = self.q.get((i, j), self.field(0))
        if not any(dz >= i and dw >= j for dw, dz in self.poly):
            rule, answer = "zero polynomial", False
        elif self.in_rho:
            rule, answer = "remainder mod m_rho", not value.is_zero()
        elif _broda_gcd_test(self.g, self.rho, i, j):
            rule, answer = "gcd test", True
        else:
            rule, answer = "field", not value.is_zero()
        if answer == value.is_zero():  # pragma: no cover - the rules are exact
            raise RuntimeError(f"zero test of q_{i}{j} inconsistent ({rule})")
        _STATS[rule] += 1
        return answer

    def branches(self) -> list[_Branch]:
        """Return the real branches at the point (all, both signs)."""
        top = _Germ(self.field, _identity, self.q, 1, [], 0)
        return _real_branches(top, self.nonzero, self.max_i, top_level=True)


def _field_of_a(
    poly: dict[tuple[int, int], Fraction], k_rho: _NumberField, a: AlgebraicReal
) -> tuple[_NumberField, _Element, _Element, bool]:
    """``K = Q(ρ)`` if ``a ∈ Q(ρ)``, else ``Q(ρ, a)`` (newton.md, section 2.3)."""
    degree = max(dw for dw, _ in poly)
    by_w: dict[int, dict[int, Fraction]] = {}
    for (dw, dz), c in poly.items():
        by_w.setdefault(dw, {})[dz] = c
    coefficients = []  # G(ρ, w) as a polynomial in w over Q(ρ)
    for dw in range(degree + 1):
        cs = by_w.get(dw, {})
        top = max(cs, default=-1)
        coefficients.append(
            k_rho.from_coefficients([cs.get(i, 0) for i in range(top + 1)])
        )
    derivative = _trim([c * i for i, c in enumerate(coefficients)][1:])
    h = _gcd(_trim(coefficients), derivative)  # monic
    e = len(h) - 1
    candidate: _Element | None = None
    if e == 1:
        candidate = -h[0]
    else:
        dh = _trim([c * i for i, c in enumerate(h)][1:])
        if len(h) - len(_gcd(h, dh)) == 1:  # h = (w - a)^e
            candidate = -h[e - 1] / e
    if candidate is not None:
        if candidate.to_real() != a:  # pragma: no cover - a is the multiple root
            raise RuntimeError("a is not the multiple root of G(ρ, ·)")
        return k_rho, k_rho.generator, candidate, True
    # Several multiple roots, conjugate over Q(ρ): find a among the real ones.
    dh = _trim([c * i for i, c in enumerate(h)][1:])
    squarefree = _monic(_divide(h, _gcd(h, dh)))
    for root in k_rho.real_roots(squarefree):
        if root.value.to_real() == a:
            assert isinstance(root.field, _NumberField)
            return root.field, root.embedding(k_rho.generator), root.value, False
    raise RuntimeError("a is not a root of gcd(G(ρ, ·), G_w(ρ, ·))")  # pragma: no cover


def _divide(a: list[_Element], b: list[_Element]) -> list[_Element]:
    quotient, remainder = _divmod(a, b)
    assert not remainder
    return quotient


def _taylor(
    poly: dict[tuple[int, int], Fraction],
    field_: _NumberField,
    rho: _Element,
    a: _Element,
) -> _Poly:
    """``q_ij``: ``sum c (ρ(1 - s))^dz (a + Y)^dw`` expanded in ``s`` and ``Y``."""
    max_w = max(dw for dw, _ in poly)
    max_z = max(dz for _, dz in poly)
    rho_powers = [field_(1)]
    for _ in range(max_z):
        rho_powers.append(rho_powers[-1] * rho)
    a_powers = [field_(1)]
    for _ in range(max_w):
        a_powers.append(a_powers[-1] * a)
    q: _Poly = {}
    for (dw, dz), c in poly.items():
        base = rho_powers[dz] * c
        for i in range(dz + 1):
            si = base * (math.comb(dz, i) * (-1) ** i)
            for j in range(dw + 1):
                term = si * a_powers[dw - j] * math.comb(dw, j)
                q[(i, j)] = q[(i, j)] + term if (i, j) in q else term
    return {key: value for key, value in q.items() if not value.is_zero()}


def _broda_gcd_test(g: sp.Poly, rho: AlgebraicReal, i: int, j: int) -> bool:
    """Broda's test (text after eq. (22)): ``True`` proves ``∂^i_z ∂^j_w G(ρ, a) != 0``.

    ``gcd(Res_w(G, D), m_ρ) = 1`` means ``D(ρ, w) != 0`` at every root ``w`` of
    ``G(ρ, ·)``, in particular at ``a``; anything else is inconclusive.
    """
    w, z = g.gens
    d = sp.diff(g.as_expr(), z, i, w, j)
    m_rho = rho.polynomial.as_expr().subs(rho.polynomial.gen, z)
    res = sp.resultant(g.as_expr(), d, w) if sp.Poly(d, w).degree() > 0 else d
    return bool(sp.gcd(sp.Poly(res, z), sp.Poly(m_rho, z)).degree() == 0)


# -- germs: the Newton-Puiseux recursion ---------------------------------------------


@dataclass
class _Germ:
    """A partial branch: ``Y = head(v) + v^offset V`` with ``s = v^kappa``.

    ``poly`` is ``P(v, V)`` over ``field`` (``P(0, 0) = 0``), ``embed`` maps ``K``
    into ``field``, ``head`` lists ``(exponent in v, coefficient)``.
    """

    field: _NumberField
    embed: Callable[[_Element], _Element]
    poly: _Poly
    kappa: int
    head: list[tuple[int, _Element]]
    offset: int
    # Hensel lifting of V = sum e_n v^n (a simple root: dP/dV(0, 0) != 0).
    _e: list[_Element] = dataclass_field(default_factory=list)
    _powers: list[list[_Element]] = dataclass_field(default_factory=list)

    def substitute(
        self, p: int, q: int, root: _Element, embed: Callable[[Any], Any]
    ) -> _Germ:
        """Return the germ one level down: ``v = u^q``, ``V = u^p (c + V')``.

        The new germ lives in ``root``'s field; ``embed`` maps this germ's field
        into it.
        """
        L = root.field
        poly = {key: embed(value) for key, value in self.poly.items()}
        beta = min(q * i + p * j for i, j in poly)
        new: _Poly = {}
        powers = [L(1)]
        top = max(j for _, j in poly)
        for _ in range(top):
            powers.append(powers[-1] * root)
        for (i, j), value in poly.items():
            e = q * i + p * j - beta
            for m in range(j + 1):
                term = value * powers[j - m] * math.comb(j, m)
                new[(e, m)] = new[(e, m)] + term if (e, m) in new else term
        new = {key: value for key, value in new.items() if not value.is_zero()}
        head = [(q * x, embed(c)) for x, c in self.head]
        offset = q * self.offset + p
        head.append((offset, root))
        composed = _compose(self.embed, embed)
        return _Germ(L, composed, new, self.kappa * q, head, offset)

    def coefficient(self, n: int) -> _Element:
        """Return the coefficient of ``v^n`` in ``Y`` (Hensel lifting of ``V``)."""
        value = self.field(0)
        for x, c in self.head:
            if x == n:
                value = c
        if n > self.offset:
            self._lift(n - self.offset)
            value = value + self._e[n - self.offset]
        return value

    def _lift(self, n: int) -> None:
        """Compute ``e_1, ..., e_n`` (undetermined coefficients, FS p. 500)."""
        L = self.field
        d = self.poly.get((0, 1))
        if d is None or d.is_zero():  # pragma: no cover - only simple roots lift
            raise RuntimeError("Hensel lifting needs a simple root")
        top = max(j for _, j in self.poly)
        if not self._e:
            self._e.append(L(0))  # e_0 = 0: V -> 0
            self._powers.append([L(1)])  # V^0
            for _ in range(top):
                self._powers.append([L(0)])  # V^j, j >= 1: no constant term
        while len(self._e) <= n:
            m = len(self._e)
            # powers[j][m] for j >= 2 use e_1..e_{m-1} only.
            for j in range(len(self._powers)):
                if j == 0:
                    self._powers[0].append(L(0))
                elif j == 1:
                    self._powers[1].append(L(0))  # e_m, set below
                else:
                    total = L(0)
                    for k in range(1, m):
                        total = total + self._e[k] * self._powers[j - 1][m - k]
                    self._powers[j].append(total)
            total = L(0)
            for (i, j), value in self.poly.items():
                if (i, j) != (0, 1) and i <= m:
                    total = total + value * self._powers[j][m - i]
            e_m = -total / d
            self._e.append(e_m)
            self._powers[1][m] = e_m


def _compose(
    first: Callable[[Any], Any], second: Callable[[Any], Any]
) -> Callable[[_Element], _Element]:
    if first is _identity:
        return second
    if second is _identity:
        return first
    return lambda x: second(first(x))


@dataclass
class _Branch:
    """A real branch at the point: ``Y ~ leading s^gamma`` as ``s → 0⁺``.

    ``germ`` is resolved (a simple root at its last level, Hensel-ready); it is
    built lazily for a root found by the binomial fast path (``make_germ``).
    ``key`` lists the leading data of every level, to order branches that share
    their first terms.
    """

    gamma: Fraction
    leading: AlgebraicReal
    kappa: int
    _germ: _Germ | None
    make_germ: Callable[[], _Germ] | None = None

    @property
    def sign(self) -> int:
        return 1 if self.leading > 0 else -1  # never 0: φ(0) != 0

    def germ(self) -> _Germ:
        if self._germ is None:
            assert self.make_germ is not None
            self._germ = self.make_germ()
        return self._germ

    def expansion(self) -> Iterator[tuple[Fraction, _Element]]:
        """``(exponent in s, coefficient)`` for every power of ``v`` from 1 on."""
        germ = self.germ()
        n = 1
        while True:
            yield Fraction(n, germ.kappa), germ.coefficient(n)
            n += 1


def _polygon(
    nonzero: Callable[[int, int], bool], max_i: int
) -> tuple[int, dict[int, int]]:
    """``r`` and the lowest point of each column ``j <= r``."""
    r = 0
    while not nonzero(0, r):
        r += 1
    lowest: dict[int, int] = {r: 0}
    for j in range(r):
        for i in range(max_i + 1):
            if nonzero(i, j):
                lowest[j] = i
                break
    return r, lowest


def _edges(lowest: dict[int, int]) -> list[tuple[int, int, int, int]]:
    """Edges ``(j1, i1, j2, i2)`` of negative slope of the lower convex hull."""
    hull: list[tuple[int, int]] = []
    for point in sorted(lowest.items()):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (point[0] - x1) >= (point[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(point)
    return [
        (j1, i1, j2, i2) for (j1, i1), (j2, i2) in itertools.pairwise(hull) if i2 < i1
    ]


def _real_branches(
    germ: _Germ,
    zero_test: Callable[[int, int], bool] | None,
    max_i: int | None,
    *,
    top_level: bool = False,
) -> list[_Branch]:
    """All real branches below ``germ`` (newton.md, section 3.1), resolved."""
    out: list[_Branch] = []
    if not any(j == 0 for _, j in germ.poly):  # V = 0 is a branch: P = V P_1
        out.append(_exact_branch(germ))
        shifted = {(i, j - 1): c for (i, j), c in germ.poly.items()}
        germ = _Germ(
            germ.field, germ.embed, shifted, germ.kappa, germ.head, germ.offset
        )
    poly = germ.poly

    def present(i: int, j: int) -> bool:
        return (i, j) in poly

    nonzero = present if zero_test is None else zero_test
    _, lowest = _polygon(nonzero, max(i for i, _ in poly) if max_i is None else max_i)
    for j1, i1, j2, i2 in _edges(lowest):
        gamma = Fraction(i1 - i2, j2 - j1)
        p, q = gamma.numerator, gamma.denominator
        phi = [germ.field(0)] * (j2 - j1 + 1)
        for j in range(j1, j2 + 1):
            if (i1 - lowest.get(j, -1)) * (j2 - j1) == (i1 - i2) * (j - j1):
                phi[j - j1] = poly.get((lowest[j], j), germ.field(0))
        binomial = top_level and math.gcd(j2 - j1, i1 - i2) == 1
        if binomial:
            out.extend(_binomial_branches(germ, phi, p, q))
            continue
        for root in germ.field.real_roots(phi):
            assert isinstance(root.field, _NumberField)
            child = germ.substitute(p, q, root.value, root.embedding)
            exponent = Fraction(child.offset, child.kappa)
            if root.multiplicity == 1:
                out.append(_Branch(exponent, root.value.to_real(), child.kappa, child))
            else:
                for sub in _real_branches(child, None, None):
                    out.append(
                        _Branch(exponent, root.value.to_real(), sub.kappa, sub.germ())
                    )
    return out


def _exact_branch(germ: _Germ) -> _Branch:  # pragma: no cover - multiple roots only
    """Return the branch ``V = 0``: ``Y = head`` exactly."""
    x, c = germ.head[-1]
    return _Branch(Fraction(x, germ.kappa), c.to_real(), germ.kappa, germ)


def _binomial_branches(
    germ: _Germ, phi: list[_Element], p: int, q: int
) -> list[_Branch]:
    """Return the branches of a binomial edge ``φ = φ_0 + φ_d c^d`` (newton.md 2.4).

    The roots are the real ``d``-th roots of ``u = -φ_0/φ_d ∈ K``, simple, as
    ``AlgebraicReal`` from the norm ``m_u(y^d)`` without building ``K(c)``; the
    field is built only when the terms are asked for.
    """
    d = len(phi) - 1
    u = -phi[0] / phi[-1]
    gamma = Fraction(p, q)
    out = []
    for value in _binomial_roots(u, d):

        def make(value: AlgebraicReal = value) -> _Germ:
            for root in germ.field.real_roots(phi):
                if root.value.to_real() == value:
                    assert isinstance(root.field, _NumberField)
                    return germ.substitute(p, q, root.value, root.embedding)
            raise RuntimeError(
                "the binomial root is not a root of φ"
            )  # pragma: no cover

        out.append(_Branch(gamma, value, germ.kappa * q, None, make))
    return out


def _binomial_roots(u: _Element, d: int) -> list[AlgebraicReal]:
    """Return the real roots of ``c^d = u`` (``u != 0``), increasing, exact."""
    sign = u.sign()
    if d % 2 == 0 and sign < 0:
        return []
    y = sp.Symbol("y")
    m_u = u.to_real().polynomial
    norm = sp.Poly(m_u.as_expr().subs(m_u.gen, y**d), y, domain=sp.QQ)
    factors = [_normalised(f) for f, _ in norm.factor_list()[1]]
    bits = 8
    while True:  # enclose |u|^(1/d), then the root itself (sign(u) |u|^(1/d))
        lo_u, hi_u = u.enclose(Fraction(1, 2**bits))
        lo_u, hi_u = (lo_u, hi_u) if sign > 0 else (-hi_u, -lo_u)
        if lo_u > 0:
            lo, hi = _root_bounds(lo_u, d, bits)[0], _root_bounds(hi_u, d, bits)[1]
            if sign < 0:
                lo, hi = -hi, -lo
            hits = [(f, f.count_roots(_rational(lo), _rational(hi))) for f in factors]
            if sum(h for _, h in hits) == 1:
                break
        bits *= 2
    f = next(f for f, h in hits if h == 1)
    index = int(f.count_roots(None, _rational(hi))) - 1
    root = AlgebraicReal._make(f, index, _intervals(f)[index])
    if d % 2 == 1:
        return [root]
    return [_negate(root), root]


def _negate(x: AlgebraicReal) -> AlgebraicReal:
    """``-x`` (substitute ``y -> -y`` in the minimal polynomial)."""
    poly = x.polynomial
    y = poly.gen
    negated = _normalised(sp.Poly(poly.as_expr().subs(y, -y), y, domain=sp.QQ))
    roots = _intervals(negated)
    index = len(roots) - 1 - x.index
    return AlgebraicReal._make(negated, index, roots[index])


def _root_bounds(x: Fraction, d: int, bits: int) -> tuple[Fraction, Fraction]:
    """Rationals ``lo <= x^{1/d} <= hi`` with ``hi - lo = 2^-bits`` (``x > 0``)."""
    scale = 2**bits
    n = x.numerator * scale**d // x.denominator
    root, _ = sp.integer_nthroot(n, d)
    return Fraction(int(root), scale), Fraction(int(root) + 1, scale)


# -- which branch is w ---------------------------------------------------------------


def _order(x: _Branch, y: _Branch) -> int:
    """``sign(Y_x - Y_y)`` as ``s → 0⁺`` (newton.md, section 3.3)."""
    if x.gamma != y.gamma:
        return x.sign if x.gamma < y.gamma else -y.sign
    if x.leading != y.leading:
        return 1 if x.leading > y.leading else -1
    # The same first term: compare deeper terms (sub-branches of a multiple root),
    # merging the exponents of the two expansions (they may differ in κ). Distinct
    # branches differ at some term (G squarefree), so this terminates.
    xs, ys = x.expansion(), y.expansion()
    (ex, cx), (ey, cy) = next(xs), next(ys)
    while True:
        if ex == ey:
            if cx != cy:
                return 1 if cx > cy else -1
            (ex, cx), (ey, cy) = next(xs), next(ys)
        elif ex < ey:
            if not cx.is_zero():
                return cx.sign()
            ex, cx = next(xs)
        else:
            if not cy.is_zero():
                return -cy.sign()
            ey, cy = next(ys)


def _choose(
    branches: list[_Branch], window: _Window, g: sp.Poly, case: Case
) -> _Branch:
    """Return the branch of ``w``: the rank test, the sign rule as an assertion."""
    w, z = g.gens
    fiber = sp.Poly(g.as_expr().subs(z, _rational(window.x0)), w, domain=sp.QQ)
    left = None if window.left is None else _rational(window.left)
    right = None if window.right is None else _rational(window.right)
    n_w = int(fiber.count_roots(left, right))
    # [lower, upper] holds w(x0) and no other root: the roots in W up to upper,
    # minus w(x0) itself, are the roots below it.
    rank = int(fiber.count_roots(left, _rational(window.upper))) - 1
    return _select(branches, case, n_w, rank)


def _candidates(branches: list[_Branch], case: Case) -> list[_Branch]:
    """Sort the branches of the window by ``w``: all (I) or those with ``V > 0`` (II).

    Case I: ascending ``Y = w - a``; Case II: descending ``V = 1/w``.
    """
    if case == "II":
        positive = [b for b in branches if b.sign > 0]
        return sorted(positive, key=functools.cmp_to_key(_reversed_order))
    return sorted(branches, key=functools.cmp_to_key(_order))


def _reversed_order(x: _Branch, y: _Branch) -> int:
    return -_order(x, y)


def _select(branches: list[_Branch], case: Case, n_w: int, rank: int) -> _Branch:
    """Return the branch of rank ``rank`` among ``n_w`` (newton.md, section 3.3)."""
    candidates = _candidates(branches, case)
    if len(candidates) != n_w:  # pragma: no cover - Claims 1 and 2
        raise RuntimeError(
            f"{len(candidates)} real branches, {n_w} real sheets in the window"
        )
    chosen = candidates[rank]
    want = -1 if case == "I" else 1
    if chosen.sign != want:  # pragma: no cover - the sign rule, section 3.2
        raise RuntimeError(f"the branch of rank {rank} violates the sign rule")
    return chosen


# -- terms ---------------------------------------------------------------------------


class _State:
    """The chosen branch and the field data needed for :meth:`LocalExpansion.terms`."""

    def __init__(self, problem: _LocalProblem, branch: _Branch) -> None:
        self.problem = problem
        self.branch = branch
        self._inverse: list[_Element] = []

    def case_one_constant(self) -> tuple[Fraction, AlgebraicReal]:
        """Case I: ``α`` (the first non-integer exponent) and ``b``."""
        branch = self.branch
        if branch.gamma.denominator > 1:
            return branch.gamma, _negate(branch.leading)
        germ = branch.germ()  # pragma: no cover - not in our scope (PS Thm 8.5)
        for n in range(1, _SCAN):  # pragma: no cover
            d = germ.coefficient(n)
            if n % germ.kappa and not d.is_zero():
                return Fraction(n, germ.kappa), (-d).to_real()
        raise RuntimeError("no singular term found")  # pragma: no cover

    def terms(self, n: int) -> list[_Element]:
        germ = self.branch.germ()
        if self.problem.case == "I":
            a = germ.embed(self.problem.a_k)
            return [a] + [germ.coefficient(i) for i in range(1, n)]
        # Case II: w = 1/V, V = sum_{i >= n0} f_i t^i.
        n0 = germ.head[0][0]
        lead = germ.coefficient(n0)
        inv = self._inverse
        while len(inv) < n:
            k = len(inv)
            if k == 0:
                inv.append(germ.field(1))
                continue
            total = germ.field(0)
            for i in range(1, k + 1):
                total = total + germ.coefficient(n0 + i) / lead * inv[k - i]
            inv.append(-total)
        return [x / lead for x in inv[:n]]


__test__ = method_doctests(LocalExpansion)
