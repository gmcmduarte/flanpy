# SPDX-License-Identifier: GPL-3.0-or-later
"""Coefficient asymptotics from the local expansion (build-order step 6).

The design is recorded in ``dev/design/transfer.md``. :class:`GammaConstant`,
:class:`Asymptotics` and :func:`asymptotics` (a fixed ``k``) are public.

References: Flajolet and Sedgewick, *Analytic Combinatorics* (2009), "FS";
Broda, Machiavelo, Moreira and Reis (2020), "Broda"; Broda, Machiavelo, Moreira and
Reis (2014), *A Hitchhiker's Guide to descriptional complexity through analytic
combinatorics*, "HG"; NIST DLMF, eq. 5.11.8 (Stirling's series).
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, NoReturn

import mpmath
import sympy as sp

from flanpy._algebraic import AlgebraicReal, KRoot
from flanpy._doctests import method_doctests
from flanpy._field import _Element
from flanpy._kset import KSet
from flanpy._newton import LocalExpansion

__all__ = ["Asymptotics", "GammaConstant", "asymptotics"]

_Y = sp.Symbol("y")


# -- the polynomials e_l of FS Thm VI.1 ------------------------------------------------

_Poly = list[Fraction]  # coefficients, lowest degree first

_BERNOULLI: list[Fraction] = [Fraction(1)]
_E: list[_Poly] = [[Fraction(1)]]  # e_0 = 1
_S: list[_Poly] = [[]]  # s_0 is not used


def _bernoulli(n: int) -> Fraction:
    """Return the Bernoulli number ``B_n`` (``B_1 = -1/2``)."""
    while len(_BERNOULLI) <= n:
        m = len(_BERNOULLI)
        total = sum(
            (math.comb(m + 1, i) * _BERNOULLI[i] for i in range(m)), Fraction(0)
        )
        _BERNOULLI.append(-total / (m + 1))
    return _BERNOULLI[n]


def _bernoulli_polynomial(n: int) -> _Poly:
    """``B_n(a) = sum_k binom(n, k) B_{n-k} a^k``."""
    return [math.comb(n, k) * _bernoulli(n - k) for k in range(n + 1)]


def _add(p: _Poly, q: _Poly) -> _Poly:
    out = [Fraction(0)] * max(len(p), len(q))
    for i, c in enumerate(p):
        out[i] += c
    for i, c in enumerate(q):
        out[i] += c
    return out


def _multiply(p: _Poly, q: _Poly) -> _Poly:
    out = [Fraction(0)] * max(len(p) + len(q) - 1, 0)
    for i, c in enumerate(p):
        for j, d in enumerate(q):
            out[i + j] += c * d
    return out


def _e_poly(ell: int) -> tuple[Fraction, ...]:
    """Return the polynomial ``e_l(a)`` of FS Thm VI.1, lowest degree first.

    ``Γ(n + a)/Γ(n + 1) = n^{a-1} (1 + Σ_{l>=1} e_l(a) n^{-l})``. By Stirling's
    series (DLMF 5.11.8) the logarithm of the left side is
    ``(a - 1) log n + Σ_j s_j n^{-j}`` with
    ``s_j = (-1)^{j+1} (B_{j+1}(a) - B_{j+1}(1)) / (j (j + 1))``; exponentiating,
    ``e_0 = 1`` and ``l e_l = Σ_{j=1}^{l} j s_j e_{l-j}`` (from ``E' = S' E``).
    Cached.

    Examples
    --------
    FS eq. (13): ``e_1 = a(a - 1)/2``.

    >>> from flanpy._transfer import _e_poly
    >>> [str(c) for c in _e_poly(1)]
    ['0', '-1/2', '1/2']
    """
    while len(_E) <= ell:
        m = len(_E)
        b = _bernoulli_polynomial(m + 1)
        b[0] -= sum(b, Fraction(0))  # B_{m+1}(a) - B_{m+1}(1)
        _S.append([(-1) ** (m + 1) * c / (m * (m + 1)) for c in b])
        total: _Poly = []
        for j in range(1, m + 1):
            total = _add(total, [j * c for c in _multiply(_S[j], _E[m - j])])
        _E.append([c / m for c in total])
    poly = _E[ell]
    while poly and poly[-1] == 0:
        poly = poly[:-1]
    return tuple(poly)


def _e_value(ell: int, a: Fraction) -> Fraction:
    """Return ``e_ell(a)`` (Horner)."""
    value = Fraction(0)
    for c in reversed(_e_poly(ell)):
        value = value * a + c
    return value


def _inverse_gamma(x: Fraction) -> tuple[Fraction, Fraction | None]:
    """``1/Γ(x) = r / Γ(ν)`` with ``r`` rational and ``ν = x - ⌊x⌋ ∈ (0, 1)``.

    For an integer ``x`` there is no ``Γ``: ``ν`` is ``None`` and ``r = 1/Γ(x)``,
    which is ``0`` for ``x <= 0`` (FS Thm VI.1, footnote 2) and ``1/(x - 1)!``
    otherwise. The reduction is the recurrence ``Γ(y + 1) = y Γ(y)``.

    Examples
    --------
    >>> from fractions import Fraction
    >>> from flanpy._transfer import _inverse_gamma
    >>> _inverse_gamma(Fraction(-1, 2))  # Γ(-1/2) = -2√π
    (Fraction(-1, 2), Fraction(1, 2))
    >>> _inverse_gamma(Fraction(3))
    (Fraction(1, 2), None)
    """
    if x.denominator == 1:
        if x <= 0:
            return Fraction(0), None
        return Fraction(1, math.factorial(int(x) - 1)), None
    nu = x - math.floor(x)
    r, y = Fraction(1), x
    while y > nu:  # 1/Γ(y) = 1/((y - 1) Γ(y - 1))
        y -= 1
        r /= y
    while y < nu:  # 1/Γ(y) = y/Γ(y + 1)
        r *= y
        y += 1
    return r, nu


# -- exact constants -------------------------------------------------------------------


def _radical(x: AlgebraicReal) -> sp.Expr:
    """``x`` as a SymPy expression: radicals up to degree 2, ``CRootOf`` above."""
    p = x.polynomial
    if p.degree() == 1:
        c1, c0 = p.all_coeffs()
        return sp.Rational(-c0, c1)
    if p.degree() == 2:
        a, b, c = p.all_coeffs()  # a > 0 (normalised); index 1 is the larger root
        root = sp.sqrt(b * b - 4 * a * c)
        return (-b + (root if x.index == 1 else -root)) / (2 * a)
    return x.as_expr()


_ZERO = AlgebraicReal(sp.Poly(_Y, _Y), 0)


class GammaConstant[X: (AlgebraicReal, KRoot)]:
    """An exact constant: an algebraic number times a product of ``Γ(ν)^m``.

    Returned by :class:`Asymptotics` (:attr:`~Asymptotics.leading`,
    :meth:`~Asymptotics.terms`); not built directly. Its value is
    ``A · Π Γ(ν)^{m_ν}`` with ``A`` an :class:`~flanpy.AlgebraicReal`
    (:meth:`algebraic`) and ``0 < ν < 1`` rational (:attr:`gamma`). The ``Γ``
    part is kept as FS and Broda et al. print it: the coefficient of
    ``[z^n] (1 - z/ρ)^x`` is ``1/Γ(-x)``, reduced to ``1/Γ(ν)`` with
    ``ν = -x mod 1`` by ``Γ(y + 1) = y Γ(y)`` (transfer.md, section 3.1).

    Notes
    -----
    **What "exact" means** (transfer.md, section 3.2). With ``κ = 2`` the only
    base is ``Γ(1/2) = √π``, which is transcendental: two constants are equal
    exactly when their algebraic parts and their ``Γ`` parts are. The constants
    of one :class:`Asymptotics` have a single base each, ``1/Γ(ν)``, and then the
    same holds for every ``κ`` in our scope; equality and hashing compare
    ``(A, Γ part)``, with ``0`` having no ``Γ`` part. Products of constants
    (ratios of two expansions) normalise ``Γ(ν)`` with ``ν > 1/2`` by reflection,
    ``Γ(ν)Γ(1 - ν) = π/sin(πν)``, for the comparison only.

    A ``GammaConstant`` is immutable, hashable and can be pickled.

    Examples
    --------
    Catalan: ``[z^n] F ~ 4^n n^{-3/2}/√π``, so the leading constant is
    ``1/Γ(1/2)``.

    >>> from sympy import symbols
    >>> from flanpy import System, asymptotics, eliminate, local_expansion
    >>> from flanpy import singularity
    >>> z, F = symbols("z F")
    >>> catalan = System({F: 1 + z * F**2}, z=z)
    >>> c = asymptotics(local_expansion(singularity(eliminate(catalan, F)))).leading
    >>> c
    GammaConstant(AlgebraicReal(y - 1, 0), gamma=((Fraction(1, 2), -1),))
    >>> print(c)
    1/sqrt(pi)
    """

    __slots__ = ("_algebraic", "_gamma")

    _algebraic: AlgebraicReal
    _gamma: tuple[tuple[Fraction, int], ...]

    def __init__(self) -> None:
        raise TypeError("GammaConstant objects are created by flanpy.asymptotics")

    @classmethod
    def _make(
        cls, algebraic: AlgebraicReal, gamma: tuple[tuple[Fraction, int], ...]
    ) -> GammaConstant[Any]:
        new = object.__new__(cls)
        if _is_zero(algebraic):
            gamma = ()
        gamma = tuple(sorted((nu, m) for nu, m in gamma if m != 0))
        object.__setattr__(new, "_algebraic", algebraic)
        object.__setattr__(new, "_gamma", gamma)
        return new

    # -- public interface ----------------------------------------------------------

    def algebraic(self) -> AlgebraicReal:
        """Return the algebraic part ``A`` (the ``Γ`` part excluded).

        Returns
        -------
        AlgebraicReal
            ``A``, exactly.

        Examples
        --------
        Motzkin (FS Example VII.18): ``[z^n] F ~ (√3/2) 3^n n^{-3/2}/√π``.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> motzkin = System({F: z + z * F + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(motzkin, F)))
        >>> asymptotics(e).leading.algebraic()
        AlgebraicReal(4*y**2 - 3, 1)
        """
        return self._algebraic

    @property
    def gamma(self) -> tuple[tuple[Fraction, int], ...]:
        """The ``Γ`` part: pairs ``(ν, m)`` for ``Π Γ(ν)^m``, ``0 < ν < 1``.

        Empty for an algebraic constant (and for ``0``).

        Examples
        --------
        ``K = U/2 + K^2``, ``U = z + U^2``, is ``K = (1 - s^{1/4})/2`` with
        ``s = 1 - 4z``: its leading constant is ``1/(8 Γ(3/4))``.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, U, K = symbols("z U K")
        >>> system = System({U: z + U**2, K: U / 2 + K**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(system, K))))
        >>> a.leading.gamma
        ((Fraction(3, 4), -1),)
        """
        return self._gamma

    def evalf(self, dps: int = 15) -> mpmath.mpf:
        """Return the value, ``Γ`` part included, as an :class:`mpmath.mpf`.

        Parameters
        ----------
        dps : int, default 15
            Decimal digits of relative accuracy.

        Returns
        -------
        mpmath.mpf
            The value.

        Examples
        --------
        >>> import mpmath
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
        >>> mpmath.nstr(a.leading.evalf(30), 30)  # 1/√π
        '0.564189583547756286948079451561'
        """
        if isinstance(dps, bool) or not isinstance(dps, int) or dps < 1:
            raise ValueError(f"dps must be a positive int, got {dps!r}")
        with mpmath.workdps(dps + 10):
            value = self._algebraic.evalf(dps + 10)
            for nu, m in self._gamma:
                value *= mpmath.gamma(mpmath.mpf(nu.numerator) / nu.denominator) ** m
        with mpmath.workdps(dps):
            return +value

    def as_expr(self) -> sp.Expr:
        """Return the value as a SymPy expression.

        The algebraic part is written with radicals when its degree is at most 2,
        as a :class:`sympy.CRootOf` otherwise; the ``Γ`` part with
        :class:`sympy.gamma`, which SymPy writes as ``sqrt(pi)`` for ``ν = 1/2``.

        Returns
        -------
        sympy.Expr
            The constant.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> motzkin = System({F: z + z * F + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(motzkin, F)))
        >>> asymptotics(e).leading.as_expr()
        sqrt(3)/(2*sqrt(pi))
        """
        expr = _radical(self._algebraic)
        for nu, m in self._gamma:
            expr *= sp.gamma(sp.Rational(nu.numerator, nu.denominator)) ** m
        return expr

    def _is_zero(self) -> bool:
        return _is_zero(self._algebraic)

    # -- value semantics -----------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (self._algebraic, self._gamma)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GammaConstant):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"GammaConstant({self._algebraic!r}, gamma={self._gamma!r})"

    def __str__(self) -> str:
        return str(self.as_expr())

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore_constant, (self._algebraic, self._gamma))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("GammaConstant is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("GammaConstant is immutable")


def _is_zero(x: AlgebraicReal) -> bool:
    return bool(x.polynomial.degree() == 1 and x.polynomial.TC() == 0)


def _restore_constant(
    algebraic: AlgebraicReal, gamma: tuple[tuple[Fraction, int], ...]
) -> GammaConstant[Any]:
    """Rebuild a pickled :class:`GammaConstant`."""
    return GammaConstant._make(algebraic, gamma)


GammaConstant.__module__ = "flanpy"


# -- the expansion in n ----------------------------------------------------------------


def _first_singular(e: LocalExpansion[Any]) -> Fraction:
    """``e_0``: ``α`` in Case I, ``-α`` in Case II (transfer.md, section 2.2)."""
    return e.alpha if e.singularity.case == "I" else -e.alpha


def _coefficients(
    e: LocalExpansion[AlgebraicReal], count: int
) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]:
    """Return the first ``count`` pairs ``(power of n, D_j)`` (transfer.md, 2.2).

    ``[z^n] w = ρ^{-n} Σ_j D_j n^{-1-p_j}``, ``p_j = e_0 + j/κ``, where
    ``D_j = Σ d_i e_l(-x_i)/Γ(-x_i)`` over the Puiseux terms ``d_i s^{x_i}`` with
    ``l = p_j - x_i`` a non-negative integer (FS Thm VI.1, term by term). Every
    ``x_i`` of one ``D_j`` lies in one class mod 1, so ``D_j = A_j/Γ(ν_j)`` with
    ``A_j`` in the branch's field ``L``, computed exactly there. The Puiseux terms
    needed are those with ``x <= p_{count-1}`` (section 2.3).
    """
    if count == 0:
        return []
    case_one = e.singularity.case == "I"
    kappa, e0 = e.ramification, _first_singular(e)
    start = Fraction(0) if case_one else -e.alpha
    needed = int((e0 - start) * kappa) + count
    puiseux = [
        (start + Fraction(i, kappa), d)
        for i, d in enumerate(e._field_terms(needed)[:needed])
    ]
    out = []
    for j in range(count):
        p = e0 + Fraction(j, kappa)
        nu = None if p.denominator == 1 else -p - math.floor(-p)
        total: _Element | None = None
        for x, d in puiseux:
            ell = p - x
            if ell < 0:
                break
            if ell.denominator != 1 or d.is_zero():
                continue
            r, base = _inverse_gamma(-x)
            if r == 0:  # x a non-negative integer: an analytic term
                continue
            assert base == nu
            term = d * (r * _e_value(int(ell), -x))
            total = term if total is None else total + term
        if total is None or total.is_zero():
            out.append((-1 - p, GammaConstant._make(_ZERO, ())))
        else:
            gamma = () if nu is None else ((nu, -1),)
            out.append((-1 - p, GammaConstant._make(total.to_real(), gamma)))
    return out


# -- the public type -------------------------------------------------------------------


class Asymptotics[X: (AlgebraicReal, KRoot)]:
    """The asymptotic expansion of the coefficients ``[z^n] w``.

    Returned by :func:`asymptotics`; not built directly. In the notation of FS
    Thm VII.8 (eq. (82)),

    ``[z^n] w = ρ^{-n} (Σ_{j < count} D_j n^{-1-p_j} + O(n^{e}))``,
    ``p_j = e_0 + j/κ``,

    with ``e_0 = α`` (Case I) or ``-α`` (Case II), ``κ`` the ramification, the
    exact constants ``D_j`` (:meth:`terms`, :class:`GammaConstant`) and the
    exponent ``e`` of the ``O``-term (:meth:`error_exponent`). The leading term
    is ``C ρ^{-n} n^β`` with ``β`` = :attr:`exponent` and ``C`` = :attr:`leading`:
    Broda et al. (2020) Thm 3.2, ``-b/Γ(-α) = αb/Γ(1 - α)`` (Case I) and
    ``1/(c Γ(α))`` (Case II).

    Notes
    -----
    Equality and hashing compare the local expansion (everything else is
    computed from it). An ``Asymptotics`` is immutable and can be pickled (the
    cached terms are recomputed after unpickling).

    Examples
    --------
    Catalan, ``F = 1 + zF^2``: ``[z^n] F ~ 4^n n^{-3/2}/√π`` (FS Thm VI.4).

    >>> from sympy import symbols
    >>> from flanpy import System, asymptotics, eliminate, local_expansion
    >>> from flanpy import singularity
    >>> z, F = symbols("z F")
    >>> catalan = System({F: 1 + z * F**2}, z=z)
    >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
    >>> a  # doctest: +NORMALIZE_WHITESPACE
    Asymptotics(exponent=Fraction(-3, 2),
                leading=GammaConstant(AlgebraicReal(y - 1, 0),
                                      gamma=((Fraction(1, 2), -1),)))
    >>> print(a)
    4**n/(sqrt(pi)*n**(3/2)) (1 + O(n^-1))
    """

    __slots__ = ("_expansion", "_leading", "_terms")

    _expansion: LocalExpansion[X]
    _leading: GammaConstant[X]
    _terms: list[tuple[Fraction, GammaConstant[AlgebraicReal]]]

    def __init__(self) -> None:
        raise TypeError("Asymptotics objects are created by flanpy.asymptotics")

    @classmethod
    def _make(cls, expansion: LocalExpansion[AlgebraicReal]) -> Asymptotics[Any]:
        new = object.__new__(cls)
        terms = _coefficients(expansion, 1)
        object.__setattr__(new, "_expansion", expansion)
        object.__setattr__(new, "_leading", terms[0][1])
        object.__setattr__(new, "_terms", terms)
        return new

    # -- public interface ----------------------------------------------------------

    @property
    def expansion(self) -> LocalExpansion[X]:
        """The local expansion at ``ρ`` the coefficients come from (step 5).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(catalan, F)))
        >>> asymptotics(e).expansion is e
        True
        """
        return self._expansion

    @property
    def rho(self) -> X:
        """``ρ``, the dominant singularity: the growth is ``ρ^{-n}``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> motzkin = System({F: z + z * F + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(motzkin, F)))
        >>> asymptotics(e).rho
        AlgebraicReal(3*z - 1, 0)
        """
        rho: X = self._expansion.singularity.rho
        return rho

    @property
    def exponent(self) -> Fraction:
        """``β``, the power of ``n`` in the leading term ``C ρ^{-n} n^β``.

        ``-α - 1`` in Case I, ``α - 1`` in Case II (Broda et al. 2020, Thm 3.2).

        Examples
        --------
        ``1/(1 - 4z)^{3/2}`` (Case II, ``α = 3/2``) has ``[z^n] ~ 2 · 4^n √n/√π``:

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, B, Y, W = symbols("z B Y W")
        >>> system = System({B: z + B**2, Y: 1 + 2 * B * Y, W: Y**3}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(system, W))))
        >>> a.exponent, a.leading.as_expr()
        (Fraction(1, 2), 2/sqrt(pi))
        """
        return -1 - _first_singular(self._expansion)

    @property
    def leading(self) -> GammaConstant[X]:
        """``C``, the constant of the leading term ``C ρ^{-n} n^β``.

        Case I: ``α b/Γ(1 - α)``; Case II: ``1/(c Γ(α))`` (Broda et al. 2020,
        Thm 3.2, with ``Γ(-α) = -Γ(1 - α)/α``). For ``α = 1/2`` these are HG
        Prop. 4's ``b/(2√π)`` and ``1/(c√π)``.

        Examples
        --------
        Broda et al. (2020) ``R_k`` at ``k = 2``: ``√(2 - 2ρ)/(8ρ√π)`` (eq. (6)).

        >>> from sympy import pi, sqrt, symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> a = asymptotics(local_expansion(singularity(eliminate(rk, R, k=2))))
        >>> a.leading.as_expr()
        CRootOf(128*y**4 - 192*y**2 + 69, 3)/sqrt(pi)
        >>> rho = a.rho.as_expr()
        >>> closed = sqrt(2 - 2 * rho) / (8 * rho * sqrt(pi))
        >>> abs((a.leading.as_expr() - closed).evalf(30)) < 1e-25
        True
        """
        return self._leading

    @property
    def k(self) -> Any:
        """The parameter: ``None`` at a fixed ``k`` (forwarded from the expansion).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(catalan, F)))
        >>> asymptotics(e).k is None
        True
        """
        return self._expansion.k

    @property
    def kset(self) -> KSet:
        """The ``k`` this result is for (forwarded from the expansion).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> asymptotics(local_expansion(singularity(eliminate(rk, R, k=2)))).kset
        KSet([(2, 2)])
        """
        return self._expansion.kset

    def terms(self, count: int) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]:
        """Return the first ``count`` pairs ``(power of n, D_j)``, exactly.

        The powers are ``-1 - p_j``, ``p_j = e_0 + j/κ``, consecutive in steps of
        ``1/κ``; zero coefficients are included (``D_j = 0`` whenever ``p_j`` is a
        non-negative integer: the analytic part transfers to nothing). Cached.

        Parameters
        ----------
        count : int
            How many coefficients, ``count >= 0`` (zeros included).

        Returns
        -------
        list of (Fraction, GammaConstant)
            ``[z^n] w = ρ^{-n} Σ D_j n^{power}``.

        Raises
        ------
        TypeError
            If ``count`` is not an ``int``.
        ValueError
            If ``count < 0``.

        Examples
        --------
        Catalan: ``[z^n] F = 4^n/√π (n^{-3/2} - 9/8 n^{-5/2} + 145/128 n^{-7/2}
        - ...)``.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(catalan, F)))
        >>> [(str(p), str(d)) for p, d in asymptotics(e).terms(5)]
        ... # doctest: +NORMALIZE_WHITESPACE
        [('-3/2', '1/sqrt(pi)'), ('-2', '0'), ('-5/2', '-9/(8*sqrt(pi))'),
         ('-3', '0'), ('-7/2', '145/(128*sqrt(pi))')]
        """
        _check_count(count)
        if len(self._terms) < count:
            object.__setattr__(self, "_terms", _coefficients(self._fixed(), count))
        return self._terms[:count]

    def error_exponent(self, count: int) -> Fraction | None:
        """Return the exponent ``e`` of the ``O``-term after ``count`` coefficients.

        ``[z^n] w - ρ^{-n} Σ_{j < count} D_j n^{-1-p_j} = O(ρ^{-n} n^e)`` (FS
        Thm VI.4 with ``τ = s^{p_count}``): ``e = -1 - p_count``, moved one step
        on when ``p_count`` is a non-negative integer (that coefficient is zero).
        ``None`` when every remaining coefficient is zero (``κ = 1``: ``w`` is
        meromorphic at ``ρ``); the remainder is then ``O((ρ + ε)^{-n})``. A
        coefficient that is zero by accident does not sharpen the exponent.

        Parameters
        ----------
        count : int
            How many coefficients are kept, ``count >= 0``.

        Returns
        -------
        Fraction or None
            ``e``, or ``None`` (an exponentially smaller remainder).

        Examples
        --------
        Catalan: after ``n^{-3/2}`` the next non-zero power is ``n^{-5/2}``.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F, Y = symbols("z F Y")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> e = local_expansion(singularity(eliminate(catalan, F)))
        >>> asymptotics(e).error_exponent(1)
        Fraction(-5, 2)

        ``1/(1 - 2z)``: ``[z^n] = 2^n`` exactly, up to nothing.

        >>> pole = System({Y: 1 + 2 * z * Y}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(pole, Y))))
        >>> a.error_exponent(1) is None
        True
        """
        _check_count(count)
        kappa = self._expansion.ramification
        p = _first_singular(self._expansion) + Fraction(count, kappa)
        while p.denominator == 1 and p >= 0:
            if kappa == 1:
                return None
            p += Fraction(1, kappa)
        return -1 - p

    def evalf(self, n: int, *, count: int = 1, dps: int = 15) -> mpmath.mpf:
        """Return ``ρ^{-n} Σ_{j < count} D_j n^{-1-p_j}`` as an :class:`mpmath.mpf`.

        The expansion is asymptotic, not convergent: it approximates ``[z^n] w``
        for large ``n``, with the relative error of :meth:`error_exponent`; for
        small ``n`` more terms can be worse. ``ρ`` is evaluated with
        ``dps + ⌈log10 n⌉ + 10`` digits, since a relative error ``δ`` in ``ρ``
        becomes ``n δ`` in ``ρ^{-n}``.

        Parameters
        ----------
        n : int
            The index, ``n >= 1``.
        count : int, default 1
            How many coefficients ``D_j`` (zeros included).
        dps : int, default 15
            Decimal digits of the evaluation.

        Returns
        -------
        mpmath.mpf
            The truncated expansion at ``n``.

        Raises
        ------
        TypeError
            If ``n`` or ``count`` is not an ``int``.
        ValueError
            If ``n < 1``, ``count < 0`` or ``dps < 1``.

        Examples
        --------
        Catalan at ``n = 1000``, against the exact coefficient:

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import series, singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
        >>> exact = series(catalan, 1000)[F][1000]
        >>> [f"{float(abs(a.evalf(1000, count=c) / exact - 1)):.1e}" for c in (1, 3, 5)]
        ['1.1e-03', '1.1e-06', '1.1e-09']
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {n!r}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if isinstance(dps, bool) or not isinstance(dps, int) or dps < 1:
            raise ValueError(f"dps must be a positive int, got {dps!r}")
        work = dps + 10
        with mpmath.workdps(work):
            total = mpmath.mpf(0)
            for power, d in self.terms(count):
                if not d._is_zero():
                    exponent = mpmath.mpf(power.numerator) / power.denominator
                    total += d.evalf(work) * mpmath.mpf(n) ** exponent
        digits = work + len(str(n))
        with mpmath.workdps(digits):
            scale = self._fixed().singularity.rho.evalf(digits) ** (-n)
        with mpmath.workdps(work):
            value = total * scale
        with mpmath.workdps(dps):
            return +value

    def as_expr(self, n: sp.Expr, *, count: int = 1) -> sp.Expr:
        """Return ``ρ^{-n} Σ_{j < count} D_j n^{-1-p_j}`` as a SymPy expression.

        ``ρ`` and the algebraic parts are written with radicals up to degree 2
        (:meth:`GammaConstant.as_expr`).

        Parameters
        ----------
        n : sympy.Expr
            The index, usually a symbol.
        count : int, default 1
            How many coefficients ``D_j`` (zeros included).

        Returns
        -------
        sympy.Expr
            The truncated expansion, without its ``O``-term.

        Raises
        ------
        TypeError
            If ``n`` is not a SymPy expression or ``count`` not an ``int``.
        ValueError
            If ``count < 0``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion
        >>> from flanpy import singularity
        >>> z, F, n = symbols("z F n")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
        >>> a.as_expr(n)
        4**n/(sqrt(pi)*n**(3/2))
        >>> a.as_expr(n, count=3)
        4**n*(1/(sqrt(pi)*n**(3/2)) - 9/(8*sqrt(pi)*n**(5/2)))
        """
        if not isinstance(n, sp.Expr):
            raise TypeError(f"n must be a SymPy expression, got {n!r}")
        total = sp.Add(
            *(
                d.as_expr() * n ** sp.Rational(p.numerator, p.denominator)
                for p, d in self.terms(count)
                if not d._is_zero()
            )
        )
        growth = sp.radsimp(1 / _radical(self._fixed().singularity.rho))
        return growth**n * total

    def _fixed(self) -> LocalExpansion[AlgebraicReal]:
        return self._expansion  # type: ignore[return-value]  # fixed k (chunk 2)

    # -- value semantics -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Asymptotics):
            return NotImplemented
        return bool(self._expansion == other._expansion)

    def __hash__(self) -> int:
        return hash(("Asymptotics", self._expansion))

    def __repr__(self) -> str:
        return f"Asymptotics(exponent={self.exponent!r}, leading={self._leading!r})"

    def __str__(self) -> str:
        lead = self.as_expr(sp.Symbol("n"))
        error = self.error_exponent(1)
        if error is None:
            return f"{lead} + O((ρ + ε)^-n)"
        return f"{lead} (1 + O(n^{error - self.exponent}))"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore, (self._expansion,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Asymptotics is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Asymptotics is immutable")


def _restore(expansion: LocalExpansion[AlgebraicReal]) -> Asymptotics[Any]:
    """Rebuild a pickled :class:`Asymptotics` (terms recomputed on demand)."""
    return Asymptotics._make(expansion)


def _check_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError(f"count must be an int, got {count!r}")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")


Asymptotics.__module__ = "flanpy"


# -- the public function ---------------------------------------------------------------


def asymptotics(
    expansion: LocalExpansion[Any], *, k: int | None = None
) -> Asymptotics[AlgebraicReal]:
    """Return the asymptotic expansion of ``[z^n] w``, fixed ``k``.

    Singularity analysis (FS Thm VI.4) of the local expansion of step 5: every
    Puiseux term ``d s^x``, ``s = 1 - z/ρ``, ``x`` not a non-negative integer,
    transfers to ``ρ^{-n} n^{-x-1}/Γ(-x) (1 + Σ_l e_l(-x) n^{-l})`` (FS Thm VI.1),
    and the terms are collected by powers of ``n`` (FS Thm VII.8, eq. (82)). The
    leading term is Broda et al. (2020) Thm 3.2, and HG Prop. 4 for ``α = 1/2``.

    Parameters
    ----------
    expansion : LocalExpansion
        From :func:`~flanpy.local_expansion` (a fixed ``k``), or a piece of
        :func:`~flanpy.local_expansion_in_k` together with ``k``.
    k : int, optional
        Required for an expansion of symbolic ``k`` (it is specialized first);
        must be omitted otherwise.

    Returns
    -------
    Asymptotics
        The exponent, the exact leading constant, the further terms, the
        ``O``-term and a numeric evaluation.

    Raises
    ------
    TypeError
        If ``expansion`` is not a :class:`~flanpy.LocalExpansion`.
    ValueError
        If ``k`` is missing or extra, or not in the expansion's ``kset``; or if
        ``ρ`` is not the only dominant singularity (``unique`` is ``False``: FS
        Thm VI.5 would be needed) or that is undecided (``unique`` is ``None``).

    See Also
    --------
    Asymptotics : the fields of the result.

    Notes
    -----
    **Hypotheses** (transfer.md, section 1.2). FS Thm VI.4 needs ``w`` analytic in
    a ``Δ``-domain at ``ρ``. ``w`` is algebraic, so its continuation can be
    singular only at finitely many points (FS Lemma VII.4); ``unique`` (step 4)
    says that no point of ``|z| = ρ`` other than ``ρ`` is one, so ``w`` extends
    across the rest of the circle (compactness), and near ``ρ`` the Puiseux
    expansion (FS Thm VII.7) converges in a slit disc. Each Puiseux term lies in
    FS's scale ``S`` and the remainder after finitely many terms is ``O(s^x)``.

    **The polynomials** ``e_l`` (FS Thm VI.1, eq. (13)) are computed exactly from
    Stirling's series, ``log Γ(n + a) - log Γ(n + 1) = (a - 1) log n +
    Σ_j s_j n^{-j}``, ``s_j = (-1)^{j+1}(B_{j+1}(a) - B_{j+1}(1))/(j(j + 1))``
    (DLMF 5.11.8, Bernoulli polynomials), by ``l e_l = Σ_j j s_j e_{l-j}``.

    **The coefficients.** ``D_j`` collects the terms with ``x ≡ p_j`` mod 1, so it
    is ``A_j/Γ(ν_j)`` with one ``ν_j ∈ (0, 1)`` (or no ``Γ``, when ``p_j`` is an
    integer); ``A_j`` is computed in the number field of the branch, exactly.
    When ``p_j`` is a non-negative integer, ``D_j = 0``: the integer exponents
    ``x >= 0`` have ``1/Γ(-x) = 0``, and a pole ``s^{-m}`` gives the polynomial
    ``binom(n + m - 1, m - 1)``, whose expansion stops at ``n^0`` (``e_l(m) = 0``
    for ``l >= m``, FS Thm VI.1, footnote 2). ``count`` coefficients need the
    Puiseux terms with ``x <= p_{count-1}``: ``count + 1`` of them in Case I,
    ``count`` in Case II.

    Examples
    --------
    Catalan, ``F = 1 + zF^2``: ``[z^n] F ~ 4^n n^{-3/2}/√π (1 - 9/(8n) + ...)``.

    >>> from sympy import symbols
    >>> from flanpy import System, asymptotics, eliminate, local_expansion
    >>> from flanpy import singularity
    >>> z, F, n = symbols("z F n")
    >>> catalan = System({F: 1 + z * F**2}, z=z)
    >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
    >>> a.as_expr(n)
    4**n/(sqrt(pi)*n**(3/2))

    A periodic series has several dominant singularities and is refused:

    >>> T = symbols("T")
    >>> trees = System({T: z + z * T**2}, z=z)  # odd powers only
    >>> e = local_expansion(singularity(eliminate(trees, T)))
    >>> asymptotics(e)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: w has 2 dominant singularities on |z| = ρ (period 2): ...

    An expansion of symbolic ``k`` needs ``k``:

    >>> from flanpy import eliminate_in_k, local_expansion_in_k, singularity_in_k
    >>> k, R = symbols("k R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> piece = local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))[2]
    >>> asymptotics(piece, k=2) == asymptotics(piece.specialize(2))
    True
    """
    if not isinstance(expansion, LocalExpansion):
        raise TypeError(
            f"expected a flanpy.LocalExpansion, got {type(expansion).__name__}"
        )
    if expansion.k is not None:
        if k is None:
            raise ValueError(
                "the expansion has the parameter k: give k=n for the coefficients "
                "at k = n"
            )
        expansion = expansion.specialize(k)
    elif k is not None:
        raise ValueError("the expansion has no parameter k; omit the k argument")
    singularity = expansion.singularity
    if singularity.unique is False:
        p = singularity.period
        if p == 1:  # pragma: no cover - step 4 proves False only through a period
            raise ValueError("ρ is not the only dominant singularity on |z| = ρ")
        raise ValueError(
            f"w has {p} dominant singularities on |z| = ρ (period {p}): the transfer "
            "needs FS Thm VI.5, which is not implemented"
        )
    if singularity.unique is None:
        raise ValueError(
            "whether ρ is the only dominant singularity on |z| = ρ is undecided "
            "(step 4): the transfer (FS Thm VI.4) needs it, and nothing is guessed"
        )
    return Asymptotics._make(expansion)


__test__ = method_doctests(GammaConstant, Asymptotics)
