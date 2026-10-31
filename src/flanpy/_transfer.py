# SPDX-License-Identifier: GPL-3.0-or-later
"""Coefficient asymptotics from the local expansion (build-order step 6).

The design is recorded in ``dev/design/transfer.md``. :class:`GammaConstant`,
:class:`Asymptotics`, :func:`asymptotics` (a fixed ``k``), :func:`asymptotics_in_k`
(symbolic ``k``), :class:`Ratio` and :func:`ratio` are public.

References: Flajolet and Sedgewick, *Analytic Combinatorics* (2009), "FS";
Broda, Machiavelo, Moreira and Reis (2020), "Broda"; Broda, Machiavelo, Moreira and
Reis (2014), *A Hitchhiker's Guide to descriptional complexity through analytic
combinatorics*, "HG"; NIST DLMF, eq. 5.11.8 (Stirling's series).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import Any, NoReturn

import mpmath
import sympy as sp

from flanpy._algebraic import AlgebraicReal, KRoot
from flanpy._doctests import method_doctests
from flanpy._field import _Element, _NumberField
from flanpy._kset import KCases, KSet
from flanpy._newton import LocalExpansion

__all__ = [
    "Asymptotics",
    "GammaConstant",
    "Ratio",
    "asymptotics",
    "asymptotics_in_k",
    "ratio",
]

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


# -- exact arithmetic on real algebraic numbers ----------------------------------------


def _rational_real(q: Fraction) -> AlgebraicReal:
    """``q`` as an :class:`AlgebraicReal`."""
    return AlgebraicReal(sp.Poly(q.denominator * _Y - q.numerator, _Y), 0)


def _as_rational(x: AlgebraicReal) -> Fraction | None:
    """``x`` as a ``Fraction`` when it is rational, else ``None``."""
    p = x.polynomial
    if p.degree() != 1:
        return None
    c1, c0 = p.all_coeffs()
    return Fraction(-int(c0), int(c1))


def _common_field(
    k1: _NumberField, k2: _NumberField
) -> tuple[Any, Callable[[_Element], Any], Callable[[_Element], Any]]:
    """One real number field ``L ⊇ K1, K2`` and the two embeddings.

    ``L`` comes from the real roots of ``m_θ2`` over ``K1`` (step 5's
    :meth:`~flanpy._field._NumberField.real_roots`, Trager norms): the root equal to
    ``θ2`` (exact comparison) gives ``L = K1(θ2)``, flattened to one generator.
    """
    if k1 == k2:
        return k1, lambda x: x, lambda y: y
    if k2.degree == 1:
        return k1, lambda x: x, lambda y: k1(y.coefficients[0])
    if k1.degree == 1:
        return k2, lambda x: k2(x.coefficients[0]), lambda y: y
    for root in k1.real_roots(list(k2.modulus)):
        if root.value.to_real() == k2.theta:
            return root.field, root.embedding, _embedding(root.field, root.value)
    raise RuntimeError("θ2 is not a root of its minimal polynomial")  # pragma: no cover


def _embedding(field: Any, theta: Any) -> Callable[[_Element], Any]:
    """Return the map ``Σ c_i θ2^i ↦ Σ c_i theta^i`` into ``field``."""

    def embed(y: _Element) -> Any:
        total, power = field(0), field(1)
        for c in y.coefficients:
            total = total + power * c
            power = power * theta
        return total

    return embed


def _monomial(q: Fraction, factors: list[tuple[Any, int]]) -> AlgebraicReal:
    """``q · Π x^m``, exactly (in one number field for all the ``x``)."""
    value: Any = _NumberField.rationals()(q)
    for x, m in factors:
        r = _as_rational(x)
        if r is not None:
            value = value * r**m
            continue
        field = _NumberField(x)
        _, embed_value, embed_x = _common_field(value.field, field)
        g = embed_x(field.generator)
        value = embed_value(value) * (g**m if m > 0 else g.inverse() ** -m)
    real: AlgebraicReal = value.to_real()
    return real


@functools.cache
def _sin_pi(nu: Fraction) -> AlgebraicReal:
    """``sin(πν)`` for rational ``ν``, as an exact :class:`AlgebraicReal`.

    ``sin(πν) = cos(2π a/N)`` with ``a/N = |1 - 2ν|/4`` in lowest terms. Its
    conjugates are ``cos(2π a'/N)`` for ``0 <= a' <= N/2`` prime to ``N``, all real
    and distinct, and decreasing in ``a'``; so its index among the real roots of
    its minimal polynomial is the number of those ``a' > a``. Exact: no numerics.
    """
    angle = abs(1 - 2 * nu) / 4
    a, n = angle.numerator, angle.denominator
    poly = sp.Poly(sp.minimal_polynomial(sp.cos(2 * sp.pi * angle), _Y), _Y)
    index = sum(1 for b in range(a + 1, n // 2 + 1) if math.gcd(b, n) == 1)
    return AlgebraicReal(poly, index)


# -- exact constants -------------------------------------------------------------------


class GammaConstant[X: (AlgebraicReal, KRoot)]:
    """An exact constant: an algebraic number times a product of ``Γ(ν)^m``.

    Returned by :class:`Asymptotics` (:attr:`~Asymptotics.leading`,
    :meth:`~Asymptotics.terms`) and :class:`Ratio`; not built directly. Its value
    is ``A · Π Γ(ν)^{m_ν}``, ``0 < ν < 1`` rational (:attr:`gamma`). The ``Γ`` part
    is kept as FS and Broda et al. print it: the coefficient of
    ``[z^n] (1 - z/ρ)^x`` is ``1/Γ(-x)``, reduced to ``1/Γ(ν)`` with
    ``ν = -x mod 1`` by ``Γ(y + 1) = y Γ(y)`` (transfer.md, section 3.1).

    The algebraic part is the monomial ``A = q · Π X_i^{m_i}`` (:attr:`rational`,
    :attr:`factors`):

    - at a fixed ``k``, it is one :class:`~flanpy.AlgebraicReal`,
      :meth:`algebraic`; ``factors`` is empty when ``A`` is rational (then
      ``q = A``) and ``((A, 1),)`` otherwise (then ``q = 1``);
    - for symbolic ``k`` (:func:`asymptotics_in_k`), the ``X_i`` are
      :class:`~flanpy.KRoot` s (``ρ_k``, ``b_k``, ``c_k``) with integer exponents,
      for example ``(1/2) b_k / Γ(1/2)``; no new ``KRoot`` is built for ``1/c_k``.
      :meth:`at` evaluates the monomial at ``k = n`` exactly; :meth:`algebraic`,
      :meth:`evalf` and :meth:`as_expr` need a fixed ``k``.

    Constants multiply and divide (``*``, ``/``, also by ``int`` and ``Fraction``);
    a fixed-``k`` product is computed exactly in a number field containing both
    algebraic parts.

    Notes
    -----
    **What "exact" means** (transfer.md, section 3.2). With ``κ = 2`` the only
    base is ``Γ(1/2) = √π``, which is transcendental, so two constants are equal
    exactly when their algebraic parts and their ``Γ`` parts are. In general,
    equality and hashing use a canonical key: ``Γ(1/2) = π^{1/2}``, and every
    ``Γ(ν)`` with ``ν > 1/2`` is rewritten by reflection,
    ``Γ(ν) = π / (sin(πν) Γ(1 - ν))``, with ``sin(πν)`` (algebraic, exact) moved
    into the algebraic part. So ``Γ(1/4) Γ(3/4)`` equals ``√2 π``. With ``κ <= 4``
    this key decides equality (``π`` and ``Γ(1/4)`` are algebraically
    independent, a known result). For symbolic ``k`` the key compares the
    monomials (the same ``KRoot`` s with the same exponents).

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
    >>> print(c * c * 4)
    4/pi
    """

    __slots__ = ("_cached", "_factors", "_gamma", "_rational", "_symbolic")

    _rational: Fraction
    _factors: tuple[tuple[X, int], ...]
    _gamma: tuple[tuple[Fraction, int], ...]
    _symbolic: bool
    _cached: tuple[Any, ...] | None

    def __init__(self) -> None:
        raise TypeError("GammaConstant objects are created by flanpy.asymptotics")

    @classmethod
    def _make(
        cls,
        rational: Fraction,
        factors: Iterable[tuple[Any, int]],
        gamma: Iterable[tuple[Fraction, int]],
        symbolic: bool,
    ) -> GammaConstant[Any]:
        """Normalise: merge equal bases, drop zero exponents; ``0`` has nothing else."""
        merged: dict[Any, int] = {}
        for x, m in factors:
            merged[x] = merged.get(x, 0) + m
        bases: dict[Fraction, int] = {}
        for nu, m in gamma:
            bases[nu] = bases.get(nu, 0) + m
        if rational == 0:
            merged, bases = {}, {}
        new = object.__new__(cls)
        _set = object.__setattr__
        _set(new, "_rational", Fraction(rational))
        _set(new, "_factors", tuple((x, m) for x, m in merged.items() if m))
        _set(new, "_gamma", tuple(sorted((nu, m) for nu, m in bases.items() if m)))
        _set(new, "_symbolic", symbolic)
        _set(new, "_cached", None)
        return new

    @classmethod
    def _fixed(
        cls, value: AlgebraicReal, gamma: Iterable[tuple[Fraction, int]] = ()
    ) -> GammaConstant[AlgebraicReal]:
        """Return a fixed-``k`` constant with algebraic part ``value``."""
        q = _as_rational(value)
        if q is not None:
            return cls._make(q, (), gamma, False)
        return cls._make(Fraction(1), ((value, 1),), gamma, False)

    # -- public interface ----------------------------------------------------------

    def algebraic(self) -> AlgebraicReal:
        """Return the algebraic part ``A`` (the ``Γ`` part excluded), fixed ``k``.

        Returns
        -------
        AlgebraicReal
            ``A``, exactly.

        Raises
        ------
        ValueError
            For symbolic ``k``: the algebraic part is the monomial
            ``rational · Π factors`` (use :attr:`rational`, :attr:`factors`, or
            :meth:`at`).

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
        self._need_fixed("algebraic()")
        if self._factors:
            x: AlgebraicReal = self._factors[0][0]  # type: ignore[assignment]
            return x
        return _rational_real(self._rational)

    @property
    def rational(self) -> Fraction:
        """The rational factor ``q`` of the algebraic part ``q · Π X_i^{m_i}``.

        At a fixed ``k`` it is ``A`` when ``A`` is rational, and ``1`` otherwise.

        Examples
        --------
        Broda et al. (2020) ``R_k`` is in Case I with ``α = 1/2``: its leading
        constant is ``α b_k/Γ(1 - α) = (1/2) b_k/Γ(1/2)``.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics_in_k, eliminate_in_k
        >>> from flanpy import local_expansion_in_k, singularity_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> e = local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
        >>> c = asymptotics_in_k(e)[2].leading
        >>> c.rational, [m for _, m in c.factors], c.gamma
        (Fraction(1, 2), [1], ((Fraction(1, 2), -1),))
        """
        return self._rational

    @property
    def factors(self) -> tuple[tuple[X, int], ...]:
        """The algebraic factors ``(X_i, m_i)`` of ``q · Π X_i^{m_i}``.

        Symbolic ``k``: :class:`~flanpy.KRoot` s. Fixed ``k``: empty, or
        ``((A, 1),)`` with ``A`` an irrational :class:`~flanpy.AlgebraicReal`.

        Examples
        --------
        Broda et al. (2020) ``R_k``: the leading constant is ``(1/2) b_k/Γ(1/2)``,
        ``b_k`` the constant of the local expansion (relative to ``ρ_k``).

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics_in_k, eliminate_in_k
        >>> from flanpy import local_expansion_in_k, singularity_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> e = local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
        >>> ((b, m),) = asymptotics_in_k(e)[2].leading.factors
        >>> b == e[2].b, m
        (True, 1)
        """
        return self._factors

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

        Raises
        ------
        ValueError
            For symbolic ``k`` (use :meth:`at`), or if ``dps < 1``.

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
        self._need_fixed("evalf()")
        with mpmath.workdps(dps + 10):
            value = self.algebraic().evalf(dps + 10)
            for nu, m in self._gamma:
                value *= mpmath.gamma(mpmath.mpf(nu.numerator) / nu.denominator) ** m
        with mpmath.workdps(dps):
            return +value

    def as_expr(self) -> sp.Expr:
        """Return the value as a SymPy expression, fixed ``k``.

        The algebraic part is written with radicals when its degree is at most 2,
        as a :class:`sympy.CRootOf` otherwise; the ``Γ`` part with
        :class:`sympy.gamma`, which SymPy writes as ``sqrt(pi)`` for ``ν = 1/2``.

        Returns
        -------
        sympy.Expr
            The constant.

        Raises
        ------
        ValueError
            For symbolic ``k`` (use :meth:`at`).

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
        self._need_fixed("as_expr()")
        expr = _radical(self.algebraic())
        for nu, m in self._gamma:
            expr *= sp.gamma(sp.Rational(nu.numerator, nu.denominator)) ** m
        return expr

    def at(self, k: int) -> GammaConstant[AlgebraicReal]:
        """Return the constant at a fixed ``k`` (symbolic ``k`` only).

        The monomial ``q · Π X_i^{m_i}`` is evaluated exactly with
        ``X_i.at(k)`` (one number field for all of them); the ``Γ`` part is
        unchanged.

        Parameters
        ----------
        k : int
            A value of ``k`` where the ``X_i`` are defined.

        Returns
        -------
        GammaConstant
            The fixed-``k`` constant.

        Raises
        ------
        ValueError
            If the constant is already at a fixed ``k``.

        Examples
        --------
        Broda et al. (2020) ``R_k`` at ``k = 2``: the same constant as the
        fixed-``k`` analysis.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, asymptotics_in_k, eliminate
        >>> from flanpy import eliminate_in_k, local_expansion, local_expansion_in_k
        >>> from flanpy import singularity, singularity_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> a = asymptotics_in_k(
        ...     local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
        ... )
        >>> fixed = asymptotics(local_expansion(singularity(eliminate(rk, R, k=2))))
        >>> a[2].leading.at(2) == fixed.leading
        True
        """
        if not self._symbolic:
            raise ValueError("the constant is at a fixed k: nothing to specialize")
        if self._rational == 0:
            return GammaConstant._make(Fraction(0), (), (), False)
        factors: list[tuple[Any, int]] = list(self._factors)
        value = _monomial(self._rational, [(x.at(k), m) for x, m in factors])
        return GammaConstant._fixed(value, self._gamma)

    def _is_zero(self) -> bool:
        return self._rational == 0

    def _need_fixed(self, what: str) -> None:
        if self._symbolic:
            raise ValueError(
                f"{what} needs a fixed k: the algebraic part is the monomial "
                "rational · Π factors; use at(n) for the constant at k = n"
            )

    # -- products ------------------------------------------------------------------

    def _times(self, other: GammaConstant[Any], sign: int) -> GammaConstant[Any]:
        if self._symbolic != other._symbolic:
            raise ValueError("cannot combine a fixed-k constant with a symbolic one")
        if other._rational == 0 and sign < 0:
            raise ZeroDivisionError("division by a zero GammaConstant")
        rational = self._rational * other._rational**sign
        gamma = [*self._gamma, *((nu, sign * m) for nu, m in other._gamma)]
        factors = [*self._factors, *((x, sign * m) for x, m in other._factors)]
        if self._symbolic or rational == 0 or not factors:
            return GammaConstant._make(rational, factors, gamma, self._symbolic)
        return GammaConstant._fixed(_monomial(rational, factors), gamma)

    def __mul__(self, other: object) -> GammaConstant[X]:
        if isinstance(other, int | Fraction):
            other = GammaConstant._make(Fraction(other), (), (), self._symbolic)
        if not isinstance(other, GammaConstant):
            return NotImplemented
        return self._times(other, 1)

    def __rmul__(self, other: object) -> GammaConstant[X]:
        return self.__mul__(other)

    def __truediv__(self, other: object) -> GammaConstant[X]:
        if isinstance(other, int | Fraction):
            other = GammaConstant._make(Fraction(other), (), (), self._symbolic)
        if not isinstance(other, GammaConstant):
            return NotImplemented
        return self._times(other, -1)

    # -- value semantics -----------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        """Return the canonical key (reflection; Notes), computed once."""
        if self._cached is not None:
            return self._cached
        if self._rational == 0:
            key: tuple[Any, ...] = ("zero",)
        else:
            pi = Fraction(0)
            bases: dict[Fraction, int] = {}
            sines: list[tuple[AlgebraicReal, int]] = []
            for nu, m in self._gamma:
                if nu == Fraction(1, 2):
                    pi += Fraction(m, 2)
                elif nu > Fraction(1, 2):  # Γ(ν)^m = π^m sin(πν)^-m Γ(1 - ν)^-m
                    pi += m
                    sines.append((_sin_pi(nu), -m))
                    bases[1 - nu] = bases.get(1 - nu, 0) - m
                else:
                    bases[nu] = bases.get(nu, 0) + m
            gamma = tuple(sorted((nu, m) for nu, m in bases.items() if m))
            if self._symbolic:
                coefficient = _monomial(self._rational, sines)
                key = (coefficient, frozenset(self._factors), pi, gamma)
            else:
                value = _monomial(self._rational, [*self._factors, *sines])
                key = (value, pi, gamma)
        object.__setattr__(self, "_cached", key)
        return key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GammaConstant):
            return NotImplemented
        if self._symbolic != other._symbolic:
            return False
        if (self._rational, self._factors, self._gamma) == (
            other._rational,
            other._factors,
            other._gamma,
        ):
            return True
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash((self._symbolic, self._key()))

    def __repr__(self) -> str:
        if not self._symbolic:
            return f"GammaConstant({self.algebraic()!r}, gamma={self._gamma!r})"
        return (
            f"GammaConstant(rational={self._rational!r}, factors={self._factors!r}, "
            f"gamma={self._gamma!r})"
        )

    def __str__(self) -> str:
        if not self._symbolic:
            return str(self.as_expr())
        parts = (
            [str(self._rational)] if self._rational != 1 or not self._factors else []
        )
        for x, m in self._factors:
            parts.append(f"[{x}]" + ("" if m == 1 else f"^{m}"))
        for nu, m in self._gamma:
            parts.append(f"Γ({nu})^{m}")
        return " · ".join(parts)

    def __reduce__(self) -> tuple[Any, ...]:
        return (
            _restore_constant,
            (self._rational, self._factors, self._gamma, self._symbolic),
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("GammaConstant is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("GammaConstant is immutable")


def _restore_constant(
    rational: Fraction,
    factors: tuple[tuple[Any, int], ...],
    gamma: tuple[tuple[Fraction, int], ...],
    symbolic: bool,
) -> GammaConstant[Any]:
    """Rebuild a pickled :class:`GammaConstant`."""
    return GammaConstant._make(rational, factors, gamma, symbolic)


GammaConstant.__module__ = "flanpy"

_ZERO_CONSTANT: GammaConstant[Any] = GammaConstant._make(Fraction(0), (), (), False)


# -- the expansion in n ----------------------------------------------------------------


def _first_singular(e: LocalExpansion[Any]) -> Fraction:
    """``e_0``: ``α`` in Case I, ``-α`` in Case II (transfer.md, section 2.2)."""
    return e.alpha if e.singularity.case == "I" else -e.alpha


def _class(p: Fraction) -> Fraction | None:
    """``ν`` with ``1/Γ(-x) = r/Γ(ν)`` for ``x ≡ p`` (mod 1); ``None`` if integer."""
    return None if p.denominator == 1 else -p - math.floor(-p)


def _raw_coefficients(
    e: LocalExpansion[AlgebraicReal], count: int
) -> list[tuple[Fraction, _Element | None, Fraction | None]]:
    """Return the first ``count`` triples ``(p_j, A_j, ν_j)`` (transfer.md, 2.2).

    ``[z^n] w = ρ^{-n} Σ_j A_j/Γ(ν_j) n^{-1-p_j}``, ``p_j = e_0 + j/κ``, where
    ``A_j/Γ(ν_j) = Σ d_i e_l(-x_i)/Γ(-x_i)`` over the Puiseux terms ``d_i s^{x_i}``
    with ``l = p_j - x_i`` a non-negative integer (FS Thm VI.1, term by term).
    Every ``x_i`` of one ``j`` lies in one class mod 1, so one ``ν_j``; ``A_j`` is
    an element of the branch's field ``L`` (``None`` for zero), computed exactly
    there. The Puiseux terms needed are those with ``x <= p_{count-1}`` (section
    2.3).
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
    out: list[tuple[Fraction, _Element | None, Fraction | None]] = []
    for j in range(count):
        p = e0 + Fraction(j, kappa)
        nu = _class(p)
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
        out.append((p, None if total is None or total.is_zero() else total, nu))
    return out


def _constant(value: _Element | None, gamma: dict[Fraction, int]) -> GammaConstant[Any]:
    if value is None or value.is_zero():
        return _ZERO_CONSTANT
    return GammaConstant._fixed(value.to_real(), gamma.items())


def _coefficients(
    e: LocalExpansion[AlgebraicReal], count: int
) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]:
    """Return the first ``count`` pairs ``(power of n, D_j)``."""
    return [
        (-1 - p, _constant(value, {} if nu is None else {nu: -1}))
        for p, value, nu in _raw_coefficients(e, count)
    ]


def _leading_in_k(e: LocalExpansion[KRoot]) -> GammaConstant[KRoot]:
    """Return the leading constant for symbolic ``k``, a monomial (section 7.1).

    Case I: ``-b/Γ(-α) = -r b/Γ(ν)``; Case II: ``1/(c Γ(α)) = r c^{-1}/Γ(ν)``,
    with ``1/Γ(x) = r/Γ(ν)`` (:func:`_inverse_gamma`).
    """
    if e.singularity.case == "I":
        r, nu = _inverse_gamma(-e.alpha)
        factors, rational = [(e.b, 1)], -r
    else:
        r, nu = _inverse_gamma(e.alpha)
        factors, rational = [(e.c, -1)], r
    gamma = () if nu is None else ((nu, -1),)
    return GammaConstant._make(rational, factors, gamma, True)


# -- the public types ------------------------------------------------------------------
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

    On a piece of :func:`asymptotics_in_k` (symbolic ``k``), ``rho`` is a
    :class:`~flanpy.KRoot` and :attr:`leading` the monomial ``α b_k/Γ(1 - α)`` or
    ``c_k^{-1}/Γ(α)`` (:class:`GammaConstant`); the exponent and the ``O``-term
    are those of every ``k`` of the piece. The further terms, :meth:`evalf` and
    :meth:`as_expr` need a fixed ``k``: :meth:`specialize`.

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
    def _make(cls, expansion: LocalExpansion[Any]) -> Asymptotics[Any]:
        new = object.__new__(cls)
        if expansion.k is None:
            terms = _coefficients(expansion, 1)
            leading: GammaConstant[Any] = terms[0][1]
        else:
            terms, leading = [], _leading_in_k(expansion)
        object.__setattr__(new, "_expansion", expansion)
        object.__setattr__(new, "_leading", leading)
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
            If ``count < 0``, or for symbolic ``k`` (use :meth:`specialize`).

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
        expansion = self._fixed("terms")
        if len(self._terms) < count:
            object.__setattr__(self, "_terms", _coefficients(expansion, count))
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

        ``1/(1 - 2z)``: ``[z^n] = 2^n`` exactly: no ``O``-term in ``n``.

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
            If ``n < 1``, ``count < 0`` or ``dps < 1``, or for symbolic ``k``.

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
        rho = self._fixed("evalf").singularity.rho
        work = dps + 10
        with mpmath.workdps(work):
            total = mpmath.mpf(0)
            for power, d in self.terms(count):
                if not d._is_zero():
                    exponent = mpmath.mpf(power.numerator) / power.denominator
                    total += d.evalf(work) * mpmath.mpf(n) ** exponent
        digits = work + len(str(n))
        with mpmath.workdps(digits):
            scale = rho.evalf(digits) ** (-n)
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
            If ``count < 0``, or for symbolic ``k``.

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
        rho = self._fixed("as_expr").singularity.rho
        total = sp.Add(
            *(
                d.as_expr() * n ** sp.Rational(p.numerator, p.denominator)
                for p, d in self.terms(count)
                if not d._is_zero()
            )
        )
        growth = sp.radsimp(1 / _radical(rho))
        return growth**n * total

    def specialize(self, k: int) -> Asymptotics[AlgebraicReal]:
        """Return the asymptotics at a fixed ``k`` (for a piece of symbolic ``k``).

        Equal to ``asymptotics(self.expansion, k=k)``; its leading constant is
        ``self.leading.at(k)``.

        Parameters
        ----------
        k : int
            A value in :attr:`kset`.

        Returns
        -------
        Asymptotics
            The asymptotics at ``k``.

        Raises
        ------
        ValueError
            If the asymptotics are already at a fixed ``k``, or ``k`` is not in
            :attr:`kset`.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics_in_k, eliminate_in_k
        >>> from flanpy import local_expansion_in_k, singularity_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> e = local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
        >>> a = asymptotics_in_k(e)[2]
        >>> a.specialize(2).leading == a.leading.at(2)
        True
        """
        if self._expansion.k is None:
            raise ValueError("the asymptotics are at a fixed k: nothing to specialize")
        return asymptotics(self._expansion, k=k)

    def _fixed(self, what: str) -> LocalExpansion[AlgebraicReal]:
        if self._expansion.k is not None:
            raise ValueError(
                f"{what} needs a fixed k: use specialize(n) for the asymptotics at "
                "k = n"
            )
        expansion: Any = self._expansion
        return expansion  # type: ignore[no-any-return]

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
        error = self.error_exponent(1)
        if self._expansion.k is not None:
            tail = "" if error is None else f" (1 + O(n^{error - self.exponent}))"
            return f"ρ^-n n^{self.exponent} · {self._leading}{tail}"
        lead = self.as_expr(sp.Symbol("n"))
        if error is None:
            return f"{lead} + O((ρ + ε)^-n)"
        return f"{lead} (1 + O(n^{error - self.exponent}))"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore, (self._expansion,))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Asymptotics is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Asymptotics is immutable")


def _restore(expansion: LocalExpansion[Any]) -> Asymptotics[Any]:
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


# -- ratios of two expansions ----------------------------------------------------------


def _ratio_terms(
    a: Asymptotics[AlgebraicReal], b: Asymptotics[AlgebraicReal], count: int
) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]:
    """Return the first ``count`` terms ``q_j`` of ``[z^n] A / [z^n] B`` (6.1).

    Both expansions in ``u = n^{-1/κ}``, ``κ = lcm(κ_A, κ_B)``; ``q`` is their
    series quotient, computed exactly in one number field holding both branch
    fields. ``q_j`` has the ``Γ`` part of ``A_j`` over that of ``B_0`` when every
    non-zero ``B_i`` (``0 < i < count``) has ``B_0``'s ``Γ`` class; otherwise
    ``q_j`` is a sum over several ``Γ`` monomials, which is not represented.
    """
    ea, eb = a._fixed("terms"), b._fixed("terms")
    kappa = math.lcm(ea.ramification, eb.ramification)
    step_a, step_b = kappa // ea.ramification, kappa // eb.ramification
    raw_a = _raw_coefficients(ea, -(-count // step_a))
    raw_b = _raw_coefficients(eb, -(-count // step_b))
    field_a, field_b = raw_a[0][1], raw_b[0][1]
    assert field_a is not None and field_b is not None  # D_0 != 0
    _, embed_a, embed_b = _common_field(field_a.field, field_b.field)

    def sequence(
        raw: list[tuple[Fraction, _Element | None, Fraction | None]],
        step: int,
        embed: Callable[[_Element], Any],
        e0: Fraction,
    ) -> list[tuple[Any, Fraction | None]]:
        out: list[tuple[Any, Fraction | None]] = []
        for j in range(count):
            if j % step:
                out.append((None, _class(e0 + Fraction(j, kappa))))
            else:
                _, value, nu = raw[j // step]
                out.append((None if value is None else embed(value), nu))
        return out

    seq_a = sequence(raw_a, step_a, embed_a, _first_singular(ea))
    seq_b = sequence(raw_b, step_b, embed_b, _first_singular(eb))
    b0, nu_b0 = seq_b[0]
    for i, (value, nu) in enumerate(seq_b):
        if value is not None and nu != nu_b0:
            raise NotImplementedError(
                f"the denominator has terms in two Γ classes (its coefficient {i}): "
                f"the ratio's terms from index {i} on are sums of several Γ "
                "monomials, which are not represented; the limit and evalf work"
            )
    exponent = a.exponent - b.exponent
    q: list[Any] = []
    out: list[tuple[Fraction, GammaConstant[AlgebraicReal]]] = []
    for j in range(count):
        total = seq_a[j][0]
        for i in range(1, j + 1):
            if seq_b[i][0] is not None and q[j - i] is not None:
                term = seq_b[i][0] * q[j - i]
                total = -term if total is None else total - term
        value = None if total is None or total.is_zero() else total / b0
        q.append(value)
        nu_a = seq_a[j][1]
        gamma: dict[Fraction, int] = {} if nu_a is None else {nu_a: -1}
        if nu_b0 is not None:
            gamma[nu_b0] = gamma.get(nu_b0, 0) + 1
        out.append((exponent - Fraction(j, kappa), _constant(value, gamma)))
    return out


class Ratio[X: (AlgebraicReal, KRoot)]:
    """The asymptotic ratio ``[z^n] A / [z^n] B`` of two expansions at one ``ρ``.

    Returned by :func:`ratio`; not built directly.
    ``[z^n] A / [z^n] B = n^{β_A - β_B} (q_0 + q_1 n^{-1/κ} + q_2 n^{-2/κ} + ...)``,
    ``κ = lcm(κ_A, κ_B)``: :attr:`exponent` is ``β_A - β_B`` and :attr:`limit` is
    ``q_0 = C_A/C_B``, exact; :meth:`terms` gives the ``q_j``. This is how Broda
    et al. (2020) compute averages (Thm 3.2, eqs. (8) and (38)).

    For symbolic ``k`` (two pieces of :func:`asymptotics_in_k` with the same
    ``ρ_k``), the limit is a monomial in ``KRoot`` s; :meth:`terms` and
    :meth:`evalf` need a fixed ``k`` (:meth:`specialize`).

    Notes
    -----
    Equality and hashing compare the two expansions. A ``Ratio`` is immutable and
    can be pickled.

    Examples
    --------
    Broda et al. (2020) section 2: ``[z^n] S_k / (n [z^n] R_k) → ρ_k`` (eq. (8)),
    the probability that a symbol of a regular expression is a star.

    >>> from sympy import symbols
    >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
    >>> from flanpy import singularity
    >>> z, k, R, S = symbols("z k R S")
    >>> rs = System(
    ...     {R: (k + 1) * z + 2 * z * R**2 + z * R, S: z * R + z * S + 4 * z * S * R},
    ...     z=z,
    ...     k=k,
    ... )
    >>> def at_2(w):
    ...     return asymptotics(local_expansion(singularity(eliminate(rs, w, k=2))))
    >>> r = ratio(at_2(S), at_2(R))
    >>> r.exponent, r.limit.algebraic() == at_2(R).rho
    (Fraction(1, 1), True)
    """

    __slots__ = ("_denominator", "_limit", "_numerator", "_terms")

    _numerator: Asymptotics[X]
    _denominator: Asymptotics[X]
    _limit: GammaConstant[X]
    _terms: list[tuple[Fraction, GammaConstant[AlgebraicReal]]]

    def __init__(self) -> None:
        raise TypeError("Ratio objects are created by flanpy.ratio")

    @classmethod
    def _make(cls, a: Asymptotics[Any], b: Asymptotics[Any]) -> Ratio[Any]:
        new = object.__new__(cls)
        object.__setattr__(new, "_numerator", a)
        object.__setattr__(new, "_denominator", b)
        object.__setattr__(new, "_limit", a.leading / b.leading)
        object.__setattr__(new, "_terms", [])
        return new

    # -- public interface ----------------------------------------------------------

    @property
    def numerator(self) -> Asymptotics[X]:
        """``A``, the expansion in the numerator.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
        >>> ratio(a, a).numerator is a
        True
        """
        return self._numerator

    @property
    def denominator(self) -> Asymptotics[X]:
        """``B``, the expansion in the denominator.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
        >>> ratio(a, a).denominator is a
        True
        """
        return self._denominator

    @property
    def exponent(self) -> Fraction:
        """``β_A - β_B``: ``[z^n] A / [z^n] B ~ limit · n^exponent``.

        Examples
        --------
        ``W = T/(1 - 4z)`` with ``T = 1 + zT^2`` (Catalan) has a pole at ``1/4``
        and grows like ``4^n``, Catalan like ``4^n n^{-3/2}``:

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, T, Y, W = symbols("z T Y W")
        >>> system = System({T: 1 + z * T**2, Y: 1 + 4 * z * Y, W: T * Y}, z=z)
        >>> def a(w):
        ...     return asymptotics(local_expansion(singularity(eliminate(system, w))))
        >>> r = ratio(a(W), a(T))
        >>> r.exponent, r.limit.as_expr()
        (Fraction(3, 2), 2*sqrt(pi))
        """
        return self._numerator.exponent - self._denominator.exponent

    @property
    def limit(self) -> GammaConstant[X]:
        """``q_0 = C_A/C_B``, the limit of ``[z^n] A / (n^exponent [z^n] B)``.

        Examples
        --------
        ``K = (1 - s^{1/4})/2`` over Catalan, both at ``ρ = 1/4``: the limit is
        ``√π/(8 Γ(3/4))`` (two ``Γ`` classes).

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, F, U, K = symbols("z F U K")
        >>> system = System({F: 1 + z * F**2, U: z + U**2, K: U / 2 + K**2}, z=z)
        >>> def a(w):
        ...     return asymptotics(local_expansion(singularity(eliminate(system, w))))
        >>> r = ratio(a(K), a(F))
        >>> r.exponent, r.limit.as_expr()
        (Fraction(1, 4), sqrt(pi)/(8*gamma(3/4)))
        """
        return self._limit

    @property
    def k(self) -> Any:
        """The parameter: ``None`` at a fixed ``k`` (forwarded from ``A``).

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, F = symbols("z F")
        >>> catalan = System({F: 1 + z * F**2}, z=z)
        >>> a = asymptotics(local_expansion(singularity(eliminate(catalan, F))))
        >>> ratio(a, a).k is None
        True
        """
        return self._numerator.k

    @property
    def kset(self) -> KSet:
        """The ``k`` this result is for: those of ``A`` and ``B``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> a = asymptotics(local_expansion(singularity(eliminate(rk, R, k=2))))
        >>> ratio(a, a).kset
        KSet([(2, 2)])
        """
        return self._numerator.kset & self._denominator.kset

    def terms(self, count: int) -> list[tuple[Fraction, GammaConstant[AlgebraicReal]]]:
        """Return the first ``count`` pairs ``(power of n, q_j)``, exactly.

        The powers are ``exponent - j/κ``; zeros are included. Cached.

        Parameters
        ----------
        count : int
            How many coefficients, ``count >= 0`` (zeros included).

        Returns
        -------
        list of (Fraction, GammaConstant)
            ``[z^n] A / [z^n] B = Σ q_j n^{power}``.

        Raises
        ------
        TypeError
            If ``count`` is not an ``int``.
        ValueError
            If ``count < 0``, or for symbolic ``k`` (use :meth:`specialize`).
        NotImplementedError
            If a ``q_j`` with ``j < count`` is a sum of several ``Γ`` monomials
            (the denominator has non-zero terms in two ``Γ`` classes).

        Examples
        --------
        Broda et al. (2020), eq. (8) at ``k = 2``: ``ρ (1 + 1/(2n) + ...)``.

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import singularity
        >>> z, k, R, S = symbols("z k R S")
        >>> rs = System(
        ...     {
        ...         R: (k + 1) * z + 2 * z * R**2 + z * R,
        ...         S: z * R + z * S + 4 * z * S * R,
        ...     },
        ...     z=z,
        ...     k=k,
        ... )
        >>> def at_2(w):
        ...     return asymptotics(local_expansion(singularity(eliminate(rs, w, k=2))))
        >>> [(str(p), str(q)) for p, q in ratio(at_2(S), at_2(R)).terms(3)]
        [('1', '-1/23 + 2*sqrt(6)/23'), ('1/2', '0'), ('0', '-1/46 + sqrt(6)/23')]
        """
        _check_count(count)
        if self._numerator.k is not None:
            raise ValueError(
                "terms need a fixed k: use specialize(n) for the ratio at k = n"
            )
        if len(self._terms) < count:
            terms = _ratio_terms(self._numerator, self._denominator, count)  # type: ignore[arg-type]
            object.__setattr__(self, "_terms", terms)
        return self._terms[:count]

    def evalf(self, n: int, *, count: int = 1, dps: int = 15) -> mpmath.mpf:
        """Return ``Σ_{j < count} q_j n^{exponent - j/κ}`` as an :class:`mpmath.mpf`.

        Computed from the two expansions numerically (series division of their
        :meth:`Asymptotics.terms`), so it works also where :meth:`terms` does not.
        Asymptotic, not convergent (:meth:`Asymptotics.evalf`).

        Parameters
        ----------
        n : int
            The index, ``n >= 1``.
        count : int, default 1
            How many coefficients ``q_j`` (zeros included).
        dps : int, default 15
            Decimal digits of the evaluation.

        Returns
        -------
        mpmath.mpf
            The truncated expansion of the ratio at ``n``.

        Raises
        ------
        TypeError
            If ``n`` or ``count`` is not an ``int``.
        ValueError
            If ``n < 1``, ``count < 0`` or ``dps < 1``, or for symbolic ``k``.

        Examples
        --------
        ``[z^n] T/(1 - 4z) / [z^n] T = 2√π n^{3/2} (1 + ...)`` (``T`` Catalan), at
        ``n = 1000`` against the exact ratio:

        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
        >>> from flanpy import series, singularity
        >>> z, T, Y, W = symbols("z T Y W")
        >>> system = System({T: 1 + z * T**2, Y: 1 + 4 * z * Y, W: T * Y}, z=z)
        >>> def a(w):
        ...     return asymptotics(local_expansion(singularity(eliminate(system, w))))
        >>> r = ratio(a(W), a(T))
        >>> c = series(system, 1000)
        >>> exact = c[W][1000] / c[T][1000]
        >>> f"{float(abs(r.evalf(1000, count=4) / exact - 1)):.1e}"
        '1.4e-07'
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {n!r}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if isinstance(dps, bool) or not isinstance(dps, int) or dps < 1:
            raise ValueError(f"dps must be a positive int, got {dps!r}")
        _check_count(count)
        a, b = self._numerator, self._denominator
        if a.k is not None:
            raise ValueError("evalf needs a fixed k: use specialize(n)")
        if count == 0:
            return mpmath.mpf(0)
        ka, kb = a.expansion.ramification, b.expansion.ramification
        kappa = math.lcm(ka, kb)
        work = dps + 10

        def sequence(x: Asymptotics[Any], step: int) -> list[mpmath.mpf]:
            terms = x.terms(-(-count // step))
            return [
                mpmath.mpf(0)
                if j % step or terms[j // step][1]._is_zero()
                else terms[j // step][1].evalf(work)
                for j in range(count)
            ]

        with mpmath.workdps(work):
            seq_a = sequence(a, kappa // ka)
            seq_b = sequence(b, kappa // kb)
            q: list[mpmath.mpf] = []
            total = mpmath.mpf(0)
            for j in range(count):
                value = seq_a[j] - sum(
                    (seq_b[i] * q[j - i] for i in range(1, j + 1)), mpmath.mpf(0)
                )
                q.append(value / seq_b[0])
                power = self.exponent - Fraction(j, kappa)
                total += q[j] * mpmath.mpf(n) ** (
                    mpmath.mpf(power.numerator) / power.denominator
                )
        with mpmath.workdps(dps):
            return +total

    def specialize(self, k: int) -> Ratio[AlgebraicReal]:
        """Return the ratio at a fixed ``k`` (for pieces of symbolic ``k``).

        Equal to ``ratio(A.specialize(k), B.specialize(k))``; its limit is
        ``self.limit.at(k)``.

        Parameters
        ----------
        k : int
            A value in :attr:`kset`.

        Returns
        -------
        Ratio
            The ratio at ``k``.

        Raises
        ------
        ValueError
            If the ratio is already at a fixed ``k``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System, asymptotics_in_k, eliminate_in_k
        >>> from flanpy import local_expansion_in_k, ratio, singularity_in_k
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> e = local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
        >>> a = asymptotics_in_k(e)[1]
        >>> r = ratio(a, a)
        >>> r.limit.rational, r.specialize(3).limit.algebraic()
        (Fraction(1, 1), AlgebraicReal(y - 1, 0))
        """
        if self._numerator.k is None:
            raise ValueError("the ratio is at a fixed k: nothing to specialize")
        return ratio(self._numerator.specialize(k), self._denominator.specialize(k))

    # -- value semantics -----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ratio):
            return NotImplemented
        return bool(
            self._numerator == other._numerator
            and self._denominator == other._denominator
        )

    def __hash__(self) -> int:
        return hash(("Ratio", self._numerator, self._denominator))

    def __repr__(self) -> str:
        return f"Ratio(exponent={self.exponent!r}, limit={self._limit!r})"

    def __str__(self) -> str:
        return f"{self._limit} · n^{self.exponent}"

    def __reduce__(self) -> tuple[Any, ...]:
        return (ratio, (self._numerator, self._denominator))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Ratio is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Ratio is immutable")


Ratio.__module__ = "flanpy"


def ratio[X: (AlgebraicReal, KRoot)](
    numerator: Asymptotics[X], denominator: Asymptotics[X]
) -> Ratio[X]:
    """Return the asymptotic ratio ``[z^n] A / [z^n] B`` (same ``ρ``).

    With ``[z^n] A ~ C_A ρ^{-n} n^{β_A}`` and ``[z^n] B ~ C_B ρ^{-n} n^{β_B}``,
    ``[z^n] A / [z^n] B ~ (C_A/C_B) n^{β_A - β_B}``, and the further terms come
    from the series quotient of the two expansions (transfer.md, section 6).
    Broda et al. (2020) compute averages so: ``[z^n] S_k / (n [z^n] R_k) → ρ_k``
    (eq. (8)) and ``[z^n] L_k / (n [z^n] F_k) → 2/(b_k c_k)`` (Thm 3.2, eq. (38),
    with ``b_k > 0`` as in eq. (20)).

    For symbolic ``k``, combine two :class:`~flanpy.KCases` with
    ``cases_a.combine(cases_b, ratio)``.

    Parameters
    ----------
    numerator, denominator : Asymptotics
        Both at the same fixed ``k``, or both pieces of :func:`asymptotics_in_k`
        with a common ``k``.

    Returns
    -------
    Ratio
        The exponent, the exact limit and the further terms.

    Raises
    ------
    TypeError
        If an argument is not an :class:`Asymptotics`.
    ValueError
        If ``ρ`` differs (the ratio then has the factor ``(ρ_B/ρ_A)^n``, which is
        not represented), or one is at a fixed ``k`` and the other not, or they
        are for different ``k``.

    Examples
    --------
    The number of internal nodes ``T`` and the Catalan numbers have different
    ``ρ``:

    >>> from sympy import symbols
    >>> from flanpy import System, asymptotics, eliminate, local_expansion, ratio
    >>> from flanpy import singularity
    >>> z, F = symbols("z F")
    >>> def a(system):
    ...     return asymptotics(local_expansion(singularity(eliminate(system, F))))
    >>> catalan = a(System({F: 1 + z * F**2}, z=z))
    >>> motzkin = a(System({F: z + z * F + z * F**2}, z=z))
    >>> ratio(catalan, catalan).limit.algebraic()
    AlgebraicReal(y - 1, 0)
    >>> ratio(catalan, motzkin)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: ρ differs (1/4 and 1/3): the ratio has the factor (ρ_B/ρ_A)^n, ...
    """
    for x in (numerator, denominator):
        if not isinstance(x, Asymptotics):
            raise TypeError(f"expected a flanpy.Asymptotics, got {type(x).__name__}")
    if (numerator.k is None) != (denominator.k is None):
        raise ValueError(
            "one expansion is at a fixed k and the other for symbolic k: use "
            "specialize(n) on the symbolic one"
        )
    if numerator.k is None:
        if numerator.kset != denominator.kset:
            raise ValueError(
                f"the expansions are for different k ({numerator.kset} and "
                f"{denominator.kset})"
            )
    elif not numerator.kset & denominator.kset:
        raise ValueError(
            f"the pieces have no k in common ({numerator.kset} and {denominator.kset})"
        )
    if numerator.rho != denominator.rho:
        a, b = numerator.rho, denominator.rho
        shown = [
            str(_radical(x)) if isinstance(x, AlgebraicReal) else str(x) for x in (a, b)
        ]
        raise ValueError(
            f"ρ differs ({shown[0]} and {shown[1]}): the ratio has the factor "
            "(ρ_B/ρ_A)^n, which is not represented"
        )
    return Ratio._make(numerator, denominator)


# -- symbolic k ------------------------------------------------------------------------


def asymptotics_in_k(
    expansions: LocalExpansion[KRoot] | KCases[LocalExpansion[Any]],
) -> KCases[Asymptotics[KRoot]]:
    """Return the asymptotics of ``[z^n] w`` for symbolic ``k``, piece by piece.

    One piece for each piece of :func:`~flanpy.local_expansion_in_k` (no new
    cuts; ``uncovered`` is kept). On a piece, ``α``, ``κ`` and the case are
    constant, so the exponent and the ``O``-term are; the leading constant is the
    monomial ``α b_k/Γ(1 - α)`` (Case I) or ``c_k^{-1}/Γ(α)`` (Case II, with
    ``1/Γ`` reduced to ``1/Γ(ν)``), in the :class:`~flanpy.KRoot` s of step 5
    (transfer.md, section 7). ``piece.specialize(n)`` (or
    ``piece.leading.at(n)``) equals the fixed-``k`` result.

    Parameters
    ----------
    expansions : LocalExpansion or KCases
        A piece of :func:`~flanpy.local_expansion_in_k` or all of them.

    Returns
    -------
    KCases of Asymptotics
        The asymptotics on each piece.

    Raises
    ------
    TypeError
        If the input is not a :class:`~flanpy.LocalExpansion` (or a ``KCases`` of
        them).
    ValueError
        If an expansion is at a fixed ``k`` (use :func:`asymptotics`), or on some
        piece ``ρ_k`` is not known to be the only dominant singularity (``unique``
        is ``False`` or ``None``); the message names those pieces, and
        ``cases.restrict(...)`` removes them.

    Examples
    --------
    Broda et al. (2020) ``R_k``: one piece, ``[z^n] R_k ~ (1/2) b_k/Γ(1/2) ρ_k^{-n}
    n^{-3/2}`` for every ``k >= 1`` (eq. (6)).

    >>> from sympy import symbols
    >>> from flanpy import System, asymptotics_in_k, eliminate_in_k
    >>> from flanpy import local_expansion_in_k, singularity_in_k
    >>> z, k, R = symbols("z k R")
    >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    >>> cases = asymptotics_in_k(
    ...     local_expansion_in_k(singularity_in_k(eliminate_in_k(rk, R)))
    ... )
    >>> [(str(part), a.exponent) for part, a in cases.pieces]
    [('[1, ∞)', Fraction(-3, 2))]
    >>> cases[5].specialize(5).leading == cases[5].leading.at(5)
    True
    """
    if isinstance(expansions, LocalExpansion):
        if expansions.k is None:
            raise ValueError(
                "the expansion is at a fixed k: use flanpy.asymptotics for it"
            )
        cases: KCases[LocalExpansion[Any]] = KCases({expansions.kset: expansions})
    elif isinstance(expansions, KCases):
        cases = expansions
    else:
        raise TypeError(
            "expected a flanpy.LocalExpansion or a KCases of them, got "
            f"{type(expansions).__name__}"
        )
    bad = []
    for part, e in cases.pieces:
        if not isinstance(e, LocalExpansion):
            raise TypeError(f"expected flanpy.LocalExpansion values, got {e!r}")
        if e.k is None:
            raise ValueError("the expansions are at a fixed k: use flanpy.asymptotics")
        if e.singularity.unique is not True:
            bad.append(str(part))
    if bad:
        raise ValueError(
            "ρ_k is not known to be the only dominant singularity on "
            f"{', '.join(bad)} (unique is False or undecided there): the transfer "
            "needs it; restrict the cases to the other pieces first "
            "(KCases.restrict)"
        )
    return cases.map(Asymptotics._make)


__test__ = method_doctests(GammaConstant, Asymptotics, Ratio)
