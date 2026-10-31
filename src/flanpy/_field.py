# SPDX-License-Identifier: GPL-3.0-or-later
"""Real number fields ``Q(θ)``: the coefficient field of the Puiseux expansions.

Private module (design: ``dev/design/newton.md``, section 4). Step 5 computes the
local expansion of a branch at its dominant singularity ``ρ``. Its coefficients are
real algebraic numbers (newton.md, Lemma R), and the Newton polygons and the branch
order need *exact* zero and sign tests on them. So the arithmetic is done in one
explicit real number field at a time:

- :class:`_NumberField` is ``Q(θ) = Q[x]/(m_θ)`` with the real embedding given by an
  :class:`~flanpy.AlgebraicReal` ``θ``. Elements (:class:`_Element`) are polynomials
  of degree ``< deg m_θ``; the zero test reduces modulo ``m_θ`` (exact), signs come
  from ``θ.sign`` (exact), and :meth:`_Element.to_real` returns the element as an
  ``AlgebraicReal`` (its minimal polynomial over ``Q`` and an isolating interval).
- :meth:`_NumberField.real_roots` returns the real roots of a polynomial over
  ``Q(θ)``, each in a field ``L = Q(θ, c)`` with a single generator (the tower is
  flattened), by Trager's norm method (Trager 1976, Algebraic factoring; Loos 1983,
  real algebraic numbers).
- :class:`_Field` is the protocol behind which a later backend (a flint number field,
  Sage) could replace :class:`_NumberField` (newton.md, decision Q3). SymPy radicals
  are *not* an implementation: their zero test is heuristic, and ``ρ`` itself is often
  not expressible in radicals (newton.md, section 4.3).

The polynomial arithmetic in ``Q[x]`` uses the backend chosen by
:func:`flanpy._rings.backend`: python-flint's ``fmpq_poly`` / ``fmpq_mat`` when it is
installed, otherwise SymPy's sparse ``QQ[x]`` and ``DomainMatrix``. Norms of
polynomials over ``Q(θ)``, factorisation over ``Q`` and real-root isolation use SymPy
in both cases (python-flint 0.9 has no exact real-root isolation; step 4).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, NoReturn, Protocol, Self

import sympy as sp

from flanpy._algebraic import (
    AlgebraicReal,
    _intervals,
    _normalised,
    _rational,
    _real_roots,
)
from flanpy._rings import BackendName, backend

__all__: list[str] = []

_X, _Y = sp.symbols("x y")
# Trager's shifts t in N_t(y) = Res_x(m(x), P(x, y - t x)), tried in this order.
_SHIFTS = (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5)

_Scalar = int | Fraction


# -- polynomials over Q in the chosen backend ------------------------------------------


class _QQx:
    """Univariate polynomials over ``Q``: the operations the fields need.

    Elements are native to the backend (``fmpq_poly`` or a SymPy ``PolyElement``);
    ``+``, ``-``, ``*``, ``%`` and ``divmod`` are the native operators. Coefficient
    lists are ``Fraction``s, lowest degree first.
    """

    name: BackendName

    def from_fractions(self, cs: Sequence[Fraction]) -> Any:
        """Return ``sum cs[i] x^i``."""
        raise NotImplementedError

    def fractions(self, p: Any) -> tuple[Fraction, ...]:
        """Return the coefficients of ``p``, lowest first, without trailing zeros."""
        raise NotImplementedError

    def xgcd(self, a: Any, b: Any) -> tuple[Any, Any]:
        """Return ``(g, s)`` with ``s a ≡ g (mod b)`` and ``g = gcd(a, b)``."""
        raise NotImplementedError

    def charpoly(self, rows: Sequence[Sequence[Fraction]]) -> tuple[Fraction, ...]:
        """Return ``det(y I - M)`` for a square matrix ``M``, lowest degree first."""
        raise NotImplementedError


class _FlintQQx(_QQx):
    name: BackendName = "flint"

    def __init__(self) -> None:
        import flint

        self.flint = flint

    def from_fractions(self, cs: Sequence[Fraction]) -> Any:
        fl = self.flint
        return fl.fmpq_poly([fl.fmpq(c.numerator, c.denominator) for c in cs])

    def fractions(self, p: Any) -> tuple[Fraction, ...]:
        return tuple(Fraction(int(c.p), int(c.q)) for c in p.coeffs())

    def xgcd(self, a: Any, b: Any) -> tuple[Any, Any]:
        g, s, _ = a.xgcd(b)
        return g, s

    def charpoly(self, rows: Sequence[Sequence[Fraction]]) -> tuple[Fraction, ...]:
        fl = self.flint
        n = len(rows)
        entries = [fl.fmpq(c.numerator, c.denominator) for row in rows for c in row]
        return self.fractions(fl.fmpq_mat(n, n, entries).charpoly())


class _PythonQQx(_QQx):
    name: BackendName = "python"

    def __init__(self) -> None:
        from sympy.polys.rings import ring

        self.ring, _ = ring("x", sp.QQ)

    def from_fractions(self, cs: Sequence[Fraction]) -> Any:
        qq = sp.QQ
        return self.ring(
            {(i,): qq(c.numerator, c.denominator) for i, c in enumerate(cs) if c}
        )

    def fractions(self, p: Any) -> tuple[Fraction, ...]:
        if not p:
            return ()
        qq = sp.QQ
        dense = p.to_dense()  # highest degree first
        return tuple(
            Fraction(int(qq.numer(c)), int(qq.denom(c))) for c in reversed(dense)
        )

    def xgcd(self, a: Any, b: Any) -> tuple[Any, Any]:
        s, _, g = a.gcdex(b)
        return g, s

    def charpoly(self, rows: Sequence[Sequence[Fraction]]) -> tuple[Fraction, ...]:
        from sympy.polys.matrices import DomainMatrix

        qq = sp.QQ
        n = len(rows)
        matrix = DomainMatrix(
            [[qq(c.numerator, c.denominator) for c in row] for row in rows], (n, n), qq
        )
        return tuple(
            Fraction(int(qq.numer(c)), int(qq.denom(c)))
            for c in reversed(matrix.charpoly())
        )


_QQX_CACHE: dict[BackendName, _QQx] = {}


def _qqx(name: BackendName | None = None) -> _QQx:
    """Return the ``Q[x]`` arithmetic of the selected backend (flint if installed)."""
    chosen = backend(name).name
    if chosen not in _QQX_CACHE:
        _QQX_CACHE[chosen] = _FlintQQx() if chosen == "flint" else _PythonQQx()
    return _QQX_CACHE[chosen]


def _horner_interval(
    cs: Sequence[Fraction], lo: Fraction, hi: Fraction
) -> tuple[Fraction, Fraction]:
    """Return a rational enclosure of ``sum cs[i] x^i`` for ``x`` in ``[lo, hi]``."""
    if not cs:
        return Fraction(0), Fraction(0)
    a = b = cs[-1]
    for c in reversed(cs[:-1]):
        products = (a * lo, a * hi, b * lo, b * hi)
        a, b = min(products) + c, max(products) + c
    return a, b


def _sympy_poly(cs: Sequence[Fraction], x: sp.Symbol) -> sp.Poly:
    """``sum cs[i] x^i`` as a :class:`sympy.Poly` over ``QQ``."""
    return sp.Poly.from_list(
        [_rational(c) for c in reversed(cs)] or [0], x, domain=sp.QQ
    )


# -- the protocol ----------------------------------------------------------------------


class _FieldElement(Protocol):
    """An element of a real number field (newton.md, section 4.1)."""

    def __add__(self, other: Self | _Scalar) -> Self: ...
    def __sub__(self, other: Self | _Scalar) -> Self: ...
    def __mul__(self, other: Self | _Scalar) -> Self: ...
    def __truediv__(self, other: Self | _Scalar) -> Self: ...
    def __neg__(self) -> Self: ...
    def __pow__(self, n: int) -> Self: ...
    def is_zero(self) -> bool: ...
    def sign(self) -> int: ...
    def enclose(self, width: Fraction) -> tuple[Fraction, Fraction]: ...
    def to_real(self) -> AlgebraicReal: ...


class _Field[E: _FieldElement](Protocol):
    """A real number field with exact arithmetic (newton.md, section 4.1).

    The operations step 5 needs: constants and the generator, arithmetic and exact
    zero and sign tests on the elements (:class:`_FieldElement`), rational
    enclosures, conversion to :class:`~flanpy.AlgebraicReal`, and the real roots of a
    polynomial over the field, each in an extension field with an embedding.
    """

    @property
    def degree(self) -> int: ...

    @property
    def generator(self) -> E: ...

    def __call__(self, value: E | _Scalar) -> E: ...

    def real_roots(self, phi: Sequence[E | _Scalar]) -> list[_RealRoot[E]]: ...


@dataclass(frozen=True, slots=True)
class _RealRoot[E]:
    """A real root ``value`` of a polynomial over ``K``, in a field ``L ⊇ K``.

    ``embedding`` maps the elements of ``K`` into ``field`` (``L``); it is the
    identity when the root lies in ``K`` already.
    """

    field: _Field[Any]
    embedding: Callable[[E], E]
    value: E
    multiplicity: int


# -- Q(θ) ------------------------------------------------------------------------------


class _NumberField:
    """The real number field ``Q(θ)``, ``θ`` an :class:`~flanpy.AlgebraicReal`.

    Elements are the polynomials in ``θ`` of degree ``< deg m_θ`` (``m_θ`` the minimal
    polynomial); arithmetic is modulo ``m_θ``, the inverse comes from the extended
    gcd. Two fields are equal when they have the same ``θ`` (the canonical pair of
    minimal polynomial and index) and the same backend. ``Q`` is the degree-1 field
    with ``θ = 0`` (:meth:`rationals`).

    Parameters
    ----------
    theta : AlgebraicReal
        The generator with its real embedding.
    backend : {"flint", "python"}, optional
        The ``Q[x]`` arithmetic (default: flint if installed).
    """

    __slots__ = ("_degree", "_m", "_m_fractions", "_qqx", "_theta")

    _theta: AlgebraicReal
    _qqx: _QQx
    _m: Any  # m_θ, native
    _m_fractions: tuple[Fraction, ...]
    _degree: int

    def __init__(
        self, theta: AlgebraicReal, *, backend: BackendName | None = None
    ) -> None:
        if not isinstance(theta, AlgebraicReal):
            raise TypeError(f"theta must be an AlgebraicReal, got {theta!r}")
        qqx = _qqx(backend)
        m = tuple(Fraction(int(c)) for c in reversed(theta.polynomial.all_coeffs()))
        _set = object.__setattr__
        _set(self, "_theta", theta)
        _set(self, "_qqx", qqx)
        _set(self, "_m_fractions", m)
        _set(self, "_m", qqx.from_fractions(m))
        _set(self, "_degree", len(m) - 1)

    @classmethod
    def rationals(cls, *, backend: BackendName | None = None) -> _NumberField:
        """Return ``Q`` as the degree-1 field ``Q(0)``."""
        x = sp.Symbol("x")
        return cls(AlgebraicReal(sp.Poly(x, x), 0), backend=backend)

    # -- structure -----------------------------------------------------------------

    @property
    def theta(self) -> AlgebraicReal:
        """The generator as an ``AlgebraicReal``."""
        return self._theta

    @property
    def degree(self) -> int:
        """``[Q(θ) : Q]``, the degree of ``m_θ``."""
        return self._degree

    @property
    def modulus(self) -> tuple[Fraction, ...]:
        """The coefficients of ``m_θ`` (integers), lowest degree first."""
        return self._m_fractions

    @property
    def backend(self) -> BackendName:
        """The ``Q[x]`` backend, ``"flint"`` or ``"python"``."""
        return self._qqx.name

    @property
    def generator(self) -> _Element:
        """``θ`` as an element (a rational for a degree-1 field)."""
        return self.from_coefficients([Fraction(0), Fraction(1)])

    def __call__(self, value: _Element | _Scalar) -> _Element:
        """Return ``value`` (an ``int``, a ``Fraction`` or an element) in this field."""
        if isinstance(value, _Element):
            if value._field != self:
                raise ValueError(f"{value!r} belongs to another field; embed it first")
            return value
        if isinstance(value, bool) or not isinstance(value, int | Fraction):
            raise TypeError(f"cannot convert {value!r} to a field element")
        return _Element._make(self, self._qqx.from_fractions([Fraction(value)]))

    def from_coefficients(self, cs: Sequence[_Scalar]) -> _Element:
        """Return ``sum cs[i] θ^i`` (any length; reduced modulo ``m_θ``)."""
        rep = self._qqx.from_fractions([Fraction(c) for c in cs]) % self._m
        return _Element._make(self, rep)

    # -- the real roots of a polynomial over the field -------------------------------

    def real_roots(
        self, phi: Sequence[_Element | _Scalar]
    ) -> list[_RealRoot[_Element]]:
        """Return the distinct real roots of ``phi`` with multiplicities, ascending.

        ``phi`` lists the coefficients of ``φ(c) = sum phi[i] c^i`` (lowest first)
        over this field ``K``; it must be non-zero. Each root comes as a
        :class:`_RealRoot`: a field ``L = Q(η) ⊇ K`` with one generator (the tower is
        flattened), the embedding ``K → L``, the root as an element of ``L`` and its
        multiplicity. A root that lies in ``K`` (rational roots included) keeps
        ``L = K``. The root ``0`` is included when ``φ(0) = 0``.

        Method (newton.md, section 4.2): the squarefree decomposition of ``φ`` over
        ``K`` (Yun); for each squarefree part ``P`` Trager's norm
        ``N_t(y) = Res_x(m_θ(x), P(x, y - t x))`` with the first shift ``t`` in
        ``0, 1, -1, 2, ...`` that makes it squarefree (Trager 1976). The irreducible
        factors ``f`` of ``N_t`` over ``Q`` give the factors ``gcd(P(c), f(c + tθ))``
        of ``P`` over ``K``. A linear factor is a root in ``K``. Otherwise, for each
        real root ``η`` of ``f``, the gcd of ``m_θ(x)`` and ``P(x, η - t x)`` over
        ``Q(η)`` is linear, ``x - R(η)``; ``η`` belongs to our ``θ`` exactly when
        ``R(η) = θ`` (decided by refining the enclosure of ``R(η)`` only: it is a root
        of ``m_θ`` and ``θ``'s isolating interval holds no other root). Then
        ``L = Q(η)``, ``θ = R(η)`` and the root is ``η - t R(η)``. Over ``K = Q`` the
        real roots of ``φ`` are isolated directly (no norm).

        Raises
        ------
        ValueError
            If ``phi`` is zero, or an element of ``phi`` belongs to another field.
        """
        coefficients = _trim([self(c) for c in phi])
        if not coefficients:
            raise ValueError("phi must be non-zero")
        if len(coefficients) == 1:
            return []
        if self._degree == 1:
            roots = self._rational_real_roots(coefficients)
        else:
            roots = []
            for multiplicity, part in _yun(coefficients):
                roots.extend(self._trager(part, multiplicity))
        return _sorted(roots)

    def _rational_real_roots(
        self, coefficients: Sequence[_Element]
    ) -> list[_RealRoot[_Element]]:
        """Real roots of ``φ`` over ``K = Q``: real-root isolation, no norm."""
        poly = _sympy_poly([c._rational() for c in coefficients], _X)
        _, parts = poly.sqf_list()
        out: list[_RealRoot[_Element]] = []
        for part, multiplicity in parts:
            for root in _real_roots(part):
                if root.polynomial.degree() == 1:
                    a, b = (Fraction(int(c)) for c in root.polynomial.all_coeffs())
                    out.append(_RealRoot(self, _identity, self(-b / a), multiplicity))
                    continue
                field = _NumberField(root, backend=self.backend)
                embedding = _Embedding(self, field, field(self._rational_theta()))
                out.append(_RealRoot(field, embedding, field.generator, multiplicity))
        return out

    def _rational_theta(self) -> Fraction:
        """``θ`` for a degree-1 field."""
        a, b = self._m_fractions[1], self._m_fractions[0]
        return -b / a

    def _trager(
        self, part: Sequence[_Element], multiplicity: int
    ) -> list[_RealRoot[_Element]]:
        """Real roots of the monic squarefree ``part`` over ``K`` (Trager 1976)."""
        if len(part) == 2:  # linear: the root lies in K
            return [_RealRoot(self, _identity, -part[0], multiplicity)]
        n, d = self._degree, len(part) - 1
        m_expr = _sympy_poly(self._m_fractions, _X).as_expr()
        columns = [c.coefficients for c in part]
        for t in _SHIFTS:
            shifted = sum(
                (
                    _sympy_poly(cs, _X).as_expr() * (_Y - t * _X) ** j
                    for j, cs in enumerate(columns)
                ),
                sp.Integer(0),
            )
            norm = sp.Poly(
                sp.resultant(m_expr, sp.expand(shifted), _X), _Y, domain=sp.QQ
            )
            if norm.degree() == n * d and norm.gcd(norm.diff(_Y)).degree() == 0:
                break
        else:  # pragma: no cover - only finitely many shifts fail
            raise RuntimeError(f"no squarefree norm for shifts {_SHIFTS}")
        out: list[_RealRoot[_Element]] = []
        theta = self.generator
        for factor, _ in norm.factor_list()[1]:
            # The factor of P over K that belongs to f: gcd(P(c), f(c + tθ)).
            f = [Fraction(int(c)) for c in reversed(_normalised(factor).all_coeffs())]
            f_shifted = _compose_linear(f, self, theta * t)
            local = _gcd(list(part), f_shifted)
            if len(local) == 2:  # a root in K: no new field
                out.append(_RealRoot(self, _identity, -local[0], multiplicity))
                continue
            for eta in _real_roots(sp.Poly.from_list(list(reversed(f)), _Y)):
                field = _NumberField(eta, backend=self.backend)
                image = self._theta_in(field, columns, t)
                if image is not None:
                    value = field.generator - image * t
                    embedding = _Embedding(self, field, image)
                    out.append(_RealRoot(field, embedding, value, multiplicity))
        return out

    def _theta_in(
        self, field: _NumberField, columns: Sequence[Sequence[Fraction]], t: int
    ) -> _Element | None:
        """``R(η)`` if it equals ``θ`` (see :meth:`real_roots`), else ``None``.

        ``R(η)`` is the root of the linear gcd of ``m_θ(x)`` and ``P(x, η - t x)``
        over ``field = Q(η)``.
        """
        eta = field.generator
        m = [field(c) for c in self._m_fractions]
        # P(x, η - t x) as a polynomial in x over Q(η).
        linear = [eta, field(-t)]  # η - t x
        total: list[_Element] = []
        power: list[_Element] = [field(1)]
        for cs in columns:
            total = _add(total, _mul([field(c) for c in cs], power))
            power = _mul(power, linear)
        g = _gcd(m, _trim(total))
        if len(g) != 2:  # pragma: no cover - the norm is squarefree
            raise RuntimeError("Trager's gcd is not linear")
        image = -g[0]
        return image if _is_value(image, self._theta) else None

    # -- value semantics --------------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (self._theta._key(), self._qqx.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _NumberField):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        return f"_NumberField({self._theta!r})"

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("_NumberField is immutable")


class _Element:
    """An element of a :class:`_NumberField`, immutable.

    Arithmetic with ``int``, ``Fraction`` and elements of the same field; ``==`` is
    exact also across fields (see :meth:`__eq__`), and so are ``<``, ``<=``, ``>``,
    ``>=``.
    """

    __slots__ = ("_field", "_real", "_rep")

    _field: _NumberField
    _rep: Any  # native polynomial in θ, reduced modulo m_θ
    _real: AlgebraicReal | None  # cache of to_real()

    @classmethod
    def _make(cls, field: _NumberField, rep: Any) -> _Element:
        new = object.__new__(cls)
        _set = object.__setattr__
        _set(new, "_field", field)
        _set(new, "_rep", rep)
        _set(new, "_real", None)
        return new

    @property
    def field(self) -> _NumberField:
        """The field of the element."""
        return self._field

    @property
    def coefficients(self) -> tuple[Fraction, ...]:
        """The element as ``sum c_i θ^i``: the ``c_i``, lowest first (``()`` for 0)."""
        return self._field._qqx.fractions(self._rep)

    # -- arithmetic --------------------------------------------------------------------

    def _other(self, other: object) -> Any:
        """Return ``other`` natively in this field, or ``None`` if not a scalar."""
        if isinstance(other, _Element):
            if other._field != self._field:
                raise ValueError("elements of different fields; embed one first")
            return other._rep
        if isinstance(other, bool) or not isinstance(other, int | Fraction):
            return None
        return self._field._qqx.from_fractions([Fraction(other)])

    def _new(self, rep: Any) -> _Element:
        return _Element._make(self._field, rep % self._field._m)

    def __add__(self, other: _Element | _Scalar) -> _Element:
        o = self._other(other)
        return NotImplemented if o is None else self._new(self._rep + o)

    __radd__ = __add__

    def __sub__(self, other: _Element | _Scalar) -> _Element:
        o = self._other(other)
        return NotImplemented if o is None else self._new(self._rep - o)

    def __rsub__(self, other: _Element | _Scalar) -> _Element:
        o = self._other(other)
        return NotImplemented if o is None else self._new(o - self._rep)

    def __mul__(self, other: _Element | _Scalar) -> _Element:
        o = self._other(other)
        return NotImplemented if o is None else self._new(self._rep * o)

    __rmul__ = __mul__

    def __neg__(self) -> _Element:
        return self._new(-self._rep)

    def inverse(self) -> _Element:
        """``1/x``, from the extended gcd with ``m_θ``.

        Raises
        ------
        ZeroDivisionError
            If the element is zero.
        """
        if self.is_zero():
            raise ZeroDivisionError("inverse of zero in a number field")
        qqx = self._field._qqx
        g, s = qqx.xgcd(self._rep, self._field._m)
        (g0,) = qqx.fractions(g)  # m_θ irreducible: the gcd is a constant
        return self._new(s * qqx.from_fractions([1 / g0]))

    def __truediv__(self, other: _Element | _Scalar) -> _Element:
        o = self._other(other)
        if o is None:
            return NotImplemented
        return self * _Element._make(self._field, o).inverse()

    def __rtruediv__(self, other: _Element | _Scalar) -> _Element:
        o = self._other(other)
        if o is None:
            return NotImplemented
        return _Element._make(self._field, o) * self.inverse()

    def __pow__(self, n: int) -> _Element:
        if isinstance(n, bool) or not isinstance(n, int):
            return NotImplemented
        base = self if n >= 0 else self.inverse()
        result = self._field(1)
        for bit in bin(abs(n))[2:]:
            result = result * result
            if bit == "1":
                result = result * base
        return result

    # -- exact tests -------------------------------------------------------------------

    def is_zero(self) -> bool:
        """Exact: the reduced representation is the zero polynomial."""
        return not self.coefficients

    def _rational(self) -> Fraction:
        """Return the value of a rational element (degree ``<= 0`` in ``θ``)."""
        cs = self.coefficients
        if len(cs) > 1:
            raise ValueError(f"{self!r} is not rational")
        return cs[0] if cs else Fraction(0)

    def is_rational(self) -> bool:
        """Exact: whether the element is a rational number."""
        return len(self.coefficients) <= 1

    def sign(self) -> int:
        """Exact sign ``-1``, ``0`` or ``1``, via ``θ.sign`` (step 4)."""
        cs = self.coefficients
        if len(cs) <= 1:
            v = cs[0] if cs else Fraction(0)
            return (v > 0) - (v < 0)
        return self._field._theta.sign(_sympy_poly(cs, _X))

    def _enclosure(self) -> tuple[Fraction, Fraction]:
        """Return an enclosure at ``θ``'s current precision (no refinement)."""
        lo, hi = self._field._theta.interval
        return _horner_interval(self.coefficients, lo, hi)

    def enclose(self, width: Fraction) -> tuple[Fraction, Fraction]:
        """Return rationals ``lo <= x <= hi`` with ``hi - lo <= width``.

        Refines ``θ`` (interval Horner evaluation on ``θ``'s isolating interval).

        Raises
        ------
        ValueError
            If ``width <= 0``.
        """
        width = Fraction(width)
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        while True:
            lo, hi = self._enclosure()
            if hi - lo <= width:
                return lo, hi
            self._field._theta._shrink()

    def to_real(self) -> AlgebraicReal:
        """Return the element as an :class:`~flanpy.AlgebraicReal`.

        The characteristic polynomial of the multiplication by ``x`` on ``Q(θ)`` is
        ``m_x^{[Q(θ) : Q(x)]}`` (``m_x`` the minimal polynomial of ``x``), so ``m_x``
        is its squarefree part; the root is identified by refining the enclosure of
        ``x`` until it holds exactly one real root of ``m_x``. Only the enclosure of
        ``x`` is refined (newton.md, section 4.2). The result is cached.
        """
        if self._real is not None:
            return self._real
        cs = self.coefficients
        if len(cs) <= 1:
            q = cs[0] if cs else Fraction(0)
            m = _normalised(sp.Poly(q.denominator * _Y - q.numerator, _Y, domain=sp.QQ))
            real = AlgebraicReal._make(m, 0, (q, q))
        else:
            real = self._real_from_norm()
        object.__setattr__(self, "_real", real)
        return real

    def _real_from_norm(self) -> AlgebraicReal:
        field = self._field
        qqx = field._qqx
        n = field._degree
        x = qqx.from_fractions([Fraction(0), Fraction(1)])
        columns = []
        column = self._rep
        for _ in range(n):  # column j: the coordinates of x θ^j
            cs = qqx.fractions(column)
            columns.append(list(cs) + [Fraction(0)] * (n - len(cs)))
            column = (column * x) % field._m
        rows = [[columns[j][i] for j in range(n)] for i in range(n)]
        chi = _sympy_poly(qqx.charpoly(rows), _Y)
        minimal = _normalised(sp.Poly(chi.quo(chi.gcd(chi.diff(_Y))), _Y))
        while True:
            lo, hi = self._enclosure()
            if minimal.count_roots(_rational(lo), _rational(hi)) == 1:
                break
            field._theta._shrink()
        index = int(minimal.count_roots(None, _rational(hi))) - 1
        return AlgebraicReal._make(minimal, index, _intervals(minimal)[index])

    # -- value semantics ---------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Exact equality, also between elements of different fields.

        In one field: the reduced representations are equal. Across fields: the
        two :meth:`to_real` values (canonical minimal polynomial and index) are
        equal. A common field ``Q(θ_1, θ_2)`` would need a primitive element, a
        Trager norm of degree up to ``d_1 d_2``, for every comparison; ``to_real``
        costs one characteristic polynomial of degree ``d`` per element, cached.
        """
        if isinstance(other, _Element):
            if other._field == self._field:
                return self.coefficients == other.coefficients
            return self.to_real() == other.to_real()
        if isinstance(other, bool) or not isinstance(other, int | Fraction):
            return NotImplemented
        return self.is_rational() and self._rational() == other

    def __hash__(self) -> int:
        return hash(self.to_real())  # consistent with == across fields

    def _compare(self, other: _Element | _Scalar) -> int:
        """``sign(self - other)``, exact, also across fields."""
        if not isinstance(other, _Element):
            return (self - other).sign()
        if other._field == self._field:
            return (self - other).sign()
        if self == other:
            return 0
        while True:  # distinct reals: refine both enclosures until they separate
            (a, b), (c, d) = self._enclosure(), other._enclosure()
            if b < c:
                return -1
            if d < a:
                return 1
            self._field._theta._shrink()
            other._field._theta._shrink()

    def __lt__(self, other: _Element | _Scalar) -> bool:
        return self._compare(other) < 0

    def __le__(self, other: _Element | _Scalar) -> bool:
        return self._compare(other) <= 0

    def __gt__(self, other: _Element | _Scalar) -> bool:
        return self._compare(other) > 0

    def __ge__(self, other: _Element | _Scalar) -> bool:
        return self._compare(other) >= 0

    def __repr__(self) -> str:
        cs = self.coefficients
        terms = [
            f"{c}" if i == 0 else f"{c}*θ" if i == 1 else f"{c}*θ**{i}"
            for i, c in enumerate(cs)
            if c
        ]
        return f"_Element({' + '.join(terms) or '0'})"

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("_Element is immutable")


class _Embedding:
    """The embedding ``K → L`` that sends the generator of ``K`` to ``image``."""

    __slots__ = ("image", "source", "target")

    def __init__(
        self, source: _NumberField, target: _NumberField, image: _Element
    ) -> None:
        self.source, self.target, self.image = source, target, image

    def __call__(self, x: _Element) -> _Element:
        """Return ``x`` (an element of ``K``) in ``L``, by Horner in ``image``."""
        if x.field != self.source:
            raise ValueError(f"{x!r} is not in the source field")
        result = self.target(0)
        for c in reversed(x.coefficients):
            result = result * self.image + c
        return result

    def __repr__(self) -> str:
        return f"_Embedding({self.source!r} -> {self.target!r}, θ -> {self.image!r})"


def _identity(x: _Element) -> _Element:
    """Embed a field into itself (the identity)."""
    return x


# -- polynomials over a field: coefficient lists, lowest first -------------------------


def _trim(p: list[_Element]) -> list[_Element]:
    while p and p[-1].is_zero():
        p = p[:-1]
    return p


def _add(a: Sequence[_Element], b: Sequence[_Element]) -> list[_Element]:
    if len(a) < len(b):
        a, b = b, a
    return _trim([a[i] + b[i] if i < len(b) else a[i] for i in range(len(a))])


def _mul(a: Sequence[_Element], b: Sequence[_Element]) -> list[_Element]:
    if not a or not b:
        return []
    out = [a[0].field(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x.is_zero():
            continue
        for j, y in enumerate(b):
            out[i + j] = out[i + j] + x * y
    return _trim(out)


def _divmod(
    a: Sequence[_Element], b: Sequence[_Element]
) -> tuple[list[_Element], list[_Element]]:
    """Division with remainder over the field (``b`` non-zero, trimmed)."""
    r = list(a)
    if len(r) < len(b):
        return [], _trim(r)
    q = [b[-1].field(0)] * (len(r) - len(b) + 1)
    inv = b[-1].inverse()
    while r and len(r) >= len(b):
        c = r[-1] * inv
        shift = len(r) - len(b)
        q[shift] = c
        for i, y in enumerate(b[:-1]):
            r[shift + i] = r[shift + i] - c * y
        r = _trim(r[:-1])  # the leading term cancels exactly
    return _trim(q), r


def _monic(a: Sequence[_Element]) -> list[_Element]:
    inv = a[-1].inverse()
    return [c * inv for c in a]


def _gcd(a: Sequence[_Element], b: Sequence[_Element]) -> list[_Element]:
    """Return the monic gcd of two polynomials over the field (not both zero)."""
    a, b = _trim(list(a)), _trim(list(b))
    while b:
        a, b = b, _divmod(a, b)[1]
    return _monic(a)


def _diff(a: Sequence[_Element]) -> list[_Element]:
    return _trim([c * i for i, c in enumerate(a)][1:])


def _yun(p: Sequence[_Element]) -> list[tuple[int, list[_Element]]]:
    """Squarefree decomposition over the field: ``p = lc · prod P_i^i`` (Yun 1976).

    Returns the pairs ``(i, P_i)`` with ``P_i`` monic of degree ``>= 1``.
    """
    f = _monic(p)
    g = _gcd(f, _diff(f))
    b = _divmod(f, g)[0]
    c = _divmod(_diff(f), g)[0]
    d = _trim(_add(c, [-x for x in _diff(b)]))
    out = []
    i = 1
    while len(b) > 1:
        a = _gcd(b, d) if d else _monic(b)
        if len(a) > 1:
            out.append((i, a))
        b = _divmod(b, a)[0]
        c = _divmod(d, a)[0] if d else []
        d = _trim(_add(c, [-x for x in _diff(b)]))
        i += 1
    return out


def _compose_linear(
    f: Sequence[Fraction], field: _NumberField, shift: _Element
) -> list[_Element]:
    """``f(c + shift)`` as a polynomial in ``c`` over the field (Horner)."""
    result: list[_Element] = []
    linear = [shift, field(1)]
    for coefficient in reversed(f):
        result = _add(_mul(result, linear), [field(coefficient)])
    return result


def _is_value(x: _Element, target: AlgebraicReal) -> bool:
    """Whether ``x == target``, given that ``x`` is a root of ``target.polynomial``.

    Exact. Only the enclosure of ``x`` is refined: ``target``'s isolating interval
    holds exactly one root of its minimal polynomial, so ``x`` equals ``target`` iff
    an enclosure of ``x`` falls inside that interval, and differs iff one falls
    outside. Refining both in lockstep can loop forever, since the enclosure of ``x``
    may stay wider than ``target``'s interval by a constant factor (newton.md,
    section 4.2, the pitfall found by the spike).
    """
    t_lo, t_hi = target.interval
    if t_lo == t_hi:  # a rational target
        return x.is_rational() and x._rational() == t_lo
    while True:
        lo, hi = x._enclosure()
        if t_lo <= lo and hi <= t_hi:
            return True
        if hi < t_lo or t_hi < lo:
            return False
        x.field.theta._shrink()


def _sorted(roots: list[_RealRoot[_Element]]) -> list[_RealRoot[_Element]]:
    """Sort roots (distinct reals, possibly in different fields) increasingly."""
    while True:
        roots.sort(key=lambda r: r.value._enclosure()[0])
        clash = [
            (r, s)
            for r, s in itertools.pairwise(roots)
            if not r.value._enclosure()[1] < s.value._enclosure()[0]
        ]
        if not clash:
            return roots
        for r, s in clash:
            r.value.field.theta._shrink()
            s.value.field.theta._shrink()
