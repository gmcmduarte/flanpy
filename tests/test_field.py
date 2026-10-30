# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the real number fields of step 5 (newton.md, section 4)."""

from __future__ import annotations

import itertools
import signal
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction

import mpmath
import pytest
import sympy as sp
from hypothesis import assume, given
from hypothesis import strategies as st

from flanpy import AlgebraicReal
from flanpy._field import (
    _Element,
    _Embedding,
    _Field,
    _is_value,
    _NumberField,
    _RealRoot,
)
from flanpy._rings import BackendName, flint_available

x, y, c = sp.symbols("x y c")

BACKENDS: list[BackendName] = ["python", "flint"]
Scalar = _Element | int | Fraction


@pytest.fixture(params=BACKENDS)
def be(request: pytest.FixtureRequest) -> BackendName:
    name: BackendName = request.param
    if name == "flint" and not flint_available():
        pytest.skip("python-flint is not installed")
    return name


def sqrt2_field(be: BackendName) -> _NumberField:
    return _NumberField(AlgebraicReal(x**2 - 2, 1), backend=be)


# The S5 tree of newton.md, section 4.3: m_rho has Galois group S_5.
QUINTIC = 3233 * x**5 - 2250 * x**4 - 27 * x**3 + 1600 * x**2 - 256


def value(e: _Element, dps: int = 50) -> mpmath.mpf:
    """The element evaluated with mpmath (independent of to_real)."""
    with mpmath.workdps(dps + 10):
        theta = e.field.theta.evalf(dps + 10)
        return sum(
            (mpmath.mpf(q.numerator) / q.denominator * theta**i)
            for i, q in enumerate(e.coefficients)
        )


def close(a: mpmath.mpf, b: mpmath.mpf, digits: int = 40) -> bool:
    return bool(abs(a - b) <= mpmath.mpf(10) ** -digits * (1 + abs(b)))


@contextmanager
def time_limit(seconds: int) -> Iterator[None]:
    """Fail instead of hanging (Unix: SIGALRM)."""

    def handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"took longer than {seconds} s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# -- the field and its elements -----------------------------------------------------


def test_protocol_conformance() -> None:
    field: _Field[_Element] = _NumberField.rationals()  # checked by mypy
    assert field.degree == 1


def test_sqrt2_arithmetic(be: BackendName) -> None:
    K = sqrt2_field(be)
    s = K.generator
    assert K.degree == 2 and K.backend == be
    assert s * s == 2 and s.coefficients == (Fraction(0), Fraction(1))
    assert (1 + s).inverse() == s - 1  # 1/(1 + √2) = √2 - 1
    assert (1 + s) ** -1 == s - 1 and s**0 == 1 and s**5 == 4 * s
    assert 1 / s == s / 2 and (s + 1) / (s - 1) == 3 + 2 * s
    assert 3 - s == -(s - 3) and Fraction(1, 2) * s * 2 == s
    assert K(0).is_zero() and not s.is_zero() and (s - s).is_zero()
    assert K(Fraction(3, 4)).is_rational() and not s.is_rational()
    assert s.sign() == 1 and (-s).sign() == -1 and K(0).sign() == 0
    assert (s - Fraction(14142, 10000)).sign() == 1
    assert (s - Fraction(14143, 10000)).sign() == -1
    assert repr(1 + 2 * s) == "_Element(1 + 2*θ)"
    assert repr(K) == "_NumberField(AlgebraicReal(x**2 - 2, 1))"


def test_errors(be: BackendName) -> None:
    K = sqrt2_field(be)
    Q = _NumberField.rationals(backend=be)
    with pytest.raises(ZeroDivisionError):
        K(0).inverse()
    with pytest.raises(ZeroDivisionError):
        K.generator / 0
    with pytest.raises(ValueError, match="different fields"):
        _ = K.generator + Q(1)
    with pytest.raises(ValueError, match="another field"):
        K(Q(1))
    with pytest.raises(TypeError):
        K(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _NumberField(sp.sqrt(2))
    with pytest.raises(ValueError, match="non-zero"):
        K.real_roots([0, K(0)])
    with pytest.raises(ValueError, match="positive"):
        K.generator.enclose(Fraction(0))
    with pytest.raises(AttributeError):
        K.generator._rep = None
    with pytest.raises(AttributeError):
        K._degree = 3


def test_value_semantics(be: BackendName) -> None:
    K = sqrt2_field(be)
    K2 = _NumberField(AlgebraicReal(2 * x**2 - 4, 1), backend=be)  # the same θ
    assert K == K2 and hash(K) == hash(K2)
    assert _NumberField(AlgebraicReal(x**2 - 2, 0), backend=be) != K
    assert K.generator == K2.generator
    assert hash(K(Fraction(1, 3))) == hash(Fraction(1, 3))
    assert K(Fraction(1, 3)) == Fraction(1, 3) and K(2) == 2 and K.generator != 2
    other: BackendName = "flint" if be == "python" else "python"
    if flint_available():
        assert _NumberField(AlgebraicReal(x**2 - 2, 1), backend=other) != K


def test_quintic_field(be: BackendName) -> None:
    theta = AlgebraicReal(QUINTIC, 0)  # its only real root: ρ = 0.4230...
    K = _NumberField(theta, backend=be)
    t = K.generator
    assert sp.Poly(QUINTIC).degree() == 5 and K.degree == 5
    assert K.from_coefficients([-256, 0, 1600, -27, -2250, 3233]).is_zero()  # m(θ) = 0
    e = 3 * t**4 - t + Fraction(1, 7)
    assert e * e.inverse() == 1
    with mpmath.workdps(60):
        assert close(value(e.inverse()), 1 / value(e))
    real = e.to_real()
    assert real.polynomial.degree() == 5  # Q(e) = Q(θ): the field has prime degree
    assert close(real.evalf(45), value(e))
    assert sp.galois_group(sp.Poly(QUINTIC, x))[0].order() == 120


def test_rationals(be: BackendName) -> None:
    Q = _NumberField.rationals(backend=be)
    assert Q.degree == 1 and Q.generator == 0
    q = Q(Fraction(2, 3))
    assert q * q.inverse() == 1 and q.to_real() == Fraction(2, 3)
    assert q.enclose(Fraction(1, 10)) == (Fraction(2, 3), Fraction(2, 3))
    # A degree-1 field with a non-zero generator (Catalan's ρ = 1/4).
    R = _NumberField(AlgebraicReal(4 * x - 1, 0), backend=be)
    assert R.generator == Fraction(1, 4) and R.generator.is_rational()


def test_enclose_and_to_real(be: BackendName) -> None:
    K = sqrt2_field(be)
    e = (1 + K.generator) ** 3
    lo, hi = e.enclose(Fraction(1, 10**30))
    assert (
        hi - lo <= Fraction(1, 10**30) and lo**2 - 14 * lo - 1 < 0 < hi**2 - 14 * hi - 1
    )
    assert e.to_real() == AlgebraicReal(y**2 - 14 * y - 1, 1)
    assert (-e).to_real() == AlgebraicReal(y**2 + 14 * y - 1, 0)
    assert K.generator.to_real() == AlgebraicReal(x**2 - 2, 1)


def test_equality_and_order_across_fields(be: BackendName) -> None:
    K = sqrt2_field(be)
    (root,) = [r for r in K.real_roots([-3, 0, 1]) if r.value > 0]  # √3 in Q(√2, √3)
    L, sqrt3 = root.field, root.value
    assert L.degree == 4
    sqrt2_in_L = root.embedding(K.generator)
    assert sqrt2_in_L == K.generator and hash(sqrt2_in_L) == hash(K.generator)
    assert sqrt3 > K.generator and K.generator < sqrt3 and K.generator <= sqrt2_in_L
    assert sqrt3 != K.generator and sqrt3 * sqrt3 == 3
    assert (sqrt3 * sqrt2_in_L).to_real() == AlgebraicReal(y**2 - 6, 1)


# -- real roots over the field --------------------------------------------------------


def check_roots(
    K: _NumberField, phi: Sequence[Scalar], roots: list[_RealRoot[_Element]]
) -> None:
    """Every root is exact (φ(c) = 0 in L), roots ascend, embeddings fix θ."""
    for r in roots:
        L = r.field
        assert isinstance(L, _NumberField)
        coefficients = [r.embedding(K(q)) for q in phi]
        total = L(0)
        for q in reversed(coefficients):
            total = total * r.value + q
        assert total.is_zero()
        assert r.embedding(K.generator) == K.generator
    values = [r.value for r in roots]
    assert all(a < b for a, b in itertools.pairwise(values))


def test_s5_over_q(be: BackendName) -> None:
    """c^10 - c^2 - 1 = psi(c^2), psi(u) = u^5 - u - 1 (Galois group S_5)."""
    Q = _NumberField.rationals(backend=be)
    phi: list[Scalar] = [-1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1]
    roots = Q.real_roots(phi)
    assert len(roots) == 2 and [r.multiplicity for r in roots] == [1, 1]
    for r in roots:
        assert r.field.degree == 10
        assert r.value.to_real().polynomial == sp.Poly(y**10 - y**2 - 1, y)
    assert roots[0].value == -roots[1].value
    assert sp.galois_group(sp.Poly(x**5 - x - 1, x))[0].order() == 120
    check_roots(Q, phi, roots)


def test_degree_20_over_sqrt2(be: BackendName) -> None:
    """c^10 - √2 c^2 - 1 over Q(√2): two real roots of degree 20 (spike `field`)."""
    K = sqrt2_field(be)
    s = K.generator
    phi: list[Scalar] = [K(-1), K(0), -s, *([K(0)] * 7), K(1)]
    roots = K.real_roots(phi)
    assert len(roots) == 2
    minimal = sp.Poly(y**20 - 2 * y**10 - 2 * y**4 + 1, y)
    for r in roots:
        assert r.field.degree == 20
        assert r.value.to_real().polynomial == minimal
    assert close(roots[1].value.to_real().evalf(30), mpmath.mpf("1.10559127187222"), 13)
    check_roots(K, phi, roots)


def test_repeated_rational_and_zero_roots(be: BackendName) -> None:
    Q = _NumberField.rationals(backend=be)
    # c (c - 1/2)^2 (c^2 - 2)^3 (c^2 + 1)
    expr = sp.expand(c * (c - sp.Rational(1, 2)) ** 2 * (c**2 - 2) ** 3 * (c**2 + 1))
    phi: list[Scalar] = [
        Fraction(int(a.p), int(a.q)) for a in reversed(sp.Poly(expr, c).all_coeffs())
    ]
    roots = Q.real_roots(phi)
    got = [(r.value.to_real(), r.multiplicity, r.field.degree) for r in roots]
    assert got == [
        (AlgebraicReal(y**2 - 2, 0), 3, 2),
        (AlgebraicReal(y, 0), 1, 1),
        (AlgebraicReal(2 * y - 1, 0), 2, 1),
        (AlgebraicReal(y**2 - 2, 1), 3, 2),
    ]
    assert roots[1].field == Q and roots[2].field == Q  # rational roots: no new field
    check_roots(Q, phi, roots)
    # The same over Q(√3): the rational roots and the root 0 stay in K.
    K = _NumberField(AlgebraicReal(x**2 - 3, 1), backend=be)
    roots = K.real_roots(phi)
    assert [r.multiplicity for r in roots] == [3, 1, 2, 3]
    assert [r.field == K for r in roots] == [False, True, True, False]
    assert roots[1].value == 0 and roots[2].value == Fraction(1, 2)
    check_roots(K, phi, roots)


def test_roots_already_in_k(be: BackendName) -> None:
    """c^2 - 2 over Q(√2): L = K, no degree doubling."""
    K = sqrt2_field(be)
    roots = K.real_roots([-2, 0, 1])
    assert [r.field for r in roots] == [K, K]
    assert [r.value for r in roots] == [-K.generator, K.generator]
    # Repeated, in K: (c - √2)^2 (c + 1) = c^3 + (1 - 2s) c^2 + (s^2 - 2s) c + s^2
    s = K.generator
    phi = [s * s, s * s - 2 * s, 1 - 2 * s, K(1)]
    roots = K.real_roots(phi)
    assert [(r.value, r.multiplicity, r.field == K) for r in roots] == [
        (K(-1), 1, True),
        (s, 2, True),
    ]
    assert K.real_roots([1, 0, 1]) == []  # c^2 + 1: no real root
    assert K.real_roots([K(5)]) == []  # a non-zero constant


def test_new_field_embedding(be: BackendName) -> None:
    """c^2 - (1 + √2) over Q(√2): roots ±√(1 + √2), degree 4; θ ↦ c^2 - 1."""
    K = sqrt2_field(be)
    s = K.generator
    phi = [-(1 + s), K(0), K(1)]
    roots = K.real_roots(phi)
    assert len(roots) == 2
    for r in roots:
        assert r.field.degree == 4
        assert isinstance(r.embedding, _Embedding)
        assert r.embedding(s) == r.value * r.value - 1
    with mpmath.workdps(50):
        expected = mpmath.sqrt(1 + mpmath.sqrt(2))
    assert close(roots[1].value.to_real().evalf(45), expected)
    check_roots(K, phi, roots)


# -- the enclosure pitfall (newton.md, section 4.2) ------------------------------------


def test_is_value_refines_the_element_only(be: BackendName) -> None:
    """R_2: a = (1 - ρ)/(4ρ) = √6/2 in Q(ρ), ρ the positive root of 23z^2 + 2z - 1.

    With the target's interval as narrow as ρ's, refining both in lockstep loops
    forever (the spike's bug); refining the element only terminates at once.
    """
    rho = AlgebraicReal(23 * x**2 + 2 * x - 1, 1).refine(Fraction(1, 10**40))
    target = AlgebraicReal(2 * y**2 - 3, 1).refine(Fraction(1, 10**40))
    K = _NumberField(rho, backend=be)
    a = (1 - K.generator) / (4 * K.generator)
    with time_limit(20):
        assert _is_value(a, target)
        assert not _is_value(-a, target)
        assert a.to_real() == target


# -- properties ---------------------------------------------------------------------


@st.composite
def fields_and_elements(draw: st.DrawFn) -> tuple[_NumberField, list[int]]:
    degree = draw(st.integers(2, 4))
    cs = draw(st.lists(st.integers(-5, 5), min_size=degree, max_size=degree))
    m = sp.Poly([1, *cs], x)
    assume(m.is_irreducible and m.count_roots() > 0)
    index = draw(st.integers(0, int(m.count_roots()) - 1))
    element = draw(st.lists(st.integers(-4, 4), min_size=1, max_size=degree))
    return _NumberField(AlgebraicReal(m, index)), element


@given(fields_and_elements())
def test_random_elements(data: tuple[_NumberField, list[int]]) -> None:
    K, cs = data
    e = K.from_coefficients(cs)
    assume(not e.is_zero())
    assert e * e.inverse() == 1
    v = value(e, 40)
    assert close(e.to_real().evalf(35), v, 30)
    assert e.sign() == (1 if v > 0 else -1)
    lo, hi = e.enclose(Fraction(1, 10**20))
    assert hi - lo <= Fraction(1, 10**20)
    with mpmath.workdps(50):
        assert mpmath.mpf(lo.numerator) / lo.denominator <= v + mpmath.mpf(10) ** -40
        assert v - mpmath.mpf(10) ** -40 <= mpmath.mpf(hi.numerator) / hi.denominator
