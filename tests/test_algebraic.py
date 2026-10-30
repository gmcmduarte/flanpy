# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.AlgebraicReal and the root helpers of flanpy._algebraic.

Design: dev/design/singularity.md, sections 2, 5.1 and 8 ("Algebraic reals").
"""

import itertools
import pickle
from fractions import Fraction
from typing import Any

import mpmath
import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

import flanpy
from flanpy._algebraic import (
    AlgebraicReal,
    _negative,
    _others_on_circle,
    _positive_roots,
    _real_roots,
    _reciprocal,
    _separators,
    _square,
)

z, w, k = sp.symbols("z w k")
SQRT2 = AlgebraicReal(z**2 - 2, 1)

# Broda et al. 2020: p_k (eq. (18)) and the second factor q_k of disc_L G
# (singularity.md, section 3.2).
P_K = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
Q_K = (
    (2 * k**3 + 36 * k**2 + 108 * k) * z**3
    + (12 * k**2 + 72 * k) * z**2
    + 15 * k * z
    - 2
)


def mp(x: AlgebraicReal, dps: int = 60) -> mpmath.mpf:
    with mpmath.workdps(dps):
        return x.evalf(dps)


# -- public API ------------------------------------------------------------------------


def test_exported() -> None:
    assert flanpy.AlgebraicReal is AlgebraicReal
    assert "AlgebraicReal" in flanpy.__all__
    assert AlgebraicReal.__module__ == "flanpy"


def test_construction_and_attributes() -> None:
    rho = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
    assert rho.polynomial == sp.Poly(23 * z**2 + 2 * z - 1, z, domain="ZZ")
    assert rho.index == 1
    lo, hi = rho.interval
    assert isinstance(lo, Fraction) and isinstance(hi, Fraction)
    assert lo <= hi
    # also from a Poly, over QQ with integer coefficients
    assert AlgebraicReal(sp.Poly(z**2 - 2, z, domain="QQ"), 1) == SQRT2


@pytest.mark.parametrize(
    ("polynomial", "index", "error", "message"),
    [
        (z**2 - 1, 0, ValueError, r"not irreducible over Q: factors z - 1, z \+ 1"),
        (z / 2 - 1, 0, ValueError, "integer coefficients; use z - 2"),
        (
            2 * z - 2,
            0,
            ValueError,
            "primitive with a positive leading coefficient; use z - 1",
        ),
        (1 - 4 * z, 0, ValueError, "use 4\\*z - 1"),
        (z**2 + 1, 0, ValueError, "has 0 real root\\(s\\); index 0 is out of range"),
        (z**2 - 2, 2, ValueError, "index 2 is out of range"),
        (z**2 - 2, -1, ValueError, "index -1 is out of range"),
        (sp.Integer(3), 0, ValueError, "degree >= 1"),
        (z * w - 1, 0, ValueError, "univariate"),
        (sp.sqrt(2) * z - 1, 0, ValueError, "rational coefficients"),
        (1 / z, 0, ValueError, "polynomial"),
        (z**2 - 2, True, TypeError, "index must be an int"),
        (z**2 - 2, 1.0, TypeError, "index must be an int"),
        ("not a polynomial (", 0, TypeError, "must be a polynomial"),
    ],
)
def test_validation(
    polynomial: object, index: object, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        AlgebraicReal(polynomial, index)  # type: ignore[arg-type]


def test_value_semantics() -> None:
    a = AlgebraicReal(z**2 - 2, 1)
    assert a == SQRT2 and hash(a) == hash(SQRT2)
    assert AlgebraicReal(w**2 - 2, 1) == SQRT2  # the variable's name does not matter
    assert hash(AlgebraicReal(w**2 - 2, 1)) == hash(SQRT2)
    assert AlgebraicReal(z**2 - 2, 0) != SQRT2
    # the cache does not affect equality or hashing
    h = hash(a)
    a.refine(Fraction(1, 10**30))
    assert a == SQRT2 and hash(a) == h
    # rational numbers equal (and hash like) the equal Fraction / int
    quarter = AlgebraicReal(4 * z - 1, 0)
    assert quarter == Fraction(1, 4) and hash(quarter) == hash(Fraction(1, 4))
    two = AlgebraicReal(z - 2, 0)
    assert two == 2 and 2 == two and hash(two) == hash(2)  # noqa: SIM300 (reflected)
    assert {quarter, Fraction(1, 4)} == {Fraction(1, 4)}
    assert SQRT2 != 1.4142135623730951  # floats are never equal (not exact)


def test_immutable_and_picklable() -> None:
    a = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
    with pytest.raises(AttributeError, match="immutable"):
        a._index = 0
    with pytest.raises(AttributeError, match="immutable"):
        del a._poly
    a.refine(Fraction(1, 10**12))
    b = pickle.loads(pickle.dumps(a))
    assert b == a and hash(b) == hash(a)
    assert b.interval == a.interval and b.polynomial == a.polynomial


def test_repr_and_str() -> None:
    a = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
    assert repr(a) == "AlgebraicReal(23*z**2 + 2*z - 1, 1)"
    assert eval(repr(a), {"AlgebraicReal": AlgebraicReal, "z": z}) == a
    assert str(a) == "0.169520847198537 (root 1 of 23*z**2 + 2*z - 1)"
    assert str(AlgebraicReal(4 * z - 1, 0)) == "0.25 (root 0 of 4*z - 1)"


def test_refine() -> None:
    a = AlgebraicReal(z**3 - 2, 0)
    assert a.refine(Fraction(1, 10**20)) is a
    lo, hi = a.interval
    assert hi - lo <= Fraction(1, 10**20) and lo**3 <= 2 <= hi**3
    assert AlgebraicReal(4 * z - 1, 0).refine(1).interval == (Fraction(1, 4),) * 2
    with pytest.raises(ValueError, match="positive"):
        a.refine(0)
    with pytest.raises(TypeError, match="int or a Fraction"):
        a.refine(0.5)  # type: ignore[arg-type]


def test_sign_exact() -> None:
    # zero exactly when the minimal polynomial divides p
    assert SQRT2.sign((z**2 - 2) * (z**5 + 3 * z - 7)) == 0
    assert SQRT2.sign(w**2 - 2) == 0  # p's variable is ignored
    # close but non-zero: sqrt 2 vs its 30-digit truncation
    t = 14142135623730950488016887242096
    assert SQRT2.sign(10**31 * z - t) == 1
    assert SQRT2.sign(10**31 * z - (t + 1)) == -1
    # constants
    assert (
        SQRT2.sign(0) == 0 and SQRT2.sign(-3) == -1 and SQRT2.sign(Fraction(1, 7)) == 1
    )
    # Broda eq. (18) at k = 54: rho_54 = 2/45 exactly, a simple root
    p54 = P_K.subs(k, 54)
    rho54 = AlgebraicReal(45 * z - 2, 0)
    assert rho54.sign(p54) == 0
    assert rho54.sign(sp.diff(p54, z)) == 1
    with pytest.raises(ValueError, match="univariate"):
        SQRT2.sign(z * w)


def test_comparisons() -> None:
    cbrt3 = AlgebraicReal(z**3 - 3, 0)
    assert SQRT2 < cbrt3 and cbrt3 > SQRT2  # noqa: SIM300 (both operators)
    assert SQRT2 <= SQRT2 and SQRT2 >= SQRT2
    # two irrationals within 1e-30 of each other
    near = AlgebraicReal(10**30 * z**2 - (2 * 10**30 + 1), 1)
    assert SQRT2 < near and not near < SQRT2 and near != SQRT2  # noqa: SIM300
    # against int and Fraction, in both directions
    assert 1 < SQRT2 < 2 and Fraction(141, 100) < SQRT2 < Fraction(142, 100)
    assert AlgebraicReal(4 * z - 1, 0) <= Fraction(1, 4) <= AlgebraicReal(4 * z - 1, 0)
    mixed: list[Any] = [cbrt3, Fraction(3, 2), SQRT2, 1]
    assert sorted(mixed) == [1, SQRT2, cbrt3, Fraction(3, 2)]
    with pytest.raises(TypeError, match="exactly"):
        _ = SQRT2 < 1.5
    with pytest.raises(TypeError):
        _ = SQRT2 < "x"


def test_evalf_against_mpmath() -> None:
    with mpmath.workdps(60):
        assert abs(mp(SQRT2) - mpmath.sqrt(2)) < mpmath.mpf(10) ** -55
        assert (
            abs(mp(AlgebraicReal(z**3 - 2, 0)) - mpmath.cbrt(2)) < mpmath.mpf(10) ** -55
        )
        rho2 = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)  # (-1 + sqrt(24)) / 23
        assert abs(mp(rho2) - (mpmath.sqrt(24) - 1) / 23) < mpmath.mpf(10) ** -55
        tiny = AlgebraicReal(10**40 * z**2 - 3, 1)  # sqrt(3) * 1e-20: relative accuracy
        assert (
            abs(mp(tiny) / (mpmath.sqrt(3) * mpmath.mpf(10) ** -20) - 1)
            < mpmath.mpf(10) ** -55
        )
    assert AlgebraicReal(z, 0).evalf() == 0
    assert float(SQRT2) == 2**0.5
    with pytest.raises(ValueError, match="dps"):
        SQRT2.evalf(0)


def test_as_expr() -> None:
    a = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
    assert a.as_expr() == sp.CRootOf(23 * z**2 + 2 * z - 1, 1)
    assert abs(sp.N(a.as_expr(), 50) - sp.Float(mp(a), 50)) < sp.Float(10) ** -45
    assert AlgebraicReal(4 * z - 1, 0).as_expr() == sp.Rational(1, 4)


# -- roots of polynomials --------------------------------------------------------------


def test_real_and_positive_roots() -> None:
    f = z * (z**2 - 2) * (3 * z - 1) ** 2 * (z**2 + 1) * (23 * z**2 + 2 * z - 1)
    roots = _real_roots(f)
    assert roots == sorted(roots)
    assert [r.polynomial.as_expr() for r in roots] == [
        z**2 - 2,
        23 * z**2 + 2 * z - 1,
        z,
        23 * z**2 + 2 * z - 1,
        3 * z - 1,
        z**2 - 2,
    ]
    assert _positive_roots(f) == roots[3:]
    assert _real_roots(z**2 + 1) == [] and _real_roots(sp.Integer(5)) == []
    with pytest.raises(ValueError, match="non-zero"):
        _real_roots(sp.Integer(0))


def test_separators() -> None:
    roots = _real_roots((z**2 - 2) * (10**20 * z**2 - (2 * 10**20 + 1)))
    qs = _separators(roots)
    assert len(qs) == len(roots) + 1
    for q, r, s in zip(qs, roots, qs[1:], strict=False):
        assert q < r < s
    for q in qs:  # never a root
        assert all(r != q for r in roots)
    assert _separators([]) == [Fraction(0)]


def test_minimal_polynomials_of_transforms() -> None:
    assert _square(SQRT2) == 2
    assert _square(AlgebraicReal(z**2 - 2, 0)) == 2
    cbrt2 = AlgebraicReal(z**3 - 2, 0)
    assert _square(cbrt2) == AlgebraicReal(z**3 - 4, 0)
    assert _negative(cbrt2) == AlgebraicReal(z**3 + 2, 0)
    assert _negative(SQRT2) == AlgebraicReal(z**2 - 2, 0)
    rho2 = AlgebraicReal(23 * z**2 + 2 * z - 1, 1)
    inv = _reciprocal(rho2)
    assert inv.polynomial.as_expr() == z**2 - 2 * z - 23
    with mpmath.workdps(50):
        assert abs(mp(inv, 50) * mp(rho2, 50) - 1) < mpmath.mpf(10) ** -45
        assert abs(mp(_square(rho2), 50) - mp(rho2, 50) ** 2) < mpmath.mpf(10) ** -45
    assert _reciprocal(AlgebraicReal(z**2 - 2, 0)) == AlgebraicReal(2 * z**2 - 1, 0)
    with pytest.raises(ZeroDivisionError):
        _reciprocal(AlgebraicReal(z, 0))


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.lists(st.integers(-6, 6), min_size=2, max_size=4).filter(
            lambda c: c[0] != 0
        ),
        min_size=1,
        max_size=3,
    )
)
def test_real_roots_property(factors: list[list[int]]) -> None:
    f = sp.Mul(*(sp.Poly(c, z).as_expr() for c in factors))
    roots = _real_roots(f)
    # exact: distinct, increasing, each a root of f with an irreducible polynomial
    assert all(r < s for r, s in itertools.pairwise(roots))
    for r in roots:
        assert r.sign(f) == 0
        assert sp.Poly(f, z).rem(sp.Poly(r.polynomial.as_expr(), z)).is_zero
    # against numerics: the distinct real roots of f
    numeric = sorted(
        round(float(sp.re(x)), 8)
        for x in sp.Poly(sp.sqf_part(f), z).nroots(n=30)
        if abs(sp.im(x)) < 1e-20
    )
    assert [round(float(r), 8) for r in roots] == numeric
    # the transforms agree with arithmetic on the values
    for r in roots:
        assert abs(float(_square(r)) - float(r) ** 2) < 1e-9 * max(1, float(r) ** 2)
        assert _negative(_negative(r)) == r
        if r != 0:
            assert _reciprocal(_reciprocal(r)) == r


# -- the circle test (singularity.md, section 5.1) ------------------------------------


def candidates_of_lk(n: int) -> tuple[sp.Expr, AlgebraicReal]:
    """lc_L G * disc_L G of eq. (17), squarefree: z p_k q_k; and rho_k."""
    p = sp.Poly(P_K.subs(k, n), z)
    (rho,) = _positive_roots(p)
    return z * P_K.subs(k, n) * Q_K.subs(k, n), rho


def test_circle_periodic_and_conjugate_pairs() -> None:
    # T = z + z T^2: curve z T^2 - T + z, candidates z (1 - 4 z^2); -1/2 on the circle
    assert _others_on_circle(z * (1 - 4 * z**2), AlgebraicReal(2 * z - 1, 0)) == [-0.5]
    # -rho a root of m_rho itself
    assert _others_on_circle(z**2 - 2, SQRT2) == [pytest.approx(-(2**0.5))]
    # a conjugate pair e^{±iπ/3} on the unit circle
    found = _others_on_circle((z - 1) * (z**2 - z + 1), AlgebraicReal(z - 1, 0))
    assert found == [
        pytest.approx(complex(0.5, -(3**0.5) / 2)),
        pytest.approx(complex(0.5, 3**0.5 / 2)),
    ]


def test_circle_none() -> None:
    one = AlgebraicReal(z - 1, 0)
    # a real pair with product rho^2 = 1 (1/2 and 2) is not on the circle
    assert _others_on_circle((z - 1) * (2 * z - 1) * (z - 2), one) == []
    # a conjugate pair on another circle (|e^{2iπ/3}| = 1 vs rho = 1/2)
    assert (
        _others_on_circle((2 * z - 1) * (z**2 + z + 1), AlgebraicReal(2 * z - 1, 0))
        == []
    )
    # the factor z is skipped
    assert _others_on_circle(z * (4 * z - 1), AlgebraicReal(4 * z - 1, 0)) == []
    with pytest.raises(ValueError, match="positive"):
        _others_on_circle(z + 1, AlgebraicReal(z + 1, 0))


@pytest.mark.parametrize("n", [1, 2, 10, 50, 53, 54, 55, 100])
def test_circle_broda_lk(n: int) -> None:
    """Broda et al. 2020 §3.4: rho_k is the only candidate on its circle."""
    candidates, rho = candidates_of_lk(n)
    assert _others_on_circle(candidates, rho) == []


@pytest.mark.slow
def test_circle_broda_lk_range() -> None:
    for n in range(1, 61):
        candidates, rho = candidates_of_lk(n)
        assert _others_on_circle(candidates, rho) == [], n
