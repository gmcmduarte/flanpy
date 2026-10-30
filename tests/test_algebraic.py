# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.AlgebraicReal, flanpy.KRoot and the root helpers of _algebraic.

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
    KRoot,
    _negative,
    _OnCircle,
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
        (z**2 + 1, 0, ValueError, "has 0 real root\\(s\\); index 0 is out of range"),
        (z**2 - 2, 2, ValueError, "index 2 is out of range"),
        (z**2 - 2, -1, ValueError, "index -1 is out of range"),
        (sp.Integer(3), 0, ValueError, "degree >= 1"),
        (sp.Integer(0), 0, ValueError, "non-zero"),
        (4 * z**2 - 4, 0, ValueError, r"z\*\*2 - 1 is not irreducible"),
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


def test_normalised_silently() -> None:
    """Rational coefficients, sign and content do not change the roots."""
    quarter = AlgebraicReal(4 * z - 1, 0)
    for same in (1 - 4 * z, 8 * z - 2, z / 2 - sp.Rational(1, 8), -12 * z + 3):
        a = AlgebraicReal(same, 0)
        assert a == quarter and a.polynomial == quarter.polynomial
    assert AlgebraicReal(z / 2 - 1, 0) == 2
    b = AlgebraicReal(sp.Poly(-2 * z**2 + 4, z, domain="QQ"), 1)
    assert b == SQRT2 and repr(b) == "AlgebraicReal(z**2 - 2, 1)"
    # the index refers to the same roots: -z**2 + 2 has roots -sqrt 2 < sqrt 2
    assert AlgebraicReal(2 - z**2, 0) < 0 < AlgebraicReal(2 - z**2, 1)


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
    assert str(AlgebraicReal(4 * z - 1, 0)) == "1/4"  # rational: exact


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
    found = _others_on_circle(z * (1 - 4 * z**2), AlgebraicReal(2 * z - 1, 0))
    assert found == _OnCircle(False, [-0.5])
    # -rho a root of m_rho itself
    found = _others_on_circle(z**2 - 2, SQRT2)
    assert not found.proved_none and found.roots == pytest.approx([-(2**0.5)])
    # a conjugate pair e^{±iπ/3} on the unit circle
    found = _others_on_circle((z - 1) * (z**2 - z + 1), AlgebraicReal(z - 1, 0))
    expected = [complex(0.5, -(3**0.5) / 2), complex(0.5, 3**0.5 / 2)]
    assert not found.proved_none and found.roots == pytest.approx(expected)


def test_circle_none_proved() -> None:
    quarter = AlgebraicReal(4 * z - 1, 0)
    # the factor z is skipped; -1 and 1/4 are on other circles
    assert _others_on_circle(z * (4 * z - 1) * (z + 1), quarter) == _OnCircle(True, [])
    # a conjugate pair on another circle (|e^{2iπ/3}| = 1 vs rho = 1/2): c = 0
    half = AlgebraicReal(2 * z - 1, 0)
    assert _others_on_circle((2 * z - 1) * (z**2 + z + 1), half) == _OnCircle(True, [])
    with pytest.raises(ValueError, match="positive"):
        _others_on_circle(z + 1, AlgebraicReal(z + 1, 0))


def test_circle_none_numerical() -> None:
    one = AlgebraicReal(z - 1, 0)
    # 1/2 and 2 have product rho^2 = 1, but they are roots of different (linear)
    # factors: pairs are only formed within one factor, so this is proved
    assert _others_on_circle((z - 1) * (2 * z - 1) * (z - 2), one) == _OnCircle(
        True, []
    )
    # the two real roots (3 ± sqrt 5)/2 of one irreducible factor have product 1:
    # c > 0, the pair is not conjugate, so nothing is on the circle, but that is a
    # numerical conclusion and proved_none is False (unique = None in step 4)
    assert _others_on_circle((z - 1) * (z**2 - 3 * z + 1), one) == _OnCircle(False, [])


@pytest.mark.parametrize("n", [1, 2, 10, 50, 53, 54, 55, 100])
def test_circle_broda_lk(n: int) -> None:
    """Broda et al. 2020 §3.4: rho_k is the only candidate on its circle, proved."""
    candidates, rho = candidates_of_lk(n)
    assert _others_on_circle(candidates, rho) == _OnCircle(True, [])


@pytest.mark.slow
def test_circle_broda_lk_range() -> None:
    for n in range(1, 61):
        candidates, rho = candidates_of_lk(n)
        assert _others_on_circle(candidates, rho) == _OnCircle(True, []), n


# -- KRoot -------------------------------------------------------------------------


def test_kroot_exported() -> None:
    assert "KRoot" in flanpy.__all__ and flanpy.KRoot is KRoot
    assert KRoot.__module__ == "flanpy"


def test_kroot_broda_rho_k() -> None:
    rho = KRoot(P_K, 0, k=k)
    assert rho.positive_index == 0
    assert rho.polynomial == sp.Poly(P_K, z, k, domain=sp.ZZ)
    assert rho.at(2) == AlgebraicReal(124 * z**3 + 36 * z**2 - z - 2, 0)
    # p_54 = 2 (45z - 2)(18z + 1)^2: the minimal polynomial at 54 is a factor
    assert rho.at(54) == Fraction(2, 45)
    assert rho.at(54).polynomial.as_expr() == 45 * z - 2
    # three real roots for k >= 55, one positive: still positive root 0
    assert len(_real_roots(P_K.subs(k, 60))) == 3
    assert rho.at(60) == _positive_roots(P_K.subs(k, 60))[0]


def test_kroot_normalised_silently() -> None:
    """Rational coefficients, sign and a content in Z[k] do not change the roots."""
    base = KRoot(P_K, 0, k=k)
    assert KRoot(-P_K / 6, 0, k=k) == base
    assert KRoot(k * (k + 1) * P_K, 0, k=k) == base
    assert KRoot(sp.Poly(P_K, z, k), 0, k=k) == base
    assert KRoot(sp.Poly(P_K, k, z), 0, k=k) == base
    # the names of the variables do not matter
    x, m = sp.symbols("x m")
    assert KRoot(P_K.subs({z: x, k: m}), 0, k=m) == base
    assert hash(KRoot(P_K.subs({z: x, k: m}), 0, k=m)) == hash(base)
    assert KRoot(P_K, 1, k=k) != base
    # constant in k is allowed
    assert KRoot(4 * z - 1, 0, k=k).at(7) == Fraction(1, 4)


def test_kroot_validation() -> None:
    with pytest.raises(TypeError, match="positive_index must be an int"):
        KRoot(P_K, True, k=k)
    with pytest.raises(TypeError, match="positive_index must be an int"):
        KRoot(P_K, 0.0, k=k)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="positive_index must be >= 0"):
        KRoot(P_K, -1, k=k)
    with pytest.raises(TypeError, match=r"k must be a sympy\.Symbol"):
        KRoot(P_K, 0, k="k")
    with pytest.raises(ValueError, match="exactly one variable besides k"):
        KRoot(k**2 - 2, 0, k=k)
    with pytest.raises(ValueError, match="exactly one variable besides k"):
        KRoot(z * w - k, 0, k=k)
    with pytest.raises(ValueError, match="polynomial in z and k over Q"):
        KRoot(z - 1 / k, 0, k=k)
    with pytest.raises(ValueError, match="polynomial in z and k over Q"):
        KRoot(z - sp.sqrt(2), 0, k=k)
    with pytest.raises(ValueError, match="not irreducible over Q"):
        KRoot((z - k) * (z + 1), 0, k=k)
    with pytest.raises(ValueError, match="not irreducible over Q"):
        KRoot((4 * z - 1) ** 2, 0, k=k)
    with pytest.raises(TypeError, match="must be a polynomial"):
        KRoot(object(), 0, k=k)


def test_kroot_at_errors() -> None:
    rho = KRoot(P_K, 0, k=k)
    with pytest.raises(TypeError, match="k must be an int"):
        rho.at(2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="k must be an int"):
        rho.at(True)
    with pytest.raises(ValueError, match="k must be >= 1"):
        rho.at(0)
    with pytest.raises(ValueError, match="has 1 positive root"):
        KRoot(P_K, 1, k=k).at(2)
    with pytest.raises(ValueError, match="has 0 positive root"):
        KRoot((k - 3) * z + 1, 0, k=k).at(3)  # the constant 1 at k = 3
    with pytest.raises(ValueError, match="has 0 positive root"):
        KRoot(z + k, 0, k=k).at(1)


def test_kroot_value_semantics() -> None:
    rho = KRoot(P_K, 0, k=k)
    assert pickle.loads(pickle.dumps(rho)) == rho
    assert {rho, KRoot(2 * P_K, 0, k=k)} == {rho}
    assert rho != AlgebraicReal(4 * z - 1, 0) and rho != "rho"
    assert repr(rho) == (
        "KRoot(8*k**2*z**3 + 108*k*z**3 + 36*k*z**2 - k*z - 4, 0, k=k)"
    )
    assert (
        str(rho) == "positive root 0 of 8*k**2*z**3 + 108*k*z**3 + 36*k*z**2 - k*z - 4"
    )
    with pytest.raises(AttributeError, match="immutable"):
        rho._index = 1
    with pytest.raises(AttributeError, match="immutable"):
        del rho._index


@pytest.mark.parametrize("n", range(1, 21))
def test_kroot_against_mpmath(n: int) -> None:
    """R_k of Broda 2020: the positive root of Δ_k, (sqrt(8 + 8k) - 1)/(7 + 8k)."""
    rho = KRoot(1 - 2 * z - (7 + 8 * k) * z**2, 0, k=k).at(n)
    with mpmath.workdps(50):
        exact = (mpmath.sqrt(8 + 8 * n) - 1) / (7 + 8 * n)
        assert abs(rho.evalf(50) - exact) < mpmath.mpf(10) ** -45
