# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the coefficient asymptotics at a fixed k (transfer.md, section 9)."""

from __future__ import annotations

import pickle
import warnings
from fractions import Fraction
from typing import Any

import mpmath
import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from flanpy import (
    AlgebraicReal,
    Asymptotics,
    GammaConstant,
    LocalExpansion,
    SingularityWarning,
    System,
    asymptotics,
    eliminate,
    eliminate_in_k,
    local_expansion,
    local_expansion_in_k,
    series,
    singularity,
    singularity_in_k,
)
from flanpy._algebraic import _reciprocal
from flanpy._transfer import _e_poly, _e_value, _inverse_gamma
from test_newton import (
    CATALAN,
    CONSTRUCTED,
    EXAMPLES,
    FK,
    HG,
    MOTZKIN,
    POLE,
    QUARTER,
    RK,
    S5_TREE,
    SSNF,
    THREE_HALVES,
    A,
    F,
    K,
    L,
    R,
    T,
    W,
    Yu,
    expand,
    z,
)
from test_singularity import small_systems

a_, y = sp.symbols("a y")
k = sp.Symbol("k")
HALF = Fraction(1, 2)


def asym(system: System, w: sp.Symbol, n: int | None = None) -> Asymptotics[Any]:
    return asymptotics(expand(system, w, n))


def rational(c: GammaConstant[Any]) -> Fraction:
    """The algebraic part of a constant, when it is rational."""
    lo, hi = c.algebraic().interval
    assert lo == hi
    return lo


def as_poly(ell: int) -> sp.Expr:
    return sum(
        sp.Rational(c.numerator, c.denominator) * a_**i
        for i, c in enumerate(_e_poly(ell))
    )


def mp(q: Fraction) -> mpmath.mpf:
    return mpmath.mpf(q.numerator) / q.denominator


def scaled(x: AlgebraicReal, q: Fraction) -> AlgebraicReal:
    """``q x`` for ``q > 0`` (same index: the roots are scaled in order)."""
    gen = x.polynomial.gen
    return AlgebraicReal(
        x.polynomial.as_expr().subs(gen, gen / sp.Rational(q)), x.index
    )


def coefficient(system: System, w: sp.Symbol, n: int, kval: int | None = None) -> Any:
    c = Fraction((series(system, n, k=kval) if kval else series(system, n))[w][n])
    return mpmath.mpf(c.numerator) / c.denominator


# -- e_l (FS Thm VI.1) and 1/Gamma -----------------------------------------------------


def test_e_polys_fs_eq_13() -> None:
    fs13 = [
        sp.Integer(1),
        a_ * (a_ - 1) / 2,
        a_ * (a_ - 1) * (a_ - 2) * (3 * a_ - 1) / 24,
        a_**2 * (a_ - 1) ** 2 * (a_ - 2) * (a_ - 3) / 48,
    ]
    for ell, expected in enumerate(fs13):
        assert sp.expand(as_poly(ell) - expected) == 0


@pytest.mark.parametrize("ell", range(1, 9))
def test_e_polys_footnote_2(ell: int) -> None:
    """Degree 2l, divisible by a(a - 1)...(a - l) (FS Thm VI.1, footnote 2)."""
    assert len(_e_poly(ell)) == 2 * ell + 1
    assert all(_e_value(ell, Fraction(i)) == 0 for i in range(ell + 1))


@pytest.mark.parametrize("ell", [1, 2])
def test_e_polys_note_vi_3_with_sign(ell: int) -> None:
    """FS Note VI.3, read with (-1)^l (transfer.md, section 2.1)."""
    v, t = sp.symbols("v t")
    expr = sp.exp(t) * (1 + v * t) ** (-1 - 1 / v)
    total = sp.Integer(0)
    for m in range(ell, 2 * ell + 1):
        inner = sp.series(expr, t, 0, m + 1).removeO().coeff(t, m)
        lam = sp.series(sp.simplify(inner), v, 0, ell + 1).removeO().coeff(v, ell)
        total += (-1) ** m * lam * sp.prod([a_ - i for i in range(1, m + 1)])
    assert sp.expand(total - as_poly(ell)) == 0


@pytest.mark.parametrize("a", [HALF, -HALF, Fraction(1, 4), Fraction(3, 2)])
def test_e_polys_against_gamma_ratios(a: Fraction) -> None:
    n = 1000
    with mpmath.workdps(50):
        exact = mpmath.gammaprod([n + mp(a)], [n + 1])
        approx = mpmath.mpf(n) ** (mp(a) - 1) * sum(
            mp(_e_value(ell, a)) / mpmath.mpf(n) ** ell for ell in range(9)
        )
        assert abs(approx / exact - 1) < mpmath.mpf(10) ** -26


@pytest.mark.parametrize(
    ("x", "r", "nu"),
    [
        (-HALF, -HALF, HALF),
        (HALF, Fraction(1), HALF),
        (Fraction(3, 2), Fraction(2), HALF),
        (Fraction(-3, 2), Fraction(3, 4), HALF),
        (Fraction(-1, 4), Fraction(-1, 4), Fraction(3, 4)),
        (Fraction(-5, 4), Fraction(5, 16), Fraction(3, 4)),
        (Fraction(7, 4), Fraction(4, 3), Fraction(3, 4)),
        (Fraction(0), Fraction(0), None),
        (Fraction(-2), Fraction(0), None),
        (Fraction(4), Fraction(1, 6), None),
    ],
)
def test_inverse_gamma(x: Fraction, r: Fraction, nu: Fraction | None) -> None:
    assert _inverse_gamma(x) == (r, nu)


@given(
    st.fractions(min_value=-12, max_value=12, max_denominator=8).filter(
        lambda x: x.denominator > 1 or x > 0
    )
)
def test_inverse_gamma_against_mpmath(x: Fraction) -> None:
    r, nu = _inverse_gamma(x)
    with mpmath.workdps(50):
        value = mp(r) if nu is None else mp(r) / mpmath.gamma(mp(nu))
        assert abs(value * mpmath.gamma(mp(x)) - 1) < mpmath.mpf(10) ** -40
    assert nu is None or 0 < nu < 1


# -- quadratic first (HG style), exact -------------------------------------------------


def test_catalan() -> None:
    a = asym(CATALAN, F)
    assert a.exponent == Fraction(-3, 2) and a.rho == AlgebraicReal(4 * y - 1, 0)
    terms = a.terms(7)
    assert [p for p, _ in terms] == [-Fraction(3 + j, 2) for j in range(7)]
    expected = [1, 0, Fraction(-9, 8), 0, Fraction(145, 128), 0, Fraction(-1155, 1024)]
    assert [rational(d) for _, d in terms] == expected
    assert [d.gamma for _, d in terms] == [((HALF, -1),) if q else () for q in expected]
    n = sp.Symbol("n")
    assert a.as_expr(n) == 4**n / (sp.sqrt(sp.pi) * n ** sp.Rational(3, 2))
    assert a.error_exponent(0) == Fraction(-3, 2)
    assert a.error_exponent(1) == a.error_exponent(2) == Fraction(-5, 2)
    assert a.error_exponent(3) == Fraction(-7, 2)
    assert str(a) == "4**n/(sqrt(pi)*n**(3/2)) (1 + O(n^-1))"


def test_motzkin_fs_example_vii_18() -> None:
    """``3^{n+1/2}/(2√π n^{3/2}) (1 - 15/(16n) + 505/(512n^2) - 8085/(8192n^3))``."""
    a = asym(MOTZKIN, F)
    terms = a.terms(7)
    lead = sp.sqrt(3) / (2 * sp.sqrt(sp.pi))
    ratios = [1, 0, Fraction(-15, 16), 0, Fraction(505, 512), 0, Fraction(-8085, 8192)]
    for (power, d), q in zip(terms, ratios, strict=True):
        assert sp.simplify(d.as_expr() - sp.Rational(q) * lead) == 0, power
    assert a.leading.algebraic() == AlgebraicReal(4 * y**2 - 3, 1)


@pytest.mark.parametrize("n", range(1, 21))
def test_hg_a_k_eq_16(n: int) -> None:
    """``[z^n] A_k ~ (2k)^{1/4} √ρ_k ρ_k^{-(n+1)} n^{-3/2}/(4√π)`` (HG eq. (16))."""
    a = asym(HG, A, n)
    with mpmath.workdps(50):
        rho = a.rho.evalf(50)
        closed = (2 * n) ** mpmath.mpf(0.25) * mpmath.sqrt(rho) / (4 * rho)
        closed /= mpmath.sqrt(mpmath.pi)
        assert abs(a.leading.evalf(45) / closed - 1) < mpmath.mpf(10) ** -40


@pytest.mark.parametrize("n", range(1, 21))
def test_broda_r_k_eq_6(n: int) -> None:
    """``[z^n] R_k ~ √(2 - 2ρ_k) ρ_k^{-n} n^{-3/2}/(8ρ_k√π)`` (Broda eq. (6))."""
    a = asym(RK, R, n)
    with mpmath.workdps(50):
        rho = a.rho.evalf(50)
        closed = mpmath.sqrt(2 - 2 * rho) / (8 * rho * mpmath.sqrt(mpmath.pi))
        assert abs(a.leading.evalf(45) / closed - 1) < mpmath.mpf(10) ** -40


# -- exponents and fields --------------------------------------------------------------


def test_alpha_quarter() -> None:
    """``K = (1 - s^{1/4})/2``: ``[z^n] K ~ 4^n n^{-5/4}/(8 Γ(3/4))``."""
    a = asym(QUARTER, K)
    assert a.exponent == Fraction(-5, 4)
    assert rational(a.leading) == Fraction(1, 8)
    assert a.leading.gamma == ((Fraction(3, 4), -1),)
    terms = a.terms(5)
    assert [d.algebraic() == 0 for _, d in terms] == [False, True, True, True, False]
    assert [d.gamma for _, d in terms][1:4] == [(), (), ()]
    # The zeros D_1..D_3 are accidental: the O-term is not sharpened by them.
    assert a.error_exponent(1) == Fraction(-3, 2)
    n = sp.Symbol("n")
    assert a.as_expr(n) == 4**n / (
        8 * sp.gamma(sp.Rational(3, 4)) * n ** sp.Rational(5, 4)
    )


def test_pole_terminates() -> None:
    """``1/(1 - 2z)``: ``[z^n] = 2^n`` exactly; every further ``D_j`` is 0."""
    a = asym(POLE, Yu)
    assert a.exponent == 0 and rational(a.leading) == 1 and a.leading.gamma == ()
    assert [(p, rational(d)) for p, d in a.terms(4)] == [
        (Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(-2), Fraction(0)),
        (Fraction(-3), Fraction(0)),
    ]
    assert a.error_exponent(1) is None
    assert a.error_exponent(0) == 0
    assert a.evalf(1000) == 2**1000 and a.evalf(1000, count=4) == 2**1000
    assert str(a) == "2**n + O((ρ + ε)^-n)"


def test_three_halves() -> None:
    """``(1 - 4z)^{-3/2}``: ``[z^n] ~ 4^n (2/√π) n^{1/2} (1 + 3/(8n) + ...)``."""
    a = asym(THREE_HALVES, W)
    assert a.exponent == HALF
    terms = a.terms(3)
    assert [rational(d) for _, d in terms] == [2, 0, Fraction(3, 4)]
    assert terms[0][1].gamma == terms[2][1].gamma == ((HALF, -1),)
    n = 1000
    with mpmath.workdps(40):
        exact = mpmath.mpf(4) ** n * mpmath.gammaprod([n + 1.5], [1.5, n + 1])
        assert abs(a.evalf(n, count=3, dps=30) / exact - 1) < 3 * 5.5e-8


@pytest.mark.parametrize(("system", "w"), [(CONSTRUCTED, F), (S5_TREE, T)])
def test_leading_constant_case_one(system: System, w: sp.Symbol) -> None:
    """``D_0 = α b/Γ(1 - α)``: ``b/(2√π)`` (deg 8 and the S5 tree, deg 10)."""
    e = expand(system, w)
    a = asymptotics(e)
    assert e.b is not None
    assert a.leading.algebraic() == scaled(e.b, HALF)
    assert a.leading.gamma == ((HALF, -1),)


def test_leading_constant_case_two() -> None:
    """``D_0 = 1/(c Γ(α))`` for ``L_2`` (Case II, ``α = 1/2``)."""
    e = expand(SSNF, L, 2)
    a = asymptotics(e)
    assert a.exponent == -HALF and e.c is not None
    assert a.leading.algebraic() == _reciprocal(e.c)
    assert a.leading.gamma == ((HALF, -1),)


# -- n = 1000 (moved from test_newton.py; now through evalf) ---------------------------

# (name, errors with count = 1, 3, 5) at n = 1000, measured; the bounds are 3 times.
MEASURED = [
    ("catalan", (1.13e-3, 1.13e-6, 1.13e-9)),
    ("motzkin", (9.37e-4, 9.86e-7, 9.87e-10)),
    ("r_2", (8.26e-4, 8.72e-7, 8.96e-10)),
    ("a_2", (9.37e-4, 9.86e-7, 9.87e-10)),
    ("quarter", (1.56e-4, 1.56e-4, 5.13e-8)),
    ("three_halves", (3.75e-4, 5.47e-8, 8.79e-12)),
    ("constructed", (2.68e-4, 2.92e-6, 2.21e-8)),
    ("s5_tree", (1.73e-5, 1.93e-8, 2.98e-10)),
    ("f_2", (7.98e-4, 8.35e-7, 8.55e-10)),
    ("l_2", (2.13e-4, 3.55e-8, 2.68e-11)),
]


@pytest.mark.slow
@pytest.mark.parametrize(("name", "errors"), MEASURED, ids=[x[0] for x in MEASURED])
def test_coefficients_at_1000(name: str, errors: tuple[float, ...]) -> None:
    _, system, w, n = next(e for e in EXAMPLES if e[0] == name)
    a = asym(system, w, n)
    with mpmath.workdps(40):
        exact = coefficient(system, w, 1000, n)
        for count, error in zip((1, 3, 5), errors, strict=True):
            assert abs(a.evalf(1000, count=count, dps=30) / exact - 1) < 3 * error


# F_k and L_k at n = 1000: the largest measured error over k in {2, 10, 50, 54, 108}.
BRODA = [(FK, F, (8.0e-4, 8.35e-7, 8.55e-10)), (SSNF, L, (2.44e-4, 3.85e-8, 3.59e-11))]


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 10, 50, 54, 108])
def test_broda_coefficients_at_1000(n: int) -> None:
    for system, w, errors in BRODA:
        a = asym(system, w, n)
        with mpmath.workdps(40):
            exact = coefficient(system, w, 1000, n)
            for count, error in zip((1, 3, 5), errors, strict=True):
                assert abs(a.evalf(1000, count=count, dps=30) / exact - 1) < 3 * error


# -- properties (hypothesis) -----------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(small_systems())
def test_random_systems(system: System) -> None:
    """Moved from test_newton.py (the 5% check at n = 300), plus the D_0 formula."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SingularityWarning)
            s = singularity(eliminate(system, F))
    except ValueError as error:  # linear systems in F can be polynomials
        assert "polynomial" in str(error)
        return
    if not s.unique:
        with pytest.raises(ValueError, match="dominant singularit"):
            asymptotics(local_expansion(s))
        return
    e = local_expansion(s)
    a = asymptotics(e)
    terms = a.terms(5)
    kappa = e.ramification
    for j, (power, d) in enumerate(terms):
        p = -1 - power
        assert p == (e.alpha if s.case == "I" else -e.alpha) + Fraction(j, kappa)
        if p.denominator == 1 and p >= 0:
            assert d.algebraic() == 0  # the analytic part transfers to nothing
    if s.case == "I":
        assert e.b is not None
        assert a.leading.algebraic() == scaled(e.b, e.alpha)
        assert a.leading.gamma == ((1 - e.alpha, -1),)
    else:
        assert e.c is not None
        r, nu = _inverse_gamma(e.alpha)
        assert a.leading.algebraic() == scaled(_reciprocal(e.c), r)
        assert a.leading.gamma == (() if nu is None else ((nu, -1),))
    if s.period == 1:
        n = 300
        c = Fraction(series(system, n)[F][n])
        value = mpmath.mpf(c.numerator) / c.denominator
        assert abs(value / a.evalf(n) - 1) < 0.05


# -- refusals and errors ---------------------------------------------------------------


def test_refuses_periodic() -> None:
    trees = System({T: z + z * T**2}, z=z)  # odd powers: dominant singularities ±1/2
    s = singularity(eliminate(trees, T))
    assert s.unique is False and s.period == 2
    with pytest.raises(ValueError, match=r"2 dominant singularities .* \(period 2\)"):
        asymptotics(local_expansion(s))


def test_refuses_undecided() -> None:
    system = System({T: z + z * T**2, W: z**2 + T}, z=z)
    with pytest.warns(SingularityWarning):
        s = singularity(eliminate(system, W))
    assert s.unique is None
    with pytest.raises(ValueError, match="undecided"):
        asymptotics(local_expansion(s))


def test_type_and_k_errors() -> None:
    s = singularity(eliminate(CATALAN, F))
    e = local_expansion(s)
    with pytest.raises(TypeError, match="LocalExpansion"):
        asymptotics(s)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no parameter k"):
        asymptotics(e, k=2)
    piece = local_expansion_in_k(singularity_in_k(eliminate_in_k(RK, R)))[2]
    assert isinstance(piece, LocalExpansion)
    with pytest.raises(ValueError, match="give k=n"):
        asymptotics(piece)
    assert asymptotics(piece, k=3) == asymptotics(piece.specialize(3))
    with pytest.raises(TypeError, match=r"created by flanpy\.asymptotics"):
        Asymptotics()
    with pytest.raises(TypeError, match=r"created by flanpy\.asymptotics"):
        GammaConstant()


def test_argument_errors() -> None:
    a = asym(CATALAN, F)
    with pytest.raises(TypeError, match="count must be an int"):
        a.terms(2.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="count must be an int"):
        a.error_exponent(True)
    with pytest.raises(ValueError, match="count must be >= 0"):
        a.terms(-1)
    with pytest.raises(TypeError, match="n must be an int"):
        a.evalf(10.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n must be >= 1"):
        a.evalf(0)
    with pytest.raises(ValueError, match="dps"):
        a.evalf(10, dps=0)
    with pytest.raises(ValueError, match="dps"):
        a.leading.evalf(0)
    with pytest.raises(TypeError, match="SymPy expression"):
        a.as_expr(3)
    assert a.terms(0) == [] and a.evalf(10, count=0) == 0


# -- API: value semantics --------------------------------------------------------------


def test_value_semantics() -> None:
    a, b = asym(CATALAN, F), asym(CATALAN, F)
    m = asym(MOTZKIN, F)
    assert a == b and hash(a) == hash(b) and a != m and a != 3
    assert a.leading == b.leading and hash(a.leading) == hash(b.leading)
    assert a.leading != m.leading and a.leading != 1
    assert a.terms(2)[1][1] == m.terms(2)[1][1]  # the zeros are equal
    with pytest.raises(AttributeError, match="immutable"):
        setattr(a, "_expansion", None)  # noqa: B010
    with pytest.raises(AttributeError, match="immutable"):
        del a._expansion
    with pytest.raises(AttributeError, match="immutable"):
        a.leading._gamma = ()
    with pytest.raises(AttributeError, match="immutable"):
        del a.leading._gamma
    assert repr(a.leading) == (
        "GammaConstant(AlgebraicReal(y - 1, 0), gamma=((Fraction(1, 2), -1),))"
    )
    assert str(m.leading) == "sqrt(3)/(2*sqrt(pi))"
    assert a.k is None and a.kset == a.expansion.kset


def test_pickle() -> None:
    a = asym(MOTZKIN, F)
    a.terms(5)
    b = pickle.loads(pickle.dumps(a))
    assert b == a and b.leading == a.leading
    assert b.terms(7) == a.terms(7)  # recomputed after unpickling
    c = pickle.loads(pickle.dumps(a.leading))
    assert c == a.leading and c.gamma == ((HALF, -1),)


def test_terms_cached() -> None:
    a = asym(CATALAN, F)
    first = a.terms(5)
    assert a.terms(3) == first[:3] and a.terms(5) == first
    assert a.terms(7)[:5] == first


def test_evalf_precision() -> None:
    a = asym(CATALAN, F)
    with mpmath.workdps(60):
        value = a.evalf(1000, count=1, dps=50)
        expected = mpmath.mpf(4) ** 1000 / (
            mpmath.sqrt(mpmath.pi) * mpmath.mpf(1000) ** 1.5
        )
        assert abs(value / expected - 1) < mpmath.mpf(10) ** -48
        assert abs(a.evalf(1000) / expected - 1) < mpmath.mpf(10) ** -14
