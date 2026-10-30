# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.singularity at a fixed k (design: dev/design/singularity.md, §8)."""

import pickle
import time
import warnings
from fractions import Fraction
from typing import Any

import mpmath
import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

import flanpy
from flanpy import (
    Singularity,
    SingularityWarning,
    System,
    UndecidedSingularityError,
    eliminate,
    eliminate_in_k,
    series,
    singularity,
)
from flanpy._algebraic import AlgebraicReal, _positive_roots
from flanpy._elimination import _closure
from flanpy._singularity import (
    _Above,
    _Below,
    _candidates,
    _Closure,
    _compare,
    _component_periods,
    _components,
    _cycle_period,
    _periods,
    _valuations,
)

z, k = sp.symbols("z k")
B, Bb, M, Mb, L, F, G, R, A, T, U, W = sp.symbols("B Bb M Mb L F G R A T U W")

CATALAN = System({F: 1 + z * F**2}, z=z)
RK = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
HG = System({B: k * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}, z=z, k=k)
SSNF = System(
    {
        B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
        Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
        L: M + Mb,
    },
    z=z,
    k=k,
)
FK = System(
    {
        B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        F: z + B + Bb,
    },
    z=z,
    k=k,
)
CONSTRUCTED = System({F: z + z * F**2 + z * G, G: z + z * G**2}, z=z)
# The same plus an unknown outside the closure of F with a smaller radius (1/40).
UNRELATED = System(
    {F: z + z * F**2 + z * G, G: z + z * G**2, U: 1 + 10 * z * U**2}, z=z
)
PERIODIC = System({T: z + z * T**2}, z=z)
# W is aperiodic but singular at ±1/2, like T.
APERIODIC_TWO = System({T: z + z * T**2, W: z**2 + T}, z=z)
# PS 2025 Example 23: linear, coordinate periods 1, cycle period 5; singular at the
# fifth roots of unity.
AE, BE = sp.symbols("A_e B_e")
PS_EXAMPLE_23 = System({AE: z**2 + z**2 * BE, BE: z**3 + z**3 * AE}, z=z)
# A critical input of period 2 (S = 1 - sqrt(1 - z^2), radius 1) feeding Y of period
# 1 and radius 1: Y is singular at -1 too (singularity.md, section 5.5).
S, Y = sp.symbols("S Y")
CRITICAL_INPUT = System({S: (z**2 + S**2) / 2, Y: z + z**2 + z * S * Y**2 / 100}, z=z)

P_K = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4  # Broda eq. (18)
Q_K = (
    (2 * k**3 + 36 * k**2 + 108 * k) * z**3
    + (12 * k**2 + 72 * k) * z**2
    + 15 * k * z
    - 2
)


def mp(x: AlgebraicReal) -> mpmath.mpf:
    return x.evalf(50)


def close(x: Any, y: Any, digits: int = 40) -> bool:
    with mpmath.workdps(50):
        return bool(abs(x - y) <= mpmath.mpf(10) ** -digits * max(1, abs(y)))


def closure_of(system: System, w: sp.Symbol, n: int | None = None) -> _Closure:
    s = system.specialize(n) if n is not None else system
    found = _closure(s, w)
    assert found is not None
    return _Closure(z, *found)


# -- public API ------------------------------------------------------------------------


def test_exported() -> None:
    for name in (
        "Singularity",
        "SingularityWarning",
        "UndecidedSingularityError",
        "singularity",
    ):
        assert name in flanpy.__all__
    for cls in (Singularity, SingularityWarning, UndecidedSingularityError):
        assert cls.__module__ == "flanpy"
    assert issubclass(UndecidedSingularityError, ValueError)
    assert issubclass(SingularityWarning, UserWarning)


def test_catalan() -> None:
    s = singularity(eliminate(CATALAN, F))
    assert s.rho == Fraction(1, 4) and s.case == "I" and s.a == 2
    assert s.period == 1 and s.unique is True
    assert s.k is None and s.kset == flanpy.KSet.all()
    assert s.curve == eliminate(CATALAN, F)


@pytest.mark.parametrize("n", range(1, 21))
def test_hitchhiker_a_k(n: int) -> None:
    """HG eqs. (8)-(9): rho_k = 1/(2(sqrt(2k) + 1)), A = (8rho^2 - 2rho + 1)/(4rho)."""
    s = singularity(eliminate(HG, A, k=n))
    delta = (4 - 8 * n) * z**2 - 4 * z + 1
    assert s.rho.sign(delta) == 0 and s.rho > 0
    with mpmath.workdps(50):
        rho = 1 / (2 * (mpmath.sqrt(2 * n) + 1))
        assert close(mp(s.rho), rho)
        assert s.case == "I" and s.a is not None
        assert close(mp(s.a), (8 * rho**2 - 2 * rho + 1) / (4 * rho))
    assert s.period == 1 and s.unique is True


@pytest.mark.parametrize("n", range(1, 21))
def test_broda_r_k(n: int) -> None:
    """Broda 2020 §3.1: rho_k is the positive root of 1 - 2z - (7 + 8k) z^2, and
    R(rho) = (1 - rho)/(4 rho) (eq. (2) with the square root 0)."""
    s = singularity(eliminate(RK, R, k=n))
    assert s.rho.sign(1 - 2 * z - (7 + 8 * n) * z**2) == 0 and s.rho > 0
    with mpmath.workdps(50):
        rho = (mpmath.sqrt(8 + 8 * n) - 1) / (7 + 8 * n)
        assert close(mp(s.rho), rho)
        assert s.case == "I" and s.a is not None
        assert close(mp(s.a), (1 - rho) / (4 * rho))
    assert s.period == 1 and s.unique is True


def test_broda_f_k() -> None:
    """F_k = z + B + Bb (eq. (38)'s denominator): Case I, rho_k = rho of L_k."""
    s = singularity(eliminate(FK, F, k=2))
    assert s.case == "I" and s.rho.sign(P_K.subs(k, 2)) == 0
    assert s.unique is True


# -- a candidate that is not a singularity of the branch -------------------------------


def test_constructed_example_skips_a_candidate() -> None:
    s = singularity(eliminate(CONSTRUCTED, F))
    positive = _positive_roots(_candidates(sp.Poly(s.curve.as_expr(), F, z)))
    assert len(positive) == 3 and s.rho == positive[1]  # the second of three
    assert s.rho.polynomial.as_expr() == 32 * z**4 + 16 * z**3 - 8 * z**2 - 4 * z + 1
    assert s.case == "I" and s.period == 1 and s.unique is True


def test_closure_only() -> None:
    """An unknown outside the closure, with a smaller radius, changes nothing; and
    the certificates of the closure say q < rho on (0.214, 0.406)."""
    assert (
        singularity(eliminate(UNRELATED, F)).rho
        == singularity(eliminate(CONSTRUCTED, F)).rho
    )
    closure = closure_of(UNRELATED, F)
    assert U not in closure.unknowns
    for q in (Fraction(22, 100), Fraction(3, 10), Fraction(40, 100)):
        assert isinstance(_compare(closure, q), _Below)
    assert isinstance(_compare(closure, Fraction(41, 100)), _Above)


# -- periods and uniqueness ----------------------------------------------------------


def test_periodic() -> None:
    s = singularity(eliminate(PERIODIC, T))
    assert s.rho == Fraction(1, 2) and s.a == 1 and s.case == "I"
    assert s.period == 2 and s.unique is False


def test_undecided_uniqueness_warns() -> None:
    """Period 1, -1/2 on the circle, T's component has period 2: unique is None."""
    with pytest.warns(SingularityWarning, match=r"undecided.*-0\.5"):
        s = singularity(eliminate(APERIODIC_TWO, W))
    assert s.period == 1 and s.unique is None and s.rho == Fraction(1, 2)


@pytest.mark.parametrize(
    ("system", "w"),
    [(CATALAN, F), (PERIODIC, T), (APERIODIC_TWO, W), (CONSTRUCTED, F)],
)
def test_period_against_series_support(system: System, w: sp.Symbol) -> None:
    import math

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularityWarning)
        s = singularity(eliminate(system, w))
    support = [n for n, c in enumerate(series(system, 60)[w]) if c]
    assert s.period == math.gcd(*(n - support[0] for n in support))


# -- signed systems, polynomial branches, errors --------------------------------------


def test_signed_refused() -> None:
    signed = System({F: 1 - z * F**2}, z=z)  # radius 1/4 at -1/4
    with pytest.raises(UndecidedSingularityError) as info:
        singularity(eliminate(signed, F))
    message = str(info.value)
    assert "sufficient" in message and "not necessary" in message
    assert "z*(4*z + 1)" in message and "-1/4, 0" in message


def test_polynomial_branch() -> None:
    with pytest.raises(ValueError, match="is a polynomial"):
        singularity(eliminate(System({F: z + z * G, G: z}, z=z), F))
    with pytest.warns(flanpy.ZeroCoordinateWarning):
        zero = System({F: z * F, G: z + z * G**2}, z=z)
    with pytest.raises(ValueError, match="identically 0"):
        singularity(eliminate(zero, F))


def test_arguments() -> None:
    tail = eliminate_in_k(RK, R).tail
    assert tail is not None
    curve = tail[1]
    with pytest.raises(TypeError, match=r"flanpy\.Curve"):
        singularity(RK)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="give k=n"):
        singularity(curve)
    with pytest.raises(ValueError, match="omit the k argument"):
        singularity(eliminate(CATALAN, F), k=2)
    # F = 1 + (k - 1) z F^2: the generic curve holds on [2, oo), k = 1 is separate
    drop = eliminate_in_k(System({F: 1 + (k - 1) * z * F**2}, z=z, k=k), F)
    with pytest.raises(ValueError, match="not in the curve's kset"):
        singularity(drop.pieces[-1][1], k=1)
    # k = n on a symbolic curve goes through curve.specialize(n)
    assert singularity(curve, k=3) == singularity(eliminate(RK, R, k=3))


def test_value_semantics() -> None:
    s = singularity(eliminate(CATALAN, F))
    t = singularity(eliminate(CATALAN, F))
    assert s == t and hash(s) == hash(t)
    assert s != singularity(eliminate(PERIODIC, T))
    assert pickle.loads(pickle.dumps(s)) == s
    with pytest.raises(AttributeError, match="immutable"):
        s._case = "II"
    with pytest.raises(TypeError, match=r"created by flanpy\.singularity"):
        Singularity()
    assert repr(s) == (
        "Singularity(rho=AlgebraicReal(4*z - 1, 0), case='I', "
        "a=AlgebraicReal(F - 2, 0), period=1, unique=True)"
    )
    assert str(singularity(eliminate(SSNF, L, k=2))).endswith(
        "Case II, w → ∞, period 1, unique"
    )


# -- the certificates (singularity.md §3.3) --------------------------------------------


@pytest.mark.parametrize(("system", "w", "n"), [(SSNF, L, 2), (CONSTRUCTED, F, None)])
def test_certificates_within_2_to_the_minus_60(
    system: System, w: sp.Symbol, n: int | None
) -> None:
    rho = singularity(eliminate(system, w, k=n) if n else eliminate(system, w)).rho
    closure = closure_of(system, w, n)
    rho.refine(Fraction(1, 2**64))
    lo, hi = rho.interval
    below, above = lo - Fraction(1, 2**62), hi + Fraction(1, 2**62)
    assert below < rho < below + Fraction(1, 2**60)
    assert above - Fraction(1, 2**60) < rho < above
    assert isinstance(_compare(closure, below), _Below)
    assert isinstance(_compare(closure, above), _Above)


def test_certificates_rational_rho() -> None:
    closure = closure_of(CATALAN, F)
    quarter = Fraction(1, 4)
    assert isinstance(_compare(closure, quarter - Fraction(1, 2**60)), _Below)
    assert isinstance(_compare(closure, quarter + Fraction(1, 2**60)), _Above)


@pytest.mark.parametrize(("system", "w", "n"), [(SSNF, L, 2), (CONSTRUCTED, F, None)])
def test_certificate_soundness_against_series(
    system: System, w: sp.Symbol, n: int | None
) -> None:
    """Below certificates: partial sums of the series (lower bounds of Y(q)) stay
    below the upper bound u, for every coordinate of the closure."""
    s = system.specialize(n) if n is not None else system
    rho = singularity(eliminate(system, w, k=n) if n else eliminate(system, w)).rho
    closure = closure_of(system, w, n)
    coefficients = series(s, 200)
    for fraction in (Fraction(1, 2), Fraction(9, 10), Fraction(99, 100)):
        q = Fraction(rho.interval[0]) * fraction
        result = _compare(closure, q)
        assert isinstance(result, _Below)
        for u, upper, lower in zip(
            closure.unknowns, result.upper, result.lower, strict=True
        ):
            partial = sum(Fraction(c) * q**i for i, c in enumerate(coefficients[u]))
            assert partial <= upper
            assert lower <= upper


# -- Broda L_k (acceptance) -----------------------------------------------------------


def check_lk(n: int) -> None:
    s = singularity(eliminate(SSNF, L, k=n))
    p = sp.Poly(P_K.subs(k, n), z)
    assert s.case == "II" and s.a is None and s.unique is True and s.period == 1
    assert p.rem(sp.Poly(s.rho.polynomial.as_expr(), z)).is_zero
    (positive,) = _positive_roots(p)  # the unique positive root of p_k (§3.4)
    assert s.rho == positive
    with mpmath.workdps(50):
        roots = [
            r
            for r in mpmath.polyroots(
                [int(c) for c in p.all_coeffs()], maxsteps=200, extraprec=200
            )
            if abs(mpmath.im(r)) < mpmath.mpf(10) ** -30 and mpmath.re(r) > 0
        ]
        assert len(roots) == 1 and close(mp(s.rho), mpmath.re(roots[0]))
    # the positive root of q_k is a candidate below rho_k, skipped (§3.2)
    (q_root,) = _positive_roots(Q_K.subs(k, n))
    candidates = _positive_roots(_candidates(sp.Poly(s.curve.as_expr(), L, z)))
    assert q_root in candidates and q_root < s.rho


@pytest.mark.parametrize("n", [1, 2, 10, 50, 53, 54, 55])
def test_broda_l_k(n: int) -> None:
    check_lk(n)


def test_broda_l_54() -> None:
    s = singularity(eliminate(SSNF, L, k=54))
    assert s.rho == Fraction(2, 45)
    assert s.rho.polynomial.as_expr() == 45 * z - 2  # p_54 = 2 (45z - 2)(18z + 1)^2


@pytest.mark.slow
def test_broda_l_k_1_to_60() -> None:
    start = time.perf_counter()
    for n in range(1, 61):
        check_lk(n)
    print(f"L_k, k = 1..60: {time.perf_counter() - start:.1f} s")


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 10, 50, 54])
def test_broda_l_k_coefficient_ratio(n: int) -> None:
    """[z^N] / [z^(N+1)] * ((N + 1)/N)^(-1/2) -> rho_k (alpha = 1/2, eq. (32))."""
    big = 1000
    c = series(SSNF, big + 1, k=n)[L]
    rho = mp(singularity(eliminate(SSNF, L, k=n)).rho)
    with mpmath.workdps(50):
        ratio = mpmath.mpf(c[big].numerator) / c[big].denominator
        ratio /= mpmath.mpf(c[big + 1].numerator) / c[big + 1].denominator
        corrected = ratio * (mpmath.mpf(big + 1) / big) ** mpmath.mpf(-0.5)
        assert abs(corrected / rho - 1) < mpmath.mpf(10) ** -6


# -- random small non-negative systems ------------------------------------------------


@st.composite
def small_systems(draw: st.DrawFn) -> System:
    """Non-negative, two unknowns, degree <= 2, every term with a factor z."""
    monomials = [F, G, F**2, F * G, G**2]
    equations = {}
    for u in (F, G):
        terms = [draw(st.sampled_from([1, 2])) * z]
        terms += [draw(st.sampled_from([0, 0, 1, 2])) * z * m for m in monomials]
        equations[u] = sum(terms)
    return System(equations, z=z)


@settings(max_examples=20, deadline=None)
@given(small_systems())
def test_random_systems(system: System) -> None:
    curve = eliminate(system, F)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SingularityWarning)
            s = singularity(curve)
    except ValueError as error:  # linear systems in F can be polynomials
        assert "polynomial" in str(error)
        return
    candidates = _candidates(sp.Poly(curve.as_expr(), F, z))
    assert s.rho in _positive_roots(candidates)
    if s.case == "II":
        lc = sp.Poly(curve.as_expr(), F).LC()
        assert s.rho.sign(sp.Poly(lc, z)) == 0  # m_rho | lc_w G (Broda §3.3)
    if s.period == 1:
        c = series(system, 201)[F]
        ratio = Fraction(c[200]) / Fraction(c[201])
        assert abs(float(ratio) / float(s.rho) - 1) < 0.1


# -- reducible closures: the component rule (singularity.md, section 5.5) -------------


def names(closure: _Closure, component: list[int]) -> list[str]:
    return [str(closure.unknowns[i]) for i in component]


def test_components_in_dependency_order() -> None:
    closure = closure_of(SSNF, L, 108)
    assert [names(closure, c) for c in _components(closure)] == [
        ["B", "Bb"],
        ["M", "Mb"],
        ["L"],
    ]
    found = [(names(closure, c), nl, p) for c, nl, p in _component_periods(closure)]
    assert found == [(["B", "Bb"], True, 1), (["M", "Mb"], False, 1)]


def test_cycle_period() -> None:
    """PS Example 23: coordinate periods 1, cycle period 5; L_k's (M, Mb): 1."""
    closure = closure_of(PS_EXAMPLE_23, AE)
    assert _periods(closure) == [1, 1]
    val, per = _valuations(closure), _periods(closure)
    assert _cycle_period(closure, [0, 1], val, per) == 5
    for n in (2, 108):
        closure = closure_of(SSNF, L, n)
        m = [closure.unknowns.index(M), closure.unknowns.index(Mb)]
        val, per = _valuations(closure), _periods(closure)
        assert _cycle_period(closure, sorted(m), val, per) == 1
    # a single equation with a self-loop: C = 1 + 2 z^4 C has cycle period 4
    C = sp.Symbol("C")
    closure = closure_of(System({C: 1 + 2 * z**4 * C}, z=z), C)
    assert _cycle_period(closure, [0], _valuations(closure), _periods(closure)) == 4


@pytest.mark.parametrize(
    ("system", "w", "n", "unique", "period"),
    [
        (SSNF, L, 108, True, 1),  # {B, Bb} nonlinear 1, {M, Mb} linear 1
        (FK, F, 108, True, 1),  # {B, Bb} nonlinear 1
        (APERIODIC_TWO, W, None, None, 1),  # {T} has period 2
        (PERIODIC, T, None, False, 2),  # the period of w
        (PS_EXAMPLE_23, AE, None, None, 1),  # cycle period 5
        (CONSTRUCTED, F, None, True, 1),  # {G} has period 2; the circle test decides
        (CRITICAL_INPUT, Y, None, None, 1),  # {S} has period 2: not True
    ],
)
def test_component_rule(
    system: System, w: sp.Symbol, n: int | None, unique: bool | None, period: int
) -> None:
    curve = eliminate(system, w, k=n) if n is not None else eliminate(system, w)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = singularity(curve)
    assert s.unique is unique and s.period == period
    warned = [w for w in caught if issubclass(w.category, SingularityWarning)]
    assert len(warned) == (unique is None)
    if unique is None:
        assert "has period" in str(warned[0].message)


def test_l_108_minus_rho_is_regular() -> None:
    """The (M, Mb) block of J at k = 108: spectral radius 1 at ρ = 1/sqrt(972), and
    both eigenvalues of modulus 0.4226 < 1 at -ρ (singularity.md, section 5.5)."""
    s_b = singularity(eliminate(SSNF, B, k=108))
    s_bb = singularity(eliminate(SSNF, Bb, k=108))
    assert s_b.rho == singularity(eliminate(SSNF, L, k=108)).rho  # {B, Bb} critical
    assert s_b.a is not None and s_bb.a is not None
    with mpmath.workdps(30):
        rho = 1 / mpmath.sqrt(972)

        def block(x: Any, b: Any, bb: Any) -> Any:
            return mpmath.matrix(
                [
                    [4 * x * b + 2 * x * bb, 2 * x * b + 2 * x],
                    [2 * x * bb, 2 * x * b + 4 * x * bb],
                ]
            )

        at_rho = mpmath.eig(block(rho, s_b.a.evalf(30), s_bb.a.evalf(30)))[0]
        assert abs(max(abs(e) for e in at_rho) - 1) < mpmath.mpf(10) ** -20
        x = -rho
        b, bb = mpmath.mpf(0), mpmath.mpf(0)
        for _ in range(200):  # Kleene at -ρ, inside the disk of (B, Bb)
            b, bb = (
                2 * x * b**2 + 2 * x * b * bb + 2 * x * bb,
                108 * x + 2 * x * b * bb + 2 * x * bb**2,
            )
        b, bb = mpmath.findroot(
            lambda u, v: (
                2 * x * u**2 + 2 * x * u * v + 2 * x * v - u,
                108 * x + 2 * x * u * v + 2 * x * v**2 - v,
            ),
            (b, bb),
        )
        moduli = [abs(e) for e in mpmath.eig(block(x, b, bb))[0]]
        assert all(
            abs(m - mpmath.mpf("0.42265")) < mpmath.mpf(10) ** -4 for m in moduli
        )


@pytest.mark.slow
def test_critical_input_is_singular_at_minus_one() -> None:
    """CRITICAL_INPUT: Y has period 1 but a (-1)^n component in its coefficients,
    ~ n^(-7/2) against n^(-3/2): Y is singular at -1, so True would be wrong."""
    y = series(CRITICAL_INPUT, 402)[Y]
    with mpmath.workdps(40):
        e = {
            n: mpmath.mpf(y[n].numerator) / y[n].denominator * mpmath.mpf(n) ** 1.5
            for n in range(380, 403)
        }
        alternating = [
            (-1) ** n * (e[n] - (e[n - 1] + e[n + 1]) / 2) * n**2
            for n in (381, 391, 401)
        ]
    assert all(
        abs(a + mpmath.mpf("0.0308")) < mpmath.mpf("0.0005") for a in alternating
    )
