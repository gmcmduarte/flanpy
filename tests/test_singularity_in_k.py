# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.singularity_in_k (design: dev/design/singularity.md, §6 and §8)."""

import pickle
import time
import warnings
from fractions import Fraction

import mpmath
import pytest
import sympy as sp

import flanpy
from flanpy import (
    KCases,
    KRoot,
    KSet,
    Singularity,
    SingularityWarning,
    System,
    UndecidedSingularityError,
    eliminate,
    eliminate_in_k,
    series,
    singularity,
    singularity_in_k,
)
from flanpy._singularity import _Family, _layout

z, k = sp.symbols("z k")
B, Bb, M, Mb, L, F, R, A, T = sp.symbols("B Bb M Mb L F R A T")

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
P_K = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4  # Broda eq. (18)
GENERIC = KSet([(1, 107), (109, None)])  # L_k and F_k: k = 108 is undecided


def in_k(curves: object) -> KCases[Singularity]:  # type: ignore[type-arg]
    """singularity_in_k, with the SingularityWarning (k = 108) ignored."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularityWarning)
        return singularity_in_k(curves)  # type: ignore[arg-type]


def generic_curve(system: System, w: sp.Symbol) -> flanpy.Curve:
    tail = eliminate_in_k(system, w).tail
    assert tail is not None
    return tail[1]


# -- public API ------------------------------------------------------------------------


def test_exported() -> None:
    assert "singularity_in_k" in flanpy.__all__ and "KRoot" in flanpy.__all__


def test_errors() -> None:
    with pytest.raises(TypeError, match=r"expected a flanpy\.Curve or KCases"):
        singularity_in_k("curve")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"expected a flanpy\.Curve"):
        singularity_in_k(KCases({KSet.all(): "curve"}))  # type: ignore[arg-type]
    catalan = eliminate(System({F: 1 + z * F**2}, z=z), F)
    with pytest.raises(ValueError, match=r"no parameter k; use flanpy\.singularity"):
        singularity_in_k(catalan)
    with pytest.raises(ValueError, match="singularity_in_k for every k"):
        singularity(generic_curve(RK, R))


def test_signed_in_k() -> None:
    """k - 2 < 0 at k = 1: the k = 1 + t test fails (sufficient, not necessary)."""
    cases = eliminate_in_k(System({F: 1 + (k - 2) * z * F**2}, z=z, k=k), F)
    with pytest.raises(UndecidedSingularityError, match="sufficient") as caught:
        singularity_in_k(cases)
    message = str(caught.value)
    assert "k - 2 of F**2*z" in message and "not necessary" in message
    assert "-z*(k - 2)*(4*k*z - 8*z - 1)" in message


def test_polynomial_piece() -> None:
    """F = 1 + (k - 1) z F^2 is F = 1 at k = 1: no singularity there."""
    cases = eliminate_in_k(System({F: 1 + (k - 1) * z * F**2}, z=z, k=k), F)
    with pytest.raises(ValueError, match=r"at k in \{1\}: .* is a polynomial"):
        singularity_in_k(cases)
    result = singularity_in_k(cases.restrict(KSet([(2, None)])))
    ((piece, s),) = result.pieces
    assert piece == KSet([(2, None)]) and result.uncovered == KSet.of(1)
    assert s.rho == KRoot(4 * (k - 1) * z - 1, 0, k=k)  # ρ_k = 1/(4(k - 1))
    assert s.a == KRoot(F - 2, 0, k=k) and s.case == "I" and s.unique is True


# -- quadratic examples first ------------------------------------------------------


def test_catalan_in_k() -> None:
    cases = singularity_in_k(eliminate_in_k(System({F: 1 + k * z * F**2}, z=z, k=k), F))
    ((piece, s),) = cases.pieces
    assert piece == KSet.all() and s.kset == KSet.all() and s.k == k
    assert s.rho == KRoot(4 * k * z - 1, 0, k=k) and s.a == KRoot(F - 2, 0, k=k)
    assert s.period == 1 and s.unique is True


def test_periodic_in_k() -> None:
    cases = singularity_in_k(eliminate_in_k(System({T: z + k * z * T**2}, z=z, k=k), T))
    ((piece, s),) = cases.pieces
    assert piece == KSet.all()
    assert s.rho == KRoot(4 * k * z**2 - 1, 0, k=k) and s.a == KRoot(
        k * T**2 - 1, 0, k=k
    )
    assert s.period == 2 and s.unique is False


def test_broda_r_k() -> None:
    """Broda 2020 §3.1: ρ_k is the positive root of Δ_k = 1 - 2z - (7 + 8k) z^2, and
    a_k = (1 - ρ_k)/(4ρ_k) = sqrt((k + 1)/2)."""
    curves = eliminate_in_k(RK, R)
    cases = singularity_in_k(curves)
    ((piece, s),) = cases.pieces
    assert piece == KSet.all()
    assert s.rho == KRoot(1 - 2 * z - (7 + 8 * k) * z**2, 0, k=k)
    assert s.case == "I" and s.a == KRoot(2 * R**2 - k - 1, 0, k=k)
    assert s.period == 1 and s.unique is True
    for n in range(1, 21):
        assert s.specialize(n) == singularity(curves[n], k=n)


def test_hitchhiker_a_k() -> None:
    """HG eq. (9): ρ_k is the positive root of (8k - 4) z^2 + 4z - 1."""
    curves = eliminate_in_k(HG, A)
    cases = singularity_in_k(curves)
    ((piece, s),) = cases.pieces
    assert piece == KSet.all()
    assert s.rho == KRoot((8 * k - 4) * z**2 + 4 * z - 1, 0, k=k)
    assert s.case == "I" and s.unique is True
    for n in range(1, 21):
        assert s.specialize(n) == singularity(curves[n], k=n)
        with mpmath.workdps(50):
            exact = 1 / (2 * (mpmath.sqrt(2 * n) + 1))
            assert abs(s.specialize(n).rho.evalf(50) - exact) < mpmath.mpf(10) ** -45


# -- acceptance: Broda L_k ------------------------------------------------------------


def test_broda_l_k_pieces() -> None:
    """ρ_k = the positive root of p_k (eq. (18)), Case II, for every k >= 1.

    k = 54 is a cut and merges. At k = 108, p_108 = 4(27z + 1)(972z^2 - 1), so
    -ρ_108 is a candidate on the circle; the closure of L is not irreducible, so
    uniqueness there is undecided (singularity.md, section 6.4).
    """
    start = time.perf_counter()
    with pytest.warns(SingularityWarning, match=r"undecided for k in \{108\}"):
        cases = singularity_in_k(eliminate_in_k(SSNF, L))
    print(f"singularity_in_k(eliminate_in_k(L_k)): {time.perf_counter() - start:.1f} s")
    (generic, s), (exceptional, s108) = cases.pieces
    assert generic == GENERIC and exceptional == KSet.of(108)
    assert cases.covered == KSet.all()
    assert s.rho == KRoot(P_K, 0, k=k) and s.case == "II" and s.a is None
    assert s.period == 1 and s.unique is True and s.kset == GENERIC
    assert s108.rho == KRoot(P_K, 0, k=k) and s108.case == "II"
    assert s108.unique is None
    assert s108.specialize(108).rho.polynomial.as_expr() == 972 * z**2 - 1


def test_broda_l_k_cuts() -> None:
    """54 is a cut; so is 17 + 7 sqrt(7) ≈ 35.52 (from disc_z q_k)."""
    family = _Family(generic_curve(SSNF, L))
    product = sp.Mul(*family.first)
    assert sp.rem(product, k**2 - 34 * k - 54, k) == 0
    points, spans = _layout(family.first, k)
    assert points == {54}
    assert spans == [(1, 35), (36, 53), (55, None)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularityWarning)
        family.run()
    (second,) = family._second.values()
    points, spans = _layout([*family.first, *second.polys], k)
    assert points == {54, 108}
    assert (1, 35) in spans and (36, 42) in spans


def check_lk(cases: KCases[Singularity], curves: KCases[flanpy.Curve], n: int) -> None:  # type: ignore[type-arg]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SingularityWarning)
        fixed = singularity(curves[n], k=n)
        assert cases[n].specialize(n) == fixed
    p = sp.Poly(P_K.subs(k, n), z)
    with mpmath.workdps(50):
        roots = [
            mpmath.re(r)
            for r in mpmath.polyroots(
                [int(c) for c in p.all_coeffs()], maxsteps=200, extraprec=200
            )
            if abs(mpmath.im(r)) < mpmath.mpf(10) ** -30 and mpmath.re(r) > 0
        ]
        assert len(roots) == 1
        assert abs(fixed.rho.evalf(50) - roots[0]) < mpmath.mpf(10) ** -40


@pytest.mark.parametrize("n", [1, 2, 35, 36, 54])
def test_broda_l_k_specialize(n: int) -> None:
    curves = eliminate_in_k(SSNF, L)
    check_lk(in_k(curves), curves, n)


@pytest.mark.slow
def test_broda_l_k_specialize_1_to_60() -> None:
    curves = eliminate_in_k(SSNF, L)
    cases = in_k(curves)
    start = time.perf_counter()
    for n in [*range(1, 61), 107, 108, 109]:
        check_lk(cases, curves, n)
    print(f"L_k, k = 1..60 and 107..109: {time.perf_counter() - start:.1f} s")
    assert cases[54].specialize(54).rho.polynomial.as_expr() == 45 * z - 2


@pytest.mark.slow
def test_broda_l_108_is_unique_numerically() -> None:
    """The undecided case is unique in truth: no (-1)^n term in [z^n] L_108."""
    c = series(SSNF, 600, k=108)[L]
    with mpmath.workdps(30):
        rho = 1 / mpmath.sqrt(972)
        scaled = [c[n] * rho**n * mpmath.sqrt(n) for n in (598, 599, 600)]
        middle = (scaled[0] + scaled[2]) / 2
        assert abs(scaled[1] / middle - 1) < mpmath.mpf(10) ** -5


# -- F_k: Case I ----------------------------------------------------------------------


def test_broda_f_k() -> None:
    """F_k = z + B + Bb (eq. (38)'s denominator): Case I, a_k a root of a cubic."""
    curves = eliminate_in_k(FK, F)
    cases = in_k(curves)
    (generic, s), (exceptional, s108) = cases.pieces
    assert generic == GENERIC and exceptional == KSet.of(108)
    assert s.rho == KRoot(P_K, 0, k=k) and s.case == "I" and s.unique is True
    assert isinstance(s.a, KRoot) and s.a.polynomial.degree(F) == 3
    assert s.a.polynomial.degree(k) > 0
    assert s108.unique is None and s108.a == s.a
    for n in (1, 2, 54, 55, 77, 108):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SingularityWarning)
            assert cases[n].specialize(n) == singularity(curves[n], k=n)


# -- KCases input ---------------------------------------------------------------------


def test_kcases_input_with_several_pieces() -> None:
    """Pieces are analysed separately (flat_map) and keep their own kset."""
    r_curve = generic_curve(RK, R)
    a_curve = generic_curve(HG, A)
    curves = KCases(
        {
            KSet([(1, 2)]): r_curve,
            KSet.of(3): eliminate(RK, R, k=3),
            KSet([(4, 9)]): a_curve,
        }
    )
    cases = singularity_in_k(curves)
    assert [piece for piece, _ in cases.pieces] == [
        KSet([(1, 2)]),
        KSet.of(3),
        KSet([(4, 9)]),
    ]
    assert cases.uncovered == KSet([(10, None)])
    (_, s1), (_, s3), (_, s4) = cases.pieces
    assert s1.kset == KSet([(1, 2)]) and s1.rho == KRoot(
        1 - 2 * z - (7 + 8 * k) * z**2, 0, k=k
    )
    assert s3.k is None and s3 == singularity(eliminate(RK, R, k=3))
    assert s4.kset == KSet([(4, 9)]) and s4.curve == a_curve
    assert s4.specialize(5) == singularity(a_curve, k=5)
    with pytest.raises(ValueError, match="not in the result's kset"):
        s4.specialize(10)


# -- Singularity on a piece ------------------------------------------------------------


def test_symbolic_singularity_value_semantics() -> None:
    s = singularity_in_k(eliminate_in_k(RK, R))[1]
    assert pickle.loads(pickle.dumps(s)) == s
    assert hash(pickle.loads(pickle.dumps(s))) == hash(s)
    assert s == singularity_in_k(eliminate_in_k(RK, R))[5]
    assert repr(s) == (
        "Singularity(rho=KRoot(8*k*z**2 + 7*z**2 + 2*z - 1, 0, k=k), case='I', "
        "a=KRoot(2*R**2 - k - 1, 0, k=k), period=1, unique=True)"
    )
    assert str(s) == (
        "ρ = positive root 0 of 8*k*z**2 + 7*z**2 + 2*z - 1, Case I, "
        "a = positive root 0 of 2*R**2 - k - 1, period 1, unique"
    )


def test_specialize_errors() -> None:
    s = singularity_in_k(eliminate_in_k(RK, R))[1]
    with pytest.raises(TypeError, match="k must be an int"):
        s.specialize(Fraction(2))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not in the result's kset"):
        s.specialize(0)
    fixed = s.specialize(2)
    assert fixed.k is None and fixed.kset == KSet.of(2)
    with pytest.raises(ValueError, match="at a fixed k"):
        fixed.specialize(2)
