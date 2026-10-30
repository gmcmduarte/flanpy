# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.eliminate, flanpy.eliminate_in_k and flanpy.Curve.

Plan: dev/design/elimination.md, section 7.
"""

import pickle
from fractions import Fraction
from functools import cache
from typing import Any

import pytest
import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

import flanpy
from flanpy import (
    Curve,
    KCases,
    KSet,
    System,
    ZeroCoordinateWarning,
    _curve,
    eliminate,
    eliminate_in_k,
    series,
)
from flanpy._curve import (
    _dense,
    _fraction,
    _order_bound,
    _select_factor,
    _truncated_value,
)
from flanpy._elimination import _BACKEND, _candidates, _content_in_z, _eliminant

z, k = sp.symbols("z k")
F, G, C, W, A, B, R, S = sp.symbols("F G C W A B R S")
Bb, M, Mb, L = sp.symbols("Bb M Mb L")
INF = None

CATALAN = {F: 1 + z * F**2}
# Broda et al. 2014 (Hitchhiker's Guide), eqs. (7)-(9): A_k = 2z + B_k.
HG = {B: k * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}
# Broda et al. 2020: R_k from grammar (1); S_k = d/du at u = 1 (pp. 4-6).
BRODA_RS = {R: (k + 1) * z + 2 * z * R**2 + z * R, S: 4 * z * R * S + z * R + z * S}
# Broda et al. 2020, L_k: eqs. (10), (11), (15), (16), (12).
BRODA_L = {
    B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
    Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
    M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
    Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
    L: M + Mb,
}
P_K = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4  # eq. (18)
EQ17 = (
    P_K * L**3
    + k * z * ((k + 12) * k * z**2 + 4 * k * z + 1) * L
    - k**2 * z**2 * (2 * z + 1)
)
# Exceptional k (elimination.md, section 2.3).
LOSES_SQF = {C: 1 + z * C**2, W: z + (k - 1) * z * C**2}
DEGREE_DROP = {F: 1 + (k - 1) * z * F**2}
CONTENT_IN_Z = {F: 2 * z * F + z - 2 * z**2 + k * z * F**2 - 2 * z**2 * F**2}
SYMMETRIC = {F: k * z + z * G**2, G: k * z + z * F**2}
# (F - z)(1 - zG) = 0, (G - z)(1 - zG) = 0: the surface zG = 1 (section 2.1).
EXCESS = {F: z + z * G * F - z**2 * G, G: z + z * G**2 - z**2 * G}


def poly(expr: Any, *gens: Any) -> sp.Poly:
    return sp.Poly(expr, *gens, domain=sp.ZZ)


def annihilates(curve: Curve, n: int = 20) -> bool:
    """G(z, w(z)) = 0 mod z^(n+1), for a curve at a fixed k (or without k)."""
    values = series(curve.system, n)[curve.w]
    coefficients = _dense(curve.as_expr(), curve.w, curve.z, _fraction)
    return not any(_truncated_value(coefficients, values, n, Fraction(0)))


@cache
def broda_l() -> KCases[Curve]:
    return eliminate_in_k(System(BRODA_L, z=z, k=k), L)


# -- quadratic examples -----------------------------------------------------------


def test_catalan() -> None:
    system = System(CATALAN, z=z)
    curve = eliminate(system, F)
    assert curve.polynomial == poly(z * F**2 - F + 1, F, z)
    assert (curve.w, curve.z, curve.k) == (F, z, None)
    assert curve.kset == KSet.all()
    assert curve.system == system
    assert annihilates(curve)
    assert str(curve) == "F**2*z - F + 1 = 0"


@pytest.mark.parametrize("method", ["auto", "groebner", "resultant"])
def test_hitchhikers_guide(method: Any) -> None:
    cases = eliminate_in_k(System(HG, z=z, k=k), A, method=method)
    assert len(cases.pieces) == 1 and cases.covered == KSet.all()
    g = cases[1].polynomial
    expected = 2 * z * A**2 - (8 * z**2 - 2 * z + 1) * A + 8 * z**3 - 4 * z**2
    assert g == poly(expected + (k + 2) * z, A, z, k)
    # eq. (9): Delta_k(z) = (4 - 8k) z^2 - 4z + 1 is the discriminant.
    delta = (4 - 8 * k) * z**2 - 4 * z + 1
    assert sp.expand(sp.discriminant(g.as_expr(), A) - delta) == 0
    # eq. (8): the root through A(0) = 0 is (8z^2 - 2z + 1 - sqrt(Delta_k)) / (4z).
    root = (8 * z**2 - 2 * z + 1 - sp.sqrt(delta)) / (4 * z)
    assert sp.simplify(g.as_expr().subs(A, root)) == 0
    for n in (1, 2, 7):
        expansion = sp.series(root.subs(k, n), z, 0, 9).removeO()
        coefficients = [expansion.coeff(z, i) for i in range(9)]
        assert tuple(coefficients) == series(System(HG, z=z, k=k), 8, k=n)[A]


def test_broda_r_and_s() -> None:
    system = System(BRODA_RS, z=z, k=k)
    r = eliminate_in_k(system, R)
    assert r.pieces == ((KSet.all(), r[1]),)
    assert r[1].polynomial == poly(2 * z * R**2 + (z - 1) * R + (k + 1) * z, R, z, k)
    s_resultant = eliminate_in_k(system, S, method="resultant")
    s_groebner = eliminate_in_k(system, S, method="groebner")
    assert s_resultant == s_groebner
    # The resultant is -z * G_S; the extraneous factor z is dropped.
    eliminant = _eliminant(system, S, "resultant")
    assert eliminant is not None
    _, factors = sp.factor_list(eliminant.g0)
    assert {f for f, _ in factors if not f.has(S)} == {z}
    g = s_resultant[1].polynomial
    assert len(sp.factor_list(g.as_expr())[1]) == 1
    assert g.degree(S) == 2


# -- acceptance: Broda et al. 2020 ---------------------------------------------------


def test_broda_eq_17() -> None:
    cases = broda_l()
    assert cases.covered == KSet.all()
    assert len(cases.pieces) == 1
    generic, curve = cases.pieces[0]
    assert generic == KSet.all() and curve.kset == generic
    # Exactly eq. (17): the unit in Q(k) is 1.
    assert curve.polynomial == poly(EQ17, L, z, k)
    assert sp.expand(sp.Poly(curve.as_expr(), L).LC() - P_K) == 0  # eq. (18)
    assert cases.tail == (generic, curve)


@pytest.mark.parametrize("n", [*range(1, 13), 54])
def test_broda_l_specialisations(n: int) -> None:
    system = System(BRODA_L, z=z, k=k)
    at_n = eliminate(system, L, k=n)
    assert broda_l()[n].specialize(n) == at_n
    assert at_n.polynomial == sp.Poly(EQ17.subs(k, n), L, z).primitive()[1]
    assert annihilates(at_n, 30)


@pytest.mark.slow
def test_broda_l_specialisations_up_to_60() -> None:
    system = System(BRODA_L, z=z, k=k)
    cases = broda_l()
    for n in range(1, 61):
        assert cases[n].specialize(n) == eliminate(system, L, k=n)


def test_broda_denominator_f() -> None:
    # F_k = z + B_k + Bb_k, the denominator series of eq. (38).
    system = System({B: BRODA_L[B], Bb: BRODA_L[Bb], F: z + B + Bb}, z=z, k=k)
    cases = eliminate_in_k(system, F)
    assert len(cases.pieces) == 1
    g = cases[1].polynomial
    assert g.degree(F) == 3
    assert sp.factor(sp.discriminant(g.as_expr(), F)).has(k)
    assert annihilates(cases[2].specialize(2))


def test_kalkbrener_criterion_fails_at_54_and_the_curve_merges() -> None:
    # elimination.md, section 2.2, consequence 2.
    system = System(BRODA_L, z=z, k=k)
    main = [B, Bb, M, Mb, L, z]
    polys = [sp.Poly(p.as_expr(), *main, k) for p in system.polynomials()]
    basis = _BACKEND.groebner_lex(polys, [*main, k])

    def lc(g: sp.Poly) -> Any:
        return sp.Poly(g.as_expr(), *main, domain=sp.QQ[k]).LC(order="lex")

    keep = [g.as_expr().subs(k, 54) for g in basis if lc(g).subs(k, 54) != 0]
    drop = [g.as_expr().subs(k, 54) for g in basis if lc(g).subs(k, 54) == 0]
    assert (len(basis), len(keep), len(drop)) == (20, 18, 2)
    assert not any(sp.reduced(d, keep, *main, order="lex")[1] == 0 for d in drop)
    eliminant = _eliminant(system, L, "auto")
    assert eliminant is not None
    g = poly(EQ17, L, z, k)
    assert _candidates(g, eliminant, L, z, k) == KSet.of(54)
    assert broda_l().pieces[0][0] == KSet.all()  # 54 merged


# -- exceptional k -------------------------------------------------------------------


def test_loses_squarefreeness_at_1() -> None:
    cases = eliminate_in_k(System(LOSES_SQF, z=z, k=k), W)
    assert [s for s, _ in cases.pieces] == [KSet.of(1), KSet([(2, INF)])]
    g = cases[2].polynomial.as_expr()
    assert sp.factor(sp.discriminant(g, W)).has(k - 1)
    assert sp.expand(g.subs(k, 1) - z * (W - z) ** 2) == 0  # generic G fails at 1
    assert cases[1].polynomial == poly(W - z, W, z)
    assert cases[1] == eliminate(System(LOSES_SQF, z=z, k=k), W, k=1)


def test_degree_drop_at_1() -> None:
    cases = eliminate_in_k(System(DEGREE_DROP, z=z, k=k), F)
    assert [s for s, _ in cases.pieces] == [KSet.of(1), KSet([(2, INF)])]
    assert cases[2].polynomial == poly((k - 1) * z * F**2 - F + 1, F, z, k)
    assert cases[1].polynomial == poly(F - 1, F, z)


def test_content_in_z_at_1() -> None:
    # Amendment B: no other candidate set sees k = 1.
    system = System(CONTENT_IN_Z, z=z, k=k)
    cases = eliminate_in_k(system, F)
    assert [s for s, _ in cases.pieces] == [KSet.of(1), KSet([(2, INF)])]
    g = cases[2].polynomial
    assert len(sp.factor_list(g.as_expr())[1]) == 1
    at_1 = (2 * z - 1) * (z * F**2 - F + z)
    assert sp.expand(g.as_expr().subs(k, 1) - at_1) == 0
    assert g.as_expr().subs({k: 1, z: sp.Rational(1, 2)}) == 0
    assert _content_in_z(g, F, z, k) == KSet.of(1)
    assert cases[1].polynomial == poly(z * F**2 - F + z, F, z)
    eliminant = _eliminant(system, F, "auto")
    assert eliminant is not None and eliminant.conditions == ()


def test_content_in_z_is_empty_elsewhere() -> None:
    assert _content_in_z(poly(EQ17, L, z, k), L, z, k) == KSet()
    assert _content_in_z(poly((k - 1) * z * F**2 - F + 1, F, z, k), F, z, k) == KSet()
    # Linear in w: a content in z at k = 3 only.
    linear = (k - 3) * F + (z - 1) * (k - 3) + z * (z - 1)
    assert _content_in_z(poly(linear, F, z, k), F, z, k) == KSet.of(3)


# -- factor selection ----------------------------------------------------------------


@pytest.mark.parametrize("method", ["groebner", "resultant"])
def test_symmetric_factor_selection(method: Any) -> None:
    system = System(SYMMETRIC, z=z, k=k)
    eliminant = _eliminant(system, F, method)
    assert eliminant is not None
    factors = [f for f, _ in sp.factor_list(eliminant.g0)[1] if f.has(F)]
    assert len(factors) == 2
    assert _order_bound(factors, F, z) == 6
    cases = eliminate_in_k(system, F, method=method)
    assert cases.pieces[0][1].polynomial == poly(z * F**2 - F + k * z, F, z, k)


def test_selection_retries_at_another_k(monkeypatch: pytest.MonkeyPatch) -> None:
    system = System(SYMMETRIC, z=z, k=k)
    g = z * F**2 - F + k * z
    wrong = g + (k - 2) * z**3  # h(z, F(z)) = (k - 2) z^3: passes at k = 2
    tried: list[Any] = []
    original = _curve._vanishing_at

    def spy(*args: Any) -> list[Any]:
        tried.append(args[-1])
        return original(*args)

    monkeypatch.setattr(_curve, "_vanishing_at", spy)
    assert _select_factor([wrong, g], system, F) == g
    assert tried == [2, 3]
    # With only k = 2 to try, the test falls back to series in Q[k].
    monkeypatch.setattr(_curve, "_TRIAL_KS", (2,))
    assert _select_factor([wrong, g], system, F) == g


def test_selection_guard() -> None:
    with pytest.raises(AssertionError, match="0 factors vanish"):
        _select_factor([F - 1], System(CATALAN, z=z), F)


# -- saturation fallback ---------------------------------------------------------------


@pytest.mark.parametrize("method", ["auto", "groebner", "resultant"])
def test_excess_component_is_saturated(method: Any) -> None:
    system = System(EXCESS, z=z)
    polys = [sp.Poly(p.as_expr(), G, F, z) for p in system.polynomials()]
    plain = _BACKEND.groebner_lex(polys, [G, F, z])
    assert not [g for g in plain if not g.as_expr().has(G) and g.degree(F) > 0]
    assert eliminate(system, F, method=method).polynomial == poly(F - z, F, z)


# -- properties ------------------------------------------------------------------------


@st.composite
def small_systems(draw: st.DrawFn) -> dict[Any, Any]:
    """Non-negative systems in 2 or 3 unknowns, degree <= 2; every term has a factor z.

    Every term has a factor ``z``, so ``c* = 0`` and ``J0 = 0``: well-posed. Each
    unknown has a term ``z`` or ``k z``, so there are no zero coordinates. The third
    unknown ``H`` (linear) forces the Groebner path.
    """
    H = sp.Symbol("H")
    third = draw(st.booleans())
    monomials = [F, G, F**2, F * G, G**2]
    out = {}
    for u in (F, G):
        terms = [draw(st.sampled_from([1, k])) * z]
        terms += [draw(st.sampled_from([0, 0, 1, k])) * z * m for m in monomials]
        out[u] = sum(terms)
    if third:
        out[F] += z * H
        out[H] = (
            draw(st.sampled_from([1, k])) * z + draw(st.sampled_from([0, 1, k])) * z * G
        )
    return out


@settings(max_examples=16)
@given(small_systems())
def test_random_systems(equations: dict[Any, Any]) -> None:
    system = System(equations, z=z, k=k)
    cases = eliminate_in_k(system, F)
    assert cases.covered == KSet.all()
    assert cases.tail is not None
    generic = cases.tail[1]
    assert len(sp.factor_list(generic.as_expr())[1]) == 1  # irreducible, primitive
    for n in range(1, 5):
        at_n = eliminate(system, F, k=n)
        assert n in cases[n].kset
        if n in generic.kset:
            assert generic.specialize(n) == at_n
        else:
            assert cases[n] == at_n
        assert annihilates(at_n, 12)
    assert eliminate_in_k(system, F, method="groebner") == cases


# -- API -------------------------------------------------------------------------------


def test_public_names() -> None:
    for name in ("Curve", "eliminate", "eliminate_in_k", "KCases"):
        assert name in flanpy.__all__
    assert Curve.__module__ == "flanpy"


def test_errors() -> None:
    catalan = System(CATALAN, z=z)
    rs = System(BRODA_RS, z=z, k=k)
    with pytest.raises(TypeError, match=r"flanpy\.System"):
        eliminate("F = 1 + z F^2", F)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not an unknown of the system"):
        eliminate(catalan, G)
    with pytest.raises(ValueError, match="method must be one of"):
        eliminate(catalan, F, method="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="omit the k argument"):
        eliminate(catalan, F, k=2)
    with pytest.raises(ValueError, match="eliminate_in_k for symbolic k"):
        eliminate(rs, R)
    with pytest.raises(ValueError, match="k must be an integer >= 1"):
        eliminate(rs, R, k=0)
    with pytest.raises(TypeError, match="k must be an int"):
        eliminate(rs, R, k=True)
    with pytest.raises(ValueError, match="use eliminate"):
        eliminate_in_k(catalan, F)
    with pytest.raises(ValueError, match="needs at most one unknown besides L"):
        eliminate_in_k(System(BRODA_L, z=z, k=k), L, method="resultant")


def test_zero_coordinate_target() -> None:
    with pytest.warns(ZeroCoordinateWarning):
        system = System({F: z + G, G: z * G}, z=z)
    assert eliminate(system, G).polynomial == poly(G, G, z)
    assert eliminate(system, F).polynomial == poly(F - z, F, z)


def test_curve_value_semantics() -> None:
    a = eliminate(System(CATALAN, z=z), F)
    b = eliminate(System(CATALAN, z=z), F)
    assert a == b and hash(a) == hash(b)
    assert a != eliminate(System({F: 1 + 2 * z * F**2}, z=z), F)
    assert pickle.loads(pickle.dumps(a)) == a
    assert repr(a) == "Curve(F**2*z - F + 1, w=F, z=z, kset=KSet([(1, None)]))"
    r = eliminate_in_k(System(BRODA_RS, z=z, k=k), R)[1]
    assert repr(r) == (
        "Curve(2*R**2*z + R*z - R + k*z + z, w=R, z=z, k=k, kset=KSet([(1, None)]))"
    )
    assert pickle.loads(pickle.dumps(r)) == r
    with pytest.raises(AttributeError, match="immutable"):
        a._w = G
    with pytest.raises(TypeError, match=r"created by flanpy\.eliminate"):
        Curve()


def test_curve_specialize_errors() -> None:
    catalan = eliminate(System(CATALAN, z=z), F)
    with pytest.raises(ValueError, match="no parameter k"):
        catalan.specialize(1)
    cases = eliminate_in_k(System(DEGREE_DROP, z=z, k=k), F)
    generic = cases[2]
    with pytest.raises(ValueError, match="not in the curve's kset"):
        generic.specialize(1)
    with pytest.raises(TypeError, match="k must be an int"):
        generic.specialize(2.0)  # type: ignore[arg-type]
