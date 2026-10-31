# SPDX-License-Identifier: GPL-3.0-or-later
"""Ratios and symbolic k for the coefficient asymptotics (transfer.md, sections 6-7).

The Broda et al. (2020) acceptance tests: Thm 3.2 / eq. (38) / Table 1, eq. (8).
"""

from __future__ import annotations

import pickle
import warnings
from fractions import Fraction
from typing import Any

import mpmath
import pytest
import sympy as sp

from flanpy import (
    AlgebraicReal,
    Asymptotics,
    KCases,
    KSet,
    System,
    asymptotics,
    asymptotics_in_k,
    eliminate,
    eliminate_in_k,
    local_expansion,
    local_expansion_in_k,
    ratio,
    series,
    singularity,
    singularity_in_k,
)
from test_newton import FK, HG, RK, SSNF, A, F, L, R, expand, k, z

y = sp.Symbol("y")
S, U, K, T = sp.symbols("S U K T")
HALF = Fraction(1, 2)
RS = System(
    {R: (k + 1) * z + 2 * z * R**2 + z * R, S: z * R + z * S + 4 * z * S * R},
    z=z,
    k=k,
)
UK = System({U: z + U**2, K: k * U / 4 + K**2}, z=z, k=k)


def asym(system: System, w: sp.Symbol, n: int) -> Asymptotics[Any]:
    return asymptotics(expand(system, w, n))


def in_k(system: System, w: sp.Symbol) -> KCases[Asymptotics[Any]]:
    return asymptotics_in_k(
        local_expansion_in_k(singularity_in_k(eliminate_in_k(system, w)))
    )


def digits(x: Any, value: str) -> bool:
    """``x`` agrees with ``value`` to all its printed digits."""
    places = len(value.split(".")[1])
    with mpmath.workdps(40):
        return bool(
            abs(x.evalf(30) - mpmath.mpf(value)) < mpmath.mpf(10) ** -places / 2
        )


# -- Broda Thm 3.2, eq. (38), Table 1 --------------------------------------------------

# k, minimal polynomial of 2/(b_k c_k), its value (Richardson on series to n = 3000,
# from review), the n^-1 coefficient (transfer.md, section 6.2).
THM_3_2 = [
    (2, 31 * y**3 - 35 * y**2 + 14 * y - 2, "0.417256345028", "0.421941"),
    (10, 47 * y**3 - 67 * y**2 + 34 * y - 6, "0.443252417412", "0.450121"),
    (50, 127 * y**3 - 227 * y**2 + 134 * y - 26, "0.465746500119", "0.473185"),
    (54, 15 * y - 7, "0.466666666667", "0.474074"),
    (108, 27 * y**2 - 36 * y + 11, "0.474216576937", "0.481125"),
]


@pytest.mark.parametrize(("n", "poly", "limit", "next_term"), THM_3_2)
def test_broda_thm_3_2_table_1(
    n: int, poly: sp.Expr, limit: str, next_term: str
) -> None:
    """``[z^n] L_k / (n [z^n] F_k) → 2/(b_k c_k)`` (Thm 3.2, eq. (38), ``b_k > 0``)."""
    lk, fk = asym(SSNF, L, n), asym(FK, F, n)
    r = ratio(lk, fk)
    assert r.exponent == 1 and r.limit.gamma == ()
    value = r.limit.algebraic()
    assert value.polynomial == sp.Poly(poly, y)
    assert digits(value, limit)
    lk_e, fk_e = lk.expansion, fk.expansion
    assert fk_e.b is not None and lk_e.c is not None
    with mpmath.workdps(40):
        assert abs(value.evalf(30) - 2 / (fk_e.b.evalf(35) * lk_e.c.evalf(35))) < 1e-28
    terms = r.terms(3)
    assert [p for p, _ in terms] == [1, HALF, 0]
    assert terms[1][1]._is_zero() and terms[2][1].gamma == ()
    assert digits(terms[2][1].algebraic(), next_term)
    if n == 54:
        assert value == AlgebraicReal(15 * y - 7, 0)


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 10, 50])
def test_broda_ratio_at_1000(n: int) -> None:
    r = ratio(asym(SSNF, L, n), asym(FK, F, n))
    exact = Fraction(series(SSNF, 1000, k=n)[L][1000]) / series(FK, 1000, k=n)[F][1000]
    with mpmath.workdps(40):
        value = mpmath.mpf(exact.numerator) / exact.denominator
        assert abs(r.evalf(1000, count=1) / value - 1) < 1.2e-3  # the limit alone
        assert abs(r.evalf(1000, count=3, dps=30) / value - 1) < 5e-8


@pytest.mark.parametrize("n", range(1, 11))
def test_broda_eq_8(n: int) -> None:
    """``[z^n] S_k / (n [z^n] R_k) = ρ_k (1 + 1/(2n) + ...)`` (eq. (8))."""
    sk, rk = asym(RS, S, n), asym(RS, R, n)
    r = ratio(sk, rk)
    rho = rk.rho
    assert r.exponent == 1 and r.limit.gamma == () and r.limit.algebraic() == rho
    terms = r.terms(3)
    assert terms[1][1]._is_zero()
    half = sp.Poly(rho.polynomial.as_expr().subs(rho.polynomial.gen, 2 * y), y)
    assert terms[2][1].algebraic() == AlgebraicReal(half, rho.index)


# -- symbolic k ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("system", "w", "case"), [(SSNF, L, "II"), (FK, F, "I"), (RK, R, "I"), (HG, A, "I")]
)
def test_one_piece(system: System, w: sp.Symbol, case: str) -> None:
    cases = in_k(system, w)
    ((part, a),) = cases.pieces
    assert part == KSet([(1, None)]) and not cases.uncovered
    e = a.expansion
    assert e.singularity.case == case and a.exponent == (
        HALF - 1 if case == "II" else -HALF - 1
    )
    lead = a.leading
    assert lead.gamma == ((HALF, -1),)
    if case == "I":
        assert (lead.rational, lead.factors) == (HALF, ((e.b, 1),))
    else:
        assert (lead.rational, lead.factors) == (Fraction(1), ((e.c, -1),))
    assert a.error_exponent(1) == a.exponent - 1
    for n in (1, 2, 54):
        assert lead.at(n) == a.specialize(n).leading == asym(system, w, n).leading


def test_broda_ratio_in_k() -> None:
    """``L_k/F_k`` by ``combine``: ``2 b_k^{-1} c_k^{-1}`` on ``[1, ∞)``."""
    cl, cf = in_k(SSNF, L), in_k(FK, F)
    cases = cl.combine(cf, ratio)
    ((part, r),) = cases.pieces
    assert part == KSet([(1, None)])
    lk, fk = cl.pieces[0][1].expansion, cf.pieces[0][1].expansion
    assert r.exponent == 1 and r.limit.rational == 2 and r.limit.gamma == ()
    assert set(r.limit.factors) == {(lk.c, -1), (fk.b, -1)}
    assert r.kset == part
    for n in (2, 54):  # all of 1..60 and 108 in the slow test below
        fixed = ratio(asym(SSNF, L, n), asym(FK, F, n))
        assert r.limit.at(n) == fixed.limit == r.specialize(n).limit
    with pytest.raises(ValueError, match="fixed k"):
        r.terms(1)
    with pytest.raises(ValueError, match="fixed k"):
        r.evalf(10)
    assert pickle.loads(pickle.dumps(r)) == r


@pytest.mark.slow
def test_broda_ratio_in_k_all() -> None:
    """``at(n)`` equals fixed ``k`` for ``k = 1..60`` and 108."""
    ((_, r),) = in_k(SSNF, L).combine(in_k(FK, F), ratio).pieces
    ((_, lk),) = in_k(SSNF, L).pieces
    for n in [*range(1, 61), 108]:
        a, b = asym(SSNF, L, n), asym(FK, F, n)
        assert r.limit.at(n) == ratio(a, b).limit
        assert lk.leading.at(n) == a.leading


def test_family_three_pieces() -> None:
    """``K = kU/4 + K^2``: ``{1}``, ``{2}`` (``α = 1/4``, ``Γ(3/4)``), ``[3, ∞)``."""
    cases = in_k(UK, K)
    parts = [(str(p), a.leading.gamma) for p, a in cases.pieces]
    assert parts == [
        ("{1}", ((HALF, -1),)),
        ("{2}", ((Fraction(3, 4), -1),)),
        ("[3, ∞)", ((HALF, -1),)),
    ]
    for n in (1, 2, 3, 4):
        a = cases[n]
        assert a.leading.at(n) == a.specialize(n).leading == asym(UK, K, n).leading
    same = cases.combine(cases, ratio)  # f only on the non-empty intersections
    assert [str(p) for p, _ in same.pieces] == ["{1}", "{2}", "[3, ∞)"]
    assert all(r.limit.rational == 1 and not r.limit.factors for _, r in same.pieces)
    with pytest.raises(ValueError, match="no k in common"):
        ratio(cases[1], cases[2])


def test_uncovered_propagated() -> None:
    e = local_expansion_in_k(singularity_in_k(eliminate_in_k(RK, R)))
    cases = asymptotics_in_k(e.restrict(KSet([(1, 10)])))
    assert cases.uncovered == KSet([(11, None)])
    single = asymptotics_in_k(e.pieces[0][1])
    assert single.pieces[0][1] == cases[1]


def test_refusals_in_k() -> None:
    periodic = System({T: k * z + z * T**2}, z=z, k=k)  # odd powers, period 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e = local_expansion_in_k(singularity_in_k(eliminate_in_k(periodic, T)))
    with pytest.raises(ValueError, match=r"dominant singularity on \[1, ∞\)"):
        asymptotics_in_k(e)
    with pytest.raises(ValueError, match=r"use flanpy\.asymptotics"):
        asymptotics_in_k(expand(RK, R, 2))
    with pytest.raises(ValueError, match=r"use flanpy\.asymptotics"):
        asymptotics_in_k(KCases({KSet.of(2): expand(RK, R, 2)}))
    with pytest.raises(TypeError, match="LocalExpansion"):
        asymptotics_in_k(singularity_in_k(eliminate_in_k(RK, R)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LocalExpansion values"):
        asymptotics_in_k(KCases({KSet.of(2): 3}))  # type: ignore[arg-type]
    rk, lk = in_k(RK, R).pieces[0][1], in_k(SSNF, L).pieces[0][1]
    with pytest.raises(ValueError, match="ρ differs"):
        ratio(rk, lk)


def test_symbolic_asymptotics_api() -> None:
    ((_, a),) = in_k(RK, R).pieces
    for call in (
        lambda: a.terms(1),
        lambda: a.evalf(10),
        lambda: a.as_expr(sp.Symbol("n")),
    ):
        with pytest.raises(ValueError, match="fixed k"):
            call()
    lead = a.leading
    for call in (lead.algebraic, lead.evalf, lead.as_expr):
        with pytest.raises(ValueError, match="fixed k"):
            call()
    assert str(lead).startswith("1/2 · [positive root 0 of") and str(lead).endswith(
        "Γ(1/2)^-1"
    )
    assert repr(lead).startswith(
        "GammaConstant(rational=Fraction(1, 2), factors=((KRoot("
    )
    assert str(a).startswith("ρ^-n n^-3/2 · 1/2 · [") and str(a).endswith(
        "(1 + O(n^-1))"
    )
    assert pickle.loads(pickle.dumps(a)) == a
    assert pickle.loads(pickle.dumps(lead)) == lead and hash(lead) == hash(lead * 1)
    one = lead / lead
    assert one.rational == 1 and one.factors == () and one.gamma == ()
    fixed = asym(RK, R, 2).leading
    with pytest.raises(ValueError, match="fixed-k constant with a symbolic one"):
        lead * fixed
    assert lead != fixed
    with pytest.raises(ValueError, match="nothing to specialize"):
        asym(RK, R, 2).specialize(2)
    assert asymptotics(eliminate_free_piece(), k=3) == asym(RK, R, 3)


def eliminate_free_piece() -> Any:
    return local_expansion_in_k(singularity_in_k(eliminate_in_k(RK, R))).pieces[0][1]


def test_fixed_singularity_has_no_k() -> None:
    a = asymptotics(local_expansion(singularity(eliminate(RK, R, k=2))))
    assert a.k is None and a.kset == KSet.of(2)
