# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the local expansion at the dominant singularity (newton.md, section 8)."""

from __future__ import annotations

import pickle
import warnings
from fractions import Fraction
from typing import Any

import mpmath
import pytest
import sympy as sp
from hypothesis import given, settings

import flanpy._newton as newton
from flanpy import (
    AlgebraicReal,
    LocalExpansion,
    Singularity,
    SingularityWarning,
    System,
    eliminate,
    eliminate_in_k,
    local_expansion,
    singularity,
    singularity_in_k,
)
from flanpy._field import _identity, _NumberField
from flanpy._newton import (
    _binomial_roots,
    _Branch,
    _candidates,
    _field_of_a,
    _Germ,
    _LocalProblem,
    _negate,
    _order,
    _real_branches,
    _select,
)
from flanpy._singularity import _window_of
from test_singularity import small_systems

z, k, s_, Y_, y = sp.symbols("z k s Y y")
F, G, R, B, Bb, M, Mb, L, A, U, K, Yu, W, T = sp.symbols(
    "F G R B Bb M Mb L A U K Y W T"
)

CATALAN = System({F: 1 + z * F**2}, z=z)
MOTZKIN = System({F: z + z * F + z * F**2}, z=z)  # FS Example VII.18
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
QUARTER = System({U: z + U**2, K: U / 2 + K**2}, z=z)  # K = (1 - s^(1/4))/2
POLE = System({Yu: 1 + 2 * z * Yu}, z=z)  # 1/(1 - 2z)
THREE_HALVES = System({B: z + B**2, Yu: 1 + 2 * B * Yu, W: Yu**3}, z=z)  # s^(-3/2)
CONSTRUCTED = System({F: z + z * F**2 + z * G, G: z + z * G**2}, z=z)
S5_TREE = System({T: z * (1 + T**2 + T**5)}, z=z)  # m_rho has Galois group S_5

EXAMPLES: list[tuple[str, System, sp.Symbol, int | None]] = [
    ("catalan", CATALAN, F, None),
    ("motzkin", MOTZKIN, F, None),
    ("r_2", RK, R, 2),
    ("a_2", HG, A, 2),
    ("quarter", QUARTER, K, None),
    ("pole", POLE, Yu, None),
    ("three_halves", THREE_HALVES, W, None),
    ("constructed", CONSTRUCTED, F, None),
    ("s5_tree", S5_TREE, T, None),
    ("l_2", SSNF, L, 2),
    ("f_2", FK, F, 2),
]


def sing(system: System, w: sp.Symbol, n: int | None = None) -> Singularity[Any]:
    curve = eliminate(system, w) if n is None else eliminate(system, w, k=n)
    return singularity(curve)


def expand(system: System, w: sp.Symbol, n: int | None = None) -> LocalExpansion[Any]:
    return local_expansion(sing(system, w, n))


def real(expr: sp.Expr, index: int) -> AlgebraicReal:
    return AlgebraicReal(sp.Poly(expr, y), index)


def rational(x: AlgebraicReal) -> Fraction:
    """The value of a rational AlgebraicReal (its interval is a point)."""
    lo, hi = x.interval
    assert lo == hi
    return lo


def rationals(
    terms: list[tuple[Fraction, AlgebraicReal]],
) -> list[tuple[Fraction, Fraction]]:
    return [(e, rational(d)) for e, d in terms]


def mp(x: AlgebraicReal, dps: int = 50) -> mpmath.mpf:
    return x.evalf(dps)


def close(a: mpmath.mpf, b: mpmath.mpf, digits: int = 40) -> bool:
    return bool(abs(a - b) <= mpmath.mpf(10) ** -digits * (1 + abs(b)))


# -- private: Newton-Puiseux on a given polynomial Q(s, Y) over Q ----------------------


def branches_of(expr: sp.Expr) -> list[_Branch]:
    """The real branches of Q(s, Y) = 0 at (0, 0), for s > 0."""
    Q = _NumberField.rationals()
    poly = {
        (i, j): Q(Fraction(int(c.p), int(c.q)))
        for (i, j), c in sp.Poly(expr, s_, Y_).terms()
    }
    return _real_branches(
        _Germ(Q, _identity, poly, 1, [], 0), None, None, top_level=True
    )


def check_expansion(expr: sp.Expr, branch: _Branch, order: int) -> None:
    """Exact: Q(t^κ, Y_N(t)) has no term of degree <= N (Y_N = Y truncated at t^N)."""
    germ = branch.germ()
    L, kappa = germ.field, germ.kappa
    ys = [L(0)] + [germ.coefficient(n) for n in range(1, order + 1)]
    total = [L(0)] * (order + 1)
    for (i, j), c in sp.Poly(expr, s_, Y_).terms():
        power = [L(1)] + [L(0)] * order
        for _ in range(j):
            power = [
                sum((power[a] * ys[n - a] for a in range(n + 1)), L(0))
                for n in range(order + 1)
            ]
        for n in range(order + 1 - kappa * i):
            total[n + kappa * i] = total[n + kappa * i] + power[n] * Fraction(
                int(c.p), int(c.q)
            )
    assert all(x.is_zero() for x in total)


def test_cube_root() -> None:
    """Y^3 = s: one real branch Y = s^(1/3) (alpha = 1/3, private level only)."""
    (b,) = branches_of(Y_**3 - s_)
    assert (b.gamma, rational(b.leading), b.kappa) == (Fraction(1, 3), Fraction(1), 3)
    check_expansion(Y_**3 - s_, b, 6)


def test_three_halves_type() -> None:
    """Y^2 = s^3: Y = ±s^(3/2)."""
    bs = _candidates(branches_of(Y_**2 - s_**3), "I")
    assert [(x.gamma, rational(x.leading)) for x in bs] == [
        (Fraction(3, 2), Fraction(-1)),
        (Fraction(3, 2), Fraction(1)),
    ]


def test_two_edges() -> None:
    """(Y - s)(Y^2 - s): edges of slopes -1 and -1/2; the steepest is not singular."""
    expr = sp.expand((Y_ - s_) * (Y_**2 - s_))
    ordered = _candidates(branches_of(expr), "I")
    assert [(x.gamma, rational(x.leading)) for x in ordered] == [
        (Fraction(1, 2), Fraction(-1)),
        (Fraction(1), Fraction(1)),
        (Fraction(1, 2), Fraction(1)),
    ]
    for b in ordered:
        check_expansion(expr, b, 6)


def test_non_radical_edge() -> None:
    """Y^10 - s^4 Y^2 - s^5: φ(c) = c^10 - c^2 - 1 = ψ(c^2), ψ = u^5 - u - 1 (S_5)."""
    expr = Y_**10 - s_**4 * Y_**2 - s_**5
    bs = _candidates(branches_of(expr), "I")
    assert len(bs) == 2
    for b in bs:
        assert b.gamma == Fraction(1, 2) and b.kappa == 2
        assert b.leading.polynomial == sp.Poly(y**10 - y**2 - 1, y)
        assert b.germ().field.degree == 10
        check_expansion(expr, b, 4)
    assert sp.galois_group(sp.Poly(y**5 - y - 1, y))[0].order() == 120


def test_same_sign_decided_by_rank() -> None:
    """(Y^2 - s)(Y^2 - 4s): two branches with Y < 0; the rank picks one."""
    expr = sp.expand((Y_**2 - s_) * (Y_**2 - 4 * s_))
    bs = branches_of(expr)
    ordered = _candidates(bs, "I")
    assert [rational(x.leading) for x in ordered] == [
        Fraction(v) for v in (-2, -1, 1, 2)
    ]
    assert _select(bs, "I", 4, 0).leading == -2
    assert _select(bs, "I", 4, 1).leading == -1
    with pytest.raises(RuntimeError, match="sign rule"):
        _select(bs, "I", 4, 2)
    with pytest.raises(RuntimeError, match="real sheets"):
        _select(bs, "I", 3, 0)
    # Case II keeps V > 0 and orders by w = 1/V: the larger V is the smaller w.
    assert [rational(x.leading) for x in _candidates(bs, "II")] == [
        Fraction(2),
        Fraction(1),
    ]


def test_tie_broken_at_the_second_level() -> None:
    """(Y - s - s^2)(Y - s + s^2): a double root c = 1, separated by ±s^2."""
    expr = sp.expand((Y_ - s_ - s_**2) * (Y_ - s_ + s_**2))
    low, high = _candidates(branches_of(expr), "I")
    assert low.leading == high.leading == 1
    assert _order(low, high) == -1 and _order(high, low) == 1
    assert low.germ().coefficient(2) == -1 and high.germ().coefficient(2) == 1


def test_binomial_roots() -> None:
    Q = _NumberField.rationals()
    assert _binomial_roots(Q(-8), 3) == [real(y + 2, 0)]
    assert _binomial_roots(Q(-8), 2) == []
    assert _binomial_roots(Q(2), 2) == [real(y**2 - 2, 0), real(y**2 - 2, 1)]
    K = _NumberField(AlgebraicReal(z**2 - 2, 1))
    (root,) = _binomial_roots(-K.generator, 5)  # -2^(1/10)
    assert root == real(y**10 - 2, 0)


# -- public: the examples --------------------------------------------------------------


def test_catalan() -> None:
    e = expand(CATALAN, F)
    assert (e.alpha, e.ramification, e.b, e.c) == (Fraction(1, 2), 2, 2, None)
    assert rationals(e.terms(5)) == [
        (Fraction(i, 2), Fraction((-1) ** i * 2)) for i in range(5)
    ]


def as_real(value: sp.Expr) -> AlgebraicReal:
    """Return ``value`` as an AlgebraicReal (identified numerically, then exact)."""
    m = sp.minimal_polynomial(value, y)
    roots = [AlgebraicReal(m, i) for i in range(int(sp.Poly(m, y).count_roots()))]
    (match,) = [r for r in roots if abs(mp(r) - sp.N(value, 50)) < 1e-30]
    return match


def test_motzkin_terms() -> None:
    """FS Example VII.18: 1 - 3δ + 9/2 δ^2 - 63/8 δ^3 + 27/2 δ^4 - 2997/128 δ^5."""
    e = expand(MOTZKIN, F)
    printed = [1, -3, sp.Rational(9, 2), -sp.Rational(63, 8), sp.Rational(27, 2)]
    printed.append(-sp.Rational(2997, 128))
    for (exponent, d), (n, c) in zip(e.terms(6), enumerate(printed), strict=True):
        assert exponent == Fraction(n, 2)
        assert d == as_real(c / sp.sqrt(3) ** n)  # δ = t/√3 with t = s^(1/2)


@pytest.mark.parametrize("n", range(1, 21))
def test_hitchhiker_a_k(n: int) -> None:
    """HG eq. (14) via A_k = (F_k + 8z^2 - 2z)/(4z): b_k^2 = (1 - 2ρ_k)/(8ρ_k^2)."""
    e = expand(HG, A, n)
    rho = e.singularity.rho
    K = _NumberField(rho)
    r = K.generator
    assert e.alpha == Fraction(1, 2)
    assert e.b == _binomial_roots((1 - 2 * r) / (8 * r * r), 2)[1]


@pytest.mark.parametrize("n", range(1, 21))
def test_broda_r_k(n: int) -> None:
    """Broda 2020 §3.1: R_k ~ -√(2 - 2ρ_k)/(4ρ_k) (1 - z/ρ_k)^(1/2)."""
    e = expand(RK, R, n)
    r = _NumberField(e.singularity.rho).generator
    assert e.alpha == Fraction(1, 2)
    assert e.b == _binomial_roots((2 - 2 * r) / (16 * r * r), 2)[1]


def test_quarter() -> None:
    e = expand(QUARTER, K)
    assert (e.alpha, e.ramification, e.b) == (Fraction(1, 4), 4, Fraction(1, 2))
    assert [d for _, d in e.terms(6)] == [Fraction(1, 2), Fraction(-1, 2), 0, 0, 0, 0]


def test_pole_and_three_halves() -> None:
    e = expand(POLE, Yu)
    assert e.alpha == 1 and e.ramification == 1 and e.b is None and e.c == 1
    assert rationals(e.terms(3)) == [
        (Fraction(-1), Fraction(1)),
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
    ]
    e = expand(THREE_HALVES, W)
    assert (e.alpha, e.ramification, e.c) == (Fraction(3, 2), 2, 1)
    assert rationals(e.terms(3)) == [
        (Fraction(-3, 2), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(-1, 2), Fraction(0)),
    ]


def test_constructed_and_s5_tree() -> None:
    e = expand(CONSTRUCTED, F)
    assert e.b is not None and e.b.polynomial.degree() == 8
    # FS Thm VII.3: T = zφ(T), φ = 1 + y^2 + y^5: b = √(2φ(τ)/φ''(τ)), τ = a.
    e = expand(S5_TREE, T)
    s = e.singularity
    assert s.rho.polynomial.degree() == 5
    assert sp.galois_group(s.rho.polynomial)[0].order() == 120
    K = _NumberField(s.rho)
    # a = τ lies in Q(ρ): ρ = τ/φ(τ), so τ is the root of ρ φ(τ) - τ near a.
    (tau,) = [r.value for r in K.real_roots(_tree_poly(K)) if r.value.to_real() == s.a]
    phi = 1 + tau**2 + tau**5
    assert e.b == _binomial_roots(2 * phi / (2 + 20 * tau**3), 2)[1]


def _tree_poly(K: _NumberField) -> list[Any]:
    """ρ φ(τ) - τ as a polynomial in τ over Q(ρ)."""
    r = K.generator
    return [r, K(-1), r, K(0), K(0), r]  # ρ - τ + ρτ^2 + ρτ^5


# -- Broda 2020 L_k and F_k ------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 10, 50, 54, 108])
def test_broda_l_k_and_f_k(n: int) -> None:
    """L_k: α = 1/2 (eq. (32)), c_k = eq. (34); F_k: b_k > 0 (eq. (37), sign note)."""
    e = expand(SSNF, L, n)
    assert (e.singularity.case, e.alpha, e.ramification, e.b) == (
        "II",
        Fraction(1, 2),
        2,
        None,
    )
    rho, kk = mp(e.singularity.rho), mpmath.mpf(n)
    with mpmath.workdps(60):
        hz = 12 * kk * (2 * kk + 27) * rho**2 + 72 * kk * rho - kk  # p_k'(ρ)
        hvv = kk * ((kk + 12) * kk * rho**2 + 4 * kk * rho + 1)
        assert e.c is not None and close(mp(e.c), mpmath.sqrt(hz / hvv))
    f = expand(FK, F, n)
    assert f.singularity.rho == e.singularity.rho
    assert (f.alpha, f.ramification) == (Fraction(1, 2), 2)
    assert f.b is not None and f.b > 0
    if n == 54:
        assert e.c == real(49 * y**2 - 15, 1)  # √15/7
        assert f.b == real(y**2 - 60, 1)  # 2√15
        assert 2 / (mp(f.b) * mp(e.c)) - mpmath.mpf(7) / 15 < mpmath.mpf(10) ** -40


# -- the branch checks on every example ------------------------------------------------


@pytest.mark.parametrize("example", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_window_counts_and_sign_rule(example: tuple[Any, ...]) -> None:
    _, system, w, n = example
    s = sing(system, w, n)
    g = sp.Poly(s.curve.polynomial.as_expr(), s.curve.w, z, domain=sp.QQ)
    window = _window_of(s)
    fiber = sp.Poly(g.as_expr().subs(z, sp.Rational(window.x0)), s.curve.w)
    left = None if window.left is None else sp.Rational(window.left)
    right = None if window.right is None else sp.Rational(window.right)
    n_w = fiber.count_roots(left, right)
    branches = _LocalProblem.build(g, s.rho, s.case, s.a).branches()
    candidates = _candidates(branches, s.case)
    assert len(candidates) == n_w
    # The sign rule picks the same branch as the rank test (one candidate each).
    want = -1 if s.case == "I" else 1
    (signed,) = [b for b in candidates if b.sign == want]
    e = local_expansion(s)
    if s.case == "I":
        assert e.b == _negate(signed.leading)
    else:
        assert e.c == signed.leading


def test_zero_test_cascade() -> None:
    newton._STATS.clear()
    expand(SSNF, L, 2)  # Case II: everything in Q(ρ)
    assert "remainder mod m_rho" in newton._STATS
    assert set(newton._STATS) <= {"zero polynomial", "remainder mod m_rho"}
    # Broda's gcd test and the field, forced on F_2 (where a is in Q(ρ)).
    s = sing(FK, F, 2)
    g = sp.Poly(s.curve.polynomial.as_expr(), F, z, domain=sp.QQ)
    problem = _LocalProblem.build(g, s.rho, s.case, s.a)
    problem.in_rho = False
    newton._STATS.clear()
    assert problem.nonzero(1, 0)  # G_z(ρ, a) != 0: proved by the gcd test
    assert not problem.nonzero(0, 1)  # G_w(ρ, a) = 0 (eq. (19)): inconclusive, field
    assert not problem.nonzero(0, 0)
    assert newton._STATS == {"gcd test": 1, "field": 2}


def test_a_in_q_rho_branches() -> None:
    """Section 2.3: a = -h_0/h_1, a from (w - a)^e, and a not in Q(ρ)."""
    rho = AlgebraicReal(z - 1, 0)  # ρ = 1
    k_rho = _NumberField(rho)
    w_ = sp.Symbol("w")

    def poly(expr: sp.Expr) -> dict[tuple[int, int], Fraction]:
        return {
            (dw, dz): Fraction(int(c.p), int(c.q))
            for (dw, dz), c in sp.Poly(sp.expand(expr), w_, z).terms()
        }

    field, _, a, in_rho = _field_of_a(
        poly((w_ - 3) ** 2 * (w_ + 1) + z - 1), k_rho, real(y - 3, 0)
    )
    assert field == k_rho and a == 3 and in_rho
    field, _, a, in_rho = _field_of_a(
        poly((w_ - 3) ** 4 + z - 1), k_rho, real(y - 3, 0)
    )
    assert a == 3 and in_rho
    sqrt2 = real(y**2 - 2, 1)
    field, rho_k, a, in_rho = _field_of_a(
        poly((w_**2 - 2) ** 2 + (z - 1) * w_), k_rho, sqrt2
    )
    assert not in_rho and field.degree == 2 and a.to_real() == sqrt2 and rho_k == 1


# -- properties ------------------------------------------------------------------------


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
    e = local_expansion(s)
    kappa = e.ramification
    if s.case == "I":
        assert kappa >= 2 and kappa & (kappa - 1) == 0  # PS Thm 8.5
        assert e.alpha == Fraction(1, kappa) and e.b is not None and e.b > 0
    else:
        assert e.c is not None and e.c > 0
    # The coefficient check at n = 300 is in test_transfer.py (test_random_systems).


# -- API -------------------------------------------------------------------------------


def test_errors() -> None:
    s = sing(CATALAN, F)
    with pytest.raises(TypeError, match="Singularity"):
        local_expansion(eliminate(CATALAN, F))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no parameter k"):
        local_expansion(s, k=2)
    piece = singularity_in_k(eliminate_in_k(RK, R))[2]
    with pytest.raises(ValueError, match="give k=n"):
        local_expansion(piece)
    with pytest.raises(ValueError, match="not in"):
        local_expansion(piece, k=0)
    e = local_expansion(s)
    with pytest.raises(TypeError):
        e.terms(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        e.terms(-1)
    with pytest.raises(ValueError, match="fixed k"):
        e.specialize(2)
    with pytest.raises(TypeError, match="created by"):
        LocalExpansion()
    with pytest.raises(AttributeError):
        e._alpha = Fraction(1)


def test_value_semantics_and_caching() -> None:
    s = sing(CATALAN, F)
    e, f = local_expansion(s), local_expansion(s)
    assert e == f and hash(e) == hash(f) and e != local_expansion(sing(MOTZKIN, F))
    assert e.k is None and e.kset == s.kset and e.singularity is s
    first = e.terms(4)
    assert e.terms(2) == first[:2] and e.terms(4) == first and e.terms(0) == []
    again = pickle.loads(pickle.dumps(e))
    assert again == e and again.terms(4) == first
    assert repr(e).startswith("LocalExpansion(alpha=Fraction(1, 2), ramification=2")
    assert str(e) == "α = 1/2, κ = 2, Case I, b = 2"
