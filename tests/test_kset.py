# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.KSet and flanpy.KCases (design: dev/design/kset.md, §6)."""

import itertools
import pickle
from fractions import Fraction
from typing import Any

import pytest
import sympy as sp
from hypothesis import given
from hypothesis import strategies as st

import flanpy
from flanpy._kset import KCases, KSet, Relation, _sign_pieces

z, k, w, x = sp.symbols("z k w x")
INF = None
RELATIONS: tuple[Relation, ...] = ("==", "!=", "<", "<=", ">", ">=")
Model = tuple[frozenset[int], bool]


def holds(sign: int, rel: str) -> bool:
    return {
        "==": sign == 0,
        "!=": sign != 0,
        "<": sign < 0,
        "<=": sign <= 0,
        ">": sign > 0,
        ">=": sign >= 0,
    }[rel]


def coefficients(p: sp.Expr) -> list[Fraction]:
    return [
        Fraction(int(c.p), int(c.q)) for c in sp.Poly(p, k, domain=sp.QQ).all_coeffs()
    ]


def exact_sign(cs: list[Fraction], n: int) -> int:
    """Sign at n of the polynomial with coefficients cs, by Horner over Fraction
    (no lambdify, no floats)."""
    v = Fraction(0)
    for c in cs:
        v = v * n + c
    return (v > 0) - (v < 0)


def check_against_brute_force(p: sp.Expr, bound: int, den: sp.Expr = 1) -> None:
    """Every relation agrees with exact signs on 1..bound; beyond bound (no roots of
    p or den there), the sign is that of the leading coefficient of p·den."""
    cp, cd = coefficients(p), coefficients(den)
    signs = {n: (exact_sign(cp, n), exact_sign(cd, n)) for n in range(1, bound + 1)}
    tail_sign = exact_sign(coefficients(sp.expand(p * den)), bound + 10**6)
    for rel in RELATIONS:
        s = KSet.where(p, rel, k=k, den=den)
        for n, (sp_, sd) in signs.items():
            assert (n in s) is (sd != 0 and holds(sp_ * sd, rel)), (p, den, rel, n)
        tail = KSet([(bound + 1, INF)])
        assert (tail <= s) if holds(tail_sign, rel) else s.isdisjoint(tail)


# -- representation -------------------------------------------------------------


def test_canonical_form() -> None:
    assert KSet([(7, 9), (1, 2), (3, 3), (8, 12), (20, INF), (25, 30)]).intervals == (
        (1, 3),
        (7, 12),
        (20, INF),
    )
    assert KSet([(5, INF), (1, INF)]).intervals == ((1, INF),)
    assert KSet([(1, 2), (4, 5)]).intervals == ((1, 2), (4, 5))
    assert KSet().intervals == ()
    assert KSet.of(3, 1, 2, 2, 10).intervals == ((1, 3), (10, 10))
    assert KSet.of().intervals == ()
    assert KSet.all().intervals == ((1, INF),)
    assert KSet([(sp.Integer(2), sp.Integer(4))]).intervals == ((2, 4),)
    assert all(
        type(e) is int for iv in KSet([(sp.Integer(2), 4)]).intervals for e in iv
    )


@pytest.mark.parametrize(
    ("intervals", "error", "match"),
    [
        ([(0, 5)], ValueError, "starts below 1"),
        ([(-3, INF)], ValueError, "starts below 1"),
        ([(5, 4)], ValueError, "lo > hi"),
        ([(INF, 5)], TypeError, "only hi can be None"),
        ([(True, 5)], TypeError, "integer"),
        ([(1, 2.0)], TypeError, "integer"),
        ([(1.0, 2)], TypeError, "integer"),
        ([(1, 2, 3)], TypeError, "pair"),
        ([5], TypeError, "pair"),
    ],
)
def test_constructor_errors(intervals: Any, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        KSet(intervals)


def test_of_errors() -> None:
    with pytest.raises(ValueError, match="below 1"):
        KSet.of(0)
    with pytest.raises(TypeError, match="integer"):
        KSet.of(2.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("s", "text", "rep"),
    [
        (KSet(), "∅", "KSet()"),
        (KSet.of(54), "{54}", "KSet([(54, 54)])"),
        (
            KSet([(1, 53), (55, INF)]),
            "[1, 53] ∪ [55, ∞)",
            "KSet([(1, 53), (55, None)])",
        ),
        (
            KSet([(2, 2), (5, 5), (7, 9)]),
            "{2} ∪ {5} ∪ [7, 9]",
            "KSet([(2, 2), (5, 5), (7, 9)])",
        ),
        (KSet([(3, 4)]), "[3, 4]", "KSet([(3, 4)])"),
        (KSet.all(), "[1, ∞)", "KSet([(1, None)])"),
    ],
)
def test_printing(s: KSet, text: str, rep: str) -> None:
    assert str(s) == text
    assert repr(s) == rep
    assert eval(rep, {"KSet": KSet}) == s


def test_value_semantics() -> None:
    a = KSet([(1, 53), (55, INF)])
    b = KSet([(55, INF), (1, 50), (51, 53)])
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b, KSet.of(54)}) == 2
    assert a != KSet.all()
    assert a != a.intervals  # no equality with other types
    assert pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(AttributeError, match="immutable"):
        a._intervals = ()
    with pytest.raises(AttributeError, match="immutable"):
        del a._intervals
    with pytest.raises(AttributeError):
        a.other = 1


def test_public_name() -> None:
    assert flanpy.KSet is KSet
    assert "KSet" in flanpy.__all__
    assert KSet.__module__ == "flanpy"
    assert flanpy.KCases is KCases
    assert "KCases" in flanpy.__all__
    assert KCases.__module__ == "flanpy"


# -- operations -----------------------------------------------------------------


def test_operations_by_hand() -> None:
    a = KSet([(1, 10), (20, INF)])
    b = KSet([(5, 25)])
    assert a & b == KSet([(5, 10), (20, 25)])
    assert a | b == KSet.all()
    assert a - b == KSet([(1, 4), (26, INF)])
    assert ~a == KSet([(11, 19)])
    assert ~KSet() == KSet.all()
    assert ~KSet.all() == KSet()
    assert ~KSet.of(1) == KSet([(2, INF)])
    assert KSet.of(5) <= b < KSet.all()
    assert KSet.all() >= b > KSet.of(5)
    assert b <= b and not b < b
    assert not a <= b
    assert KSet([(1, 4)]).isdisjoint(b)
    assert not a.isdisjoint(b)
    assert bool(a) and not bool(KSet())


def test_operators_with_other_types() -> None:
    a = KSet.all()
    for op in ("__and__", "__or__", "__sub__", "__le__", "__lt__", "__ge__", "__gt__"):
        assert getattr(a, op)({1, 2}) is NotImplemented
    with pytest.raises(TypeError):
        a & {1}
    with pytest.raises(TypeError, match="expected a KSet"):
        a.isdisjoint({1})  # type: ignore[arg-type]


def test_membership() -> None:
    a = KSet([(2, 4), (10, INF)])
    assert [n for n in range(-2, 13) if n in a] == [2, 3, 4, 10, 11, 12]
    assert 10**40 in a
    assert sp.Integer(3) in a
    assert 0 not in KSet.all()
    for bad in (True, 2.5, "3", sp.Rational(1, 2), k):
        with pytest.raises(TypeError, match="integer"):
            bad in a  # noqa: B015


def test_tail_finite_part_and_iteration() -> None:
    a = KSet([(1, 3), (7, 7), (10, INF)])
    assert a.tail_start == 10
    assert not a.is_finite
    assert a.finite_part == KSet([(1, 3), (7, 7)])
    assert a.finite_part | KSet([(a.tail_start, INF)]) == a
    f = KSet([(2, 4), (8, 8)])
    assert f.tail_start is None and f.is_finite
    assert f.finite_part == f
    assert list(f) == [2, 3, 4, 8]
    assert list(KSet()) == []
    assert KSet().is_finite and KSet().tail_start is None
    assert KSet.all().tail_start == 1
    assert KSet.all().finite_part == KSet()
    # The error is raised by iter(), before any element is requested.
    with pytest.raises(ValueError, match="infinite"):
        iter(a)
    with pytest.raises(ValueError, match="infinite"):
        list(KSet.all())


# -- laws (property-based) --------------------------------------------------------

B = 50  # every finite endpoint drawn below is < B


@st.composite
def ksets_with_model(draw: st.DrawFn) -> tuple[KSet, Model]:
    """A KSet and an independent model: (points in [1, B], contains (B, ∞))."""
    pairs = draw(st.lists(st.tuples(st.integers(1, 40), st.integers(0, 6)), max_size=6))
    raw: list[tuple[int, int | None]] = [(lo, lo + width) for lo, width in pairs]
    tail = draw(st.none() | st.integers(1, 45))
    if tail is not None:
        raw.append((tail, INF))
    draw(st.randoms()).shuffle(raw)
    points = {n for lo, hi in raw for n in range(lo, (B if hi is None else hi) + 1)}
    return KSet(raw), (frozenset(points), tail is not None)


ksets = ksets_with_model().map(lambda pair: pair[0])
UNIVERSE = frozenset(range(1, B + 1))


def model_of(s: KSet) -> Model:
    """The model of a KSet, read off its canonical intervals."""
    points = {
        n for lo, hi in s.intervals for n in range(lo, (B if hi is None else hi) + 1)
    }
    return frozenset(points), s.tail_start is not None


def assert_canonical(s: KSet) -> None:
    ivs = s.intervals
    for lo, hi in ivs:
        assert lo >= 1 and (hi is None or lo <= hi)
    for (_, hi), (lo, _) in itertools.pairwise(ivs):
        assert hi is not None and hi + 1 < lo
    assert s == KSet(ivs)


@given(ksets_with_model(), ksets_with_model())
def test_operations_match_the_model(
    am: tuple[KSet, Model], bm: tuple[KSet, Model]
) -> None:
    (a, (pa, ta)), (b, (pb, tb)) = am, bm
    assert model_of(a) == (pa, ta)
    for result, expected in [
        (a & b, (pa & pb, ta and tb)),
        (a | b, (pa | pb, ta or tb)),
        (a - b, (pa - pb, ta and not tb)),
        (~a, (UNIVERSE - pa, not ta)),
    ]:
        assert_canonical(result)
        assert model_of(result) == expected
    assert (a <= b) == (pa <= pb and (tb or not ta))
    assert a.isdisjoint(b) == (not (pa & pb) and not (ta and tb))
    assert [n for n in range(-1, B + 3) if n in a] == sorted(pa) + (
        [B + 1, B + 2] if ta else []
    )
    assert a.is_finite == (not ta)
    if a.is_finite:
        assert list(a) == sorted(pa)
    else:
        assert a.tail_start is not None
        assert a.finite_part | KSet([(a.tail_start, INF)]) == a
        assert a.finite_part.is_finite


@given(ksets, ksets, ksets)
def test_boolean_algebra_laws(a: KSet, b: KSet, c: KSet) -> None:
    assert ~~a == a
    assert ~(a & b) == ~a | ~b
    assert ~(a | b) == ~a & ~b
    assert a & (b | c) == (a & b) | (a & c)
    assert a | (b & c) == (a | b) & (a | c)
    assert a & (a | b) == a
    assert a | (a & b) == a
    assert a - b == a & ~b
    assert a & ~a == KSet()
    assert a | ~a == KSet.all()
    assert (a <= b) == (a & b == a)
    assert (a < b) == (a <= b and a != b)
    assert (a == b) == (hash(a) == hash(b) and a.intervals == b.intervals)


# -- where: hand-made conditions --------------------------------------------------

# name, polynomial, expected sets for <, ==, > (the others follow)
HAND_MADE = [
    (
        "integer roots",
        (k - 2) * (k - 5),
        [(3, 4)],
        [(2, 2), (5, 5)],
        [(1, 1), (6, INF)],
    ),
    # roots (7 ± sqrt 17)/4 ~ 0.72, 2.78
    ("between integers", 2 * k**2 - 7 * k + 4, [(1, 2)], [], [(3, INF)]),
    (
        "double root at 3",
        (k - 3) ** 2 * (k - 10),
        [(1, 2), (4, 9)],
        [(3, 3), (10, 10)],
        [(11, INF)],
    ),
    ("double root between", (2 * k - 7) ** 2, [], [], [(1, INF)]),
    ("no real roots", k**2 + 1, [], [], [(1, INF)]),
    ("negative constant", sp.Integer(-3), [(1, INF)], [], []),
    ("zero", sp.Integer(0), [], [(1, INF)], []),
    (
        "close roots",
        (k - sp.Rational(1001, 1000)) * (k - sp.Rational(1002, 1000)),
        [],
        [],
        [(1, INF)],
    ),
    ("root at 1", k - 1, [], [(1, 1)], [(2, INF)]),
    ("roots below 1", (k + 5) * (2 * k - 1), [], [], [(1, INF)]),
    (
        "large root",
        k - 10**30,
        [(1, 10**30 - 1)],
        [(10**30, 10**30)],
        [(10**30 + 1, INF)],
    ),
    ("rational coefficients", k / 3 - sp.Rational(7, 2), [(1, 10)], [], [(11, INF)]),
    # Regression (kset.md, §3): SymPy isolates 61/3 in (20, 21), and 20 is a root.
    ("endpoint root", (k - 20) * (3 * k - 61), [], [(20, 20)], [(1, 19), (21, INF)]),
    (
        "negative leading",
        -(k - 4) * (k - sp.Rational(9, 2)),
        [(1, 3), (5, INF)],
        [(4, 4)],
        [],
    ),
]


@pytest.mark.parametrize(
    ("p", "lt", "eq", "gt"),
    [case[1:] for case in HAND_MADE],
    ids=[case[0] for case in HAND_MADE],
)
def test_where_hand_made(p: sp.Expr, lt: Any, eq: Any, gt: Any) -> None:
    lt, eq, gt = KSet(lt), KSet(eq), KSet(gt)
    assert KSet.where(p, "<", k=k) == lt
    assert KSet.where(p, "==", k=k) == eq
    assert KSet.where(p, ">", k=k) == gt
    assert KSet.where(p, "<=", k=k) == lt | eq
    assert KSet.where(p, ">=", k=k) == gt | eq
    assert KSet.where(p, "!=", k=k) == lt | gt
    assert lt | eq | gt == KSet.all()
    assert lt.isdisjoint(eq) and lt.isdisjoint(gt) and eq.isdisjoint(gt)


def test_where_input_forms() -> None:
    p = (k - 2) * (k - 5)
    expected = KSet.of(2, 5)
    assert KSet.where(sp.Poly(p, k), "==", k=k) == expected
    assert KSet.where(sp.Poly(p, k, domain=sp.QQ), "==", k=k) == expected
    assert KSet.where(sp.expand(p), "==", k=k) == expected
    assert KSet.where(3, ">", k=k) == KSet.all()
    assert KSet.where(Fraction(-1, 2), ">=", k=k) == KSet()
    assert KSet.where(0, "==", k=k) == KSet.all()


def test_where_is_memoised() -> None:
    _sign_pieces.cache_clear()
    p = 32 * k**2 * (k - 54) ** 3
    for rel in RELATIONS:
        KSet.where(p, rel, k=k)
    info = _sign_pieces.cache_info()
    assert info.misses == 1 and info.hits == 5


# -- where: rational functions -----------------------------------------------------


def test_where_den_excludes_poles() -> None:
    # (k - 3)/(k - 5): pole at 5, zero at 3.
    p, q = k - 3, k - 5
    assert KSet.where(p, "<", k=k, den=q) == KSet([(4, 4)])
    assert KSet.where(p, "==", k=k, den=q) == KSet.of(3)
    assert KSet.where(p, ">", k=k, den=q) == KSet([(1, 2), (6, INF)])
    for rel in RELATIONS:
        assert 5 not in KSet.where(p, rel, k=k, den=q)
    assert KSet.where(0, "==", k=k, den=q) == KSet.all() - KSet.of(5)


def test_where_den_is_not_cancelled() -> None:
    assert KSet.where(k - 1, ">", k=k, den=k - 1) == KSet([(2, INF)])
    assert KSet.where(k - 1, "==", k=k, den=k - 1) == KSet()
    # Lowest terms, if wanted, by cancelling first:
    num, den = sp.fraction(sp.cancel((k**2 - 1) / (k - 1)))
    assert KSet.where(num, ">", k=k, den=den) == KSet.all()
    assert KSet.where(k**2 - 1, ">", k=k, den=k - 1) == KSet([(2, INF)])


def test_where_den_constant_and_without_integer_roots() -> None:
    assert KSet.where(k - 3, ">", k=k, den=-2) == KSet([(1, 2)])
    assert KSet.where(k - 3, ">", k=k, den=Fraction(1, 7)) == KSet([(4, INF)])
    assert KSet.where(k - 3, ">=", k=k, den=k**2 + 1) == KSet([(3, INF)])
    assert KSet.where(k - 3, ">", k=k, den=2 * k - 9) == KSet([(1, 2), (5, INF)])


@pytest.mark.parametrize(
    ("p", "den"),
    [
        ((k - 2) * (k - 7), (2 * k - 11) * (k - 4)),
        (k**3 - 30 * k + 1, (k - 1) * (k - 9)),
        ((k - 6) ** 2, (k - 6) * (3 * k - 8)),
    ],
)
def test_where_den_against_brute_force(p: sp.Expr, den: sp.Expr) -> None:
    check_against_brute_force(p, 20, den)


# -- where: errors ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p", "kwargs", "error", "match"),
    [
        (1 / k, {}, ValueError, r"not a polynomial in k over Q: 1/k; use den="),
        (sp.sqrt(k), {}, ValueError, "not a polynomial"),
        ((k**2 - 1) / (k - 1), {}, ValueError, "not a polynomial"),
        (k + x, {}, ValueError, "in k alone; has x"),
        (sp.Poly(k * x, k, x), {}, ValueError, "in k alone"),
        (sp.Poly(x, x), {}, ValueError, "in k alone"),
        (2.0 * k, {}, ValueError, "over Q"),
        (sp.sqrt(2) * k, {}, ValueError, "over Q"),
        (k, {"den": 0}, ValueError, "den is the zero polynomial"),
        (k, {"den": 1 / k}, ValueError, "den is not a polynomial"),
        (True, {}, TypeError, "polynomial"),
        ("k", {}, TypeError, "sympy expression"),
        (2.0, {}, TypeError, "sympy expression"),
    ],
)
def test_where_errors(
    p: Any, kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        KSet.where(p, ">", k=k, **kwargs)


def test_where_bad_rel_and_k() -> None:
    with pytest.raises(ValueError, match="rel must be one of"):
        KSet.where(k, "=", k=k)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"sympy\.Symbol"):
        KSet.where(k, ">", k="k")
    with pytest.raises(TypeError):
        KSet.where(k, ">")  # type: ignore[call-arg]


# -- where: against brute force (property-based) ---------------------------------------

rationals = st.builds(
    sp.Rational, st.integers(-20, 120), st.sampled_from([1, 1, 2, 3, 7])
)


@given(
    st.lists(rationals, min_size=1, max_size=4),
    st.integers(0, 3) | st.none(),
    st.sampled_from([1, -1, sp.Rational(2, 3)]),
)
def test_where_from_roots(
    roots: list[sp.Rational], quadratic: int | None, scale: Any
) -> None:
    # Roots known exactly, so the bound is exact: no root of p beyond max root + 2
    # (the optional k^2 + c has no real root for c >= 1, and a double root 0 for c = 0).
    p = scale * sp.prod([k - r for r in roots])
    if quadratic is not None:
        p *= k**2 + quadratic
    bound = max(2, int(sp.ceiling(max(roots))) + 2)
    check_against_brute_force(sp.expand(p), bound)


@given(
    st.integers(1, 30),
    st.integers(2, 7).flatmap(lambda d: st.tuples(st.just(d), st.integers(1, d - 1))),
    st.integers(1, 2),
    st.booleans(),
)
def test_where_integer_root_next_to_another(
    n: int, frac: tuple[int, int], mult: int, below: bool
) -> None:
    # An integer root n and a non-integer root in (n, n + 1) or (n - 1, n): the
    # configuration of the "endpoint root" and "negative leading" regressions.
    d, j = frac
    other = sp.Rational(d * n - j if below else d * n + j, d)
    p = sp.expand((k - n) ** mult * (k - other) * (n + 2 - k))
    check_against_brute_force(p, n + 4)


@given(st.lists(st.integers(-50, 50), min_size=1, max_size=5))
def test_where_random_coefficients(coeffs: list[int]) -> None:
    # Cauchy bound: every root satisfies |r| < 1 + max |c_i / c_lead| <= 51.
    p = sum(c * k**i for i, c in enumerate(coeffs))
    check_against_brute_force(sp.expand(p), 52)


# -- acceptance: Broda et al. 2020 ---------------------------------------------------

# Broda et al. 2020, eq. (18).
P_K = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4


def test_broda_2020_eq_18() -> None:
    disc = sp.discriminant(P_K, z)
    assert sp.factor(disc) == 32 * k**2 * (k - 54) ** 3
    lt = KSet.where(disc, "<", k=k)
    eq = KSet.where(disc, "==", k=k)
    gt = KSet.where(disc, ">", k=k)
    assert lt == KSet([(1, 53)])
    assert eq == KSet.of(54)
    assert gt == KSet([(55, INF)])
    assert (str(lt), str(eq), str(gt)) == ("[1, 53]", "{54}", "[55, ∞)")
    assert lt | eq | gt == KSet.all()
    assert lt.isdisjoint(eq | gt) and eq.isdisjoint(gt)


def test_broda_2020_eq_17_derived_root_between_integers() -> None:
    # Derived by us, not printed in the paper: D = disc_z of the squarefree part of
    # res_w(G, dG/dw), for G from Broda et al. 2020, eq. (17), as printed:
    #   p_k(z) w^3 + k z ((k + 12) k z^2 + 4 k z + 1) w - k^2 z^2 (2z + 1) = 0,
    # with p_k from eq. (18). D and its factor k^2 - 34k - 54 (root 17 + 7 sqrt 7,
    # about 35.52, not an integer) come from this computation.
    G = P_K * w**3 + k * z * ((k + 12) * k * z**2 + 4 * k * z + 1) * w
    G -= k**2 * z**2 * (2 * z + 1)
    res = sp.resultant(sp.expand(G), sp.diff(G, w), w)
    D = sp.discriminant(sp.sqf_part(res), z)
    c, factors = sp.factor_list(D)
    assert c == 764411904
    assert dict(factors) == {
        k: 22,
        k + 2: 8,
        k - 54: 9,
        k**2 - 34 * k - 54: 1,
    }
    assert KSet.where(D, ">", k=k) == KSet([(1, 35), (55, INF)])
    assert KSet.where(D, "<", k=k) == KSet([(36, 53)])
    assert KSet.where(D, "==", k=k) == KSet.of(54)


# -- KCases ---------------------------------------------------------------------------


def test_kcases_basics() -> None:
    generic, exceptional = KSet([(1, 53), (55, INF)]), KSet.of(54)
    cases = KCases({generic: "generic", exceptional: "k = 54"})
    assert cases.pieces == ((generic, "generic"), (exceptional, "k = 54"))
    assert cases.covered == KSet.all()
    assert cases.uncovered == KSet()
    assert cases[1] == cases[10**9] == "generic"
    assert cases[54] == "k = 54"
    assert str(cases) == "[1, 53] ∪ [55, ∞): generic\n{54}: k = 54"


def test_kcases_uncovered_and_errors() -> None:
    cases = KCases([(KSet.of(3), 1), (KSet([(1, 2)]), 0), (KSet(), 99)])
    assert cases.pieces == ((KSet([(1, 2)]), 0), (KSet.of(3), 1))  # sorted, ∅ dropped
    assert cases.uncovered == KSet([(4, INF)])
    assert str(cases) == "[1, 2]: 0\n{3}: 1\nundecided: [4, ∞)"
    with pytest.raises(KeyError, match=r"k = 4 is not covered \(undecided\)"):
        cases[4]
    with pytest.raises(KeyError, match="outside the domain"):
        cases[0]
    with pytest.raises(TypeError, match="integer"):
        cases[2.0]  # type: ignore[index]
    with pytest.raises(ValueError, match=r"pieces overlap on \[3, 5\]"):
        KCases({KSet([(1, 5)]): "a", KSet([(3, 9)]): "b"})
    with pytest.raises(TypeError, match="keyed by KSet"):
        KCases({(1, 2): "a"})  # type: ignore[arg-type]


def test_kcases_tail() -> None:
    cases = KCases({KSet.of(1): "a", KSet([(2, INF)]): "b"})
    assert cases.tail == (KSet([(2, INF)]), "b")
    assert KCases({KSet([(1, 3), (7, INF)]): "b", KSet.of(5): "a"}).tail == (
        KSet([(1, 3), (7, INF)]),
        "b",
    )
    assert KCases({KSet([(1, 9)]): "a"}).tail is None
    assert KCases({}).tail is None


def test_kcases_combine_only_nonempty_intersections() -> None:
    calls = []

    def f(u: int, v: int) -> int:
        calls.append((u, v))
        return u + v

    a = KCases({KSet.of(1): 1, KSet([(2, None)]): 2})
    b = KCases({KSet.of(1): 10, KSet([(2, None)]): 20})
    assert [v for _, v in a.combine(b, f).pieces] == [11, 22]
    assert calls == [(1, 10), (2, 20)]


def test_kcases_map_restrict_combine() -> None:
    a = KCases({KSet([(1, 10)]): 1, KSet([(11, INF)]): 2})
    b = KCases({KSet([(5, 20)]): 10, KSet([(30, 40)]): 20})
    assert a.map(lambda v: v * 3) == KCases({KSet([(1, 10)]): 3, KSet([(11, INF)]): 6})
    r = a.restrict(KSet([(8, 12)]))
    assert r == KCases({KSet([(8, 10)]): 1, KSet([(11, 12)]): 2})
    assert r.uncovered == ~KSet([(8, 12)])
    c = a.combine(b, lambda u, v: u + v)
    assert c == KCases(
        {KSet([(5, 10)]): 11, KSet([(11, 20)]): 12, KSet([(30, 40)]): 22}
    )
    assert c.covered == a.covered & b.covered


def test_kcases_flat_map() -> None:
    """singularity.md §7: each value's own KCases, restricted to its piece."""
    a = KCases({KSet([(1, 10)]): "x", KSet([(20, INF)]): "y"})  # [11, 19] uncovered

    def split(v: str) -> KCases[str]:
        if v == "x":  # extends beyond its piece: restricted to [1, 10]
            return KCases({KSet([(1, 4)]): "x1", KSet([(5, 50)]): "x2"})
        return KCases({KSet([(1, 30)]): "y1"})  # nothing beyond 30

    r = a.flat_map(split)
    assert r == KCases(
        {KSet([(1, 4)]): "x1", KSet([(5, 10)]): "x2", KSet([(20, 30)]): "y1"}
    )
    # uncovered: the input's, plus each piece's part that its KCases leaves out
    assert r.uncovered == KSet([(11, 19), (31, INF)])
    # identity
    assert a.flat_map(lambda v: KCases({KSet.all(): v})) == a
    # a value whose KCases covers nothing of its piece
    assert a.flat_map(lambda v: KCases({KSet.of(15): v})) == KCases({})
    assert KCases({}).flat_map(split) == KCases({})
    with pytest.raises(TypeError, match="returning KCases"):
        a.flat_map(lambda v: v)  # type: ignore[arg-type, return-value]


def test_kcases_value_semantics() -> None:
    a = KCases({KSet.of(1): sp.Integer(2), KSet([(2, INF)]): k})
    b = KCases([(KSet([(2, INF)]), k), (KSet.of(1), sp.Integer(2))])
    assert a == b and hash(a) == hash(b)
    assert a != KCases({KSet.of(1): 2})
    assert pickle.loads(pickle.dumps(a)) == a
    assert repr(KCases({KSet.of(1): "x"})) == "KCases([(KSet([(1, 1)]), 'x')])"
    with pytest.raises(AttributeError, match="immutable"):
        a._pieces = ()
