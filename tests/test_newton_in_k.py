# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the local expansion for symbolic k (newton.md, section 6)."""

from __future__ import annotations

import pickle
from fractions import Fraction
from typing import Any

import pytest
import sympy as sp

import flanpy._newton as newton
from flanpy import (
    AlgebraicReal,
    KCases,
    KRoot,
    KSet,
    LocalExpansion,
    System,
    eliminate,
    eliminate_in_k,
    local_expansion,
    local_expansion_in_k,
    singularity,
    singularity_in_k,
)
from flanpy._field import _NumberField
from flanpy._newton import _binomial_roots, _FamilyExpansion, _negate
from flanpy._singularity import _Family, _layout
from test_newton import FK, HG, RK, SSNF, A, F, L, R, k, z

y = sp.Symbol("y")
U, K = sp.symbols("U K")
# A cut from P5 inside one step 4 piece: K = kU/4 + K^2, U = z + U^2.
UK = System({U: z + U**2, K: k * U / 4 + K**2}, z=z, k=k)
P_K = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4  # Broda eq. (18)
EQ34 = ((k**2 + 12 * k) * z**2 + 4 * k * z + 1) * y**2 - (
    (24 * k + 324) * z**2 + 72 * z - 1
)

_CACHE: dict[str, Any] = {}


def family(name: str) -> tuple[Any, KCases[LocalExpansion[Any]]]:
    """(singularity_in_k, local_expansion_in_k) of a family, cached per session."""
    if name not in _CACHE:
        system, w = {"L": (SSNF, L), "F": (FK, F), "R": (RK, R), "A": (HG, A)}.get(
            name, (UK, K)
        )
        s4 = singularity_in_k(eliminate_in_k(system, w))
        _CACHE[name] = (s4, local_expansion_in_k(s4))
    result: tuple[Any, KCases[LocalExpansion[Any]]] = _CACHE[name]
    return result


def fixed(system: System, w: sp.Symbol, n: int) -> LocalExpansion[AlgebraicReal]:
    return local_expansion(singularity(eliminate(system, w, k=n)))


def constant(e: LocalExpansion[Any]) -> Any:
    return e.b if e.b is not None else e.c


# -- Broda L_k and F_k: one piece ------------------------------------------------------


def test_l_k_one_piece() -> None:
    s4, cases = family("L")
    ((piece, e),) = cases.pieces
    assert piece == KSet.all() and cases.uncovered == KSet()
    assert (e.singularity.case, e.alpha, e.ramification, e.b) == (
        "II",
        Fraction(1, 2),
        2,
        None,
    )
    rho = KRoot(P_K, 0, k=k)
    assert e.singularity.rho == rho
    assert e.c == KRoot(EQ34, 0, k=k, over=rho)  # Broda eq. (34)
    # 54 is a root of the cut set (Res_z(p_k, num) has the factor (k - 54)^3),
    # instantiated and merged.
    fam = _FamilyExpansion(s4.pieces[0][1])
    fam._setup()
    roots, _ = _layout(fam.polys, k)
    assert {54, 108} <= roots


def test_f_k_one_piece() -> None:
    _, cases = family("F")
    ((piece, e),) = cases.pieces
    assert piece == KSet.all()
    assert (e.singularity.case, e.alpha, e.ramification, e.c) == (
        "I",
        Fraction(1, 2),
        2,
        None,
    )
    b = constant(e)
    assert b.over == KRoot(P_K, 0, k=k) and b.positive_index == 0


@pytest.mark.parametrize("n", [1, 2, 10, 50, 54, 55, 108])
def test_l_k_and_f_k_at_n(n: int) -> None:
    for name, system, w in (("L", SSNF, L), ("F", FK, F)):
        _, cases = family(name)
        e = cases[n]
        assert constant(e).at(n) == constant(fixed(system, w, n))
        assert e.specialize(n) == local_expansion(e.singularity, k=n)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["L", "F"])
def test_l_k_and_f_k_at_every_n(name: str) -> None:
    system, w = {"L": (SSNF, L), "F": (FK, F)}[name]
    _, cases = family(name)
    for n in [*range(1, 61), 108]:
        e = cases[n]
        assert constant(e).at(n) == constant(fixed(system, w, n)), n


# -- R_k and HG A_k: one piece, closed forms -------------------------------------------


@pytest.mark.parametrize(
    ("name", "closed"),
    [
        ("R", lambda r: (2 - 2 * r) / (16 * r * r)),  # Broda 2020 §3.1
        ("A", lambda r: (1 - 2 * r) / (8 * r * r)),  # HG eq. (14)
    ],
)
def test_r_k_and_a_k(name: str, closed: Any) -> None:
    _, cases = family(name)
    ((piece, e),) = cases.pieces
    assert piece == KSet.all() and e.alpha == Fraction(1, 2)
    for n in range(1, 21):
        r = _NumberField(e.singularity.rho.at(n)).generator
        assert constant(e).at(n) == _binomial_roots(closed(r), 2)[1]


# -- a cut from P5 inside a step 4 piece -----------------------------------------------


def test_cut_inside_a_step_4_piece() -> None:
    s4, cases = family("UK")
    assert [str(p) for p, _ in s4.pieces] == ["{1}", "[2, ∞)"]
    got = [(str(p), e.alpha, e.ramification) for p, e in cases.pieces]
    assert sorted(got) == [
        ("[3, ∞)", Fraction(1, 2), 2),
        ("{1}", Fraction(1, 2), 2),
        ("{2}", Fraction(1, 4), 4),
    ]
    for n in range(1, 9):
        e, f = cases[n], fixed(UK, K, n)
        assert (e.alpha, e.ramification) == (f.alpha, f.ramification)
        assert constant(e).at(n) == f.b
    assert constant(cases[2]).at(2) == AlgebraicReal(2 * y - 1, 0)  # 1/2
    assert constant(cases[3]).at(3) == AlgebraicReal(2 * y**2 - 1, 1)  # 1/√2
    assert constant(cases[4]).at(4) == AlgebraicReal(8 * y**2 - 3, 1)  # √6/4


def test_which_polynomials_cut_at_2() -> None:
    """The (r, 0) vertex: G_ww(ρ_k, a_k) = 0 at k = 2, where a_2 is a 4-fold root."""
    s4, _ = family("UK")
    piece = s4.pieces[-1][1]
    fam = _FamilyExpansion(piece)
    fam._setup()
    assert fam.edges == [(0, 1, 2, 0)]

    def order_at_2(p: Any) -> int:
        return int(sp.roots(sp.Poly(p, k)).get(sp.Integer(2), 0))

    assert order_at_2(fam._cache[(0, 2)]) == 13  # vertex (2, 0): Res_z(f, q_02)
    assert order_at_2(fam._cache[(1, 0)]) == 12  # vertex (0, 1): Res_z(f, q_10)
    assert order_at_2(fam.extra[0]) == 3  # Res_z(f, s_11): a_2 is no double root
    # Step 4 cut there too (ρ_2 = 1/4 meets the radius of U), and merged.
    assert any(order_at_2(p) > 0 for p in _Family(piece.curve).first if p != 0)


# -- the API ---------------------------------------------------------------------------


def test_terms_specialize_and_value_semantics() -> None:
    _, cases = family("R")
    e = cases[2]
    with pytest.raises(ValueError, match=r"specialize\(n\)"):
        e.terms(3)
    fixed_e = e.specialize(2)
    assert fixed_e.b is not None and fixed_e.terms(2)[1][1] == _negate(fixed_e.b)
    with pytest.raises(ValueError, match="not in"):
        e.specialize(0)
    assert pickle.loads(pickle.dumps(e)) == e and hash(e) == hash(e)


def test_inputs() -> None:
    s4, cases = family("R")
    assert local_expansion_in_k(s4.pieces[0][1]) == cases
    restricted = local_expansion_in_k(s4.restrict(KSet([(1, 10)])))
    assert restricted.covered == KSet([(1, 10)])
    assert restricted.uncovered == KSet([(11, None)])
    (_, e) = restricted.pieces[0]
    assert e.kset == KSet([(1, 10)]) and e.singularity.kset == KSet([(1, 10)])
    one = singularity(eliminate(RK, R, k=2))  # a piece at one fixed k
    single = local_expansion_in_k(KCases({one.kset: one}))
    assert single[2] == local_expansion(one)
    with pytest.raises(ValueError, match=r"use flanpy\.local_expansion"):
        local_expansion_in_k(singularity(eliminate(System({F: 1 + z * F**2}, z=z), F)))
    with pytest.raises(TypeError, match="Singularity"):
        local_expansion_in_k(eliminate_in_k(RK, R))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Singularity"):
        local_expansion_in_k("R_k")  # type: ignore[arg-type]


def test_gaps_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    s4, _ = family("R")
    piece = s4.pieces[0][1]
    monkeypatch.setattr(sp, "subresultants", lambda *args: [])
    with pytest.raises(NotImplementedError, match=r"\[1, ∞\).*fixed k"):
        local_expansion_in_k(piece)
    monkeypatch.undo()
    original = newton._analyse

    def tied(s: Any) -> Any:
        e, state = original(s)
        state.branches.append(
            newton._Branch(
                state.branch.gamma, state.branch.leading, state.branch.kappa, None
            )
        )
        return e, state

    monkeypatch.setattr(newton, "_analyse", tied)
    with pytest.raises(NotImplementedError, match=r"ties with another"):
        local_expansion_in_k(piece)
