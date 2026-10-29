# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.System (design: dev/design/system.md, section 10)."""

import pickle
from collections.abc import Mapping
from typing import Any

import pytest
import sympy as sp

import flanpy
from flanpy import (
    IllPosedSystemError,
    System,
    UndecidedSystemError,
    WellPosednessError,
    ZeroCoordinateWarning,
)

z, k, u = sp.symbols("z k u")
F, G, R, S, Y1, Y2 = sp.symbols("F G R S Y1 Y2")
B, Bb, M, Mb, L = sp.symbols("B Bb M Mb L")

# Broda et al. 2020: R_k from grammar (1) (p. 4); S_k = d/du at u = 1 (pp. 5-6).
BRODA_R = {R: (k + 1) * z + 2 * z * R**2 + z * R}
BRODA_RS = {**BRODA_R, S: 4 * z * R * S + z * R + z * S}
# Broda et al. 2020, L_k: eqs. (10), (11), (15), (16), (12).
BRODA_L = {
    B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
    Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
    M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
    Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
    L: M + Mb,
}
# Signed system whose constant terms need 2 > m = 1 steps: 0 -> 2 -> 1 -> 1.
TWO_STEP = {F: z + 1 - (F - 1) ** 2 * (F - 2) / 2}

# name, equations, k, constant terms c*, non-negative
WELL_POSED = [
    ("linear", {F: 1 + 2 * z * F}, None, (1,), True),
    ("catalan", {F: 1 + z * F**2}, None, (1,), True),
    ("motzkin", {F: 1 + z * F + z * F**2}, None, (1,), True),
    ("leaves", {F: z + F**2}, None, (0,), True),
    ("two-equations", {F: 1 + z * G, G: 1 + z * F**2}, None, (1, 1), True),
    # Pivoteau, Salvy & Soria 2012, Example 8 (well founded).
    ("pss-example-8", {Y1: 1 + z * Y1, Y2: 1 + Y1**2}, None, (1, 2), True),
    ("broda-R", BRODA_R, k, (0,), True),
    ("broda-RS", BRODA_RS, k, (0, 0), True),
    ("broda-L", BRODA_L, k, (0,) * 5, True),
    ("two-step-signed", TWO_STEP, None, (1,), False),
]


@pytest.mark.parametrize(
    ("equations", "kk", "constants", "nonnegative"),
    [case[1:] for case in WELL_POSED],
    ids=[case[0] for case in WELL_POSED],
)
def test_well_posed(
    equations: Mapping[Any, Any], kk: Any, constants: tuple[int, ...], nonnegative: bool
) -> None:
    system = System(equations, z=z, k=kk)
    assert system.unknowns == tuple(equations)
    assert system.z == z
    assert system.k == kk
    assert dict(system.equations) == {f: sp.expand(e) for f, e in equations.items()}
    assert system._constant_terms == tuple(sp.Integer(c) for c in constants)
    assert system._nonnegative is nonnegative
    m = len(equations)
    jac = system._jacobian0
    assert jac.shape == (m, m)
    assert (jac**m).is_zero_matrix
    assert 0 <= system._nilpotency_index <= m
    params = () if kk is None else (kk,)
    for f, poly in zip(system.unknowns, system.polynomials(), strict=True):
        assert poly.gens == (*system.unknowns, z, *params)
        assert poly.domain == sp.QQ
        assert poly.as_expr() == sp.expand(f - equations[f])


def test_broda_L_polynomials() -> None:
    system = System(BRODA_L, z=z, k=k)
    polys = system.polynomials()
    assert polys[4] == sp.Poly(L - M - Mb, B, Bb, M, Mb, L, z, k, domain=sp.QQ)
    assert polys[1] == sp.Poly(
        Bb - k * z - 2 * z * B * Bb - 2 * z * Bb**2, B, Bb, M, Mb, L, z, k, domain=sp.QQ
    )
    assert system._nilpotency_index == 2  # L depends on M, Mb at z = 0


@pytest.mark.parametrize("value", [1, 2, 7])
@pytest.mark.parametrize("equations", [BRODA_R, BRODA_RS, BRODA_L])
def test_specialize(equations: Mapping[Any, Any], value: Any) -> None:
    system = System(equations, z=z, k=k).specialize(value)
    assert system.k is None
    assert all(k not in phi.free_symbols for phi in system.equations.values())
    assert system == System({f: e.subs(k, value) for f, e in equations.items()}, z=z)
    assert all(len(p.gens) == len(equations) + 1 for p in system.polynomials())


def test_specialize_zero_creates_zero_coordinates() -> None:
    assert System(BRODA_R, z=z, k=k).specialize(0).unknowns == (R,)  # no warning
    with pytest.warns(ZeroCoordinateWarning, match="B, Bb, M, Mb, L are identically"):
        System(BRODA_L, z=z, k=k).specialize(0)


@pytest.mark.parametrize(
    ("equations", "zeros"),
    [
        # Pivoteau, Salvy & Soria 2012, Example 4: excluded there, accepted here.
        ({F: F}, "F is"),
        ({F: z * F}, "F is"),
        ({F: F + z * F}, "F is"),
        ({F: z + G, G: G}, "G is"),
        ({F: F * G, G: z * F}, "F, G are"),
    ],
    ids=["F=F", "F=zF", "F=F+zF", "zero-G", "zero-FG"],
)
def test_zero_coordinates_warn(equations: Mapping[Any, Any], zeros: str) -> None:
    with pytest.warns(ZeroCoordinateWarning, match=f"{zeros} identically 0"):
        system = System(equations, z=z)
    assert len(system._active) == len(equations) - zeros.count(",") - 1


# name, equations, k, phrases that the message must contain
ILL_POSED = [
    ("F=F+z", {F: F + z}, None, ["[[1]]", "cycle F -> F", "Thm 5.5"]),
    ("F=z+F+F^2", {F: z + F + F**2}, None, ["[[1]]", "not nilpotent"]),
    ("F=G+z,G=F", {F: G + z, G: F}, None, ["cycle F -> G -> F"]),
    ("F=1+F", {F: 1 + F}, None, ["do not stabilise", "F(0) = 2", "Thm 5.7"]),
    ("F=1+F^2", {F: 1 + F**2}, None, ["do not stabilise"]),
    ("F=kF+z", {F: k * F + z}, k, ["[[k]]", "every integer k >= 1", "k = 0"]),
    # Pivoteau, Salvy & Soria 2012, Example 8 (not well founded).
    ("pss-example-8", {Y1: 1 + Y2 * Y1, Y2: 1}, None, ["Y1(0) = 3, Y2(0) = 1"]),
    ("F=1-F", {F: 1 - F}, None, ["cycle", "F(0) = 0 -> F(0) = 1 -> F(0) = 0"]),
]


@pytest.mark.parametrize(
    ("equations", "kk", "phrases"),
    [case[1:] for case in ILL_POSED],
    ids=[case[0] for case in ILL_POSED],
)
def test_ill_posed(equations: Mapping[Any, Any], kk: Any, phrases: list[str]) -> None:
    with pytest.raises(IllPosedSystemError, match=r"^the system is ill-posed") as info:
        System(equations, z=z, k=kk)
    message = str(info.value)
    for phrase in phrases:
        assert phrase in message
    assert isinstance(info.value, WellPosednessError)
    assert isinstance(info.value, ValueError)


def test_undecided_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The signed two-step example needs 2 iterations; a cap of 1 cannot decide it.
    monkeypatch.setattr("flanpy._system._signed_cap", lambda m: 1)
    with pytest.raises(UndecidedSystemError, match="did not stabilise within 1 "):
        System(TWO_STEP, z=z)


def test_undecided_unbounded_constant_terms() -> None:
    with pytest.raises(UndecidedSystemError, match="within 32 iterations"):
        System({F: F + 1 - z * F**2}, z=z)


def test_undecided_signed_jacobian() -> None:
    with pytest.raises(UndecidedSystemError, match=r"\[\[-1\]\]") as info:
        System({F: z - F}, z=z)
    assert not isinstance(info.value, IllPosedSystemError)


@pytest.mark.parametrize(
    ("equations", "kwargs", "error", "match"),
    [
        ({F: 1 / (1 - z * F)}, {}, ValueError, "not a polynomial"),
        ({F: sp.sqrt(z)}, {}, ValueError, "not a polynomial"),
        ({F: sp.exp(z)}, {}, ValueError, "not a polynomial"),
        ({F: sp.sqrt(2) * z}, {}, ValueError, "not rational"),
        ({F: sp.Float(0.5) * z}, {}, ValueError, "floating-point"),
        ({F: u * z}, {}, ValueError, "Marking variables are not supported"),
        ({F: sp.Symbol("z", positive=True)}, {}, ValueError, "different assumptions"),
        ({F: 1 + z * F}, {"k": F}, ValueError, "both as an unknown"),
        ({z: 1 + z}, {}, ValueError, "both as an unknown"),
        ({}, {}, ValueError, "at least one equation"),
        ({F: 1, sp.Symbol("F", positive=True): 1}, {}, ValueError, "named 'F'"),
        ({F: 1}, {"k": z}, ValueError, "different symbols"),
        ({"F": 1}, {}, TypeError, "unknowns must be sympy.Symbol"),
        ({F: "z"}, {}, TypeError, "SymPy expression or an int"),
        ({F: 1}, {"k": "k"}, TypeError, "k must be"),
        ([(F, 1)], {}, TypeError, "mapping"),
    ],
)
def test_malformed_input(
    equations: Mapping[Any, Any],
    kwargs: dict[str, Any],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        System(equations, z=z, **kwargs)


def test_z_must_be_symbol() -> None:
    with pytest.raises(TypeError, match="z must be"):
        System({F: 1}, z="z")


@pytest.mark.parametrize(
    ("value", "error"), [(-1, ValueError), (2.0, TypeError), (True, TypeError)]
)
def test_specialize_rejects(value: Any, error: type[Exception]) -> None:
    with pytest.raises(error):
        System(BRODA_R, z=z, k=k).specialize(value)


def test_specialize_without_k() -> None:
    with pytest.raises(ValueError, match="no parameter"):
        System({F: 1 + z * F**2}, z=z).specialize(1)


def test_value_semantics() -> None:
    a = System(BRODA_RS, z=z, k=k)
    b = System(
        {R: z + k * z + z * R + 2 * z * R**2, S: z * (R + S + 4 * R * S)}, z=z, k=k
    )
    assert a == b
    assert hash(a) == hash(b)
    catalan = {F: 1 + z * F**2}
    assert System(catalan, z=z) != System(catalan, z=z, k=k)  # k differs
    assert a != System({S: BRODA_RS[S], R: BRODA_RS[R]}, z=z, k=k)  # order differs
    assert pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(AttributeError, match="immutable"):
        a._rhs = ()
    with pytest.raises(TypeError):
        a.equations[R] = z  # type: ignore[index]


def test_repr_and_public_names() -> None:
    system = System(BRODA_R, z=z, k=k)
    assert repr(system) == "System({R: 2*R**2*z + R*z + k*z + z}, z=z, k=k)"
    for name in (
        "System",
        "WellPosednessError",
        "IllPosedSystemError",
        "UndecidedSystemError",
        "ZeroCoordinateWarning",
    ):
        assert getattr(flanpy, name).__module__ == "flanpy"
        assert name in flanpy.__all__
