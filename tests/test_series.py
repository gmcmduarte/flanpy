# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for flanpy.series and flanpy.series_in_k (plan: series.md, section 7)."""

import time
from collections.abc import Mapping
from fractions import Fraction
from math import comb, factorial
from typing import Any

import pytest
import sympy as sp

from _enumerate import broda_counts
from flanpy import System, ZeroCoordinateWarning, series, series_in_k
from flanpy._rings import flint_available
from flanpy._series import _series

z, k = sp.symbols("z k")
F, G, R, S = sp.symbols("F G R S")
B, Bb, M, Mb, L = sp.symbols("B Bb M Mb L")

BACKENDS = [
    "python",
    pytest.param(
        "flint",
        marks=pytest.mark.skipif(not flint_available(), reason="python-flint missing"),
    ),
]

# Broda et al. 2020: R_k from grammar (1); S_k = d/du at u = 1 (pp. 4-6).
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
BRODA = {"R": BRODA_R, "RS": BRODA_RS, "L": BRODA_L}
TWO_STEP = {F: z + 1 - (F - 1) ** 2 * (F - 2) / 2}  # signed, c* = 1


def catalan(n: int) -> int:
    return comb(2 * n, n) // (n + 1)


def motzkin(n_max: int) -> list[int]:
    out = [1, 1]
    for n in range(2, n_max + 1):
        out.append(((2 * n + 1) * out[-1] + 3 * (n - 1) * out[-2]) // (n + 2))
    return out[: n_max + 1]


def fibonacci(n_max: int) -> list[int]:
    out = [1, 1]
    while len(out) <= n_max:
        out.append(out[-1] + out[-2])
    return out[: n_max + 1]


def broda_eq3(n: int, kk: int) -> Fraction:
    """[z^n] R_k, Broda et al. 2020, eq. (3) (n >= 1)."""
    total = sum(
        Fraction(1, 2 * (n - j) + 1)
        * comb(2 * (n - j + 1), n - j + 1)
        * comb(n - j + 1, j)
        * (7 + 8 * kk) ** j
        for j in range((n + 1) // 2 + 1)
    )
    result: Fraction = total / 2 ** (n + 3)
    return result


def broda_eq5(n: int, kk: int) -> Fraction:
    """[z^n] S_k, Broda et al. 2020, eq. (5) (n >= 2), with the odd case corrected.

    Erratum (to be confirmed with the authors): for odd n the paper prints the
    prefactor (n - 1)/2^(n+2); the correct one is (n - 1)/2^(n+1). For k = 1, n = 3
    the printed formula gives 2, but there are 4 stars (in eps** and a**).
    """

    def term(j: int) -> Fraction:
        den = factorial(n - j - 1) * factorial(j) * factorial(n - 2 * j)
        value: Fraction = (
            Fraction(factorial(2 * n - 2 * j - 2), den) * (7 + 8 * kk) ** j
        )
        return value

    if n % 2 == 0:
        head: Fraction = Fraction(comb(n, n // 2), 2 ** (n + 2)) * (7 + 8 * kk) ** (
            n // 2
        )
        return head + Fraction(n - 1, 2 ** (n + 1)) * sum(
            term(j) for j in range(n // 2)
        )
    return Fraction(n - 1, 2 ** (n + 1)) * sum(term(j) for j in range((n - 1) // 2 + 1))


def residual_is_zero(system: System, values: Mapping[Any, tuple[Any, ...]]) -> bool:
    """Phi(z, Y) - Y = 0 mod z^(n+1), checked independently with SymPy."""
    n = len(next(iter(values.values()))) - 1
    ys = {
        f: sum(sp.Rational(c) * z**i for i, c in enumerate(cs))
        for f, cs in values.items()
    }
    for f, phi in system.equations.items():
        diff = sp.expand(phi.xreplace(ys) - ys[f])
        if any(diff.coeff(z, i) != 0 for i in range(n + 1)):
            return False
    return True


# -- hand-made systems with known coefficients ---------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_known_coefficients(backend: Any) -> None:
    def run(eqs: dict[Any, Any], n: int) -> Mapping[Any, tuple[Any, ...]]:
        return _series(System(eqs, z=z), n, None, backend_name=backend)

    assert run({F: 1 + 2 * z * F}, 30)[F] == tuple(2**i for i in range(31))
    assert run({F: 1 + z * F + z**2 * F}, 30)[F] == tuple(fibonacci(30))
    half = run({F: 1 + z * F / 2}, 20)[F]
    assert half == tuple(Fraction(1, 2**i) for i in range(21))
    assert all(type(c) is Fraction for c in half)
    assert run({F: 1 + z * F**2}, 200)[F] == tuple(catalan(i) for i in range(201))
    assert run({F: 1 + z * F + z**2 * F**2}, 200)[F] == tuple(motzkin(200))
    both = run({F: 1 + z * G**2, G: 1 + z * F**2}, 60)
    assert both[F] == both[G] == tuple(catalan(i) for i in range(61))
    signed = run({F: z - z * F}, 30)[F]
    assert signed == (0, *((-1) ** (i + 1) for i in range(1, 31)))
    two_step = run(TWO_STEP, 6)[F]
    assert two_step[:3] == (1, 1, Fraction(1, 2))


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_coordinates(backend: Any) -> None:
    with pytest.warns(ZeroCoordinateWarning):
        system = System({F: z + G, G: G}, z=z)
    values = _series(system, 5, None, backend_name=backend)
    assert values[F] == (0, 1, 0, 0, 0, 0)
    assert values[G] == (0,) * 6
    # Zero coordinates that appear only at k = 1 (no warning: System is not rebuilt).
    sys_k = System({F: z + (k - 1) * G, G: (k - 1) * z}, z=z, k=k)
    assert _series(sys_k, 3, 1, backend_name=backend)[G] == (0, 0, 0, 0)
    assert _series(sys_k, 3, 3, backend_name=backend)[F] == (0, 5, 0, 0)


@pytest.mark.parametrize(
    "equations",
    [
        {F: 1 + 2 * z * F},
        {F: 1 + z * F + z**2 * F},
        {F: 1 + z * F / 2},
        {F: 1 + z * F**2},
        {F: 1 + z * F + z**2 * F**2},
        {F: 1 + z * G**2, G: 1 + z * F**2},
        {F: z - z * F},
        TWO_STEP,
        {F: z**2 * F**3 + 1 - z * F},  # cubic, signed, product nodes of length 3
    ],
    ids=["linear", "fibonacci", "half", "catalan", "motzkin", "two-eq", "signed",
         "two-step", "cubic"],
)  # fmt: skip
def test_residual(equations: dict[Any, Any]) -> None:
    system = System(equations, z=z)
    assert residual_is_zero(system, series(system, 25))


# -- Broda et al. 2020 ---------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kk", [1, 2, 3])
def test_broda_against_enumeration(backend: Any, kk: int) -> None:
    want = broda_counts(kk, 7)
    for name, eqs in BRODA.items():
        got = _series(System(eqs, z=z, k=k), 7, kk, backend_name=backend)
        for f, cs in got.items():
            assert list(cs) == want[str(f)], (name, f, kk)


@pytest.mark.parametrize("kk", [1, 2, 10, 50])
def test_broda_closed_forms(kk: int) -> None:
    got = series(System(BRODA_RS, z=z, k=k), 200, k=kk)
    assert all(got[R][n] == broda_eq3(n, kk) for n in range(1, 201))
    assert all(got[S][n] == broda_eq5(n, kk) for n in range(2, 201))


def test_eq5_erratum_example() -> None:
    # Broda et al. 2020, eq. (5) as printed, odd case, k = 1, n = 3:
    # (n - 1)/2^(n+2) * sum_{j=0}^{1} (2n-2j-2)!/((n-j-1)! j! (n-2j)!) * 15^j.
    terms = [
        Fraction(factorial(4), factorial(2) * factorial(0) * factorial(3)),
        Fraction(factorial(2), factorial(1) * factorial(1) * factorial(1)) * 15,
    ]
    printed = Fraction(3 - 1, 2 ** (3 + 2)) * sum(terms)
    assert printed == 2
    assert series(System(BRODA_RS, z=z, k=k), 3, k=1)[S][3] == 4 == broda_eq5(3, 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", list(BRODA))
def test_symbolic_k_matches_fixed_k(backend: Any, name: str) -> None:
    system = System(BRODA[name], z=z, k=k)
    symbolic = _series(system, 40, None, backend_name=backend)
    for kk in range(1, 11):
        fixed = series(system, 40, k=kk)
        for f in system.unknowns:
            assert [p.eval(kk) for p in symbolic[f]] == list(fixed[f]), (f, kk)


def test_symbolic_closed_form() -> None:
    values = series_in_k(System(BRODA_L, z=z, k=k), 5)
    assert values[L][5] == sp.Poly(8 * k**2 * (3 * k + 4), k, domain=sp.ZZ)


@pytest.mark.skipif(not flint_available(), reason="python-flint missing")
@pytest.mark.parametrize("name", list(BRODA))
def test_backends_identical(name: str) -> None:
    system = System(BRODA[name], z=z, k=k)
    for kk in (None, 10):
        n = 30 if kk is None else 300
        python = _series(system, n, kk, backend_name="python")
        flint = _series(system, n, kk, backend_name="flint")
        assert python == flint
        assert all(
            type(a) is type(b)
            for f in python
            for a, b in zip(python[f], flint[f], strict=True)
        )


# -- API -----------------------------------------------------------------------------


def test_result_types_and_shape() -> None:
    catalan_sys = System({F: 1 + z * F**2}, z=z)
    values = series(catalan_sys, 4)
    assert list(values) == [F]
    assert len(values[F]) == 5  # indices 0..4 inclusive
    assert all(type(c) is int for c in values[F])
    assert series(catalan_sys, 0)[F] == (1,)
    with pytest.raises(TypeError):
        values[F] = ()  # type: ignore[index]
    rs = series(System(BRODA_RS, z=z, k=k), 3, k=2)
    assert list(rs) == [R, S]
    poly = series_in_k(System(BRODA_R, z=z, k=k), 3)[R]
    assert all(isinstance(p, sp.Poly) and p.domain == sp.ZZ for p in poly)
    rational = series_in_k(System({F: z + k * z * F**2 / 2}, z=z, k=k), 3)[F]
    assert all(p.domain == sp.QQ for p in rational)
    assert rational[3] == sp.Poly(k / 2, k, domain=sp.QQ)


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda s, c: series(s, -1), ValueError, "n must be >= 0"),
        (lambda s, c: series(s, True), TypeError, "n must be an int"),
        (lambda s, c: series(s, 2.0), TypeError, "n must be an int"),  # type: ignore[arg-type]
        (lambda s, c: series(s, 3, k=2), ValueError, "omit the k"),
        (lambda s, c: series(c, 3), ValueError, "series_in_k"),
        (lambda s, c: series(c, 3, k=0), ValueError, ">= 1"),
        (lambda s, c: series(c, 3, k=2.0), TypeError, "k must be an int"),  # type: ignore[arg-type]
        (lambda s, c: series_in_k(s, 3), ValueError, "needs a system with a parameter"),
        (lambda s, c: series("system", 3), TypeError, "flanpy.System"),  # type: ignore[arg-type]
    ],
)
def test_errors(call: Any, error: type[Exception], match: str) -> None:
    plain = System({F: 1 + z * F**2}, z=z)
    with_k = System(BRODA_R, z=z, k=k)
    with pytest.raises(error, match=match):
        call(plain, with_k)


# -- N = 1000 (slow) -------------------------------------------------------------------

# Time budgets in seconds: about 5x the times measured on an Apple M1 (series.md, 6).
BUDGET = {"python": 60.0, "flint": 10.0}


def truncated_product(a: list[int], b: list[int], n: int) -> list[int]:
    out = [0] * (n + 1)
    for i, x in enumerate(a[: n + 1]):
        if x:
            for j, y in enumerate(b[: n + 1 - i]):
                out[i + j] += x * y
    return out


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kk", [2, 10, 50])
def test_n1000_R_S(backend: Any, kk: int) -> None:
    start = time.perf_counter()
    got = _series(System(BRODA_RS, z=z, k=k), 1000, kk, backend_name=backend)
    assert time.perf_counter() - start < BUDGET[backend]
    for n in (997, 998, 999, 1000):
        assert got[R][n] == broda_eq3(n, kk)
        assert got[S][n] == broda_eq5(n, kk)


@pytest.mark.slow
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kk", [2, 10, 50])
def test_n1000_L(backend: Any, kk: int) -> None:
    n = 1000
    start = time.perf_counter()
    got = _series(System(BRODA_L, z=z, k=k), n, kk, backend_name=backend)
    assert time.perf_counter() - start < BUDGET[backend]
    # Broda et al. 2020, eqs. (17)-(18): p_k w^3 + k z ((k+12) k z^2 + 4 k z + 1) w
    # - k^2 z^2 (2 z + 1) = 0, with p_k = 4k(2k+27) z^3 + 36k z^2 - k z - 4.
    w = list(got[L])
    w3 = truncated_product(truncated_product(w, w, n), w, n)
    p = [-4, -kk, 36 * kk, 4 * kk * (2 * kk + 27)]
    q = [0, kk, 4 * kk * kk, kk * kk * (kk + 12)]
    lhs = [
        a + b
        for a, b in zip(
            truncated_product(p, w3, n), truncated_product(q, w, n), strict=True
        )
    ]
    lhs[2] -= kk * kk
    lhs[3] -= 2 * kk * kk
    assert lhs == [0] * (n + 1)
