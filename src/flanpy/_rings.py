# SPDX-License-Identifier: GPL-3.0-or-later
"""Coefficient rings for exact computations, with a pure-Python and a flint backend.

Private module (design: ``dev/design/series.md``, section 4). A backend supplies, for
each coefficient domain, a :class:`_Ring`: a scalar type with ``+`` and ``*`` plus
conversions in and out. Algorithms are written once against plain ``+`` and ``*``.

Domains:

- ``"ZZ"``, ``"QQ"``: integers and rationals (fixed ``k``);
- ``"ZZ[k]"``, ``"QQ[k]"``: polynomials in ``k`` (symbolic ``k``).

Only scalar rings exist so far. A Newton iteration for large orders (series.md,
section 3.2) would add truncated series multiplication to :class:`_Backend`, next to
:meth:`_Backend.ring`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import sympy as sp

Domain = Literal["ZZ", "QQ", "ZZ[k]", "QQ[k]"]
BackendName = Literal["python", "flint"]


@dataclass(frozen=True)
class _Ring:
    """A coefficient ring: its zero and one, and conversions.

    ``from_coeffs(cs)`` builds the element ``sum cs[i] k^i`` (for ``ZZ`` and ``QQ``
    ``cs`` has at most one entry); ``to_public`` converts an element to the public
    result type (``int``, ``Fraction`` or ``sympy.Poly`` in ``k``).
    """

    domain: Domain
    zero: Any
    one: Any
    from_coeffs: Callable[[Sequence[Fraction]], Any]
    to_public: Callable[[Any], Any]


class _Backend:
    """Base class of the backends: builds a :class:`_Ring` for each domain."""

    name: BackendName

    def ring(self, domain: Domain, k: sp.Symbol | None = None) -> _Ring:
        """Return the ring for ``domain`` (``k`` names the variable of ``*[k]``)."""
        raise NotImplementedError


def _constant(cs: Sequence[Fraction]) -> Fraction:
    if len(cs) > 1:
        raise ValueError("a constant ring cannot hold a polynomial in k")
    return cs[0] if cs else Fraction(0)


def _as_int(c: Fraction) -> int:
    if c.denominator != 1:
        raise ValueError(f"{c} is not an integer")
    return c.numerator


def _poly(coeffs: Sequence[Any], k: sp.Symbol, domain: Domain) -> sp.Poly:
    """``sympy.Poly`` in ``k`` from ``int``/``Fraction`` coefficients, lowest first."""
    rep = {
        (i,): sp.Rational(c.numerator, c.denominator) for i, c in enumerate(coeffs) if c
    }
    return sp.Poly.from_dict(
        rep or {(0,): 0}, k, domain=sp.ZZ if domain == "ZZ[k]" else sp.QQ
    )


class _PythonBackend(_Backend):
    """``int`` / ``Fraction`` scalars; SymPy sparse polynomials for ``Q[k]``."""

    name: BackendName = "python"

    def ring(self, domain: Domain, k: sp.Symbol | None = None) -> _Ring:
        if domain == "ZZ":
            return _Ring(domain, 0, 1, lambda cs: _as_int(_constant(cs)), int)
        if domain == "QQ":
            return _Ring(domain, Fraction(0), Fraction(1), _constant, Fraction)
        if k is None:
            raise ValueError(f"domain {domain} needs the symbol k")
        from sympy.polys.rings import ring as poly_ring

        ground = sp.ZZ if domain == "ZZ[k]" else sp.QQ
        R, _ = poly_ring(str(k), ground)

        def element(c: Fraction) -> Any:
            if domain == "ZZ[k]":
                return ground(_as_int(c))
            return ground(c.numerator, c.denominator)

        def from_coeffs(cs: Sequence[Fraction]) -> Any:
            return R({(i,): element(c) for i, c in enumerate(cs) if c})

        def to_public(x: Any) -> sp.Poly:
            degree = max((e[0] for e in x), default=-1)
            coeffs = [Fraction(0)] * (degree + 1)
            for (e,), c in x.items():
                coeffs[e] = Fraction(int(ground.numer(c)), int(ground.denom(c)))
            return _poly(coeffs, k, domain)

        return _Ring(domain, R.zero, R.one, from_coeffs, to_public)


class _FlintBackend(_Backend):
    """python-flint scalars: ``fmpz``, ``fmpq``, ``fmpz_poly``, ``fmpq_poly``."""

    name: BackendName = "flint"

    def __init__(self) -> None:
        import flint

        self.flint = flint

    def ring(self, domain: Domain, k: sp.Symbol | None = None) -> _Ring:
        fl = self.flint
        if domain == "ZZ":
            return _Ring(
                domain,
                fl.fmpz(0),
                fl.fmpz(1),
                lambda cs: fl.fmpz(_as_int(_constant(cs))),
                int,
            )
        if domain == "QQ":

            def q(c: Fraction) -> Any:
                return fl.fmpq(c.numerator, c.denominator)

            return _Ring(
                domain,
                fl.fmpq(0),
                fl.fmpq(1),
                lambda cs: q(_constant(cs)),
                lambda x: Fraction(int(x.p), int(x.q)),
            )
        if k is None:
            raise ValueError(f"domain {domain} needs the symbol k")
        if domain == "ZZ[k]":
            return _Ring(
                domain,
                fl.fmpz_poly([]),
                fl.fmpz_poly([1]),
                lambda cs: fl.fmpz_poly([_as_int(c) for c in cs]),
                lambda x: _poly([Fraction(int(c)) for c in x.coeffs()], k, domain),
            )
        return _Ring(
            domain,
            fl.fmpq_poly([]),
            fl.fmpq_poly([1]),
            lambda cs: fl.fmpq_poly([fl.fmpq(c.numerator, c.denominator) for c in cs]),
            lambda x: _poly(
                [Fraction(int(c.p), int(c.q)) for c in x.coeffs()], k, domain
            ),
        )


def flint_available() -> bool:
    """Whether python-flint can be imported."""
    try:
        import flint  # noqa: F401
    except ImportError:
        return False
    return True


def backend(name: BackendName | None = None) -> _Backend:
    """Return the named backend; by default flint if installed, else pure Python."""
    if name is None:
        name = "flint" if flint_available() else "python"
    if name == "python":
        return _PythonBackend()
    if name == "flint":
        if not flint_available():
            raise ImportError(
                "the flint backend needs python-flint (pip install flanpy[fast])"
            )
        return _FlintBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'python' or 'flint'")
