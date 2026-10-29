# SPDX-License-Identifier: GPL-3.0-or-later
"""Polynomial systems ``F = Phi(z, k, F)`` and their well-posedness check.

The design is recorded in ``dev/design/system.md``. Pivoteau, Salvy & Soria
(2012) are cited with the numbering of arXiv:1109.2688v2 (definitions,
theorems, lemmas, algorithms).
"""

from __future__ import annotations

import os
import re
import textwrap
import warnings
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, NoReturn

import sympy as sp

__all__ = [
    "IllPosedSystemError",
    "System",
    "UndecidedSystemError",
    "WellPosednessError",
    "ZeroCoordinateWarning",
]


def _signed_cap(m: int) -> int:
    """Return the iteration cap for the constant terms of a signed system.

    Non-negative systems need no cap: Pivoteau, Salvy & Soria (2012,
    arXiv:1109.2688v2 numbering), Thm 5.7,
    bound the number of steps by ``m``. Signed systems may need more than ``m``
    steps (see dev/design/system.md, section 5.3), and no bound is known.
    """
    return max(32, 4 * m)


class WellPosednessError(ValueError):
    """The well-posedness check of a :class:`System` did not pass.

    Base class of :class:`IllPosedSystemError` (proven ill-posed) and
    :class:`UndecidedSystemError` (not decided).

    Examples
    --------
    One ``except`` clause catches both outcomes:

    >>> from sympy import symbols
    >>> from flanpy import System, WellPosednessError
    >>> z, F = symbols("z F")
    >>> try:
    ...     System({F: 1 + F**2}, z=z)
    ... except WellPosednessError as error:
    ...     print(type(error).__name__)
    IllPosedSystemError
    """


class IllPosedSystemError(WellPosednessError):
    """The system is proven ill-posed: the iteration from 0 does not converge.

    Examples
    --------
    The Catalan equation without its ``z``, ``F = 1 + F^2``: the constant terms
    grow forever.

    >>> from sympy import symbols
    >>> from flanpy import System
    >>> z, F = symbols("z F")
    >>> System({F: 1 + F**2}, z=z)
    Traceback (most recent call last):
        ...
    flanpy.IllPosedSystemError: the system is ill-posed.
      The constant terms do not stabilise, so the iteration does not converge.
      After 1 iteration(s) F(0) = 1, after 2: F(0) = 2. A convergent system with
      non-negative coefficients reaches its constant terms within 1 iteration(s)
      (Pivoteau, Salvy & Soria 2012, arXiv:1109.2688v2, Thm 5.7).
      Equations:
        F = F**2 + 1
    """


class UndecidedSystemError(WellPosednessError):
    """The check could not decide whether the system is well-posed.

    This is never a claim that the system is ill-posed.

    Examples
    --------
    ``F = z - F`` is in fact ill-posed: its iterates alternate ``0, z, 0, z, ...``
    and never converge. But the check is conservative for signed systems: it
    proves ill-posedness only when the constant terms cycle, and here they are
    constantly 0, so it reports "undecided" rather than guessing.

    >>> from sympy import symbols
    >>> from flanpy import System
    >>> z, F = symbols("z F")
    >>> System({F: z - F}, z=z)
    Traceback (most recent call last):
        ...
    flanpy.UndecidedSystemError: could not decide whether the system is well-posed.
      It has negative coefficients, and the nilpotency test fails. At z = 0 the
      constant terms are F(0) = 0 and the Jacobian dPhi/dF is [[-1]], which is not
      nilpotent: the dependency cycle F -> F has no factor z. For systems with
      negative coefficients this failure does not prove divergence.
      Equations:
        F = -F + z
    """


class ZeroCoordinateWarning(UserWarning):
    """Some unknowns of a :class:`System` are identically zero.

    Examples
    --------
    >>> import warnings
    >>> from sympy import symbols
    >>> from flanpy import System
    >>> z, F, G = symbols("z F G")
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     system = System({F: z + G, G: G}, z=z)
    >>> print(caught[0].message)
    G is identically 0 (zero coordinate).
      Every iterate from 0 is 0 there, so the solution keeps it as 0. Pivoteau,
      Salvy & Soria (2012, arXiv:1109.2688v2, Def. 5.3) exclude such systems.
      Remove this unknown if this is not intended.
    """


_Exprs = tuple[Any, ...]  # tuples of sympy expressions (sympy is untyped)


class System:
    r"""Polynomial system ``F = Phi(z, k, F)`` for generating functions.

    The unknowns ``F_1..F_m`` are formal power series in ``z``, and ``k`` (the
    alphabet size) is a parameter whose domain is the integers ``k >= 1``.
    Each right-hand side ``Phi_i`` is a polynomial in the unknowns, ``z`` and
    ``k`` with rational coefficients. This is the form the symbolic method produces
    (Flajolet & Sedgewick 2009, Def. I.7, p. 33, and eq. (VII.61), p. 482).

    Parameters
    ----------
    equations : Mapping[sympy.Symbol, sympy.Expr | int]
        Maps each unknown ``F_i`` to ``Phi_i``. The mapping order fixes the
        order of the unknowns.
    z : sympy.Symbol
        The size variable.
    k : sympy.Symbol or None, optional
        The parameter (alphabet size). ``None`` if the system has no parameter.

    Raises
    ------
    TypeError
        If a key, ``z`` or ``k`` is not a :class:`sympy.Symbol`, or a value is
        not a SymPy expression or an integer.
    ValueError
        If the input is malformed: no equations, clashing symbols, a symbol
        other than the unknowns, ``z`` and ``k``, or a right-hand side that is
        not a polynomial with rational coefficients.
    IllPosedSystemError
        If the system is proven ill-posed (see Notes).
    UndecidedSystemError
        If the check cannot decide whether the system is well-posed.

    Warns
    -----
    ZeroCoordinateWarning
        If some unknowns are identically zero.

    Notes
    -----
    The system is *well-posed* if the iteration ``F^(0) = 0``,
    ``F^(n+1) = Phi(z, k, F^(n))`` converges in the ``z``-adic topology, for
    symbolic ``k``. The limit is the solution that the rest of the package
    uses. This is the semantics of recursion of Flajolet & Sedgewick
    (2009, pp. 32-33).

    The check runs with ``k`` symbolic. It works in three steps.

    1. *Zero coordinates.* Support propagation finds the unknowns whose
       iterates are all 0. They are set to 0, and a warning names them.
       Pivoteau, Salvy & Soria (2012, Def. 5.3) exclude such systems; here they
       are accepted, because their iteration converges.
    2. *Constant terms.* On the remaining ``m'`` unknowns, the constant terms
       ``c^(h+1) = Phi(0, k, c^(h))``, ``c^(0) = 0``, must become stationary,
       at some ``c*``.
    3. *Jacobian.* ``J0 = dPhi/dF (0, k, c*)`` must be nilpotent.

    If steps 2 and 3 pass, the system is well-posed, whatever the signs of
    the coefficients.

    The system is *non-negative* if all its coefficients are ``>= 0`` for
    every ``k`` in the domain. The test is sufficient and does not sample
    ``k``: substitute ``k = 1 + t`` and require every coefficient, as a
    polynomial in the unknowns, ``z`` and ``t``, to be ``>= 0``. For example,
    ``F = z + (k - 1) z F^2`` is non-negative.

    For non-negative systems the check is a decision procedure. It follows
    Pivoteau, Salvy & Soria (2012): Thm 3.6 and Thm 5.5 (characterisation of
    well-founded systems), and Thm 5.7 (``c* = c^(m')``). A failure in step 2
    or 3 then proves that the system is ill-posed for every integer
    ``k >= 2``. It also proves this for ``k = 1``, unless some coefficient
    vanishes at ``k = 1``; the error message says which case applies.

    When the system is not non-negative (*signed*), the check is only
    partial:

    - a cycle in the constant terms proves the system ill-posed;
    - otherwise a failure raises :class:`UndecidedSystemError`.

    A system that passes for symbolic ``k`` passes for every ``k``. The
    iterates and the identity ``J0^m' = 0`` are polynomial identities in
    ``k``, and they survive substitution. Specialising may create zero
    coordinates, e.g. ``G = (k - 1) z`` at ``k = 1``.

    References
    ----------
    .. [1] P. Flajolet and R. Sedgewick, *Analytic Combinatorics*, Cambridge
       University Press, 2009.
    .. [2] C. Pivoteau, B. Salvy and M. Soria, "Algorithms for combinatorial
       structures: Well-founded systems and Newton iterations", *J. Combin.
       Theory Ser. A* 119 (2012) 1711-1773. Cited with the numbering of
       arXiv:1109.2688v2.
    .. [3] S. Broda, A. Machiavelo, N. Moreira and R. Reis, "Analytic
       Combinatorics and Descriptional Complexity of Regular Languages on
       Average", *ACM SIGACT News* 51(1) (2020) 38-56.

    Examples
    --------
    Catalan numbers, ``F = 1 + z F^2``:

    >>> from sympy import symbols
    >>> from flanpy import System
    >>> z, k, F, R = symbols("z k F R")
    >>> catalan = System({F: 1 + z * F**2}, z=z)
    >>> catalan
    System({F: F**2*z + 1}, z=z)
    >>> catalan.polynomials()
    (Poly(-F**2*z + F - 1, F, z, domain='QQ'),)

    All regular expressions over ``k`` letters, from grammar (1) of
    Broda et al. (2020):

    >>> System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
    System({R: 2*R**2*z + R*z + k*z + z}, z=z, k=k)

    An ill-posed system: ``F = F + z`` never fixes the coefficient of ``z``.

    >>> System({F: F + z}, z=z)
    Traceback (most recent call last):
        ...
    flanpy.IllPosedSystemError: the system is ill-posed.
      The iteration from 0 does not converge. At z = 0 the constant terms are
      F(0) = 0 and the Jacobian dPhi/dF is [[1]], which is not nilpotent: the
      dependency cycle F -> F has no factor z. (Non-negative system: Pivoteau,
      Salvy & Soria 2012, arXiv:1109.2688v2, Thm 5.5 and Thm 3.6.)
      Equations:
        F = F + z
    """

    __slots__ = (
        "_active",
        "_constant_terms",
        "_jacobian0",
        "_k",
        "_nilpotency_index",
        "_nonnegative",
        "_params",
        "_rhs",
        "_unknowns",
        "_z",
    )

    _unknowns: tuple[Any, ...]
    _rhs: _Exprs
    _z: Any
    _k: Any
    _params: tuple[Any, ...]
    _active: tuple[int, ...]
    _constant_terms: _Exprs
    _jacobian0: Any
    _nilpotency_index: int
    _nonnegative: bool

    def __init__(
        self,
        equations: Mapping[sp.Symbol, sp.Expr | int],
        *,
        z: sp.Symbol,
        k: sp.Symbol | None = None,
    ) -> None:
        unknowns, rhs, params = _parse(equations, z, k)
        _set = object.__setattr__
        _set(self, "_unknowns", unknowns)
        _set(self, "_rhs", rhs)
        _set(self, "_z", z)
        _set(self, "_k", k)
        _set(self, "_params", params)
        _Check(self).run()

    # -- public interface ------------------------------------------------------

    @property
    def unknowns(self) -> tuple[sp.Symbol, ...]:
        """The unknowns ``F_1..F_m``, in the order of the input mapping.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System
        >>> z, F = symbols("z F")
        >>> System({F: 1 + z * F**2}, z=z).unknowns
        (F,)
        """
        return self._unknowns

    @property
    def equations(self) -> Mapping[sp.Symbol, sp.Expr]:
        """Read-only mapping ``F_i -> Phi_i`` (right-hand sides expanded).

        This is the fixed-point form ``F = Phi(z, k, F)``.

        Examples
        --------
        >>> from sympy import symbols
        >>> from flanpy import System
        >>> z, F = symbols("z F")
        >>> System({F: 1 + z * F**2}, z=z).equations
        mappingproxy({F: F**2*z + 1})
        """
        return MappingProxyType(dict(zip(self._unknowns, self._rhs, strict=True)))

    @property
    def z(self) -> sp.Symbol:
        """The size variable."""
        return self._z

    @property
    def k(self) -> sp.Symbol | None:
        """The parameter (alphabet size), or ``None``."""
        return self._k

    def polynomials(self) -> tuple[sp.Poly, ...]:
        """Return the system as polynomial equations ``P_i = F_i - Phi_i = 0``.

        This is the form used for elimination (Flajolet & Sedgewick 2009,
        eq. (VII.77), p. 494). Each ``P_i`` is a :class:`sympy.Poly` over
        ``QQ`` in the generators ``(F_1, ..., F_m, z, k)``, with ``k`` last
        (``k`` is omitted if it is ``None``).

        Returns
        -------
        tuple of sympy.Poly
            One polynomial per unknown, in the order of :attr:`unknowns`.

        Examples
        --------
        All regular expressions over ``k`` letters, Broda et al. (2020),
        grammar (1):

        >>> from sympy import symbols
        >>> from flanpy import System
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> rk.polynomials()
        (Poly(-2*R**2*z - R*z + R - z*k - z, R, z, k, domain='QQ'),)
        """
        gens = (*self._unknowns, self._z, *self._params)
        return tuple(
            sp.Poly(f - phi, *gens, domain=sp.QQ)
            for f, phi in zip(self._unknowns, self._rhs, strict=True)
        )

    def specialize(self, k: int) -> System:
        """Substitute a value for the parameter ``k``.

        Parameters
        ----------
        k : int
            An integer ``k >= 1`` (the domain of ``k``).

        Returns
        -------
        System
            The system with ``k`` replaced by the value and no parameter. It is
            well-posed whenever ``self`` is (see the Notes of :class:`System`),
            but it may have zero coordinates, and then it warns.

        Raises
        ------
        ValueError
            If the system has no parameter, or ``k < 1``.
        TypeError
            If ``k`` is not an ``int``.

        Examples
        --------
        Regular expressions over 2 letters, Broda et al. (2020), grammar (1):

        >>> from sympy import symbols
        >>> from flanpy import System
        >>> z, k, R = symbols("z k R")
        >>> rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
        >>> rk.specialize(2)
        System({R: 2*R**2*z + R*z + 3*z}, z=z)
        """
        if self._k is None:
            raise ValueError("the system has no parameter k to specialize")
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an int, got {k!r} of type {type(k).__name__}")
        if k < 1:
            raise ValueError(f"k must be an integer >= 1 (alphabet size), got {k}")
        value = sp.Integer(k)
        return System(
            {
                f: sp.expand(phi.xreplace({self._k: value}))
                for f, phi in zip(self._unknowns, self._rhs, strict=True)
            },
            z=self._z,
        )

    # -- value semantics -------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (self._unknowns, self._rhs, self._z, self._k)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, System):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        body = ", ".join(
            f"{sp.sstr(f)}: {sp.sstr(phi)}"
            for f, phi in zip(self._unknowns, self._rhs, strict=True)
        )
        tail = "" if self._k is None else f", k={sp.sstr(self._k)}"
        return f"System({{{body}}}, z={sp.sstr(self._z)}{tail})"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_restore, (tuple((n, getattr(self, n)) for n in System.__slots__),))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("System is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("System is immutable")


def _restore(state: tuple[tuple[str, Any], ...]) -> System:
    """Rebuild a pickled :class:`System` without repeating the check."""
    system = object.__new__(System)
    for name, value in state:
        object.__setattr__(system, name, value)
    return system


# Public names report flanpy as their module (tracebacks, reprs, pickling).
for _cls in (
    System,
    WellPosednessError,
    IllPosedSystemError,
    UndecidedSystemError,
    ZeroCoordinateWarning,
):
    _cls.__module__ = "flanpy"


# -- input parsing ---------------------------------------------------------------


def _parse(
    equations: Mapping[Any, Any], z: object, k: object
) -> tuple[tuple[Any, ...], _Exprs, tuple[Any, ...]]:
    """Validate the input of :class:`System`; return unknowns, rhs, params."""
    if not isinstance(z, sp.Symbol):
        raise TypeError(f"z must be a sympy.Symbol, got {z!r}")
    if k is not None and not isinstance(k, sp.Symbol):
        raise TypeError(f"k must be a sympy.Symbol or None, got {k!r}")
    if not isinstance(equations, Mapping):
        raise TypeError(
            "equations must be a mapping {unknown: right-hand side}, "
            f"got {type(equations).__name__}"
        )
    if not equations:
        raise ValueError("a System needs at least one equation")
    params = () if k is None else (k,)
    unknowns = tuple(equations)
    for f in unknowns:
        if not isinstance(f, sp.Symbol):
            raise TypeError(f"unknowns must be sympy.Symbol, got {f!r}")
        if f in (z, k):
            raise ValueError(f"{f} is used both as an unknown and as z or k")
    names: dict[str, Any] = {}
    for s in (*unknowns, z, *params):
        if s.name in names and names[s.name] != s:
            raise ValueError(
                f"two different symbols are named {s.name!r} (SymPy symbols with "
                "different assumptions are different symbols)"
            )
        names[s.name] = s
    if k is not None and k == z:
        raise ValueError("z and k must be different symbols")

    gens = (*unknowns, z, *params)
    allowed = set(gens)
    rhs = []
    for f in unknowns:
        try:
            phi = sp.sympify(equations[f], strict=True)
        except sp.SympifyError:
            raise TypeError(
                f"the right-hand side for {f} must be a SymPy expression or an "
                f"int, got {equations[f]!r}"
            ) from None
        if not isinstance(phi, sp.Expr):
            raise TypeError(f"the right-hand side for {f} is not an expression: {phi}")
        stray = phi.free_symbols - allowed
        if stray:
            names_ = ", ".join(sorted(str(s) for s in stray))
            hint = (
                " A symbol has the name of an unknown, z or k but different "
                "assumptions; SymPy treats it as a different symbol."
                if any(s.name in names for s in stray)
                else " Marking variables are not supported yet."
            )
            raise ValueError(
                f"the right-hand side for {f} contains symbols other than the "
                f"unknowns, z and k: {names_}.{hint}"
            )
        if phi.has(sp.Float):
            raise ValueError(
                f"the right-hand side for {f} has a floating-point coefficient: "
                f"{phi}. Use exact rationals, e.g. Rational(1, 2)."
            )
        if not phi.is_polynomial(*gens):
            variables = ", ".join(str(g) for g in gens)
            raise ValueError(
                f"the right-hand side for {f} is not a polynomial in ({variables}): "
                f"{phi}. Write SEQ-like terms with an auxiliary unknown, e.g. "
                f"S = 1 + z*{f}*S."
            )
        if sp.Poly(phi, *gens).domain not in (sp.ZZ, sp.QQ):
            raise ValueError(
                f"the right-hand side for {f} has coefficients that are not "
                f"rational: {phi}."
            )
        rhs.append(sp.expand(phi))
    return unknowns, tuple(rhs), params


# -- well-posedness check --------------------------------------------------------


def _equal(a: _Exprs, b: _Exprs) -> bool:
    return all(sp.expand(x - y) == 0 for x, y in zip(a, b, strict=True))


class _Check:
    """The well-posedness check of a :class:`System` (see its Notes)."""

    def __init__(self, system: System) -> None:
        self.s = system
        self.m = len(system._unknowns)
        gens = (*system._unknowns, system._z, *system._params)
        self.polys = [sp.Poly(phi, *gens, domain=sp.QQ) for phi in system._rhs]
        k = system._k
        self.uses_k = k is not None and any(
            k in phi.free_symbols for phi in system._rhs
        )
        # Non-negative on the domain k >= 1: with k = 1 + t, every coefficient
        # in (F, z, t) is >= 0 (a sufficient test that does not sample k).
        if self.uses_k:
            t = sp.Dummy("t")
            shifted = [
                sp.Poly(sp.expand(phi.xreplace({k: 1 + t})), *gens[:-1], t)
                for phi in system._rhs
            ]
        else:
            shifted = self.polys
        self.nonnegative = all(c >= 0 for p in shifted for c in p.coeffs())
        # Whether some coefficient (a polynomial in k) vanishes at k = 1. Such a
        # polynomial, non-negative in t = k - 1, can vanish on the domain only
        # there, so elsewhere the support is the same as for symbolic k.
        self.vanishes_at_1 = self.uses_k and any(
            c != 0 and c.subs(k, 1) == 0
            for phi in system._rhs
            for c in sp.Poly(phi, *gens[:-1]).coeffs()
        )

    def run(self) -> None:
        s = self.s
        active = self._zero_coordinates()
        syms = [s._unknowns[i] for i in active]
        zero = {s._unknowns[i]: sp.Integer(0) for i in range(self.m) if i not in active}
        red = [sp.expand(s._rhs[i].xreplace(zero)) for i in active]
        at0 = [phi.xreplace({s._z: sp.Integer(0)}) for phi in red]

        def step(c: _Exprs) -> _Exprs:
            sub = dict(zip(syms, c, strict=True))
            return tuple(sp.expand(phi.xreplace(sub)) for phi in at0)

        c_star = self._constant_terms(step, len(active), syms)
        sub = dict(zip(syms, c_star, strict=True))
        sub[s._z] = sp.Integer(0)
        jac = sp.ImmutableMatrix(
            len(active),
            len(active),
            lambda i, j: sp.expand(sp.diff(red[i], syms[j]).xreplace(sub)),
        )
        index = _nilpotency_index(jac)
        if index is None:
            self._jacobian_failure(jac, syms, c_star)

        constants = [sp.Integer(0)] * self.m
        for i, c in zip(active, c_star, strict=True):
            constants[i] = c
        _set = object.__setattr__
        _set(s, "_active", tuple(active))
        _set(s, "_constant_terms", tuple(constants))
        _set(s, "_jacobian0", jac)
        _set(s, "_nilpotency_index", index)
        _set(s, "_nonnegative", self.nonnegative)

    def _zero_coordinates(self) -> list[int]:
        """Return the indices of the unknowns that are not identically zero.

        Support propagation (Pivoteau, Salvy & Soria 2012, arXiv:1109.2688v2,
        Lemma 3.4 and Algorithm 0-coord): an unknown becomes non-zero once some
        monomial of its right-hand side has all its unknowns non-zero. An
        unknown never marked is 0 in every iterate, whatever the signs.
        """
        m = self.m
        supports = [
            [
                frozenset(j for j in range(m) if monom[j] > 0)
                for monom, coeff in p.terms()
                if coeff != 0
            ]
            for p in self.polys
        ]
        nonzero = [False] * m
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if not nonzero[i] and any(
                    all(nonzero[j] for j in mono) for mono in supports[i]
                ):
                    nonzero[i] = changed = True
        zeros = [str(self.s._unknowns[i]) for i in range(m) if not nonzero[i]]
        if zeros:
            one = len(zeros) == 1
            warnings.warn(
                f"{', '.join(zeros)} {'is' if one else 'are'} identically 0 "
                f"(zero coordinate{'' if one else 's'}).\n"
                + _wrap(
                    "Every iterate from 0 is 0 there, so the solution keeps "
                    f"{'it' if one else 'them'} as 0. Pivoteau, Salvy & Soria (2012, "
                    "arXiv:1109.2688v2, Def. 5.3) exclude such systems. Remove "
                    f"{'this unknown' if one else 'these unknowns'} if this is not "
                    "intended."
                ),
                ZeroCoordinateWarning,
                # Point at the caller's code, from any depth inside flanpy.
                skip_file_prefixes=(os.path.dirname(__file__),),
            )
        return [i for i in range(m) if nonzero[i]]

    def _constant_terms(
        self, step: Callable[[_Exprs], _Exprs], m: int, syms: Sequence[Any]
    ) -> _Exprs:
        """Iterate the constant terms ``c^(h+1) = Phi(0, k, c^(h))`` from 0."""
        c: _Exprs = (sp.Integer(0),) * m
        if self.nonnegative:
            # Pivoteau, Salvy & Soria (2012, arXiv:1109.2688v2), Thm 5.7:
            # S(0) = H^m(0, 0).
            for _ in range(m):
                c = step(c)
            nxt = step(c)
            if not _equal(c, nxt):
                self._fail(
                    IllPosedSystemError,
                    "the system is ill-posed: the constant terms do not stabilise, "
                    f"so the iteration does not converge. After {m} iteration(s) "
                    f"{_fmt(syms, c)}, after {m + 1}: {_fmt(syms, nxt)}. A "
                    "convergent system with non-negative coefficients reaches its "
                    f"constant terms within {m} iteration(s) (Pivoteau, Salvy & "
                    "Soria 2012, arXiv:1109.2688v2, Thm 5.7).",
                    by_support=True,
                )
            return c
        cap = _signed_cap(m)
        history = [c]
        for _ in range(cap):
            nxt = step(c)
            if _equal(c, nxt):
                return c
            for j, old in enumerate(history[:-1]):
                if _equal(old, nxt):
                    cycle = " -> ".join(_fmt(syms, x) for x in [*history[j:], nxt])
                    self._fail(
                        IllPosedSystemError,
                        "the system is ill-posed: the constant terms cycle, so the "
                        "coefficient of z^0 never settles and the iteration does "
                        f"not converge: {cycle}.",
                        by_support=False,
                    )
            history.append(nxt)
            c = nxt
        self._fail(
            UndecidedSystemError,
            "could not decide whether the system is well-posed: it has negative "
            f"coefficients, and its constant terms did not stabilise within {cap} "
            f"iterations from 0 (last two: {_fmt(syms, history[-2])}, then "
            f"{_fmt(syms, history[-1])}).",
        )

    def _jacobian_failure(self, jac: Any, syms: Sequence[Any], c_star: _Exprs) -> None:
        cycle = _keep(" -> ".join(str(syms[i]) for i in _support_cycle(jac)))
        where = (
            f"At z = 0 the constant terms are {_fmt(syms, c_star)} and the Jacobian "
            f"dPhi/dF is {_keep(sp.sstr(jac.tolist()))}, which is not nilpotent: the "
            f"dependency cycle {cycle} has no factor z."
        )
        if self.nonnegative:
            self._fail(
                IllPosedSystemError,
                "the system is ill-posed: the iteration from 0 does not converge. "
                f"{where} (Non-negative system: Pivoteau, Salvy & Soria 2012, "
                "arXiv:1109.2688v2, Thm 5.5 and Thm 3.6.)",
                by_support=True,
            )
        self._fail(
            UndecidedSystemError,
            "could not decide whether the system is well-posed: it has negative "
            f"coefficients, and the nilpotency test fails. {where} For systems "
            "with negative coefficients this failure does not prove divergence.",
        )

    def _fail(
        self,
        kind: type[WellPosednessError],
        message: str,
        *,
        by_support: bool = False,
    ) -> NoReturn:
        s = self.s
        # A short headline, then the details wrapped and indented.
        headline, _, detail = message.partition(": ")
        lines = [headline + ".", _wrap(detail[0].upper() + detail[1:]), "  Equations:"]
        lines += [
            f"    {f} = {sp.sstr(phi)}"
            for f, phi in zip(s._unknowns, s._rhs, strict=True)
        ]
        if kind is IllPosedSystemError and self.uses_k:
            if by_support and not self.vanishes_at_1:
                note = (
                    "This holds for every integer k >= 1 (no coefficient vanishes "
                    "there, so the support of the system does not change)."
                )
            elif by_support:
                note = (
                    "This holds for every integer k >= 2. At k = 1 some "
                    "coefficients vanish, so k = 1 is not covered: use "
                    "specialize(1) to check it."
                )
            else:
                note = (
                    "This holds for symbolic k; finitely many values of k may "
                    "behave differently."
                )
            lines.append(_wrap(note))
        raise kind("\n".join(lines))


_NBSP = "\u00a0"  # keeps formulas such as "F(0) = 0" on one line when wrapping


def _keep(text: str) -> str:
    """Make the spaces in ``text`` unbreakable for :func:`_wrap`."""
    return text.replace(" ", _NBSP)


def _wrap(text: str) -> str:
    """Wrap a message paragraph at 78 columns, indented by two spaces."""
    # Keep citations such as "Thm 5.5" and "Def. 5.3" on one line.
    text = re.sub(r"\b(Thm|Def\.) (\d)", rf"\1{_NBSP}\2", text)
    wrapped = textwrap.fill(
        text,
        width=78,
        initial_indent="  ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped.replace(_NBSP, " ")


def _fmt(syms: Sequence[Any], values: _Exprs) -> str:
    return (
        ", ".join(
            _keep(f"{f}(0) = {sp.sstr(v)}") for f, v in zip(syms, values, strict=True)
        )
        or "(none)"
    )


def _nilpotency_index(jac: Any) -> int | None:
    """Return the least ``nu`` with ``jac**nu == 0``, or ``None`` if not nilpotent."""
    n = jac.shape[0]
    if n == 0:
        return 0
    power = jac
    for nu in range(1, n + 1):
        if power.is_zero_matrix:
            return nu
        power = (power * jac).applyfunc(sp.expand)
    return None


def _support_cycle(jac: Any) -> list[int]:
    """Return a cycle ``[i, ..., i]`` in the support graph of ``jac``.

    A matrix whose support graph is acyclic is nilpotent, so a non-nilpotent
    matrix always has such a cycle.
    """
    n = jac.shape[0]
    succ = [[j for j in range(n) if jac[i, j] != 0] for i in range(n)]
    state = [0] * n  # 0 new, 1 on stack, 2 done
    stack: list[int] = []

    def dfs(i: int) -> list[int] | None:
        state[i] = 1
        stack.append(i)
        for j in succ[i]:
            if state[j] == 1:
                return [*stack[stack.index(j) :], j]
            if state[j] == 0 and (found := dfs(j)) is not None:
                return found
        stack.pop()
        state[i] = 2
        return None

    for start in range(n):
        if state[start] == 0 and (found := dfs(start)) is not None:
            return found
    raise AssertionError("a non-nilpotent matrix has a cycle in its support graph")
