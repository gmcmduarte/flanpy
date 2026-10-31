# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike for build-order step 6 (transfer), dev/design/transfer.md.

Throwaway code (ruff excludes dev/spikes). Run from the repo root:

    .venv/bin/python dev/spikes/transfer_spike.py            # all sections
    .venv/bin/python dev/spikes/transfer_spike.py fixed      # one section

Sections:
  gamma   the polynomials e_l(a) of FS Thm VI.1 from the Bernoulli polynomials (log-Gamma,
          DLMF 5.11.8), against FS eq. (13) and FS Note VI.3, and numerically against
          Gamma ratios; the reduction 1/Gamma(x) = r / Gamma(nu), nu = frac(x).
  fixed   every step 5 example at a fixed k: the exact coefficients D_j of
          [z^n] w = rho^-n sum_j D_j n^(-1 - e0 - j/kappa) (each D_j = algebraic / Gamma(nu)),
          computed in step 5's number field; closed forms (Catalan, Motzkin = FS Example
          VII.18, HG eq. (15), Broda eq. (6), alpha = 1/4, pole, alpha = 3/2); the error at
          n = 1000 of J terms of the n-expansion against series, and of the Gamma-ratio
          form (step 5's test helper) for comparison.
  broda   L_k, F_k for k in {2, 10, 50, 54, 108}: leading constants, the exact limit
          2/(b_k c_k) of [z^n] L_k / (n [z^n] F_k) (Broda Thm 3.2, eq. (38)) and its next
          term (composite number field), Table 1, against series at n = 1000.
  ratio   Broda section 2: [z^n] S_k / (n [z^n] R_k) -> rho_k (eq. (8)), exactly, k = 1..6.
  symk    symbolic k: the leading constants on the pieces of local_expansion_in_k as
          monomials in KRoots, at(n) against the fixed-k results.
"""

from __future__ import annotations

import math
import sys
import time
from fractions import Fraction

import mpmath
import sympy as sp

from flanpy import (
    AlgebraicReal,
    System,
    eliminate,
    eliminate_in_k,
    local_expansion,
    local_expansion_in_k,
    series,
    singularity,
    singularity_in_k,
)
from flanpy._field import _NumberField

mpmath.mp.dps = 50
z, k = sp.symbols("z k")
F, G, R, S, B, Bb, M, Mb, L, A, U, K, Yu, W, T = sp.symbols(
    "F G R S B Bb M Mb L A U K Y W T"
)

CATALAN = System({F: 1 + z * F**2}, z=z)
MOTZKIN = System({F: z + z * F + z * F**2}, z=z)
RK = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
RS = System({R: (k + 1) * z + 2 * z * R**2 + z * R, S: z * R + z * S + 4 * z * S * R}, z=z, k=k)
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
QUARTER = System({U: z + U**2, K: U / 2 + K**2}, z=z)
POLE = System({Yu: 1 + 2 * z * Yu}, z=z)
THREE_HALVES = System({B: z + B**2, Yu: 1 + 2 * B * Yu, W: Yu**3}, z=z)
CONSTRUCTED = System({F: z + z * F**2 + z * G, G: z + z * G**2}, z=z)
S5_TREE = System({T: z * (1 + T**2 + T**5)}, z=z)


# ---------------------------------------------------------------------------
# gamma: e_l(a), FS Thm VI.1
# ---------------------------------------------------------------------------

a_ = sp.Symbol("a")


def e_polys(lmax: int) -> list[sp.Poly]:
    """e_0..e_lmax: Gamma(n + a)/Gamma(n + 1) = n^(a-1) (1 + sum_l e_l(a) n^-l).

    log Gamma(n + h) ~ (n + h - 1/2) log n - n + log(2 pi)/2
                       + sum_{j>=1} (-1)^(j+1) B_{j+1}(h) / (j (j+1) n^j)   (DLMF 5.11.8),
    so log of the ratio is (a - 1) log n + sum_j s_j n^-j with
    s_j = (-1)^(j+1) (B_{j+1}(a) - B_{j+1}(1)) / (j (j+1)); exponentiate by
    l e_l = sum_{j=1}^l j s_j e_{l-j} (E' = S' E).
    """
    s = [sp.Integer(0)] + [
        (-1) ** (j + 1) * (sp.bernoulli(j + 1, a_) - sp.bernoulli(j + 1, 1)) / (j * (j + 1))
        for j in range(1, lmax + 1)
    ]
    e = [sp.Integer(1)]
    for l in range(1, lmax + 1):
        e.append(sp.expand(sum(j * s[j] * e[l - j] for j in range(1, l + 1)) / l))
    return [sp.Poly(x, a_, domain=sp.QQ) for x in e]


def note_vi3(kk: int) -> sp.Expr:
    """FS Note VI.3: e_k = sum_{l=k}^{2k} lambda_{k,l} (a-1)...(a-l).

    As extracted the note has no sign; eq. (13) needs (-1)^l (the Hankel integral of
    t^l (-t)^-a e^-t gives (-1)^l (a-1)...(a-l) / Gamma(a)), so we use it.
    """
    v, t = sp.symbols("v t")
    # lambda_{k,l} = [v^k t^l] e^t (1 + v t)^(-1-1/v)
    expr = sp.exp(t) * sp.exp((-1 - 1 / v) * sp.log(1 + v * t))
    ser = sp.series(expr, t, 0, 2 * kk + 1).removeO()
    total = 0
    for l in range(kk, 2 * kk + 1):
        coeff = sp.series(sp.simplify(ser.coeff(t, l)), v, 0, kk + 1).removeO().coeff(v, kk)
        total += (-1) ** l * coeff * sp.prod([a_ - i for i in range(1, l + 1)])
    return sp.expand(total)


def e_at_poly(p: sp.Poly, x: Fraction) -> Fraction:
    v = p.eval(sp.Rational(x.numerator, x.denominator))
    return Fraction(int(v.p), int(v.q))


def inv_gamma(x: Fraction) -> tuple[Fraction, Fraction | None]:
    """1/Gamma(x) = r / Gamma(nu), nu = x - floor(x) in (0, 1); nu = None if x is an integer."""
    if x.denominator == 1:
        return (Fraction(0) if x <= 0 else Fraction(1, math.factorial(int(x) - 1))), None
    nu = x - math.floor(x)
    r, y = Fraction(1), x
    while y > nu:  # Gamma(y) = (y - 1) Gamma(y - 1)
        y -= 1
        r /= y
    while y < nu:  # 1/Gamma(y) = y / Gamma(y + 1)
        r *= y
        y += 1
    return r, nu


def section_gamma() -> None:
    t0 = time.time()
    e = e_polys(8)
    print(f"e_0..e_8 in {time.time() - t0:.2f} s")
    fs13 = [
        1,
        a_ * (a_ - 1) / 2,
        a_ * (a_ - 1) * (a_ - 2) * (3 * a_ - 1) / 24,
        a_**2 * (a_ - 1) ** 2 * (a_ - 2) * (a_ - 3) / 48,
    ]
    for l in range(4):
        assert sp.expand(e[l].as_expr() - fs13[l]) == 0, l
    print("e_1..e_3 = FS eq. (13): ok")
    for kk in (1, 2, 3, 4):
        assert sp.expand(note_vi3(kk) - e[kk].as_expr()) == 0, kk
    print("e_1..e_4 = FS Note VI.3 (lambda_{k,l}): ok")
    for l in range(1, 9):
        p = e[l]
        assert p.degree() == 2 * l
        assert all(p.eval(i) == 0 for i in range(0, l + 1)), l  # divisible by a(a-1)...(a-l)
    print("deg e_l = 2l and a(a-1)...(a-l) | e_l (FS footnote 2), l = 1..8: ok")
    print("e_4 =", sp.factor(e[4].as_expr()))
    # numeric: Gamma(n+a)/(Gamma(a) Gamma(n+1)) vs n^(a-1)/Gamma(a) (1 + sum e_l n^-l)
    for av in (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 4), Fraction(3, 2), Fraction(-5, 4)):
        n = 1000
        am = mpmath.mpf(av.numerator) / av.denominator
        exact = mpmath.gammaprod([n + am], [n + 1])
        def mq(q: Fraction) -> mpmath.mpf:
            return mpmath.mpf(q.numerator) / q.denominator

        approx = mpmath.mpf(n) ** (am - 1) * sum(
            mq(e_at_poly(e[l], av)) / mpmath.mpf(n) ** l for l in range(9)
        )
        print(f"  a = {str(av):>5}: relative error with e_0..e_8 at n = 1000: "
              f"{mpmath.nstr(abs(approx / exact - 1), 3)}")
    for x in (Fraction(-1, 2), Fraction(1, 2), Fraction(3, 2), Fraction(-1, 4), Fraction(-3, 2),
              Fraction(-5, 4), Fraction(7, 4)):
        r, nu = inv_gamma(x)
        lhs = 1 / mpmath.gamma(mpmath.mpf(x.numerator) / x.denominator)
        rhs = mpmath.mpf(r.numerator) / r.denominator / mpmath.gamma(mpmath.mpf(nu.numerator) / nu.denominator)
        assert abs(lhs - rhs) < mpmath.mpf(10) ** -45
        print(f"  1/Gamma({x}) = ({r}) / Gamma({nu})")


# ---------------------------------------------------------------------------
# fixed: the exact n-expansion from step 5's terms
# ---------------------------------------------------------------------------

E = e_polys(12)


def e_at(l: int, x: Fraction) -> Fraction:
    v = E[l].eval(sp.Rational(x.numerator, x.denominator))
    return Fraction(int(v.p), int(v.q))


def puiseux(e, count: int):
    """(exponent, field element) for the first `count` terms of step 5 (same field)."""
    e.terms(count)  # builds the state
    ds = e._state.terms(count)
    start = Fraction(0) if e.singularity.case == "I" else -e.alpha
    return [(start + Fraction(i, e.ramification), d) for i, d in enumerate(ds)]


def first_singular(e) -> Fraction:
    return e.alpha if e.singularity.case == "I" else -e.alpha


def n_expansion(e, J: int):
    """[(power of n, D_j as a field element, nu or None)], j < J.

    [z^n] s^x = rho^-n n^(-x-1) / Gamma(-x) (1 + sum_l e_l(-x) n^-l) (FS Thm VI.1), so
    n^(-1-p) collects the terms x = p - l, l >= 0 an integer; 1/Gamma(-x) = r / Gamma(nu)
    with one nu for the whole class of p mod 1.
    """
    kappa, e0 = e.ramification, first_singular(e)
    start = Fraction(0) if e.singularity.case == "I" else -e.alpha
    count = int((e0 + Fraction(J - 1, kappa) - start) * kappa) + 1
    terms = puiseux(e, count)
    field = terms[0][1].field
    out = []
    for j in range(J):
        p = e0 + Fraction(j, kappa)
        total, base = field(0), None
        for x, d in terms:
            l = p - x
            if l < 0 or l.denominator != 1:
                continue
            r, nu = inv_gamma(-x)
            base = nu
            if r == 0 or d.is_zero():
                continue
            total = total + d * (r * e_at(int(l), -x))
        out.append((-1 - p, total, base))
    return out


def mp_alg(x: AlgebraicReal, dps: int = 60) -> mpmath.mpf:
    with mpmath.workdps(dps):
        return mpmath.mpf(x.evalf(dps))


def evaluate(e, n: int, J: int) -> mpmath.mpf:
    with mpmath.workdps(60):
        rho = mp_alg(e.singularity.rho, 60 + len(str(n)))
        total = mpmath.mpf(0)
        for power, d, nu in n_expansion(e, J):
            if d.is_zero():
                continue
            g = 1 if nu is None else mpmath.gamma(mpmath.mpf(nu.numerator) / nu.denominator)
            total += mp_alg(d.to_real()) / g * mpmath.mpf(n) ** (mpmath.mpf(power.numerator) / power.denominator)
        return total * rho ** (-n)


def gamma_ratio(e, n: int, count: int) -> mpmath.mpf:
    """step 5's test helper: sum of d [z^n] s^x exactly (Gamma ratios), x not in Z>=0."""
    with mpmath.workdps(60):
        rho = mp_alg(e.singularity.rho, 60 + len(str(n)))
        total = mpmath.mpf(0)
        for x, d in e.terms(count):
            if x.denominator == 1 and x >= 0:
                continue
            xm = mpmath.mpf(x.numerator) / x.denominator
            total += mp_alg(d) * mpmath.gammaprod([n - xm], [-xm, n + 1])
        return total * rho ** (-n)


def coefficient(system, w, n, kval=None) -> mpmath.mpf:
    c = (series(system, n, k=kval) if kval is not None else series(system, n))[w][n]
    c = Fraction(c)
    with mpmath.workdps(60):
        return mpmath.mpf(c.numerator) / c.denominator


def expand(system, w, kval=None):
    curve = eliminate(system, w, k=kval) if kval is not None else eliminate(system, w)
    return local_expansion(singularity(curve))


def show(D, J=None):
    parts = []
    for power, d, nu in D[:J]:
        g = "" if nu is None else f" / Γ({nu})"
        parts.append(f"n^{power}: {d.to_real()}{g}")
    return "; ".join(parts)


def rational(d) -> Fraction | None:
    x = d.to_real()
    lo, hi = x.interval
    return lo if lo == hi else None


def section_fixed() -> None:
    examples = [
        ("catalan", CATALAN, F, None),
        ("motzkin", MOTZKIN, F, None),
        ("r_2", RK, R, 2),
        ("a_2", HG, A, 2),
        ("quarter", QUARTER, K, None),
        ("pole", POLE, Yu, None),
        ("three_halves", THREE_HALVES, W, None),
        ("constructed", CONSTRUCTED, F, None),
        ("s5_tree", S5_TREE, T, None),
        ("f_2", FK, F, 2),
        ("l_2", SSNF, L, 2),
    ]
    n = 1000
    for name, system, w, kval in examples:
        t0 = time.time()
        e = expand(system, w, kval)
        t1 = time.time()
        D = n_expansion(e, 6)
        t2 = time.time()
        print(f"{name}: case {e.singularity.case}, alpha {e.alpha}, kappa {e.ramification}, "
              f"rho {e.singularity.rho} (step 5 {t1 - t0:.2f} s, D_0..D_5 {t2 - t1:.3f} s)")
        print("   ", show(D, 4))
        if name == "catalan":
            rs = [rational(d) for _, d, _ in D]
            assert rs[:5] == [1, 0, Fraction(-9, 8), 0, Fraction(145, 128)], rs
            print("    = 4^n n^-3/2 / sqrt(pi) (1 - 9/(8n) + 145/(128 n^2) ...): ok")
        if name == "motzkin":
            lead = D[0][1]
            ratios = [rational(D[j][1] / lead) for j in (2, 4, 6 - 0) if j < len(D)]
            more = n_expansion(e, 7)
            ratios = [rational(more[j][1] / lead) for j in (2, 4, 6)]
            assert ratios == [Fraction(-15, 16), Fraction(505, 512), Fraction(-8085, 8192)], ratios
            assert lead.to_real() == AlgebraicReal(sp.Poly(4 * sp.Symbol("y") ** 2 - 3), 1)
            print("    = 3^(n+1/2)/(2 sqrt(pi) n^3/2) (1 - 15/(16n) + 505/(512n^2) - 8085/(8192n^3)) "
                  "(FS Example VII.18): ok")
        if name == "quarter":
            assert rational(D[0][1]) == Fraction(1, 8) and D[0][2] == Fraction(3, 4)
            print("    D_0 = 1/(8 Gamma(3/4)) (K = (1 - s^(1/4))/2): ok")
        if name == "pole":
            assert rational(D[0][1]) == 1 and all(d.is_zero() for _, d, _ in D[1:])
            print("    [z^n] = 2^n exactly (terminating): ok")
        if name == "three_halves":
            assert rational(D[0][1]) == 2 and D[0][2] == Fraction(1, 2)
            assert rational(D[2][1]) == Fraction(3, 4)
            print("    D_0 = 2/sqrt(pi), D_2 = (3/8) D_0 (e_1(3/2) = 3/8): ok")
        if kval is not None and name in ("r_2", "a_2"):
            rho = mp_alg(e.singularity.rho)
            d0 = mp_alg(D[0][1].to_real()) / mpmath.sqrt(mpmath.pi)
            if name == "r_2":
                closed = mpmath.sqrt(2 - 2 * rho) / (8 * rho * mpmath.sqrt(mpmath.pi))  # Broda eq. (6)
            else:
                # HG eq. (16): [z^n] A_k ~ (2k)^(1/4) sqrt(rho)/(4 sqrt(pi)) rho^-(n+1) n^-3/2
                closed = (2 * kval) ** mpmath.mpf(0.25) * mpmath.sqrt(rho) / (4 * mpmath.sqrt(mpmath.pi) * rho)
            assert abs(d0 / closed - 1) < mpmath.mpf(10) ** -40
            print(f"    D_0 = {'Broda eq. (6)' if name == 'r_2' else 'HG eq. (16)'}: ok")
        # errors at n = 1000
        exact = coefficient(system, w, n, kval)
        errs = []
        for J in range(1, 6):
            errs.append(mpmath.nstr(abs(evaluate(e, n, J) / exact - 1), 2))
        gr = []
        for count in range(2, 7):
            gr.append(mpmath.nstr(abs(gamma_ratio(e, n, count) / exact - 1), 2))
        print(f"    n = 1000, J = 1..5 terms of the n-expansion: {errs}")
        print(f"    n = 1000, Gamma-ratio form, terms(2..6):      {gr}")
        print(f"    [{time.time() - t0:.1f} s]")


# ---------------------------------------------------------------------------
# broda: L_k, F_k, Thm 3.2, eq. (38), Table 1
# ---------------------------------------------------------------------------


def composite(x, y):
    """Embed elements x in K1 and y in K2 in one field L (real roots of m_theta2 over K1)."""
    K1, K2 = x.field, y.field
    if K2.degree == 1:
        return x, K1(Fraction(y.coefficients[0]))
    if K1.degree == 1:
        return K2(Fraction(x.coefficients[0])), y
    roots = K1.real_roots(list(K2.modulus))
    for root in roots:
        if root.value.to_real() == K2.theta:
            Lf, emb, th = root.field, root.embedding, root.value
            yy = Lf(0)
            for i, c in enumerate(y.coefficients):
                yy = yy + th**i * c
            return emb(x), yy
    raise RuntimeError("theta_2 not found")


def series_divide(num, den, J):
    """Coefficients of (sum num_j u^j) / (sum den_j u^j), u = n^(-1/kappa)."""
    out = []
    for j in range(J):
        total = num[j]
        for i in range(1, j + 1):
            total = total - den[i] * out[j - i]
        out.append(total / den[0])
    return out


def ratio(eA, eB, J):
    """[z^n] A / [z^n] B = n^shift (q_0 + q_1 n^-1/kappa + ...), same rho required."""
    assert eA.singularity.rho == eB.singularity.rho, "rho differs: exponential factor"
    assert eA.ramification == eB.ramification
    DA, DB = n_expansion(eA, J), n_expansion(eB, J)
    shift = DA[0][0] - DB[0][0]
    # Gamma bases: equal in our examples (nu = 1/2 both); a general version keeps a monomial
    assert DA[0][2] == DB[0][2]
    pairs = [composite(a, b) for (_, a, _), (_, b, _) in zip(DA, DB)]
    # all pairs live in the same L only if the composite is built once; rebuild per pair is
    # fine here because every pair uses the same K1, K2 and so the same L (same roots)
    num = [p[0] for p in pairs]
    den = [p[1] for p in pairs]
    L0 = num[0].field
    num = [x if x.field == L0 else L0(0) for x in num]
    den = [x if x.field == L0 else L0(0) for x in den]
    return shift, series_divide(num, den, J)


def section_broda() -> None:
    for kval in (2, 10, 50, 54, 108):
        t0 = time.time()
        eL, eF = expand(SSNF, L, kval), expand(FK, F, kval)
        t1 = time.time()
        DL, DF = n_expansion(eL, 3), n_expansion(eF, 3)
        print(f"k = {kval}: L_k lead {DL[0][1].to_real()} / Γ({DL[0][2]}) n^{DL[0][0]}, "
              f"F_k lead {DF[0][1].to_real()} / Γ({DF[0][2]}) n^{DF[0][0]}  (step 5 {t1 - t0:.2f} s)")
        shift, q = ratio(eL, eF, 3)
        q0 = q[0].to_real()
        bc = 2 / (mp_alg(eF.b) * mp_alg(eL.c))
        assert abs(mp_alg(q0) - bc) < mpmath.mpf(10) ** -45 and shift == 1
        approx = 0.5 - 1 / (4 * math.sqrt(2 * kval))
        print(f"    [z^n]L/(n [z^n]F) -> 2/(b c) = {mpmath.nstr(mp_alg(q0), 6)} (deg {q0.polynomial.degree()}: "
              f"{q0.polynomial.as_expr()}), eq. (38) approx {approx:.4f}; "
              f"next: q_1 = {mpmath.nstr(mp_alg(q[1].to_real()), 6) if not q[1].is_zero() else 0}, "
              f"q_2 = {mpmath.nstr(mp_alg(q[2].to_real()), 6)} (n^-1)  [{time.time() - t1:.2f} s]")
        if kval in (2, 10, 50):
            n = 1000
            t2 = time.time()
            cL = coefficient(SSNF, L, n, kval)
            cF = coefficient(FK, F, n, kval)
            actual = cL / (n * cF)
            lim = mp_alg(q0)
            two = lim + mp_alg(q[2].to_real()) / n
            print(f"    n = 1000: actual {mpmath.nstr(actual, 8)}, limit error {mpmath.nstr(abs(lim / actual - 1), 2)}, "
                  f"with q_2/n {mpmath.nstr(abs(two / actual - 1), 2)}  (series {time.time() - t2:.1f} s)")
        if kval == 54:
            assert q0 == AlgebraicReal(sp.Poly(15 * sp.Symbol("y") - 7), 0)
            print("    k = 54: limit exactly 7/15: ok")


def section_ratio() -> None:
    for kval in range(1, 7):
        eR, eS = expand(RS, R, kval), expand(RS, S, kval)
        shift, q = ratio(eS, eR, 3)
        assert shift == 1
        assert q[0].to_real() == eR.singularity.rho, (q[0].to_real(), eR.singularity.rho)
        print(f"k = {kval}: [z^n] S_k / (n [z^n] R_k) -> {q[0].to_real()} = rho_k exactly "
              f"(eq. (8)); next term q_2 = {q[2].to_real()} (n^-1)")


# ---------------------------------------------------------------------------
# symk: leading constants on the pieces, as monomials in KRoots
# ---------------------------------------------------------------------------


def section_symk() -> None:
    t0 = time.time()
    cL = local_expansion_in_k(singularity_in_k(eliminate_in_k(SSNF, L)))
    cF = local_expansion_in_k(singularity_in_k(eliminate_in_k(FK, F)))
    print(f"local_expansion_in_k L_k, F_k: {time.time() - t0:.1f} s")
    for (pL, eL), (pF, eF) in zip(cL.pieces, cF.pieces):
        print(f"  L_k on {pL}: case {eL.singularity.case}, alpha {eL.alpha}: "
              f"leading = (1/c_k) / Γ(alpha) n^(alpha-1), c_k = {eL.c}")
        print(f"  F_k on {pF}: case {eF.singularity.case}, alpha {eF.alpha}: "
              f"leading = alpha b_k / Γ(1-alpha) n^(-alpha-1), b_k = {eF.b}")
        print(f"  same rho_k: {eL.singularity.rho == eF.singularity.rho}")
    _, eL = cL.pieces[0]
    _, eF = cF.pieces[0]
    for n in (1, 2, 3, 10, 50, 54, 55, 108):
        fL, fF = expand(SSNF, L, n), expand(FK, F, n)
        assert eL.c.at(n) == fL.c and eF.b.at(n) == fF.b
        shift, q = ratio(fL, fF, 1)
        x = 2 / (mp_alg(eF.b.at(n)) * mp_alg(eL.c.at(n)))
        assert abs(mp_alg(q[0].to_real()) - x) < mpmath.mpf(10) ** -45
    print("  b_k.at(n), c_k.at(n) = fixed k, and 2 b_k^-1 c_k^-1 at n = the fixed-k limit, "
          "n in {1, 2, 3, 10, 50, 54, 55, 108}: ok")
    print(f"  [{time.time() - t0:.1f} s]")


SECTIONS = {
    "gamma": section_gamma,
    "fixed": section_fixed,
    "broda": section_broda,
    "ratio": section_ratio,
    "symk": section_symk,
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SECTIONS)
    for nm in names:
        t = time.time()
        SECTIONS[nm]()
        print(f"[{nm}: {time.time() - t:.1f} s]\n")
