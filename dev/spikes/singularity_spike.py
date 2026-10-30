# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike for build-order step 4 (singularity), dev/design/singularity.md.

Throwaway code (ruff excludes dev/spikes). Run from the repo root:

    .venv/bin/python dev/spikes/singularity_spike.py            # all sections
    .venv/bin/python dev/spikes/singularity_spike.py fixed      # one section

Sections:
  fixed   fixed-k singularity of every example: candidates, rho by certified
          bracketing (PS 2025 Thm 4.1, Prop 3.11, Lemma 3.5), Case I/II and a by the
          window test, period and the circle check; checked against mpmath roots.
  lk      Broda L_k for k = 1..60 (fixed-k algorithm), against mpmath roots of p_k.
  ratio   [z^n] ratios at n = 1000 against rho_k (L_k, k in {2, 10, 50, 54}).
  symk    symbolic k: projection polynomials, pieces, samples, merge (R_k, A_k, F_k, L_k).
  signed  F = 1 - z F^2: no positive candidate, radius 1/4 (dominant singularity -1/4).
"""

from __future__ import annotations

import math
import sys
import time
from fractions import Fraction
from functools import reduce

import mpmath
import sympy as sp

from flanpy import System, eliminate, series

z, k = sp.symbols("z k")
QQ = sp.QQ

# ---------------------------------------------------------------------------
# Exact real algebraic numbers: (irreducible f in Q[x], isolating interval).
# ---------------------------------------------------------------------------


class Alg:
    """The unique root of the irreducible ``f`` in ``[lo, hi]`` (rationals)."""

    def __init__(self, f: sp.Poly, lo: Fraction, hi: Fraction) -> None:
        self.f, self.lo, self.hi = f, Fraction(lo), Fraction(hi)

    def refine(self, eps: Fraction) -> None:
        if self.hi - self.lo <= eps:
            return
        a, b = self.f.refine_root(self.lo, self.hi, eps=eps)
        self.lo, self.hi = Fraction(str(a)), Fraction(str(b))

    def sign_of(self, g: sp.Poly) -> int:
        """Sign of g at the number: 0 iff f | g (f irreducible), else refine."""
        g = sp.Poly(g.as_expr(), self.f.gen, domain=QQ)
        if g.rem(self.f).is_zero:
            return 0
        while g.count_roots(self.lo, self.hi) > 0:
            self.refine((self.hi - self.lo) / 4)
        v = g.eval(sp.Rational(self.lo.numerator, self.lo.denominator))
        return 1 if v > 0 else -1

    def mpf(self, dps: int = 30) -> mpmath.mpf:
        self.refine(Fraction(1, 10 ** (dps + 5)))
        return mpmath.mpf(self.lo.numerator) / self.lo.denominator

    def __repr__(self) -> str:
        return f"root of {self.f.as_expr()} in [{float(self.lo):.6g}, {float(self.hi):.6g}]"


def real_roots(f: sp.Poly) -> list[Alg]:
    """All real roots of f (any f != 0), increasing, each with its irreducible factor."""
    out: list[Alg] = []
    for fac, _ in sp.factor_list(f.as_expr(), f.gen)[1]:
        p = sp.Poly(fac, f.gen, domain=QQ)
        if p.degree() <= 0:
            continue
        for (a, b), _ in p.intervals():
            out.append(Alg(p, Fraction(str(a)), Fraction(str(b))))
    # sort by refining overlapping intervals (distinct roots, so this terminates)
    changed = True
    while changed:
        changed = False
        out.sort(key=lambda r: r.lo)
        for r, s in zip(out, out[1:], strict=False):
            if r.hi >= s.lo:
                r.refine((r.hi - r.lo) / 4)
                s.refine((s.hi - s.lo) / 4)
                changed = True
    return out


def positive(rs: list[Alg]) -> list[Alg]:
    out = []
    for r in rs:
        if r.f.degree() == 1 and r.f.eval(0) == 0:
            continue  # the root 0
        while r.lo <= 0 <= r.hi:
            r.refine((r.hi - r.lo) / 4)
        if r.lo > 0:
            out.append(r)
    return out


def separators(rs: list[Alg]) -> list[Fraction]:
    """Rationals q_0 < r_1 < q_1 < ... < r_s < q_s, none a root."""
    if not rs:
        return [Fraction(0)]
    qs = [rs[0].lo - 1]
    for r, s in zip(rs, rs[1:], strict=False):
        qs.append((r.hi + s.lo) / 2)
    qs.append(rs[-1].hi + 1)
    return qs


# ---------------------------------------------------------------------------
# Certified comparison q vs rho for a non-negative polynomial system (fixed k).
# ---------------------------------------------------------------------------


class PosSystem:
    """Y = H(z, Y), polynomial with non-negative integer coefficients (k fixed)."""

    def __init__(self, system: System, kval: int | None) -> None:
        s = system.specialize(kval) if system.k is not None else system
        self.Y = list(s.unknowns)
        self.H = [sp.Poly(s.equations[y], z, *self.Y, domain=QQ) for y in self.Y]
        assert all(c >= 0 for h in self.H for c in h.coeffs()), "signed system"
        self.J = [[h.diff(y) for y in self.Y] for h in self.H]
        self.m = len(self.Y)

    def h(self, x: Fraction, y: list[Fraction]) -> list[Fraction]:
        pt = [x, *y]
        return [Fraction(str(p.eval(dict(zip(p.gens, pt, strict=True))))) for p in self.H]

    def jac(self, x: Fraction, y: list[Fraction]) -> list[list[Fraction]]:
        pt = [x, *y]
        return [
            [Fraction(str(p.eval(dict(zip(p.gens, pt, strict=True))))) for p in row]
            for row in self.J
        ]


def lam_ge_1(J: list[list[Fraction]]) -> bool:
    """Spectral radius of the non-negative J is >= 1 (exact: charpoly root in [1, oo))."""
    M = sp.Matrix([[sp.Rational(c.numerator, c.denominator) for c in r] for r in J])
    chi = sp.Poly(M.charpoly(sp.Symbol("lam")).as_expr(), sp.Symbol("lam"), domain=QQ)
    return chi.count_roots(1, None) > 0


def solve(J: list[list[Fraction]], r: list[Fraction]) -> list[Fraction]:
    """Solve (I - J) d = r exactly."""
    m = len(r)
    A = [[(Fraction(i == j) - J[i][j]) for j in range(m)] + [r[i]] for i in range(m)]
    for c in range(m):
        p = next(i for i in range(c, m) if A[i][c] != 0)
        A[c], A[p] = A[p], A[c]
        for i in range(m):
            if i != c and A[i][c] != 0:
                f = A[i][c] / A[c][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c], strict=True)]
    return [A[i][m] / A[i][i] for i in range(m)]


def down(x: Fraction, bits: int) -> Fraction:
    return Fraction(math.floor(x * 2**bits), 2**bits)


def up(x: Fraction, bits: int) -> Fraction:
    return Fraction(math.ceil(x * 2**bits), 2**bits)


def lower_bounds(S: PosSystem, x: Fraction, bits: int, steps: int):
    """Rounded-down Newton from 0. Yields l with l <= Y(x) whenever x <= rho.

    Returns ("above", l) if Lambda(J(x, l)) >= 1 (then x > rho, see note), else
    ("iter", l) after `steps` steps. Soundness (x <= rho => l <= Y(x)): induction,
    Newton step N(l) = l + (I-J)^-1 (H(l) - l) and
    Y - N(l) = (I-J)^-1 [H(Y) - H(l) - J(l)(Y - l)] >= 0 by convexity (PS Lemma 3.5)
    and (I-J)^-1 >= 0 when Lambda < 1.
    """
    y = [Fraction(0)] * S.m
    for _ in range(steps):
        J = S.jac(x, y)
        if lam_ge_1(J):
            return "above", y
        d = solve(J, [a - b for a, b in zip(S.h(x, y), y, strict=True)])
        y = [max(down(a + b, bits), a) for a, b in zip(y, d, strict=True)]
    return "iter", y


def upper_bound(S: PosSystem, x: Fraction, approx: list[Fraction], bits: int):
    """A rational u >= 0 with H(x, u) <= u (exact check), near approx; or None.

    H(x, u) <= u implies the Kleene iterates from 0 stay <= u, hence Y(x) <= u and
    x <= rho (monotone convergence; cf. PS Prop 3.11)."""
    J = S.jac(x, approx)
    v = solve(J, [Fraction(1)] * S.m)  # (I - J)^-1 1 > 0 when Lambda < 1
    for e in range(bits // 2, 2, -2):
        delta = Fraction(1, 2**e)
        u = [up(a + delta * b, bits) for a, b in zip(approx, v, strict=True)]
        if all(b >= 0 for b in u) and all(
            hb <= ub for hb, ub in zip(S.h(x, u), u, strict=True)
        ):
            return u
    return None


def compare(S: PosSystem, x: Fraction) -> tuple[str, list, list | None]:
    """'below' (x < rho, with l <= Y(x) <= u) or 'above' (x > rho). x must not be rho."""
    bits, steps = 64, 12
    while True:
        how, l = lower_bounds(S, x, bits, steps)
        if how == "above":
            return "above", l, None
        u = upper_bound(S, x, l, bits)
        if u is not None:
            return "below", l, u
        bits, steps = 2 * bits, steps + 8
        if bits > 4096:
            raise RuntimeError("no certificate")


# ---------------------------------------------------------------------------
# Fixed-k singularity of the branch on a curve.
# ---------------------------------------------------------------------------


def candidates(G: sp.Poly, w: sp.Symbol) -> sp.Poly:
    """R = squarefree part of z-free-content-free lc_w(G) * disc_w(G)."""
    g = G.as_expr()
    lc = sp.Poly(g, w).LC()
    disc = sp.discriminant(g, w) if sp.Poly(g, w).degree() > 1 else sp.Integer(1)
    R = sp.Poly(sp.expand(lc * disc), z, domain=QQ)
    return sp.Poly(sp.sqf_part(R.as_expr()), z, domain=QQ)


def singular_fixed(system: System, w: sp.Symbol, kval: int | None, verbose=True):
    t0 = time.perf_counter()
    curve = eliminate(system, w, k=kval)
    G = sp.Poly(curve.polynomial.as_expr(), w, z, domain=QQ)
    S = PosSystem(system, kval)
    R = candidates(G, w)
    pos = positive(real_roots(R))
    qs = [Fraction(0)] + [(r.hi + s.lo) / 2 for r, s in zip(pos, pos[1:], strict=False)]
    # binary search: rho is the candidate i with qs[i] < rho < qs[i+1] (qs[r] = oo)
    lo, hi = 0, len(pos) - 1
    decisions = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        verdict, *_ = compare(S, qs[mid])
        decisions += 1
        if verdict == "below":
            lo = mid
        else:
            hi = mid - 1
    rho = pos[lo]
    prev = qs[lo]
    # Case and a: windows around the real roots of N(w) = Res_z(m_rho, G).
    N = sp.Poly(sp.resultant(rho.f.as_expr(), G.as_expr(), z), w, domain=QQ)
    nroots = real_roots(N)
    seps = separators(nroots)
    # x0 in (prev, rho) with no root of G(., q_j) in [x0, rho)
    bound = prev
    for q in seps:
        gq = sp.Poly(G.as_expr().subs(w, sp.Rational(q.numerator, q.denominator)), z)
        if gq.degree() <= 0:
            continue
        for r in real_roots(gq):
            while r.lo <= rho.hi and r.hi >= rho.lo:  # separate (closed intervals)
                r.refine((r.hi - r.lo) / 4)
                rho.refine((rho.hi - rho.lo) / 4)
            if r.hi <= rho.lo:
                bound = max(bound, r.hi)
    x0 = (bound + rho.lo) / 2 if bound < rho.lo else None
    assert x0 is not None and x0 < rho.lo
    verdict, l, u = compare(S, x0)
    assert verdict == "below"
    iw = S.Y.index(w)
    lw, uw = l[iw], u[iw]
    g0 = sp.Poly(G.as_expr().subs(z, sp.Rational(x0.numerator, x0.denominator)), w)
    while g0.count_roots(lw, uw) != 1:  # tighten: rerun with more bits (rare)
        raise RuntimeError("enclosure not tight enough")
    # which window holds the unique root in [lw, uw]?
    win = None
    for j in range(len(seps) + 1):
        a = seps[j - 1] if j > 0 else None
        b = seps[j] if j < len(seps) else None
        lo_ = lw if a is None else max(lw, a)
        hi_ = uw if b is None else min(uw, b)
        if lo_ <= hi_ and g0.count_roots(lo_, hi_) == 1:
            win = j
    assert win is not None
    if win == len(seps):
        case, a_val = "II", None
    else:
        assert 0 < win, "branch tends to -oo"
        case, a_val = "I", nroots[win - 1]
    dt = time.perf_counter() - t0
    out = dict(curve=curve, rho=rho, case=case, a=a_val, candidates=pos, decisions=decisions,
               x0=x0, time=dt, R=R)
    if verbose:
        print(f"  candidates on (0,oo): {[float(c.mpf(12)) for c in pos]}")
        print(f"  rho = {rho.f.as_expr()} ~ {float(rho.mpf(20)):.15g}  "
              f"(candidate #{lo + 1}, {decisions} certified comparisons)")
        print(f"  Case {case}" + (f", a = {a_val.f.as_expr()} ~ {float(a_val.mpf(20)):.15g}"
                                   if a_val else "") + f"   [{dt:.2f} s]")
    return out


# ---------------------------------------------------------------------------
# Period (support gcd fixed point, PS 2025 Sec 7.1 rules) and the circle check.
# ---------------------------------------------------------------------------


def valuations(system: System, kval):
    s = system.specialize(kval) if system.k is not None else system
    ser = series(s, 40)
    return s, {y: next(i for i, c in enumerate(ser[y]) if c != 0) for y in s.unknowns}


def periods(system: System, kval) -> dict:
    s, val = valuations(system, kval)
    per = {y: 0 for y in s.unknowns}
    while True:
        new = {}
        for y in s.unknowns:
            g = 0
            for mon, c in sp.Poly(s.equations[y], z, *s.unknowns).terms():
                if c == 0:
                    continue
                v = mon[0] + sum(e * val[u] for e, u in zip(mon[1:], s.unknowns, strict=True))
                pm = reduce(math.gcd, [per[u] for e, u in zip(mon[1:], s.unknowns, strict=True)
                                       if e], 0)
                g = math.gcd(g, math.gcd(pm, v - val[y]))
            new[y] = g
        if new == per:
            return per
        per = new


def on_circle(R: sp.Poly, rho: Alg) -> list[complex]:
    """Roots sigma != rho of R with |sigma| = rho, exactly (gcd test) + numerics."""
    y = sp.Symbol("y")
    m_rho2 = sp.Poly(sp.resultant(rho.f.as_expr(), y - z**2, z), y, domain=QQ)
    hits = []
    for fac, _ in sp.factor_list(R.as_expr(), z)[1]:
        f = sp.Poly(fac, z, domain=QQ)
        d = f.degree()
        if d <= 0:
            continue
        P = sp.Poly(sp.resultant(f.as_expr(), sp.expand(z**d * f.as_expr().subs(z, y / z)), z),
                    y, domain=QQ)  # roots sigma_i sigma_j, all ordered pairs
        Gr = sp.Poly(sp.resultant(f.as_expr(), y - z**2, z), y, domain=QQ)  # sigma_i^2
        Q2 = P.quo(Gr)  # pairs i != j (each twice)
        # f != m_rho: any pair, including sigma^2 = rho^2 (sigma = -rho real);
        # f == m_rho: pairs i != j, or -rho a root of f.
        same = f.monic() == rho.f.monic()
        test = Q2 if same else P
        minus = same and rho.sign_of(sp.Poly(f.as_expr().subs(z, -z), z)) == 0
        if minus or sp.gcd(test, m_rho2).degree() > 0:
            r0 = complex(rho.mpf(30))
            for s in sp.Poly(f).nroots(n=30):
                s = complex(s)
                if abs(abs(s) - abs(r0)) < 1e-20 and abs(s - r0) > 1e-20:
                    hits.append(s)
    return hits


# ---------------------------------------------------------------------------
# Examples.
# ---------------------------------------------------------------------------

B, Bb, M, Mb, L, F, Gs, R, A, T = sp.symbols("B Bb M Mb L F G R A T")
ssnf = System(
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
fk = System(
    {
        B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        F: z + B + Bb,
    },
    z=z,
    k=k,
)
catalan = System({F: 1 + z * F**2}, z=z)
rk = System({R: (k + 1) * z + 2 * z * R**2 + z * R}, z=z, k=k)
hg = System({B: k * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}, z=z, k=k)
constructed = System({F: z + z * F**2 + z * Gs, Gs: z + z * Gs**2}, z=z)
periodic = System({T: z + z * T**2}, z=z)

EXAMPLES = [
    ("Catalan F = 1 + zF^2", catalan, F, None),
    ("Broda R_k, k = 2", rk, R, 2),
    ("HG A_k, k = 2", hg, A, 2),
    ("constructed F = z + zF^2 + zG, G = z + zG^2", constructed, F, None),
    ("periodic T = z + zT^2", periodic, T, None),
    ("Broda F_k = z + B + Bb, k = 2", fk, F, 2),
    ("Broda L_k, k = 2", ssnf, L, 2),
    ("Broda L_k, k = 54", ssnf, L, 54),
]


def section_fixed() -> None:
    print("== fixed k ==")
    for name, system, w, kval in EXAMPLES:
        print(name)
        out = singular_fixed(system, w, kval)
        per = periods(system, kval)[w]
        hits = on_circle(out["R"], out["rho"])
        print(f"  period {per}; other candidates on |z| = rho: {hits}")


def section_lk() -> None:
    mpmath.mp.dps = 40
    print("== L_k, k = 1..60, against mpmath roots of p_k ==")
    p = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
    worst, t0 = 0.0, time.perf_counter()
    for n in range(1, 61):
        out = singular_fixed(ssnf, L, n, verbose=False)
        assert out["case"] == "II"
        assert sp.Poly(p.subs(k, n), z).rem(out["rho"].f).is_zero
        roots = [r for r in mpmath.polyroots([int(c) for c in sp.Poly(p.subs(k, n), z).all_coeffs()],
                                             maxsteps=200, extraprec=200)
                 if abs(mpmath.im(r)) < mpmath.mpf(10) ** -20 and mpmath.re(r) > 0]
        assert len(roots) == 1
        err = abs(out["rho"].mpf(40) - mpmath.re(roots[0]))  # mp.dps = 40 below
        worst = max(worst, float(err))
        hits = on_circle(out["R"], out["rho"])
        assert not hits, (n, hits)
    print(f"  all 60: Case II, rho = unique positive root of p_k, unique on the circle; "
          f"max |rho - mpmath| = {worst:.2e}  [{time.perf_counter() - t0:.1f} s]")


def section_ratio() -> None:
    print("== [z^n] L_k ratios at n = 1000 ==")
    n = 1000
    for kval in (2, 10, 50, 54):
        t0 = time.perf_counter()
        c = series(ssnf, n + 1, k=kval)[L]
        rho = singular_fixed(ssnf, L, kval, verbose=False)["rho"].mpf(30)
        ratio = mpmath.mpf(c[n].numerator) / c[n].denominator / (
            mpmath.mpf(c[n + 1].numerator) / c[n + 1].denominator)
        # [z^n] ~ C rho^-n n^(alpha - 1), alpha = 1/2 (eq. (32)): correct the ratio
        corr = ratio * (mpmath.mpf(n + 1) / n) ** (-0.5)
        print(f"  k = {kval}: rho = {mpmath.nstr(rho, 12)}, plain ratio rel. err "
              f"{mpmath.nstr(abs(ratio / rho - 1), 3)}, corrected {mpmath.nstr(abs(corr / rho - 1), 3)}"
              f"  [{time.perf_counter() - t0:.1f} s]")


def kfactors(expr) -> list[sp.Poly]:
    """Distinct irreducible factors in Q[k, z] of positive z-degree."""
    out = []
    for fac, _ in sp.factor_list(sp.expand(expr), z, k)[1]:
        p = sp.Poly(fac, z, k, domain=QQ)
        if p.degree(z) > 0:
            out.append(p)
    return out


def projection(G: sp.Poly, w) -> tuple[dict[str, sp.Expr], list[sp.Poly]]:
    """The polynomials in k whose real roots cut [1, oo) into pieces (design Sec 5)."""
    g = G.as_expr()
    d = sp.Poly(g, w).degree()
    coeffs = sp.Poly(g, w).all_coeffs()  # a_d, ..., a_0 in Q[k][z]
    lc = coeffs[0]
    disc = sp.discriminant(g, w)
    fs = kfactors(lc * disc)
    Rt = sp.prod([f.as_expr() for f in fs])
    Rz = sp.prod([f.as_expr() for f in fs if f.as_expr() != z])
    P: dict[str, sp.Expr] = {}
    P["lc_z(R~)"] = sp.Poly(Rt, z).LC()
    P["disc_z(R~)"] = sp.discriminant(Rt, z)
    P["R~'(0, k)"] = Rz.subs(z, 0)
    P["cont_z(lc_w G)"] = sp.gcd_list(sp.Poly(lc, z).all_coeffs())
    P["cont_z(disc_w G)"] = sp.gcd_list(sp.Poly(disc, z).all_coeffs())
    for f in kfactors(lc):
        delta = next(i for i in range(1, d + 1)
                     if not sp.Poly(coeffs[i], z, k).rem(f).is_zero)
        P[f"drop {delta} at roots of {f.as_expr()}"] = sp.resultant(f.as_expr(), coeffs[delta], z)
    return P, fs


def equal_modulus(f_rho: sp.Poly, fs: list[sp.Poly]) -> sp.Expr:
    """E(k): vanishes when some other candidate meets |z| = rho_k (design Sec 4)."""
    y = sp.Symbol("y")
    m2 = sp.resultant(f_rho.as_expr(), y - z**2, z)
    E = sp.Integer(1)
    for f in fs:
        fe = f.as_expr()
        dz = f.degree(z)
        if fe == z:
            continue
        P = sp.resultant(fe, sp.expand(z**dz * fe.subs(z, y / z)), z)
        Gr = sp.resultant(fe, y - z**2, z)
        same = sp.Poly(fe, z, k).monic() == sp.Poly(f_rho.as_expr(), z, k).monic()
        test = sp.cancel(P / Gr) if same else P
        E *= sp.resultant(m2, test, y)
        if same:
            E *= sp.resultant(fe, fe.subs(z, -z), z)
    return E


def real_cuts(exprs) -> list[Alg]:
    """Real roots >= 1 of the product (squarefree), increasing."""
    prod = sp.Integer(1)
    for e in exprs:
        e = sp.factor(e)
        if e.free_symbols:
            prod *= sp.sqf_part(e)
    rs = [r for r in real_roots(sp.Poly(sp.sqf_part(prod), k, domain=QQ))]
    out = []
    for r in rs:
        while r.lo < 1 <= r.hi and r.sign_of(sp.Poly(k - 1, k)) != 0:
            r.refine((r.hi - r.lo) / 4)
        if r.hi >= 1:
            out.append(r)
    return out


def describe_a(out: dict, nfs: list[sp.Expr], w, n: int):
    """(factor of N~, index among its positive roots at n) for a_n (Case I)."""
    if out["case"] == "II":
        return None
    a = out["a"]
    for f in nfs:
        fn = sp.Poly(f.subs(k, n), w, domain=QQ)
        if fn.degree() > 0 and fn.rem(sp.Poly(a.f.as_expr(), w)).is_zero:
            pos = positive(real_roots(fn))
            idx = [i for i, r in enumerate(pos) if r.f.monic() == sp.Poly(a.f.as_expr(), w).monic()
                   and not (r.hi < a.lo or r.lo > a.hi)]
            assert len(idx) == 1
            return (f, idx[0])
    raise AssertionError


def describe(out: dict, fs: list[sp.Poly], n: int):
    """(factor of R~, index among its positive roots at n) for rho_n."""
    rho = out["rho"]
    for f in fs:
        fn = sp.Poly(f.as_expr().subs(k, n), z, domain=QQ)
        if fn.degree() > 0 and fn.rem(rho.f).is_zero:
            pos = positive(real_roots(fn))
            idx = [i for i, r in enumerate(pos) if r.f.monic() == rho.f.monic()
                   and not (r.hi < rho.lo or r.lo > rho.hi)]
            while len(idx) != 1:
                for r in pos:
                    r.refine((r.hi - r.lo) / 4)
                rho.refine((rho.hi - rho.lo) / 4)
                idx = [i for i, r in enumerate(pos) if r.f.monic() == rho.f.monic()
                       and not (r.hi < rho.lo or r.lo > rho.hi)]
            return (f.as_expr(), idx[0], out["case"])
    raise AssertionError


def section_symk() -> None:
    from flanpy import eliminate_in_k

    print("== symbolic k ==")
    for name, system, w in [("R_k", rk, R), ("HG A_k", hg, A), ("F_k", fk, F), ("L_k", ssnf, L)]:
        t0 = time.perf_counter()
        cases = eliminate_in_k(system, w)
        assert len(cases.pieces) == 1
        G = sp.Poly(cases.pieces[0][1].polynomial.as_expr(), w, z, k, domain=QQ)
        P, fs = projection(G, w)
        print(f"{name}: R~ factors {[f.as_expr() for f in fs]}")
        for key, val in P.items():
            print(f"    {key}: {sp.factor(val)}")
        cuts = real_cuts(P.values())
        # sample rho at k = 2 to get m_rho for the equal-modulus polynomial
        out2 = singular_fixed(system, w, 2, verbose=False)
        frho = next(f for f in fs
                    if sp.Poly(f.as_expr().subs(k, 2), z, domain=QQ).rem(out2["rho"].f).is_zero)
        E = equal_modulus(frho, fs)
        Ef = sp.factor_list(E)[1]
        print(f"    E(k) factors (degree in k): {[sp.Poly(f, k).degree() for f, _ in Ef]}")
        # Case I: a_k is a root of N(w, k) = Res_z(m_rho, G); its factors and their
        # projection polynomials (lc, disc of the squarefree part, value at 0).
        Nk = sp.resultant(frho.as_expr(), G.as_expr(), z)
        nfs = [fac for fac, _ in sp.factor_list(Nk, w, k)[1] if sp.Poly(fac, w).degree() > 0]
        PN = []
        if out2["case"] == "I":
            Nt = sp.prod(nfs)
            PN = [sp.Poly(Nt, w).LC(), sp.discriminant(Nt, w), Nt.subs(w, 0)]
            print(f"    N~ factors (w-degree): {[sp.Poly(f, w).degree() for f in nfs]}; "
                  f"cuts from N~: {[float(c.mpf(10)) for c in real_cuts(PN)]}")
        cuts = real_cuts([*P.values(), E, *PN])
        print(f"    real cuts in [1, oo): {[float(c.mpf(10)) for c in cuts]}")
        # pieces: integer runs between consecutive cuts; exceptional = integer cuts
        bounds, exceptional = [], []
        for c in cuts:
            if c.f.degree() == 1:
                exceptional.append(int(c.lo))
            bounds.append(c)
        pts = []  # (kind, n)
        lo = 1
        for c in bounds:
            hi = math.floor(c.hi) if c.f.degree() > 1 else int(c.lo) - 1
            if c.f.degree() > 1:
                while math.floor(c.lo) != math.floor(c.hi):
                    c.refine((c.hi - c.lo) / 4)
                hi = math.floor(c.lo)
            if lo <= hi:
                pts.append(("interval", lo, hi))
            if c.f.degree() == 1:
                pts.append(("exceptional", int(c.lo), int(c.lo)))
                lo = int(c.lo) + 1
            else:
                lo = hi + 1
        pts.append(("interval", lo, None))
        descs = []
        for kind, a, b in pts:
            n = a if b is None or a == b else (a + b) // 2
            out = singular_fixed(system, w, n, verbose=False)
            dsc = (*describe(out, fs, n), describe_a(out, nfs, w, n))
            descs.append(dsc)
            adesc = (f", a = positive root #{dsc[3][1] + 1} of a factor of N of w-degree "
                     f"{sp.Poly(dsc[3][0], w).degree()}" if dsc[3] else "")
            print(f"    {kind} [{a}, {b if b is not None else 'oo'}] sample k = {n}: "
                  f"rho = positive root #{dsc[1] + 1} of {dsc[0]}, Case {dsc[2]}{adesc}")
        merged = len(set(descs)) == 1
        print(f"    merged into one piece [1, oo): {merged}  [{time.perf_counter() - t0:.1f} s]")


def section_signed() -> None:
    print("== signed: F = 1 - z F^2 ==")
    signed = System({F: 1 - z * F**2}, z=z)
    curve = eliminate(signed, F)
    G = sp.Poly(curve.polynomial.as_expr(), F, z, domain=QQ)
    R_ = candidates(G, F)
    print(f"  G = {curve.polynomial.as_expr()}, candidates R = {sp.factor(R_.as_expr())}")
    print(f"  positive candidates: {positive(real_roots(R_))} (none), yet the radius is 1/4:")
    c = series(signed, 200)[F]
    print(f"  |[z^200] / [z^199]| = {float(abs(c[200] / c[199])):.6f} (-> 4), sign of [z^199]:"
          f" {'+' if c[199] > 0 else '-'}")
    try:
        PosSystem(signed, None)
    except AssertionError as e:
        print(f"  certified bracketing refuses it: {e}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["fixed", "lk", "ratio", "symk", "signed"]
    for name in which:
        globals()[f"section_{name}"]()
