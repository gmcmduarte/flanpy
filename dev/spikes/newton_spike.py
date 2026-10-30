# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike for build-order step 5 (newton), dev/design/newton.md.

Throwaway code (ruff excludes dev/spikes). Run from the repo root:

    .venv/bin/python dev/spikes/newton_spike.py              # all sections
    .venv/bin/python dev/spikes/newton_spike.py fixed        # one section

Sections:
  field       the real number field Q(theta) (flint polynomials mod m_theta, signs via
              AlgebraicReal) and Trager's norm method for the real roots of a
              polynomial over Q(theta); a characteristic polynomial c^10 - c^2 - 1
              (psi(u) = u^5 - u - 1, Galois group S5: no radicals).
  fixed       every example at fixed k: local problem at (rho, a), Newton polygon (exact
              zero tests, and how often Broda's gcd test is inconclusive), real branches,
              branch of w, alpha, b or c exactly, a few Puiseux terms by Hensel lifting,
              checked against closed forms and [z^n] at n = 1000 (2000 for alpha = 1/4).
  broda       L_k, F_k for k in {2, 10, 50, 54, 108}: c_k against eq. (34) and eq. (36),
              b_k against eq. (37) (sign), 2 / (b_k c_k) against eq. (38) and Table 1.
  symk        symbolic k, absolute description: c_k (L_k) and b_k (F_k) as KRoots of the
              norm over Q(k), the cut polynomials, the positive index against the fixed-k
              result for n = 1..60 (L_k: the index changes at 54/55).
  relative    symbolic k, relative description: c_k = the positive root of
              den(rho_k) y^2 - num(rho_k) (eq. (34)), against fixed k for n = 1..60, 108.
"""

from __future__ import annotations

import math
import sys
import time
from fractions import Fraction

import flint
import mpmath
import sympy as sp

from flanpy import AlgebraicReal, KRoot, System, eliminate, eliminate_in_k, series, singularity
from flanpy._singularity import _analysis, _compare, _Below, _points, _q
from flanpy._algebraic import _real_roots, _separators

X, Yv, C = sp.symbols("x y c")
z, k = sp.symbols("z k")
mpmath.mp.dps = 40


def fr(c) -> Fraction:
    c = sp.Rational(c)
    return Fraction(int(c.p), int(c.q))


def fq(p_expr, var) -> flint.fmpq_poly:
    """SymPy expression in var (rational coefficients) -> fmpq_poly."""
    p = sp.Poly(p_expr, var, domain=sp.QQ)
    cs = [fr(c) for c in reversed(p.all_coeffs())]
    return flint.fmpq_poly([flint.fmpq(c.numerator, c.denominator) for c in cs])


def fq_to_expr(p: flint.fmpq_poly, var):
    return sum(sp.Rational(int(c.p), int(c.q)) * var**i for i, c in enumerate(p.coeffs()))


def ifrac(c) -> Fraction:
    return Fraction(int(c.p), int(c.q))


def interval_eval(p: flint.fmpq_poly, lo: Fraction, hi: Fraction) -> tuple[Fraction, Fraction]:
    """Rational interval Horner: an enclosure of p([lo, hi])."""
    cs = [ifrac(c) for c in p.coeffs()] or [Fraction(0)]
    a = b = cs[-1]
    for c in reversed(cs[:-1]):
        prods = (a * lo, a * hi, b * lo, b * hi)
        a, b = min(prods) + c, max(prods) + c
    return a, b


# ---------------------------------------------------------------------------
# The real number field Q(theta) (newton.md, section 4)
# ---------------------------------------------------------------------------


class NF:
    """Q(theta) = Q[x]/(m), with the real embedding given by theta (an AlgebraicReal)."""

    def __init__(self, theta: AlgebraicReal):
        self.theta = theta
        self.m = fq(theta.polynomial.as_expr(), theta.polynomial.gen)
        self.deg = self.m.degree()

    def __call__(self, c) -> flint.fmpq_poly:
        if isinstance(c, flint.fmpq_poly):
            return c % self.m
        f = fr(c)
        return flint.fmpq_poly([flint.fmpq(f.numerator, f.denominator)])

    def gen(self):
        return flint.fmpq_poly([0, 1]) % self.m

    def mul(self, a, b):
        return (a * b) % self.m

    def inv(self, a):
        assert not a.is_zero(), "division by zero in Q(theta)"
        g, s, _ = a.xgcd(self.m)
        return (s / g.coeffs()[0]) % self.m

    def div(self, a, b):
        return self.mul(a, self.inv(b))

    def pow(self, a, e):
        r = self(1)
        for _ in range(e):
            r = self.mul(r, a)
        return r

    def sign(self, a) -> int:
        if a.is_zero():
            return 0
        return self.theta.sign(sp.Poly(fq_to_expr(a, X), X))

    def enclose(self, a) -> tuple[Fraction, Fraction]:
        lo, hi = self.theta.interval
        return interval_eval(a, lo, hi)

    def real(self, a) -> AlgebraicReal:
        """The element as an AlgebraicReal: minimal polynomial from the norm
        Res_x(m(x), y - a(x)), root identified by a rational enclosure."""
        if a.degree() <= 0:
            c = ifrac(a.coeffs()[0]) if a.degree() == 0 else Fraction(0)
            return AlgebraicReal(sp.Poly(Yv - sp.Rational(c.numerator, c.denominator), Yv), 0)
        norm = sp.Poly(sp.resultant(fq_to_expr(self.m, X), Yv - fq_to_expr(a, X), X), Yv)
        factors = [f for f, _ in sp.factor_list(norm)[1]]
        while True:
            lo, hi = self.enclose(a)
            hits = [(f, f.count_roots(_q(lo), _q(hi))) for f in factors]
            if sum(h for _, h in hits) == 1:
                f = next(f for f, h in hits if h == 1)
                return AlgebraicReal(f, int(f.count_roots(None, _q(hi))) - 1)
            self.theta._shrink()


def poly_gcd(K: NF, a: list, b: list) -> list:
    """Monic gcd over K of two polynomials given as coefficient lists (low -> high)."""

    def trim(p):
        while p and p[-1].is_zero():
            p = p[:-1]
        return p

    a, b = trim(a), trim(b)
    while b:
        # a mod b
        r = list(a)
        inv_lc = K.inv(b[-1])
        while len(r) >= len(b) and r:
            q = K.mul(r[-1], inv_lc)
            shift = len(r) - len(b)
            for i, bi in enumerate(b):
                r[shift + i] = (r[shift + i] - K.mul(q, bi)) % K.m
            r = trim(r)
        a, b = b, r
    inv_lc = K.inv(a[-1])
    return [K.mul(c, inv_lc) for c in a]


def same_real(K: NF, a, target: AlgebraicReal) -> bool:
    """Exact: is a (an element of K, known to be a root of target's minimal polynomial)
    equal to target? Refine until the enclosure is inside or outside target's
    isolating interval."""
    while True:
        lo, hi = K.enclose(a)
        tlo, thi = target.interval
        if tlo < lo and hi < thi:
            return True
        if hi < tlo or thi < lo:
            return False
        # refine the source only: the target's isolating interval holds exactly one
        # root of its minimal polynomial, and a is one of those roots
        K.theta._shrink()


def real_roots_over(K: NF, phi: list) -> list[tuple[NF, object, object]]:
    """Real roots of phi(c) = sum phi[j] c^j over K = Q(theta) (Trager 1976).

    Returns (L, theta_in_L, c_in_L) with L = Q(eta) = Q(theta, c), eta = c + t theta.
    """
    Phi = sum(fq_to_expr(p, X) * C**j for j, p in enumerate(phi))
    m = fq_to_expr(K.m, X)
    for t in (0, 1, -1, 2, -2, 3, -3):
        norm = sp.Poly(sp.resultant(m, Phi.subs(C, Yv - t * X), X), Yv)
        if norm.degree() > 0 and sp.gcd(norm, norm.diff(Yv)).degree() == 0:
            break
    else:
        raise RuntimeError("no squarefree norm")
    out = []
    for f, _ in sp.factor_list(norm)[1]:
        nreal = int(f.count_roots())
        for i in range(nreal):
            eta = AlgebraicReal(f, i)
            L = NF(eta)
            yL = L.gen()
            # gcd over L of m(x) and Phi(x, eta - t x): degree 1 (squarefree norm)
            mx = [L(c) for c in reversed(sp.Poly(m, X).all_coeffs())]
            Bx_poly = sp.Poly(sp.expand(Phi.subs(C, Yv - t * X)), X)
            Bx = []
            for c in reversed(Bx_poly.all_coeffs()):
                Bx.append(L(fq(sp.expand(c), Yv)) if sp.sympify(c).free_symbols else L(c))
            g = poly_gcd(L, mx, Bx)
            assert len(g) == 2, f"gcd of degree {len(g) - 1}"
            R = (-g[0]) % L.m
            if K.deg == 1 or same_real(L, R, K.theta):
                out.append((L, R, (yL - t * R) % L.m))
    return out


def embed(K: NF, L: NF, R, a):
    """Map a in K = Q(theta) to L, where theta -> R."""
    r = L(0)
    for c in reversed(a.coeffs()):
        r = (L.mul(r, R) + L(ifrac(c))) % L.m
    return r


def section_field():
    print("== field ==")
    t0 = time.time()
    sqrt2 = AlgebraicReal(sp.Poly(X**2 - 2, X), 1)
    K = NF(sqrt2)
    s = K.gen()
    print("  sqrt2^2 =", K.mul(s, s), " 1/(1+sqrt2) =", K.inv(s + 1))
    print("  sign(sqrt2 - 1.4142) =", K.sign(s - K(Fraction(14142, 10000))))
    print("  (1+sqrt2)^3 as AlgebraicReal:", K.real(K.pow(s + 1, 3)))
    # psi(u) = u^5 - u - 1 (Galois S5), phi(c) = psi(c^2) over Q(sqrt 2) scaled by sqrt 2
    phi = [K(-1), K(0), -s % K.m, K(0), K(0), K(0), K(0), K(0), K(0), K(0), K(1)]
    roots = real_roots_over(K, phi)
    print("  real roots of c^10 - sqrt2 c^2 - 1 over Q(sqrt2):", len(roots))
    for L, R, c in roots:
        r = L.real(c)
        print(f"    c = {float(r.evalf(20)):+.15f}, [Q(theta, c):Q] = {L.deg},",
              "minpoly of c:", r.polynomial.as_expr())
    print("  Galois group of u^5 - u - 1:", sp.galois_group(sp.Poly(X**5 - X - 1, X))[0].order(), "(S5 has order 120)")
    print(f"  ({time.time() - t0:.2f} s)")


# ---------------------------------------------------------------------------
# The local problem, the Newton polygon, the branch of w
# ---------------------------------------------------------------------------


def window(g: sp.Poly, closure, rho: AlgebraicReal, previous: Fraction, case: str):
    """Step 4's window test, returning the window data: x0, the separators, the window
    index j, and the rank of w(x0) among the real roots of G(x0, .) in its window."""
    w, zz = g.gens
    m_rho = rho.polynomial.as_expr().subs(rho.polynomial.gen, zz)
    norm = sp.Poly(sp.resultant(m_rho, g.as_expr(), zz), w, domain=sp.QQ)
    roots = _real_roots(norm) if not norm.is_zero else []
    seps = _separators(roots)
    bound = previous
    for q in seps:
        col = sp.Poly(g.as_expr().subs(w, _q(q)), zz)
        if col.is_zero or col.degree() < 1:
            continue
        for r in _real_roots(col):
            if r < rho:
                while not r.interval[1] < rho.interval[0]:
                    r._shrink()
                    rho._shrink()
                bound = max(bound, r.interval[1])
    while not bound < rho.interval[0]:
        rho._shrink()
    for x0 in _points(bound, rho.interval[0]):
        res = _compare(closure, x0)
        if res is not None:
            break
    assert isinstance(res, _Below)
    i = closure.unknowns.index(w)
    fiber = sp.Poly(g.as_expr().subs(zz, _q(x0)), w)
    rnd = 0
    while fiber.count_roots(_q(res.lower[i]), _q(res.upper[i])) > 1:
        rnd += 1
        res = _compare(closure, x0, first_round=rnd)
    lo, hi = res.lower[i], res.upper[i]
    edges = [None, *seps, None]
    for j in range(len(edges) - 1):
        left = lo if edges[j] is None else max(lo, edges[j])
        right = hi if edges[j + 1] is None else min(hi, edges[j + 1])
        if left <= right and fiber.count_roots(_q(left), _q(right)) == 1:
            break
    wl, wr = edges[j], edges[j + 1]
    in_window = fiber.count_roots(None if wl is None else _q(wl), None if wr is None else _q(wr))
    below = fiber.count_roots(None if wl is None else _q(wl), _q(lo))
    # rank = number of real roots of G(x0, .) in the window strictly below w(x0)
    return dict(x0=x0, in_window=in_window, rank=below)


def local_coefficients(g: sp.Poly, K: NF, rho, a, reciprocal: bool):
    """q_ij in K: Q(s, Y) = sum q_ij s^i Y^j = G(rho(1 - s), a + Y) (Case I), or the
    reciprocal H(z, v) = v^m G(z, 1/v) at (rho(1 - s), Y) (Case II, a = 0)."""
    w, zz = g.gens
    G = g.as_expr()
    if reciprocal:
        m = g.degree(w)
        G = sp.expand(w**m * G.subs(w, 1 / w))
    P = sp.Poly(G, w, zz)
    q = {}
    for (dw, dz), co in P.terms():
        co = fr(co)
        # co * (rho (1 - s))^dz * (a + Y)^dw
        for i in range(dz + 1):
            for j in range(dw + 1):
                term = K(co * math.comb(dz, i) * (-1) ** i * math.comb(dw, j))
                term = K.mul(term, K.pow(rho, dz))
                term = K.mul(term, K.pow(a, dw - j))
                q[(i, j)] = (q.get((i, j), K(0)) + term) % K.m
    return {key: v for key, v in q.items()}


def lower_hull(points):
    hull = []
    for p in sorted(points):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) >= (p[1] - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def newton_edges(q, K):
    """Edges of negative slope of the Newton polygon of Q(s, Y) (points (j, i))."""
    support = {(j, i) for (i, j), v in q.items() if not v.is_zero()}
    r = min(j for j, i in support if i == 0)  # ord_Y Q(0, Y): branches through (rho, a)
    colmin = {}
    for j, i in support:
        if j <= r:
            colmin[j] = min(colmin.get(j, 10**9), i)
    hull = lower_hull(colmin.items())
    edges = []
    for (j1, i1), (j2, i2) in zip(hull, hull[1:]):
        gamma = Fraction(i1 - i2, j2 - j1)
        if gamma <= 0:
            continue
        on = [(j, i) for j, i in colmin.items() if j1 <= j <= j2 and (i - i1) * (j2 - j1) == (i2 - i1) * (j - j1)]
        edges.append((gamma, j1, j2, sorted(on)))
    return r, colmin, edges


def branches(q, K, edges):
    out = []
    for gamma, j1, j2, on in edges:
        phi = [K(0)] * (j2 - j1 + 1)
        for j, i in on:
            phi[j - j1] = q[(i, j)]
        for L, R, c in real_roots_over(K, phi):
            out.append((gamma, L, R, c, phi, j1))
    return out


def hensel(q, K, L, R, gamma, c, nterms):
    """Y = t^p (c + Y1(t)), s = t^q; Y1 = sum e_n t^n by undetermined coefficients over L
    (c a simple root). Returns [c, e_1, ..., e_nterms] (coefficients of t^p, t^(p+1), ...)."""
    p, qq = gamma.numerator, gamma.denominator
    beta = min(qq * i + p * j for (i, j), v in q.items() if not v.is_zero())
    # Q1(t, Y1) = t^-beta Q(t^q, t^p (c + Y1)) = sum_l A_l(t) Y1^l
    Q1 = {}
    for (i, j), v in q.items():
        if v.is_zero():
            continue
        vL = embed(K, L, R, v)
        e = qq * i + p * j - beta
        for l in range(j + 1):
            co = L.mul(L(math.comb(j, l)), L.mul(vL, L.pow(c, j - l)))
            Q1[(e, l)] = (Q1.get((e, l), L(0)) + co) % L.m
    D = Q1.get((0, 1), L(0))
    assert not D.is_zero(), "multiple root: not a simple Hensel lift"
    coeffs = [c]
    Y1 = [L(0)] * (nterms + 1)  # series in t, index = power
    for n in range(1, nterms + 1):
        # coefficient of t^n in Q1(t, Y1) with e_n = 0
        def mulser(a, b):
            r = [L(0)] * (nterms + 1)
            for i, ai in enumerate(a):
                if ai.is_zero():
                    continue
                for j, bj in enumerate(b[: nterms + 1 - i]):
                    r[i + j] = (r[i + j] + L.mul(ai, bj)) % L.m
            return r

        total = L(0)
        powers = [[L(1)] + [L(0)] * nterms]
        maxl = max(l for (_, l) in Q1)
        for _ in range(maxl):
            powers.append(mulser(powers[-1], Y1))
        for (e, l), v in Q1.items():
            if e <= n:
                total = (total + L.mul(v, powers[l][n - e])) % L.m
        Y1[n] = L.div(-total % L.m, D)
        coeffs.append(Y1[n])
    return coeffs


def gcd_test(g, rho: AlgebraicReal, i, j, reciprocal):
    """Broda's test: gcd(Res_w(G, d^i_z d^j_w G), m_rho) == 1 -> nonzero."""
    w, zz = g.gens
    G = g.as_expr()
    if reciprocal:
        m = g.degree(w)
        G = sp.expand(w**m * G.subs(w, 1 / w))
        D = sp.diff(G, zz, i, w, j).subs(w, 0)
        if D == 0:
            return "zero"
        m_rho = rho.polynomial.as_expr().subs(rho.polynomial.gen, zz)
        return "nonzero" if sp.rem(sp.expand(D), m_rho, zz) != 0 else "zero (exact)"
    D = sp.diff(G, zz, i, w, j)
    if sp.expand(D) == 0:
        return "zero"
    m_rho = rho.polynomial.as_expr().subs(rho.polynomial.gen, zz)
    res = sp.resultant(G, D, w) if sp.Poly(D, w).degree() > 0 else D
    gg = sp.gcd(sp.Poly(res, zz), sp.Poly(m_rho, zz))
    return "nonzero" if gg.degree() == 0 else "inconclusive"


def analyse(name, system, w, kval=None, nterms=3, ncheck=1000, expect=None, verbose=True, stats=True):
    t0 = time.time()
    curve = eliminate(system, w, k=kval) if kval is not None else eliminate(system, w)
    sing, closure = _analysis(curve.specialize(kval) if kval is not None and curve.k is not None else curve)
    g = sp.Poly(curve.polynomial.as_expr().subs(k, kval) if kval is not None else curve.polynomial.as_expr(), w, z, domain=sp.QQ)
    rho, case = sing.rho, sing.case
    Krho = NF(rho)
    # the field of (rho, a)
    if case == "II":
        K, rhoK, aK, reciprocal = Krho, Krho.gen(), Krho(0), True
        how_a = "-"
    else:
        a = sing.a
        # gcd over Q(rho) of G(rho, w) and G_w(rho, w)
        Gw = sp.Poly(g.as_expr(), w)
        coeffs = [Krho(fq(sp.expand(c), z)) if sp.sympify(c).free_symbols else Krho(c) for c in reversed(Gw.all_coeffs())]
        dcoeffs = [Krho.mul(Krho(i), c) for i, c in enumerate(coeffs)][1:]
        h = poly_gcd(Krho, coeffs, dcoeffs)  # monic gcd(G(rho, .), G_w(rho, .)) over Q(rho)
        hd = [Krho.mul(Krho(i), c) for i, c in enumerate(h)][1:]
        e = len(h) - 1
        a_in_rho = None
        if e == 1:
            a_in_rho = (-h[0]) % Krho.m
        elif len(h) - len(poly_gcd(Krho, h, hd)) == 1:
            # squarefree part of degree 1: h = (w - a)^e, so a = -h_{e-1} / e
            a_in_rho = (-Krho.div(h[e - 1], Krho(e))) % Krho.m
        if a_in_rho is not None:
            K, rhoK, aK = Krho, Krho.gen(), a_in_rho
            assert same_real(K, aK, a) if K.deg > 1 else True
            how_a = "in Q(rho)"
        else:
            roots = real_roots_over(Krho, h)
            match = [(L, R, c) for L, R, c in roots if L.real(c) == a]
            L, R, c = match[0]
            K, rhoK, aK = L, R, c
            how_a = f"primitive element, degree {L.deg}"
        reciprocal = False
    q = local_coefficients(g, K, rhoK, aK, reciprocal)
    r, colmin, edges = newton_edges(q, K)
    bs = branches(q, K, edges)
    # real branches for s > 0 and the sign rule: Case I Y < 0 (w < a), Case II v > 0
    want = -1 if case == "I" else 1
    cands = [(gamma, L, R, c, phi, j1) for gamma, L, R, c, phi, j1 in bs if L.sign(c) == want]
    previous = Fraction(0)
    win = window(g, closure, rho, previous, case)
    # (previous should be the candidate below rho; harmless here since the window test
    # only needs x0 above the bad points, recomputed from the separators)
    assert len(cands) >= 1
    gamma, L, R, c, phi, j1 = cands[0]
    terms = hensel(q, K, L, R, gamma, c, nterms)
    lead = L.real(c)
    if case == "I":
        alpha = gamma
        const = L.real((-c) % L.m)  # b
    else:
        alpha = gamma
        const = lead  # c
    dt = time.time() - t0
    # Broda gcd test statistics on the points of the hull columns
    gstats = {}
    for j, i in (colmin.items() if stats else ()):
        for ii in range(0, i + 1):
            if (ii, j) in ((0, 0), (0, 1)) and not reciprocal:
                continue
            res = gcd_test(g, rho, ii, j, reciprocal)
            gstats[res] = gstats.get(res, 0) + 1
    if verbose:
        print(f"-- {name}" + (f", k = {kval}" if kval is not None else ""))
        print(f"   rho = {float(rho.evalf(20)):.12g} (deg {rho.polynomial.degree()}), case {case}, a {how_a},"
              f" field degree {K.deg} -> {L.deg}")
        print(f"   polygon columns (j: min i) {dict(sorted(colmin.items()))}, r = {r},"
              f" edges {[(str(e[0]), e[3]) for e in edges]}")
        print(f"   real branches for s > 0: {len(bs)} (signs {[L_.sign(c_) for _, L_, _, c_, _, _ in bs]}),"
              f" candidates {len(cands)}; window: {win['in_window']} real sheets, rank of w {win['rank']}")
        print(f"   alpha = {alpha}, {'b' if case == 'I' else 'c'} = {mpmath.nstr(const.evalf(30), 20)}"
              f" (minpoly deg {const.polynomial.degree()}); gcd test on hull columns: {gstats}")
        tt = [mpmath.nstr(L.real(e).evalf(20), 12) for e in terms]
        print(f"   Puiseux coefficients of t^{gamma.numerator}.. (t = s^(1/{gamma.denominator})): {tt}")
    if ncheck:
        cs = series(system, ncheck, k=kval)[w] if kval is not None else series(system, ncheck)[w]
        n = ncheck
        rho_f = rho.evalf(40)
        # transfer of the singular terms w = a + sum d_e s^e (Case I) / 1/v (Case II)
        if case == "I":
            approx = 0
            p, qq = gamma.numerator, gamma.denominator
            for idx, e_co in enumerate(terms):
                e = Fraction(p + idx, qq)
                if e.denominator == 1:
                    continue
                d = L.real(e_co).evalf(40)
                approx += d * rho_f ** (-n) * mpmath.gammaprod([n - e], [-e, n + 1])
        else:
            # leading term only: (1/c) s^-alpha
            approx = (1 / const.evalf(40)) * rho_f ** (-n) * mpmath.gammaprod([n + alpha], [alpha, n + 1])
        val = mpmath.mpf(Fraction(cs[n]).numerator) / Fraction(cs[n]).denominator
        print(f"   [z^{n}] w / transfer({len(terms)} terms) - 1 = {mpmath.nstr(val / approx - 1, 5)}")
    if expect is not None:
        exp_val = expect(rho.evalf(40))
        print(f"   closed form: {mpmath.nstr(exp_val, 20)}, diff {mpmath.nstr(const.evalf(40) - exp_val, 3)}")
    print(f"   ({dt:.2f} s analysis)")
    return dict(rho=rho, case=case, alpha=alpha, const=const, terms=terms, L=L)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

F, Gs, R_, B, Bb, M, Mb, L_, A, U, K_, Y, W, T = sp.symbols("F G R B Bb M Mb L A U K Y W T")

CATALAN = System({F: 1 + z * F**2}, z=z)
MOTZKIN = System({F: z + z * F + z * F**2}, z=z)  # FS Example VII.18
RK = System({R_: (k + 1) * z + 2 * z * R_**2 + z * R_}, z=z, k=k)
HG = System({B: k * z + 2 * z * B**2 + 2 * z * B, A: 2 * z + B}, z=z, k=k)
SSNF = System(
    {
        B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
        Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
        L_: M + Mb,
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
QUARTER = System({U: z + U**2, K_: U / 2 + K_**2}, z=z)  # K = (1 - s^(1/4))/2, alpha = 1/4
POLE = System({Y: 1 + 2 * z * Y}, z=z)  # 1/(1 - 2z): Case II, alpha = 1
THREE_HALVES = System({B: z + B**2, Y: 1 + 2 * B * Y, W: Y**3}, z=z)  # (1-4z)^(-3/2)
S5TREE = System({T: z * (1 + T**2 + T**5)}, z=z)  # rho, a, b not in radicals
CONSTRUCTED = System({F: z + z * F**2 + z * Gs, Gs: z + z * Gs**2}, z=z)


def section_fixed():
    print("== fixed ==")
    analyse("Catalan", CATALAN, F, expect=lambda r: mpmath.mpf(2))
    analyse("Motzkin (FS Ex. VII.18)", MOTZKIN, F, nterms=4, expect=lambda r: mpmath.sqrt(3))
    analyse("R_k", RK, R_, kval=2, expect=lambda r: mpmath.sqrt(2 - 2 * r) / (4 * r))
    analyse("HG A_k", HG, A, kval=2, expect=lambda r: (2 * 2) ** mpmath.mpf(0.25) / (2 * mpmath.sqrt(r)))
    analyse("alpha = 1/4: K = U/2 + K^2", QUARTER, K_, nterms=4, ncheck=2000, expect=lambda r: mpmath.mpf(1) / 2)
    analyse("pole: Y = 1 + 2zY", POLE, Y, expect=lambda r: mpmath.mpf(1))
    analyse("Case II alpha = 3/2: W = Y^3", THREE_HALVES, W, expect=lambda r: mpmath.mpf(1))
    analyse("constructed (step 4)", CONSTRUCTED, F)

    def tree_b(r):
        tau = mpmath.findroot(lambda y: 4 * y**5 + y**2 - 1, 0.6)
        phi = 1 + tau**2 + tau**5
        phi2 = 2 + 20 * tau**3
        return mpmath.sqrt(2 * phi / phi2)

    res = analyse("S5 tree T = z(1 + T^2 + T^5)", S5TREE, T, expect=tree_b)
    mrho = res["rho"].polynomial
    print("   Galois group of m_rho:", sp.galois_group(mrho)[0].order(), " m_rho =", mrho.as_expr())
    print("   sympy.roots(m_rho) gives", len(sp.roots(mrho)), "radical roots")


def section_broda():
    print("== broda ==")
    for n in (2, 10, 50, 54, 108):
        Lk = analyse("L_k", SSNF, L_, kval=n, nterms=2, ncheck=1000 if n in (2, 10, 50) else 0, verbose=False)
        rho = Lk["rho"].evalf(40)
        kk = mpmath.mpf(n)
        # eq. (34): c_k = sqrt(2 rho H_z / H_vv) at (rho, 0); H = eq. (32)
        Hz = -(12 * kk * (2 * kk + 27) * rho**2 + 72 * kk * rho - kk)
        Hvv = -2 * kk * rho * ((kk + 12) * kk * rho**2 + 4 * kk * rho + 1)
        c34 = mpmath.sqrt(2 * rho * Hz / Hvv)
        c = Lk["const"].evalf(40)
        print(f"   L_{n}: c_k = {mpmath.nstr(c, 15)}  eq.(34) {mpmath.nstr(c34, 15)}  4/sqrt(k) {mpmath.nstr(4 / mpmath.sqrt(kk), 6)}"
              f"  minpoly deg {Lk['const'].polynomial.degree()}")
        Fk = analyse("F_k", FK, F, kval=n, nterms=2, ncheck=1000 if n in (2, 10, 50) else 0, verbose=False)
        b = Fk["const"].evalf(40)
        print(f"   F_{n}: b_k = {mpmath.nstr(b, 15)}  b_k/sqrt(k) = {mpmath.nstr(b / mpmath.sqrt(kk), 8)}"
              f"  (eq. (37) prints b_k = -k^(1/2) + o(k^(1/2)))  minpoly deg {Fk['const'].polynomial.degree()}")
        print(f"   ratio [z^n]L_k/(n[z^n]F_k) -> 2/(b_k c_k) = {mpmath.nstr(2 / (b * c), 10)};"
              f" eq. (38) approx 1/2 - 1/(4 sqrt(2k)) = {mpmath.nstr(0.5 - 1 / (4 * mpmath.sqrt(2 * kk)), 6)}")


def section_symk():
    print("== symk ==")
    t0 = time.time()
    # L_k, Case II: H = reciprocal of G; c_k^2 = 2 rho H_z / H_vv at (rho, 0)
    curve = eliminate_in_k(SSNF, L_).tail[1]
    G = curve.polynomial.as_expr()
    m = sp.degree(G, L_)
    H = sp.expand(L_**m * G.subs(L_, 1 / L_))
    v = L_
    Hz = sp.diff(H, z).subs(v, 0)
    Hvv = sp.diff(H, v, 2).subs(v, 0)
    pk = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
    print("   H(z, 0) =", sp.factor(H.subs(v, 0)), "  H_v(z, 0) =", sp.factor(sp.diff(H, v).subs(v, 0)))
    norm = sp.Poly(sp.resultant(pk, Yv**2 * Hvv - 2 * z * Hz, z), Yv, k)
    facs = sp.factor_list(norm.as_expr())
    print("   Res_z(p_k, y^2 H_vv - 2 z H_z) factors:", [(sp.Poly(f, Yv).degree(), e) for f, e in facs[1]], "content", facs[0])
    h = max((f for f, e in facs[1] if f.has(Yv)), key=lambda f: sp.Poly(f, Yv).degree())
    cuts = {
        "Res_z(p_k, H_z(z,0))": sp.factor(sp.resultant(pk, Hz, z)),
        "Res_z(p_k, H_vv(z,0))": sp.factor(sp.resultant(pk, Hvv, z)),
        "lc_y h": sp.factor(sp.Poly(h, Yv).LC()),
        "h(0, k)": sp.factor(h.subs(Yv, 0)),
        "disc_y h": sp.factor(sp.discriminant(h, Yv)),
    }
    for name, val in cuts.items():
        print(f"   {name}: {val}")
    hp = sp.Poly(h, Yv, k)
    print("   h =", sp.Poly(h, Yv).degree(), "in y; terms", len(hp.terms()))
    # positive index of c_k among the positive roots of h(., k), against fixed k
    idx = set()
    bad = []
    for n in range(1, 61):
        Lk = analyse("L_k", SSNF, L_, kval=n, nterms=1, ncheck=0, verbose=False, stats=False)
        hn = sp.Poly(h.subs(k, n), Yv)
        pos = [r for r in _real_roots(hn) if r > AlgebraicReal(sp.Poly(Yv, Yv), 0)]
        j = [i for i, r in enumerate(pos) if r == Lk["const"]]
        if not j:
            bad.append(n)
        idx.update(j)
    print(f"   L_k: c_k = KRoot(h, j) with j in {idx}; mismatches {bad}  ({time.time() - t0:.1f} s)")
    kr = KRoot(sp.Poly(h, Yv, k), min(idx), k=k)
    print("   KRoot.at(54):", kr.at(54), " at(108):", kr.at(108))
    symk_fk()


def real_roots_ge1(expr):
    """Real roots >= 1 of the irreducible factors of a polynomial in k (floats, display)."""
    out = []
    for f, _ in sp.factor_list(sp.expand(expr), k)[1]:
        if not f.has(k):
            continue
        for r in sp.Poly(f, k).nroots(n=12):
            if r.is_real and r >= 1:
                out.append(float(r))
    return sorted(out)


def symk_fk():
    """F_k, Case I: a_k in Q(k)(rho_k) via the first subresultant; b_k^2 = 2 rho G_z / G_ww."""
    t0 = time.time()
    curve = eliminate_in_k(FK, F).tail[1]
    G = curve.polynomial.as_expr()
    pk = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
    Gw = sp.diff(G, F)
    print("   F_k: deg_w G =", sp.degree(G, F), " deg_z G =", sp.degree(G, z))
    prs = sp.subresultants(sp.Poly(G, F), sp.Poly(Gw, F))
    S1 = next(p for p in prs if p.degree() == 1)
    s11, s10 = S1.all_coeffs()
    s11, s10 = sp.expand(s11.as_expr() if hasattr(s11, "as_expr") else s11), sp.expand(s10)
    r11 = sp.factor(sp.resultant(pk, s11, z))
    print("   Res_z(p_k, s_11) =", r11, " real roots >= 1:", real_roots_ge1(r11))
    # b^2 = 2 z G_z / G_ww at w = -s10/s11, reduced mod p_k over Q(k)
    Aw = -s10 / s11
    num_z = sp.together(sp.diff(G, z).subs(F, Aw))
    num_ww = sp.together(sp.diff(G, F, 2).subs(F, Aw))
    nz, dz = sp.fraction(num_z)
    nww, dww = sp.fraction(num_ww)
    # y^2 * G_ww - 2 z G_z = 0  <=>  y^2 nww dz - 2 z nz dww = 0
    rel = sp.expand(Yv**2 * nww * dz - 2 * z * nz * dww)
    rel = sp.rem(sp.Poly(rel, z), sp.Poly(pk, z)).as_expr()  # reduce mod p_k (over Q(k)(y))
    rel = sp.expand(sp.numer(sp.together(rel)))
    cz = sp.factor(sp.resultant(pk, sp.numer(sp.together(sp.rem(sp.Poly(nz, z), sp.Poly(pk, z)).as_expr())), z))
    cww = sp.factor(sp.resultant(pk, sp.numer(sp.together(sp.rem(sp.Poly(nww, z), sp.Poly(pk, z)).as_expr())), z))
    print("   Res_z(p_k, num G_z(rho, a)) real roots >= 1:", real_roots_ge1(cz))
    print("   Res_z(p_k, num G_ww(rho, a)) real roots >= 1:", real_roots_ge1(cww))
    norm = sp.resultant(pk, rel, z)
    facs = sp.factor_list(norm)
    print("   Res_z(p_k, relation) factors (deg in y, mult):",
          [(sp.Poly(f, Yv).degree(), e) for f, e in facs[1] if f.has(Yv)], f"({time.time() - t0:.1f} s)")
    h = max((f for f, e in facs[1] if f.has(Yv)), key=lambda f: sp.Poly(f, Yv).degree())
    hp = sp.Poly(h, Yv)
    print("   h: deg_y", hp.degree(), " deg_k", sp.degree(h, k))
    for name, val in {"lc_y h": hp.LC(), "h(0, k)": h.subs(Yv, 0), "disc_y h": sp.discriminant(h, Yv)}.items():
        print(f"   {name}: real roots >= 1: {real_roots_ge1(val)}")
    idx, bad = set(), []
    for n in range(1, 61):
        Fk = analyse("F_k", FK, F, kval=n, nterms=1, ncheck=0, verbose=False, stats=False)
        hn = sp.Poly(h.subs(k, n), Yv)
        pos = [r for r in _real_roots(hn) if r > AlgebraicReal(sp.Poly(Yv, Yv), 0)]
        j = [i for i, r in enumerate(pos) if r == Fk["const"]]
        if not j:
            bad.append(n)
        idx.update(j)
    print(f"   F_k: b_k = KRoot(h, j) with j in {idx}; mismatches {bad}  ({time.time() - t0:.1f} s)")


def relative_root(rho: AlgebraicReal, num, den, q=2):
    """The positive root of y^q den(rho) - num(rho), over Q(rho) (the relative description)."""
    K = NF(rho)
    phi = [(-fq(num, z)) % K.m] + [K(0)] * (q - 1) + [fq(den, z) % K.m]
    pos = [(L, c) for L, R, c in real_roots_over(K, phi) if L.sign(c) > 0]
    assert len(pos) == 1
    return pos[0][0].real(pos[0][1])


def section_relative():
    """c_k (L_k) and b_k (F_k) relative to rho_k: one positive root over Q(k)(rho_k)."""
    print("== relative ==")
    t0 = time.time()
    num_L = 24 * k * z**2 + 324 * z**2 + 72 * z - 1
    den_L = k**2 * z**2 + 12 * k * z**2 + 4 * k * z + 1
    pk = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
    print("   L_k cuts: Res_z(p_k, num) =", sp.factor(sp.resultant(pk, num_L, z)),
          " Res_z(p_k, den) =", sp.factor(sp.resultant(pk, den_L, z)))
    bad = []
    for n in list(range(1, 61)) + [108]:
        Lk = analyse("L_k", SSNF, L_, kval=n, nterms=1, ncheck=0, verbose=False, stats=False)
        c = relative_root(Lk["rho"], num_L.subs(k, n), den_L.subs(k, n))
        if c != Lk["const"]:
            bad.append(n)
    print(f"   L_k: c_k = positive root of den(rho_k) y^2 - num(rho_k), k = 1..60, 108: mismatches {bad}"
          f" ({time.time() - t0:.1f} s)")


SECTIONS = {"field": section_field, "fixed": section_fixed, "broda": section_broda, "symk": section_symk,
            "relative": section_relative}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SECTIONS)
    for nm in names:
        t = time.time()
        SECTIONS[nm]()
        print(f"[{nm}: {time.time() - t:.1f} s]\n")
