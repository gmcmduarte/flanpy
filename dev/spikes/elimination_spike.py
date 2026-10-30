# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike for build-order step 3 (elimination); see dev/design/elimination.md.

Run: python dev/spikes/elimination_spike.py [case ...]
Ground types: SYMPY_GROUND_TYPES=python (pure) or default (flint when installed).
"""

import sys
import time

import sympy as sp
from sympy import QQ, Poly, symbols

from flanpy import System, series, series_in_k

z, k = symbols("z k")


def closure(system, w):
    """Unknowns that w depends on (transitively), in system order."""
    need, todo = {w}, [w]
    while todo:
        u = todo.pop()
        for v in system.equations[u].free_symbols & set(system.unknowns):
            if v not in need:
                need.add(v)
                todo.append(v)
    return [u for u in system.unknowns if u in need]


def lc_main(g, gens_main, k):
    """Leading coefficient in Q[k] of g w.r.t. lex on gens_main (k is the ground)."""
    if k is None:
        return sp.Integer(1)
    return Poly(g, *gens_main, domain=QQ[k]).LC(order="lex").as_expr()


def eliminate_gb(polys, others, w, z, k, method="f5b"):
    gens = [*others, w, z] + ([k] if k is not None else [])
    t = time.perf_counter()
    gb = sp.groebner(polys, *gens, order="lex", method=method)
    dt = time.perf_counter() - t
    E = [g for g in gb.exprs if not (g.free_symbols & set(others))]
    main = [*others, w, z]
    lcs = [sp.factor(lc_main(g, main, k)) for g in gb.exprs]
    lcs_E = [sp.factor(lc_main(g, [w, z], k)) for g in E]
    return gb, E, lcs, lcs_E, dt


def candidates(g0, w):
    """Irreducible factors of g0 over Q that involve w (squarefree, distinct)."""
    _, fl = sp.factor_list(g0)
    return [f for f, _ in fl if f.has(w)]


def res_bound(hs, w, z):
    """B = max over i != j of deg_w h_i deg_z h_j + deg_w h_j deg_z h_i."""
    b = 0
    for i, hi in enumerate(hs):
        for hj in hs[i + 1 :]:
            pi, pj = Poly(hi, w, z), Poly(hj, w, z)
            b = max(b, pi.degree(w) * pj.degree(z) + pj.degree(w) * pi.degree(z))
    return b


def vanishes(h, wser, w, z, n):
    """h(z, w(z)) == 0 mod z^(n+1), w(z) given by its coefficients 0..n."""
    ws = sum(c * z**i for i, c in enumerate(wser[: n + 1]))
    p = Poly(sp.expand(h.subs(w, ws)), z)
    return all(sp.expand(c) == 0 for (e,), c in p.terms() if e <= n)


def normalise(G, w, z, k):
    gens = [w, z] + ([k] if k is not None else [])
    p = Poly(G, *gens, domain=QQ)
    _, p = p.primitive()  # integer content
    if k is not None:
        pk = Poly(p.as_expr(), w, z, domain=QQ[k])
        cont = pk.content()
        p = Poly(sp.cancel(p.as_expr() / cont.as_expr()), *gens, domain=QQ)
        _, p = p.primitive()
    if p.LC(order="lex") < 0:  # lex w > z > k: lc_w, then lc_z, then lc_k
        p = -p
    return p


def select(system, w, hs, k):
    if len(hs) == 1:
        B = 0
    else:
        B = res_bound(hs, w, system.z)
    n = max(B, 4)
    t = time.perf_counter()
    if k is not None and system.k is not None:
        ws = [c.as_expr() for c in series_in_k(system, n)[w]]
    else:
        ws = list(series(system, n)[w])
    picked = [h for h in hs if vanishes(h, ws, w, system.z, n)]
    dt = time.perf_counter() - t
    return picked, B, dt


def exceptional(G, w, z, k, lcs_E):
    """Integer k >= 1 where the generic curve may not specialise."""
    out = {}
    if k is None:
        return out
    P = Poly(G, w, z, domain=QQ[k])
    lcw = Poly(P.as_expr(), w, domain=QQ[z, k]).LC().as_expr()
    c_lcw = Poly(lcw, z, domain=QQ[k]).content().as_expr()
    disc = sp.discriminant(G, w) if P.degree(w) > 1 else sp.Integer(1)
    c_disc = Poly(disc, z, domain=QQ[k]).content().as_expr()

    def int_roots(e):
        e = sp.expand(e)
        if e.is_number:
            return []
        return sorted(r for r in sp.roots(Poly(e, k), filter="Z") if r >= 1)

    for name, e in [("lc_E", sp.Mul(*lcs_E)), ("cont_z lc_w", c_lcw), ("cont_z disc_w", c_disc)]:
        out[name] = (sp.factor(e), int_roots(e))
    return out


def run(name, system, w, *, method="f5b", show=True):
    print(f"\n=== {name}: w = {w}")
    unk = closure(system, w)
    print("  closure:", unk, f"({len(unk)} of {len(system.unknowns)})")
    sub = System({u: system.equations[u] for u in unk}, z=system.z, k=system.k)
    others = [u for u in unk if u != w]
    polys = [p.as_expr() for p in sub.polynomials()]
    kk = system.k
    if others:
        gb, E, lcs, lcs_E, dt = eliminate_gb(polys, others, w, system.z, kk, method)
        print(f"  groebner ({method}): {dt:.3f} s, {len(gb.exprs)} elements, {len(E)} in Q[w,z,k]")
        print("  lc in k (all):", lcs)
        print("  lc in k (E):  ", lcs_E)
        if not E:
            print("  ELIMINATION IDEAL IS ZERO")
            return None
        g0 = min(E, key=lambda g: (Poly(g, w).degree(), len(sp.Add.make_args(g))))
    else:
        g0, lcs_E = polys[0], []
    t = time.perf_counter()
    hs = candidates(g0, w)
    dtf = time.perf_counter() - t
    print(f"  factor: {dtf:.3f} s, {len(hs)} factor(s) with w, w-degrees", [Poly(h, w).degree() for h in hs])
    picked, B, dts = select(sub, w, hs, kk)
    print(f"  selection: bound B = {B}, series+test {dts:.3f} s, picked {len(picked)}")
    assert len(picked) == 1
    G = normalise(picked[0], w, system.z, kk)
    if show:
        print("  G =", sp.factor(G.as_expr()) if len(str(G.as_expr())) < 400 else str(G.as_expr())[:400])
    for key, (f, rts) in exceptional(G.as_expr(), w, system.z, kk, lcs_E).items():
        print(f"  {key}: {f}  -> integer k >= 1: {rts}")
    return G


def resultant_path(name, system, w, other):
    print(f"\n=== {name} (resultant): w = {w}, eliminating {other}")
    P = [p.as_expr() for p in system.polynomials()]
    t = time.perf_counter()
    r = sp.resultant(P[0], P[1], other)
    dt = time.perf_counter() - t
    print(f"  resultant {dt:.3f} s:", sp.factor(r))


cases = sys.argv[1:] or ["catalan", "hg", "rs", "sym", "exc", "sat", "lk"]
F, G_, C, W, A, Bk = symbols("F G C W A B")
if "catalan" in cases:
    run("Catalan", System({F: 1 + z * F**2}, z=z), F)
if "hg" in cases:
    hg = System({Bk: k * z + 2 * z * Bk**2 + 2 * z * Bk, A: 2 * z + Bk}, z=z, k=k)
    run("Hitchhiker's Guide eqs. (7)-(9): A_k = 2z + B_k", hg, A)
    resultant_path("HG", hg, A, Bk)
if "rs" in cases:
    R, S = symbols("R S")
    rs = System({R: (k + 1) * z + 2 * z * R**2 + z * R, S: 4 * z * R * S + z * R + z * S}, z=z, k=k)
    run("Broda 2020 R_k", rs, R)
    run("Broda 2020 S_k", rs, S)
    resultant_path("Broda S_k", rs, S, R)
if "sym" in cases:
    sym = System({F: k * z + z * G_**2, G_: k * z + z * F**2}, z=z, k=k)
    run("symmetric F = kz + zG^2, G = kz + zF^2", sym, F)
    resultant_path("symmetric", System({G_: k * z + z * F**2, F: k * z + z * G_**2}, z=z, k=k), F, G_)
if "exc" in cases:
    run("degree drop F = 1 + (k-1) z F^2", System({F: 1 + (k - 1) * z * F**2}, z=z, k=k), F)
    exc = System({C: 1 + z * C**2, W: z + (k - 1) * z * C**2}, z=z, k=k)
    run("loses squarefreeness: W = z + (k-1) z C^2", exc, W)
    run("  ... at k = 1", exc.specialize(1), W)
    run("  ... at k = 2", exc.specialize(2), W)
if "sat" in cases:
    import warnings

    sat = System({F: z + z * G_ * F - z**2 * G_, G_: z + z * G_**2 - z**2 * G_}, z=z)
    run("excess component (1 - zG)", sat, F)
    P = [p.as_expr() for p in sat.polynomials()]
    Y = symbols("Y")
    Delta = sp.Matrix(P).jacobian([F, G_]).det()
    print("  det dP/dF =", sp.factor(Delta))
    gb = sp.groebner([*P, Y * Delta - 1], Y, G_, F, z, order="lex", method="f5b")
    print("  saturated elimination:", [g for g in gb.exprs if not g.has(Y, G_)])
if "lk" in cases:
    B, Bb, M, Mb, L = symbols("B Bb M Mb L")
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
    Gl = run("Broda 2020 L_k", ssnf, L, show=False)
    w = L
    pk = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
    eq17 = pk * w**3 + k * z * ((k + 12) * k * z**2 + 4 * k * z + 1) * w - k**2 * z**2 * (2 * z + 1)
    q = sp.cancel(Gl.as_expr() / eq17)
    print("  G / eq.(17) =", q)
    for n in (1, 2, 54):
        t = time.perf_counter()
        run(f"L_k at k = {n}", ssnf.specialize(n), L, show=False)
    run("Broda 2020 F_k = z + B + Bb", System({**{u: ssnf.equations[u] for u in (B, Bb)}, F: z + B + Bb}, z=z, k=k), F)
