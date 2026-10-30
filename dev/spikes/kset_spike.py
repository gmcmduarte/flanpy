# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike for dev/design/kset.md: sign of p in Q[k] at the integers k >= 1, exactly.

Algorithm (kset.md section 3): isolate the real roots >= 1 of p with SymPy's exact
VAS isolation (Poly.intervals), refine each isolating interval until it contains no
integer or the integer it contains is the root, then evaluate p exactly at one integer
per piece. Checks against brute force and times it.
"""
import math
import random
import time
from fractions import Fraction

import sympy as sp

k, z, w = sp.symbols("k z w")


def sign_pieces(p):
    """[(lo, hi | None, sign)] covering k >= 1, maximal pieces of constant sign."""
    P = sp.Poly(p, k, domain=sp.QQ)
    if P.is_zero:
        return [(1, None, 0)]
    coeffs = [Fraction(int(c.p), int(c.q)) for c in P.all_coeffs()]

    def sgn(n):
        v = Fraction(0)
        for c in coeffs:
            v = v * n + c
        return (v > 0) - (v < 0)

    # Integer roots >= 1, exactly (rational roots of p). Dividing them out leaves a
    # squarefree q with no integer root, so every isolating interval of q can be
    # refined until it holds no integer: no reliance on SymPy's endpoint convention
    # (an interval (20, 21) for the root 61/3 has the other root 20 as an endpoint).
    Q = P.sqf_part()
    int_roots = sorted(int(r) for r in Q.ground_roots() if r.is_integer and r >= 1)
    for n in int_roots:
        Q = Q.exquo(sp.Poly(k - n, k, domain=sp.QQ))
    roots = [("int", n) for n in int_roots]
    for (a, b), _mult in Q.intervals(inf=1):
        a, b = Fraction(int(a.p), int(a.q)), Fraction(int(b.p), int(b.q))
        while math.ceil(a) <= math.floor(b):  # [a, b] holds an integer
            a, b = (Fraction(int(x.p), int(x.q)) for x in
                    Q.refine_root(a, b, eps=(b - a) / 4))
        roots.append(("between", math.floor(a)))
    roots.sort(key=lambda r: (r[1], r[0] == "between"))
    pieces, lo = [], 1
    for n, is_int in [(n, kind == "int") for kind, n in roots]:
        if is_int:
            if lo <= n - 1:
                pieces.append((lo, n - 1))
            pieces.append((n, n))
            lo = n + 1
        elif lo <= n:
            pieces.append((lo, n))
            lo = n + 1
    pieces.append((lo, None))
    out = []
    for a, b in pieces:
        s = sgn(a)
        if out and out[-1][2] == s and out[-1][1] is not None and out[-1][1] + 1 == a:
            out[-1] = (out[-1][0], b, s)
        else:
            out.append((a, b, s))
    return out


def brute(p, M):
    """Exact signs at k = 1..M (Horner over Fraction; no lambdify, which can emit
    float division such as 3896750/3)."""
    cs = [Fraction(int(c.p), int(c.q)) for c in sp.Poly(p, k, domain=sp.QQ).all_coeffs()]
    out = []
    for n in range(1, M + 1):
        v = Fraction(0)
        for c in cs:
            v = v * n + c
        out.append((v > 0) - (v < 0))
    return out


def expand(pieces, M):
    out = []
    for a, b, s in pieces:
        hi = M if b is None else min(b, M)
        out += [s] * max(0, hi - a + 1)
    return out[:M]


def root_bound(p):
    """Fujiwara: every root has |r| <= 2 max |c_i / c_0|^(1/i)."""
    c = sp.Poly(p, k).all_coeffs()
    return 2 * max((float(abs(x / c[0])) ** (1 / i) for i, x in enumerate(c) if i), default=0)


# 1. Broda 2020 eq. (18)
p_k = 4 * k * (2 * k + 27) * z**3 + 36 * k * z**2 - k * z - 4
disc = sp.discriminant(p_k, z)
assert sp.factor(disc) == 32 * k**2 * (k - 54) ** 3
print("disc p_k =", sp.factor(disc), "->", sign_pieces(disc))
assert sign_pieces(disc) == [(1, 53, -1), (54, 54, 0), (55, None, 1)]

# 2. hand-made cases
cases = {
    "integer roots": (k - 2) * (k - 5),
    "between integers": 2 * k**2 - 7 * k + 4,  # roots (7 +- sqrt 17)/4 ~ 0.72, 2.78
    "double root at 3": (k - 3) ** 2 * (k - 10),
    "double between": (2 * k - 7) ** 2,
    "no real roots": k**2 + 1,
    "constant": sp.Integer(-3),
    "close roots": (k - sp.Rational(1001, 1000)) * (k - sp.Rational(1002, 1000)),
    "root at 1": k - 1,
    "roots below 1": (k + 5) * (2 * k - 1),
}
for name, p in cases.items():
    pcs = sign_pieces(p)
    M = int(root_bound(p)) + 5
    assert expand(pcs, M) == brute(p, M), name
    print(f"{name:18s} {pcs}")

# 3. random polynomials vs brute force (small coefficients, so Cauchy bound is small)
rng = random.Random(1)
for _ in range(1000):
    d = rng.randint(1, 6)
    roots = [sp.Rational(rng.randint(-20, 120), rng.choice([1, 1, 2, 3, 7])) for _ in range(d)]
    p = sp.expand(rng.choice([-1, 1]) * sp.prod([k - r for r in roots])
                  * (k**2 + rng.randint(0, 3)) ** rng.randint(0, 1))
    pcs = sign_pieces(p)
    M = int(root_bound(p)) + 3
    assert expand(pcs, M) == brute(p, M), p
print("1000 random polynomials agree with brute force")

# 4. timings
def t(p, label):
    t0 = time.perf_counter()
    pcs = sign_pieces(p)
    dt = time.perf_counter() - t0
    print(f"{label:46s} deg {int(sp.degree(p, k)):4d}  {dt:8.3f} s  pieces {len(pcs)}")

G = sp.Poly(sp.expand(sp.Poly(p_k, z).as_expr() * w**3
            + k * z * ((k + 12) * k * z**2 + 4 * k * z + 1) * w
            - k**2 * z**2 * (2 * z + 1)), w)
R = sp.resultant(G.as_expr(), sp.diff(G.as_expr(), w), w)  # in z, k
t(sp.Poly(R, z).LC(), "lc_z res_w(G, G_w), eq. (17)")
t(sp.discriminant(sp.sqf_part(R), z), "disc_z sqf res_w(G, G_w), eq. (17)")
for d in (10, 50, 100, 200):
    p = sum(rng.randint(-2**30, 2**30) * k**i for i in range(d + 1))
    t(p, f"random, 30-bit coefficients")
for d in (10, 30, 60):
    p = sp.expand(sp.prod([k - sp.Rational(2 * i + 1, 2) for i in range(d)]))
    t(p, "d roots at 1/2, 3/2, ... (all real, all >= 1 but one)")
p = sp.expand((k**2 - 2) * sp.prod([k - i for i in range(1, 41)]))
t(p, "40 integer roots, times k^2 - 2")
