# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike: (1) Newton-Puiseux over number fields, (2) alpha via derivative support + gcd with m_rho.
Targets: Broda et al. 2020, eqs. (32), (35), (36)."""
import sympy as sp
from sympy import Rational as R, symbols, sqrt
from fractions import Fraction

def iszero(c):
    c = sp.expand(c)
    return c == 0 or sp.simplify(c) == 0

def lower_hull(pts):
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) > (p[1] - y1) * (x2 - x1):   # strictly above the chord
                hull.pop()
            else:
                break
        hull.append(p)
    return hull

def np_roots(G, nterms):
    """G: {(i,j): coeff} = sum coeff*x^j*y^i (j>=0 int). Roots y(x) of positive valuation,
    as lists [(coef, Rational exponent)] with nterms terms."""
    jmin = {}
    for (i, j), a in G.items():
        if not iszero(a):
            jmin[i] = min(jmin.get(i, 10**9), j)
    pts = sorted(jmin.items())
    pts = [(i, j) for i, j in pts]
    coef = lambda i, j: G[(i, j)]
    out = []
    if pts[0][0] > 0:
        out.append([])                       # y = 0 is a root (terminating series)
    hull = lower_hull(pts)
    groups = []
    for a, b in zip(hull, hull[1:]):
        sl = Fraction(b[1] - a[1], b[0] - a[0])
        if sl >= 0:
            continue
        if groups and groups[-1][0] == sl:
            groups[-1][1].append(b)
        else:
            groups.append([sl, [a, b]])
    c = sp.Symbol('c')
    for sl, seg in groups:
        gam = -sl; p, q = gam.numerator, gam.denominator
        i0 = seg[0][0]
        cp = sum(coef(i, j) * c**(i - i0) for i, j in seg)
        rts = sp.roots(sp.Poly(sp.expand(cp), c))
        assert sum(rts.values()) == sp.Poly(sp.expand(cp), c).degree(), "char. poly roots not in radicals"
        for c0, mult in rts.items():
            if iszero(c0):
                continue
            if nterms == 1:
                out.append([(c0, R(p, q))]); continue
            s, y1 = sp.symbols('s y1')
            expr = sum(a * s**(q * j + p * i) * (c0 + y1)**i for (i, j), a in G.items())
            P = sp.Poly(sp.expand(expr), s, y1)
            terms = {m: sp.nsimplify(co) if False else co for m, co in P.terms() if not iszero(co)}
            beta = min(m[0] for m in terms)
            G1 = {(m[1], m[0] - beta): co for m, co in terms.items()}
            for ys in np_roots(G1, nterms - 1):
                out.append([(c0, R(p, q))] + [(cc, R(p, q) + e / q) for cc, e in ys])
    return out

x, z, k, t = symbols('x z k t', positive=True)

# ---------- (2) Puiseux expansion of rho_k in 1/k  (paper eq. (35)) ----------
p_k = 4*k*(2*k+27)*z**3 + 36*k*z**2 - k*z - 4
P = sp.expand(x**2 * p_k.subs(k, 1/x))          # x = 1/k, cleared denominators
Pp = sp.Poly(P, z, x)
G = {(i, j): co for (i, j), co in Pp.terms()}
roots = np_roots(G, 3)
print("branches found:", len(roots))
def ser(r): return sum(co * x**e for co, e in r)
for r in roots:
    print("  ", sp.nsimplify(ser(r)))
paper35 = x**R(1,2)/sqrt(8) - x/4 + 3*x**R(3,2)/(4*sqrt(2))
match = [r for r in roots if iszero(ser(r) - paper35)]
print("matches paper eq (35):", bool(match))

# ---------- c_k  (paper eq. (34)-(36)) ----------
rho = ser(match[0]).subs(x, t**2)               # t = k^{-1/2}
kk = 1/t**2
pprime = 12*kk*(2*kk+27)*rho**2 + 72*kk*rho - kk
c2 = pprime / (kk*((kk+12)*kk*rho**2 + 4*kk*rho + 1))
print("c_k^2 series:", sp.series(c2, t, 0, 4))
print("c_k   series:", sp.series(sp.sqrt(c2), t, 0, 3), " (paper: 4*k^(-1/2)=4t)")

# ---------- (1) alpha from derivative support (paper eq. (32), (22)-(27)) ----------
v = sp.Symbol('v')
H = k**2*z**2*(2*z+1)*v**3 - k*z*((k+12)*k*z**2 + 4*k*z + 1)*v**2 - p_k
K = sp.QQ.frac_field(k)
mp = sp.Poly(p_k, z, domain=K)
supp = []
for i in range(0, 4):
    for j in range(0, 4):
        D = sp.diff(H, z, i, v, j).subs(v, 0)
        if sp.expand(D) == 0:
            status = 'zero'
        else:
            g = sp.gcd(sp.Poly(D, z, domain=K), mp)
            status = 'nonzero' if g.degree() == 0 else 'inconclusive'
            if status != 'zero' and (i, j) not in [(0, 0), (0, 1)]:
                supp.append((j - 1, i))
        print(f"  d^{i}_z d^{j}_v H at (rho,0): {status}")
best = {}
for xx, yy in supp:
    best[xx] = min(best.get(xx, yy), yy)
supp = sorted(best.items())
hull = lower_hull(supp)
slopes = [Fraction(b[1]-a[1], b[0]-a[0]) for a, b in zip(hull, hull[1:])]
print("support points (j-1, i):", supp)
print("lower hull:", hull, "slopes:", slopes, "=> alpha =", -min(slopes))
