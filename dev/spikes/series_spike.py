# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike: power-series coefficients of polynomial systems (design: dev/design/series.md).

Algorithms (all start from the stabilised constant terms c* and use the constant
matrix (I - J0)^-1, which exists because J0 is nilpotent):

- (a) fixed point:   Y <- Phi(z, Y) mod z^(N+1) until it stops changing;
- (b) recurrence:    (I - J0) y_n = r_n(y_0..y_{n-1}), products updated online;
- (c) Newton:        Y <- Y + U (Phi(Y) - Y), U <- U + U (I - (I - J(Y)) U),
                     doubling the precision (Pivoteau-Salvy-Soria 2012, sec. 6 / 8.2).

Backends: pure Python (int coefficients, naive truncated products) and python-flint
(fmpz scalars / fmpz_poly with mul_low). Symbolic k: (b) with coefficients in Z[k],
as SymPy ring elements (ZZ[k]) or flint fmpz_poly in k.

Only integer systems with c* = 0 (the Broda 2020 ones) are handled; this is a
measurement spike, not the implementation. Run:

    python dev/spikes/series_spike.py            # full timings (several minutes)
    python dev/spikes/series_spike.py --quick    # cross-checks only
"""

import argparse
import platform
import subprocess
import sys
import time

import flint
import sympy as sp

z, k = sp.symbols("z k")
R, S, B, Bb, M, Mb, L = sp.symbols("R S B Bb M Mb L")

SYSTEMS = {
    "R": {R: (k + 1) * z + 2 * z * R**2 + z * R},
    "RS": {R: (k + 1) * z + 2 * z * R**2 + z * R, S: 4 * z * R * S + z * R + z * S},
    "L": {
        B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
        Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
        M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
        Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
        L: M + Mb,
    },
}
# From the brute-force enumerator (dev/spikes/broda2020_enumerate.py), n = 1..7, k = 2.
CHECK_K2 = {
    R: [3, 3, 21, 57, 327, 1263, 6753],
    S: [0, 3, 6, 63, 228, 1635, 7578],
    L: [2, 4, 16, 96, 320, 1920, 8320],
}


# -- compile a system to terms (coef, a, exps) ------------------------------------


def compile_terms(eqs, to_scalar, kval=None):
    unknowns = list(eqs)
    m = len(unknowns)
    out = []
    for f in unknowns:
        phi = eqs[f] if kval is None else eqs[f].subs(k, kval)
        poly = sp.Poly(sp.expand(phi), *unknowns, z)
        terms = []
        for monom, coef in poly.terms():
            exps = tuple(monom[:m])
            terms.append((to_scalar(coef), monom[m], exps))
        out.append(terms)
    # J0 at c* = 0: only monomials with a = 0 and total degree 1 contribute.
    j0 = [[0] * m for _ in range(m)]
    for i, terms in enumerate(out):
        for coef, a, exps in terms:
            if a == 0 and sum(exps) == 1:
                j0[i][exps.index(1)] = coef
    return unknowns, out, j0


def inverse_unipotent(j0, zero, one):
    """(I - J0)^-1 = I + J0 + ... + J0^(m-1) for nilpotent J0."""
    m = len(j0)
    ident = [[one if i == j else zero for j in range(m)] for i in range(m)]

    def matmul(a, b):
        return [[sum((a[i][t] * b[t][j] for t in range(m)), zero) for j in range(m)] for i in range(m)]

    total, power = ident, ident
    for _ in range(m - 1):
        power = matmul(power, j0)
        total = [[total[i][j] + power[i][j] for j in range(m)] for i in range(m)]
    return total


# -- (b) coefficient-by-coefficient recurrence ---------------------------------------


def recurrence(terms, m, n_max, zero, one):
    j0 = [[zero] * m for _ in range(m)]
    for i, ts in enumerate(terms):
        for coef, a, exps in ts:
            if a == 0 and sum(exps) == 1:
                j0[i][exps.index(1)] = coef
    minv = inverse_unipotent(j0, zero, one)
    # Product nodes: sorted tuples of unknown indices of length >= 2.
    keys = set()
    for ts in terms:
        for _, _, exps in ts:
            key = tuple(j for j in range(m) for _ in range(exps[j]))
            for length in range(2, len(key) + 1):
                keys.add(key[:length])
    nodes = sorted(keys, key=len)
    y = [[zero] for _ in range(m)]  # c* = 0
    node = {t: [zero] for t in nodes}

    def series_of(key):
        return y[key[0]] if len(key) == 1 else node[key]

    term_keys = [
        [(coef, a, tuple(j for j in range(m) for _ in range(exps[j]))) for coef, a, exps in ts]
        for ts in terms
    ]
    for n in range(1, n_max + 1):
        partial, sums = {}, {}
        for t in nodes:
            left, right = series_of(t[:-1]), y[t[-1]]
            s = zero
            for i in range(1, n):
                s += left[i] * right[n - i]
            sums[t] = s
            left_n = partial[t[:-1]] if len(t) > 2 else zero
            partial[t] = s + left_n * right[0]
        r = []
        for tks in term_keys:
            acc = zero
            for coef, a, key in tks:
                if not key:
                    if a == n:
                        acc += coef
                elif a == 0:
                    if len(key) > 1:
                        acc += coef * partial[key]
                elif a <= n:
                    acc += coef * series_of(key)[n - a]
            r.append(acc)
        for i in range(m):
            y[i].append(sum((minv[i][j] * r[j] for j in range(m)), zero))
        for t in nodes:
            left, right = series_of(t[:-1]), y[t[-1]]
            node[t].append(sums[t] + left[n] * right[0] + left[0] * right[n])
    return y


# -- series backends for (a) and (c) -------------------------------------------------


class PySeries:
    name = "python"

    @staticmethod
    def const(c, q):
        return [c] + [0] * (q - 1)

    @staticmethod
    def zero(q):
        return [0] * q

    @staticmethod
    def add(a, b):
        return [x + y for x, y in zip(a, b)]

    @staticmethod
    def sub(a, b):
        return [x - y for x, y in zip(a, b)]

    @staticmethod
    def scale(c, a):
        return [c * x for x in a]

    @staticmethod
    def shift(a, s, q):
        return ([0] * s + a)[:q]

    @staticmethod
    def mul(a, b, q):
        out = [0] * q
        nza = [(i, x) for i, x in enumerate(a[:q]) if x]
        nzb = [(j, y) for j, y in enumerate(b[:q]) if y]
        for i, x in nza:
            for j, y in nzb:
                if i + j >= q:
                    break
                out[i + j] += x * y
        return out

    @staticmethod
    def extend(a, q):
        return a + [0] * (q - len(a))

    @staticmethod
    def coeffs(a, q):
        return list(a[:q]) + [0] * (q - len(a))

    @staticmethod
    def equal(a, b):
        return a == b


class FlintSeries:
    name = "flint"

    @staticmethod
    def const(c, q):
        return flint.fmpz_poly([c])

    @staticmethod
    def zero(q):
        return flint.fmpz_poly([])

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def scale(c, a):
        return a * c

    @staticmethod
    def shift(a, s, q):
        return (a * flint.fmpz_poly([0] * s + [1])).truncate(q) if s else a.truncate(q)

    @staticmethod
    def mul(a, b, q):
        return a.mul_low(b, q)

    @staticmethod
    def extend(a, q):
        return a

    @staticmethod
    def coeffs(a, q):
        cs = [int(c) for c in a.coeffs()][:q]
        return cs + [0] * (q - len(cs))

    @staticmethod
    def equal(a, b):
        return a == b


def evaluate(be, terms, m, Y, q, want_jacobian):
    """Phi(Y) and optionally J(Y) mod z^q, with memoised monomial products."""
    cache = {}

    def prod(key):
        if not key:
            return be.const(1, q)
        if len(key) == 1:
            return Y[key[0]]
        if key not in cache:
            cache[key] = be.mul(prod(key[:-1]), Y[key[-1]], q)
        return cache[key]

    def mono(coef, a, key):
        return be.scale(coef, be.shift(prod(key), a, q))

    phi = []
    for ts in terms:
        acc = be.zero(q)
        for coef, a, exps in ts:
            key = tuple(j for j in range(m) for _ in range(exps[j]))
            acc = be.add(acc, mono(coef, a, key))
        phi.append(acc)
    if not want_jacobian:
        return phi, None
    jac = [[None] * m for _ in range(m)]
    for i, ts in enumerate(terms):
        for coef, a, exps in ts:
            for j in range(m):
                if exps[j]:
                    e = list(exps)
                    e[j] -= 1
                    key = tuple(t for t in range(m) for _ in range(e[t]))
                    d = mono(coef * exps[j], a, key)
                    jac[i][j] = d if jac[i][j] is None else be.add(jac[i][j], d)
    return phi, jac


def fixed_point(be, terms, m, n_max):
    q = n_max + 1
    Y = [be.zero(q) for _ in range(m)]
    passes = 0
    while True:
        new, _ = evaluate(be, terms, m, Y, q, False)
        passes += 1
        if all(be.equal(be.extend(a, q), be.extend(b, q)) for a, b in zip(new, Y)):
            return [be.coeffs(s, q) for s in Y], passes
        Y = new


def newton(be, terms, m, n_max):
    j0 = [[0] * m for _ in range(m)]
    for i, ts in enumerate(terms):
        for coef, a, exps in ts:
            if a == 0 and sum(exps) == 1:
                j0[i][exps.index(1)] = coef
    minv = inverse_unipotent(j0, 0, 1)
    prec = 1
    Y = [be.zero(1) for _ in range(m)]  # c* = 0
    U = [[be.const(minv[i][j], 1) for j in range(m)] for i in range(m)]
    while prec < n_max + 1:
        q = min(2 * prec, n_max + 1)
        Y = [be.extend(s, q) for s in Y]
        U = [[be.extend(s, q) for s in row] for row in U]
        phi, jac = evaluate(be, terms, m, Y, q, True)
        # A = I - J(Y); E = I - A U = J U - (U - I)  ->  U <- U + U E
        AU = [[None] * m for _ in range(m)]
        for i in range(m):
            for j in range(m):
                acc = U[i][j]
                for t in range(m):
                    if jac[i][t] is not None:
                        acc = be.sub(acc, be.mul(jac[i][t], U[t][j], q))
                AU[i][j] = acc
        E = [[be.sub(be.const(1 if i == j else 0, q), AU[i][j]) for j in range(m)] for i in range(m)]
        U = [
            [
                be.add(U[i][j], _dot(be, [U[i][t] for t in range(m)], [E[t][j] for t in range(m)], q))
                for j in range(m)
            ]
            for i in range(m)
        ]
        delta = [be.sub(phi[i], Y[i]) for i in range(m)]
        Y = [be.add(Y[i], _dot(be, U[i], delta, q)) for i in range(m)]
        prec = q
    return [be.coeffs(s, n_max + 1) for s in Y]


def _dot(be, xs, ys, q):
    acc = be.zero(q)
    for x, y in zip(xs, ys):
        acc = be.add(acc, be.mul(x, y, q))
    return acc


# -- driver ------------------------------------------------------------------------


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def fixed_k(system, kval, n_max, algos):
    unknowns, terms_int, _ = compile_terms(SYSTEMS[system], int, kval)
    _, terms_fmpz, _ = compile_terms(SYSTEMS[system], lambda c: flint.fmpz(int(c)), kval)
    m = len(unknowns)
    results, times = {}, {}
    for algo in algos:
        if algo == "b-python":
            out, t = timed(recurrence, terms_int, m, n_max, 0, 1)
        elif algo == "b-flint":
            out, t = timed(recurrence, terms_fmpz, m, n_max, flint.fmpz(0), flint.fmpz(1))
            out = [[int(c) for c in s] for s in out]
        elif algo == "c-flint":
            out, t = timed(newton, FlintSeries, terms_int, m, n_max)
        elif algo == "c-python":
            out, t = timed(newton, PySeries, terms_int, m, n_max)
        elif algo == "a-flint":
            (out, passes), t = timed(fixed_point, FlintSeries, terms_int, m, n_max)
        elif algo == "a-python":
            (out, passes), t = timed(fixed_point, PySeries, terms_int, m, n_max)
        results[algo] = out
        times[algo] = t
    first = next(iter(results.values()))
    for algo, out in results.items():
        assert out == first, f"{system} k={kval} N={n_max}: {algo} disagrees"
    return dict(zip(unknowns, first)), times


def symbolic(system, n_max, backend):
    if backend == "flint":
        to_scalar = _flint_poly
        zero, one = flint.fmpz_poly([]), flint.fmpz_poly([1])
    else:
        ring, kk = sp.ring("k", sp.ZZ)

        def to_scalar(c):
            return ring(sp.Poly(c, k).as_expr())

        zero, one = ring(0), ring(1)
    unknowns, terms, _ = compile_terms(SYSTEMS[system], to_scalar)
    out, t = timed(recurrence, terms, len(unknowns), n_max, zero, one)
    return dict(zip(unknowns, out)), t


def _flint_poly(c):
    coeffs = sp.Poly(c, k).all_coeffs()[::-1]
    return flint.fmpz_poly([int(x) for x in coeffs])


def machine():
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        cpu = platform.processor()
    return f"{cpu}, {platform.system()} {platform.release()}, Python {platform.python_version()}, python-flint {flint.__version__}, SymPy {sp.__version__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--sections",
        default="fixed,slow,large,symbolic",
        help="comma-separated subset of: fixed, slow, large, symbolic",
    )
    args = parser.parse_args()
    sections = set(args.sections.split(","))
    print(machine())

    # Cross-checks: every algorithm and backend agree, and match the enumerator.
    for system in SYSTEMS:
        res, _ = fixed_k(system, 2, 40, ["b-python", "b-flint", "c-flint", "c-python", "a-flint", "a-python"])
        for f, want in CHECK_K2.items():
            if f in res:
                assert res[f][1:8] == want, (system, f, res[f][1:8])
        sym, _ = symbolic(system, 30, "python")
        symf, _ = symbolic(system, 30, "flint")
        for f in sym:
            for kv in (1, 2, 3, 7):
                fk, _ = fixed_k(system, kv, 30, ["b-python"])
                assert [c(kv) for c in sym[f]] == fk[f], (system, f, kv)
                assert [int(c(kv)) for c in symf[f]] == fk[f], (system, f, kv)
    print("cross-checks passed (all algorithms/backends agree; enumerator values; symbolic = fixed k)")
    if args.quick:
        return

    if "fixed" in sections:
        section_fixed()
    if "slow" in sections:
        section_slow()
    if "large" in sections:
        section_large()
    if "symbolic" in sections:
        section_symbolic()


def section_fixed():
    print("\n# fixed k, N = 1000 (seconds)")
    print(f"{'system':6} {'k':>3} {'b-python':>9} {'b-flint':>9} {'c-flint':>9}  bits([z^1000])")
    for system in SYSTEMS:
        for kval in (2, 10, 50):
            res, times = fixed_k(system, kval, 1000, ["b-python", "b-flint", "c-flint"])
            last = list(res.values())[-1][1000]
            print(
                f"{system:6} {kval:>3} {times['b-python']:9.2f} {times['b-flint']:9.2f} "
                f"{times['c-flint']:9.2f}  {last.bit_length()}"
            )
            sys.stdout.flush()



def section_slow():
    print("\n# scaling of the slow variants, system L, k = 10 (seconds)")
    for n_max in (125, 250, 500):
        _, times = fixed_k("L", 10, n_max, ["c-python", "a-flint", "a-python"] if n_max <= 250 else ["c-python", "a-flint"])
        print(f"N = {n_max:4}: " + ", ".join(f"{a} {t:.2f}" for a, t in times.items()))
        sys.stdout.flush()



def section_large():
    print("\n# larger N, system L, k = 10 (seconds): where Newton (c) pulls ahead")
    for n_max in (1000, 2000, 4000):
        algos = ["b-flint", "c-flint"] + (["b-python"] if n_max <= 2000 else [])
        _, times = fixed_k("L", 10, n_max, algos)
        print(f"N = {n_max:4}: " + ", ".join(f"{a} {t:.2f}" for a, t in times.items()))
        sys.stdout.flush()
    # Cost of converting flint results to Python ints (the public result type).
    _, terms, _ = compile_terms(SYSTEMS["L"], lambda c: flint.fmpz(int(c)), 10)
    out = recurrence(terms, 5, 1000, flint.fmpz(0), flint.fmpz(1))
    _, t = timed(lambda: [[int(c) for c in s] for s in out])
    print(f"converting 5 x 1001 fmpz coefficients to int: {t * 1000:.1f} ms")


def section_symbolic():
    print("\n# symbolic k, recurrence (b) (seconds; '-' = not run)")
    for system in SYSTEMS:
        for n_max in (25, 50, 100, 200, 400, 1000):
            tp = symbolic(system, n_max, "python")[1] if n_max <= 200 else None
            tf = symbolic(system, n_max, "flint")[1] if n_max <= 400 or system == "R" else None
            fmt = lambda t: f"{t:8.2f}" if t is not None else f"{'-':>8}"
            print(f"{system:6} N = {n_max:4}: python(ZZ[k]) {fmt(tp)}   flint(fmpz_poly) {fmt(tf)}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
