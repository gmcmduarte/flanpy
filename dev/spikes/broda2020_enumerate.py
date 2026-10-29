# SPDX-License-Identifier: GPL-3.0-or-later
"""Spike: brute-force enumerators for the Broda et al. 2020 classes R_k, S_k, L_k.

Counts the objects described in dev/design/system.md (section 3) by explicit
enumeration of syntax trees, and compares them with the coefficients obtained by
iterating the transcribed polynomial systems:

- R_k, S_k: grammar (1), system on pp. 4-6 (S_k = d/du at u = 1, stars marked).
- L_k: grammar (9), eqs. (10), (11), (15), (16), (12); F_k = z + B_k + Bbar_k (p. 15).

Throwaway code (chunk 4 turns it into tests). Run: python dev/spikes/broda2020_enumerate.py
"""

from functools import cache

import sympy as sp

# Trees are tuples: ("eps",), ("a", i), ("+", l, r), (".", l, r), ("*", s), ("?", s).


@cache
def re_trees(n, k):
    """Grammar (1): r := eps | sigma | r + r | r . r | r*; size = number of symbols."""
    out = []
    if n == 1:
        out += [("eps",)] + [("a", i) for i in range(k)]
    if n >= 2:
        out += [("*", s) for s in re_trees(n - 1, k)]
    for i in range(1, n - 1):
        for left in re_trees(i, k):
            for right in re_trees(n - 1 - i, k):
                out += [("+", left, right), (".", left, right)]
    return tuple(out)


def count(t, tag):
    """Number of nodes with the given tag in tree t."""
    return (t[0] == tag) + sum(count(c, tag) for c in t[1:] if isinstance(c, tuple))


def nullable(t):
    """Whether the language of t contains the empty word (t has no eps leaf)."""
    tag = t[0]
    if tag == "a":
        return False
    if tag in "*?":
        return True
    if tag == ".":
        return nullable(t[1]) and nullable(t[2])
    return nullable(t[1]) or nullable(t[2])


@cache
def ssnf_trees(n, k):
    """Grammar (9) minus the lone expression eps: eps-free trees over sigma, +, ., *, ?
    in which every * and ? node has a non-nullable child."""
    out = []
    if n == 1:
        out += [("a", i) for i in range(k)]
    if n >= 2:
        for s in ssnf_trees(n - 1, k):
            if not nullable(s):
                out += [("*", s), ("?", s)]
    for i in range(1, n - 1):
        for left in ssnf_trees(i, k):
            for right in ssnf_trees(n - 1 - i, k):
                out += [("+", left, right), (".", left, right)]
    return tuple(out)


def series(equations, z, n_max):
    """Coefficients [z^0..z^n_max] by fixed-point iteration from 0 (k already substituted)."""
    y = {f: sp.Integer(0) for f in equations}
    for _ in range((n_max + 2) * (len(equations) + 1)):
        y = {
            f: sp.series(sp.expand(phi.subs(y, simultaneous=True)), z, 0, n_max + 1).removeO()
            for f, phi in equations.items()
        }
    return {f: [sp.expand(s).coeff(z, n) for n in range(n_max + 1)] for f, s in y.items()}


def main(n_max=7, ks=(1, 2)):
    z = sp.Symbol("z")
    R, S, B, Bb, M, Mb, L = sp.symbols("R S B Bb M Mb L")
    ok = True
    for k in ks:
        sec2 = {R: (k + 1) * z + 2 * z * R**2 + z * R, S: 4 * z * R * S + z * R + z * S}
        got = series(sec2, z, n_max)
        want = {
            R: [0] + [len(re_trees(n, k)) for n in range(1, n_max + 1)],
            S: [0] + [sum(count(t, "*") for t in re_trees(n, k)) for n in range(1, n_max + 1)],
        }
        ssnf = {
            B: 2 * z * B**2 + 2 * z * B * Bb + 2 * z * Bb,
            Bb: k * z + 2 * z * B * Bb + 2 * z * Bb**2,
            M: 4 * z * B * M + 2 * z * M * Bb + 2 * z * B * Mb + 2 * z * Mb,
            Mb: k * z + 2 * z * M * Bb + 2 * z * B * Mb + 4 * z * Bb * Mb,
            L: M + Mb,
        }
        got |= series(ssnf, z, n_max)

        def total(pred, weight, k=k):
            return [0] + [
                sum(weight(t) for t in ssnf_trees(n, k) if pred(t)) for n in range(1, n_max + 1)
            ]

        def letters(t):
            return count(t, "a")

        want |= {
            B: total(nullable, lambda t: 1),
            Bb: total(lambda t: not nullable(t), lambda t: 1),
            M: total(nullable, letters),
            Mb: total(lambda t: not nullable(t), letters),
            L: total(lambda t: True, letters),
        }
        for f in want:
            match = got[f] == want[f]
            ok &= match
            print(f"k={k} {f!s:2s} {'ok  ' if match else 'FAIL'} {want[f][1:]}")
    print("all match" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
