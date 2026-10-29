# SPDX-License-Identifier: GPL-3.0-or-later
"""Brute-force enumerators for the Broda et al. 2020 classes (test helper).

Moved from ``dev/spikes/broda2020_enumerate.py``; the objects are described in
``dev/design/system.md``, section 3. Size = number of symbols, parentheses excluded.

- ``R_k``: syntax trees of grammar (1), ``r := eps | sigma | r + r | r . r | r*``;
- ``S_k``: total number of ``*`` over those trees;
- ``B_k`` / ``Bb_k``: nullable / non-nullable trees of grammar (9) (strong star normal
  form, without the lone expression ``eps``); ``M_k`` / ``Mb_k``: their total numbers of
  letters; ``L_k = M_k + Mb_k``.
"""

from functools import cache

# A tree is a tuple: ("eps",), ("a", i), ("+", l, r), (".", l, r), ("*", s), ("?", s).
Tree = tuple[object, ...]


@cache
def re_trees(n: int, k: int) -> tuple[Tree, ...]:
    """Trees of grammar (1) with ``n`` symbols over ``k`` letters."""
    out: list[Tree] = []
    if n == 1:
        out += [("eps",)] + [("a", i) for i in range(k)]
    if n >= 2:
        out += [("*", s) for s in re_trees(n - 1, k)]
    for i in range(1, n - 1):
        for left in re_trees(i, k):
            for right in re_trees(n - 1 - i, k):
                out += [("+", left, right), (".", left, right)]
    return tuple(out)


def count(t: Tree, tag: str) -> int:
    """Number of nodes with the given tag in ``t``."""
    children = (c for c in t[1:] if isinstance(c, tuple))
    return (t[0] == tag) + sum(count(c, tag) for c in children)


def nullable(t: Tree) -> bool:
    """Whether the language of ``t`` (no ``eps`` leaves) contains ``eps``."""
    tag = t[0]
    if tag == "a":
        return False
    if tag in ("*", "?"):
        return True
    left, right = t[1], t[2]
    assert isinstance(left, tuple) and isinstance(right, tuple)
    if tag == ".":
        return nullable(left) and nullable(right)
    return nullable(left) or nullable(right)


@cache
def ssnf_trees(n: int, k: int) -> tuple[Tree, ...]:
    """Trees of grammar (9) other than ``eps``: ``eps``-free, and every ``*`` / ``?``
    node has a non-nullable child."""
    out: list[Tree] = []
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


def broda_counts(k: int, n_max: int) -> dict[str, list[int]]:
    """Coefficients ``[z^0..z^n_max]`` of ``R, S, B, Bb, M, Mb, L`` by enumeration."""
    sizes = range(1, n_max + 1)

    def ssnf(pred: object, letters: bool) -> list[int]:
        return [0] + [
            sum(
                count(t, "a") if letters else 1
                for t in ssnf_trees(n, k)
                if pred is None or nullable(t) is pred
            )
            for n in sizes
        ]

    return {
        "R": [0] + [len(re_trees(n, k)) for n in sizes],
        "S": [0] + [sum(count(t, "*") for t in re_trees(n, k)) for n in sizes],
        "B": ssnf(True, False),
        "Bb": ssnf(False, False),
        "M": ssnf(True, True),
        "Mb": ssnf(False, True),
        "L": ssnf(None, True),
    }
