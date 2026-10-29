# pyflan

**pyflan** is a Python implementation of the analytic-combinatorics toolchain
used to study the average descriptional complexity of regular languages and
related objects. It goes from a combinatorial specification to the asymptotics
of the coefficients of an algebraic generating function, possibly for a family
indexed by the alphabet size $k$.

```{warning}
pyflan is pre-alpha: it has no usable functionality yet. Nothing beyond the
package skeleton is implemented, and the public API is not stable.
```

## Pipeline

1. **Symbolic method**: from a grammar to a polynomial system for the
   generating functions.
2. **Elimination**: Gröbner bases and resultants, giving a single algebraic
   equation $G(z, w) = 0$.
3. **Singularity analysis**: dominant singularity $\rho$ and the value
   $a = w(\rho)$.
4. **Local expansion**: Newton polygon and Newton–Puiseux expansion at
   $(\rho, a)$.
5. **Transfer**: coefficient asymptotics
   $[z^n]\,w(z) \sim C \rho^{-n} n^{-\alpha-1}$
   {cite:p}`flajolet-sedgewick-2009`.
6. **Families in $k$**: expansions of $\rho_k$ and the constants in powers
   of $1/k$ {cite:p}`broda-2020-sigact`.

The methods follow {cite:t}`broda-2020-sigact` and
{cite:t}`broda-2014-hitchhiker`.

## Installation

```console
$ pip install pyflan
```

## API

```{eval-rst}
.. automodule:: pyflan
```

## References

```{bibliography}
```
