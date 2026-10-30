# flanpy

**flanpy** is a Python implementation of the analytic-combinatorics toolchain
used to study the average descriptional complexity of regular languages and
related objects. It goes from a combinatorial specification to the asymptotics
of the coefficients of an algebraic generating function, possibly for a family
indexed by the alphabet size $k$.

```{warning}
flanpy is pre-alpha. Only the input type {class}`flanpy.System`, its
power-series coefficients ({func}`flanpy.series`, {func}`flanpy.series_in_k`)
and sets of alphabet sizes ({class}`flanpy.KSet`) are implemented so far, and
the public API is not stable.
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

## User guide

```{toctree}
:maxdepth: 1

guide/system
guide/kset
```

## Running examples

Each example is a Python script in `examples/` (run it with `python examples/<name>.py`)
and an executed page here. Every build-order step extends both with a section that
reproduces that step's results.

```{toctree}
:maxdepth: 1

examples/catalan
examples/broda2020
```

## Installation

```console
$ pip install flanpy
```

## API

```{eval-rst}
.. automodule:: flanpy

.. autoclass:: flanpy.System
   :members: unknowns, equations, z, k, polynomials, specialize

.. autoexception:: flanpy.WellPosednessError

.. autoexception:: flanpy.IllPosedSystemError

.. autoexception:: flanpy.UndecidedSystemError

.. autoexception:: flanpy.ZeroCoordinateWarning

.. autofunction:: flanpy.series

.. autofunction:: flanpy.series_in_k

.. autoclass:: flanpy.KSet
   :members: where, of, all, intervals, tail_start, is_finite, finite_part, isdisjoint
```

Well-posedness of a {class}`~flanpy.System` follows
{cite:t}`flajolet-sedgewick-2009` and {cite:t}`pivoteau-salvy-soria-2012`.

## References

```{bibliography}
```
