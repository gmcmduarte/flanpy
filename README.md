# flanpy

Analytic combinatorics for algebraic generating functions, as used in the
study of the average descriptional complexity of regular languages.

flanpy implements the toolchain of Broda, Machiavelo, Moreira and Reis:
grammar → polynomial system → elimination (Gröbner bases, resultants) →
dominant singularity → Newton–Puiseux expansion → coefficient asymptotics,
including families indexed by the alphabet size `k`. Its goal is that papers
can ship a runnable script that reproduces their results.

## Status

**Pre-alpha: no usable functionality yet.** The releases currently on PyPI
contain only the package skeleton; they exist to reserve the name. Nothing in
the pipeline above is implemented yet, and the API is not stable.

## Installation

Requires Python 3.13 or newer.

```console
pip install flanpy              # SymPy backend
pip install "flanpy[fast]"      # adds python-flint for faster polynomial arithmetic
```

From a clone, for development:

```console
pip install -e ".[dev,docs]"
```

## Licence

GPL-3.0-or-later. See [LICENSE](LICENSE). If you use flanpy in research,
please cite it (see [CITATION.cff](CITATION.cff)).
