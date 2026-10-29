# SPDX-License-Identifier: GPL-3.0-or-later
"""flanpy: analytic combinatorics for algebraic generating functions.

The package implements the pipeline used in descriptional-complexity papers
by Broda, Machiavelo, Moreira and Reis: from a grammar to a polynomial system,
elimination to a single algebraic curve ``G(z, w) = 0``, singularity analysis,
Newton-Puiseux expansions, and transfer theorems for coefficient asymptotics,
possibly for families indexed by the alphabet size ``k``.

Pre-alpha: nothing is implemented yet; this is the package skeleton.

Examples
--------
>>> from importlib.metadata import version
>>> import flanpy
>>> flanpy.__version__ == version("flanpy")
True
"""

from importlib.metadata import version as _version

__version__: str = _version("flanpy")

__all__ = ["__version__"]
