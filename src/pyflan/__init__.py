# SPDX-License-Identifier: GPL-3.0-or-later
"""pyflan: analytic combinatorics for algebraic generating functions.

The package implements the pipeline used in descriptional-complexity papers
by Broda, Machiavelo, Moreira and Reis: from a grammar to a polynomial system,
elimination to a single algebraic curve ``G(z, w) = 0``, singularity analysis,
Newton-Puiseux expansions, and transfer theorems for coefficient asymptotics,
possibly for families indexed by the alphabet size ``k``.

Pre-alpha: nothing is implemented yet; this is the package skeleton.

Examples
--------
>>> import pyflan
>>> pyflan.__version__
'0.0.1.dev0'
"""

from importlib.metadata import version as _version

__version__: str = _version("pyflan")

__all__ = ["__version__"]
