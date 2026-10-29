# SPDX-License-Identifier: GPL-3.0-or-later
"""flanpy: analytic combinatorics for algebraic generating functions.

The package implements the pipeline used in descriptional-complexity papers
by Broda, Machiavelo, Moreira and Reis: from a grammar to a polynomial system,
elimination to a single algebraic curve ``G(z, w) = 0``, singularity analysis,
Newton-Puiseux expansions, and transfer theorems for coefficient asymptotics,
possibly for families indexed by the alphabet size ``k``.

Pre-alpha: only the input type :class:`System` is implemented so far.

Examples
--------
>>> from importlib.metadata import version
>>> import flanpy
>>> flanpy.__version__ == version("flanpy")
True
"""

from importlib.metadata import version as _version

from flanpy._system import (
    IllPosedSystemError,
    System,
    UndecidedSystemError,
    WellPosednessError,
    ZeroCoordinateWarning,
)

__version__: str = _version("flanpy")

__all__ = [
    "IllPosedSystemError",
    "System",
    "UndecidedSystemError",
    "WellPosednessError",
    "ZeroCoordinateWarning",
    "__version__",
]
