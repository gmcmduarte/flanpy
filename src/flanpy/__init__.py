# SPDX-License-Identifier: GPL-3.0-or-later
"""flanpy: analytic combinatorics for algebraic generating functions.

The package implements the pipeline used in descriptional-complexity papers
by Broda, Machiavelo, Moreira and Reis: from a grammar to a polynomial system,
elimination to a single algebraic curve ``G(z, w) = 0``, singularity analysis,
Newton-Puiseux expansions, and transfer theorems for coefficient asymptotics,
possibly for families indexed by the alphabet size ``k``.

Pre-alpha: the input type :class:`System`, its power-series coefficients
(:func:`series`, :func:`series_in_k`), and sets of alphabet sizes :class:`KSet` are
implemented so far.

Examples
--------
>>> from importlib.metadata import version
>>> import flanpy
>>> flanpy.__version__ == version("flanpy")
True
"""

from importlib.metadata import version as _version

from flanpy._kset import KSet
from flanpy._series import series, series_in_k
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
    "KSet",
    "System",
    "UndecidedSystemError",
    "WellPosednessError",
    "ZeroCoordinateWarning",
    "__version__",
    "series",
    "series_in_k",
]
