# SPDX-License-Identifier: GPL-3.0-or-later
"""flanpy: analytic combinatorics for algebraic generating functions.

The package implements the pipeline used in descriptional-complexity papers
by Broda, Machiavelo, Moreira and Reis: from a grammar to a polynomial system,
elimination to a single algebraic curve ``G(z, w) = 0``, singularity analysis,
Newton-Puiseux expansions, and transfer theorems for coefficient asymptotics,
possibly for families indexed by the alphabet size ``k``.

Pre-alpha: implemented so far are the input type :class:`System`, its power-series
coefficients (:func:`series`, :func:`series_in_k`), sets of alphabet sizes
:class:`KSet` and piecewise results :class:`KCases`, elimination to a :class:`Curve`
(:func:`eliminate`, :func:`eliminate_in_k`), exact real algebraic numbers
(:class:`AlgebraicReal`), and the dominant singularity at a fixed ``k``
(:func:`singularity`).

Examples
--------
>>> from importlib.metadata import version
>>> import flanpy
>>> flanpy.__version__ == version("flanpy")
True
"""

from importlib.metadata import version as _version

from flanpy._algebraic import AlgebraicReal
from flanpy._curve import Curve
from flanpy._elimination import eliminate, eliminate_in_k
from flanpy._kset import KCases, KSet
from flanpy._series import series, series_in_k
from flanpy._singularity import (
    Singularity,
    SingularityWarning,
    UndecidedSingularityError,
    singularity,
)
from flanpy._system import (
    IllPosedSystemError,
    System,
    UndecidedSystemError,
    WellPosednessError,
    ZeroCoordinateWarning,
)

__version__: str = _version("flanpy")

__all__ = [
    "AlgebraicReal",
    "Curve",
    "IllPosedSystemError",
    "KCases",
    "KSet",
    "Singularity",
    "SingularityWarning",
    "System",
    "UndecidedSingularityError",
    "UndecidedSystemError",
    "WellPosednessError",
    "ZeroCoordinateWarning",
    "__version__",
    "eliminate",
    "eliminate_in_k",
    "series",
    "series_in_k",
    "singularity",
]
