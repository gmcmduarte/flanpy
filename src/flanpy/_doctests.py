# SPDX-License-Identifier: GPL-3.0-or-later
"""Doctest collection for the methods of the public classes (private helper).

The public classes report ``flanpy`` as their module. pytest's ``--doctest-modules``
then finds their class docstrings and properties through ``src/flanpy/__init__.py``,
but not their methods: doctest attributes a function to the module whose dict is its
``__globals__``, i.e. the private module, where the class itself is skipped because
its ``__module__`` is ``flanpy``. Each private module therefore lists the methods
(and only them, so nothing runs twice) in ``__test__``.
"""

from __future__ import annotations

import inspect
from typing import Any

__all__: list[str] = []


def method_doctests(*classes: type) -> dict[str, Any]:
    """Return the functions, classmethods and staticmethods of ``classes``.

    Keyed ``"Class.method"``, for a module's ``__test__``. Properties and class
    docstrings are left out: ``__init__.py`` collects them.
    """
    out: dict[str, Any] = {}
    for cls in classes:
        for name, value in vars(cls).items():
            if isinstance(value, classmethod | staticmethod):
                value = value.__func__
            if inspect.isfunction(value) and value.__doc__ and ">>>" in value.__doc__:
                out[f"{cls.__name__}.{name}"] = value
    return out
