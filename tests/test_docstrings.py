# SPDX-License-Identifier: GPL-3.0-or-later
"""Every public, non-trivial API member has an ``Examples`` section (numpydoc EX01).

Walks ``flanpy.__all__`` (and the public methods and properties of public classes)
and runs numpydoc's docstring validator, checking only EX01. The examples
themselves are run as doctests by ``pytest --doctest-modules``.
"""

import inspect

import pytest
from numpydoc.validate import validate

import flanpy

# Trivial members without an Examples section, with the reason.
ALLOWLIST = {
    "flanpy.__version__": "a string, not a callable; it has no docstring of its own",
    "flanpy.System.z": "returns the constructor's z unchanged",
    "flanpy.System.k": "returns the constructor's k (or None) unchanged",
}


def public_members() -> list[str]:
    names = []
    for name in flanpy.__all__:
        obj = getattr(flanpy, name)
        names.append(f"flanpy.{name}")
        # Exceptions and warnings: only the class itself (their members are inherited).
        if inspect.isclass(obj) and not issubclass(obj, BaseException):
            for member, value in vars(obj).items():
                if not member.startswith("_") and (
                    isinstance(value, property) or inspect.isfunction(value)
                ):
                    names.append(f"flanpy.{name}.{member}")
    return names


def test_validator_detects_a_missing_examples_section() -> None:
    # A private helper without examples: the check below would catch it.
    codes = [code for code, _ in validate("flanpy._series._series")["errors"]]
    assert "EX01" in codes


def test_allowlist_is_current() -> None:
    assert set(ALLOWLIST) <= set(public_members())


@pytest.mark.parametrize("name", public_members())
def test_has_examples(name: str) -> None:
    if name in ALLOWLIST:
        pytest.skip(ALLOWLIST[name])
    codes = [code for code, _ in validate(name)["errors"]]
    assert "EX01" not in codes, f"{name} has no Examples section"
