# SPDX-License-Identifier: GPL-3.0-or-later
"""Run the running examples (examples/*.py) as scripts, so CI fails if one breaks."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.py"))


def test_examples_found() -> None:
    assert {p.name for p in EXAMPLES} >= {"catalan.py", "broda2020.py"}


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-W", "error", str(script)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
