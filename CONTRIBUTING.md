# Contributing to flanpy

Contributions are welcome. Please open an issue to discuss larger changes first.

## Setup

```console
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Before opening a pull request

```console
ruff check .
ruff format --check .
mypy
pytest
sphinx-build -W --keep-going docs docs/_build/html
```

All of these must pass; CI runs the same commands.

## Conventions

- Exact arithmetic inside the algebraic pipeline; floating point only for
  evaluation and cross-checks in tests.
- Every algorithm cites its source (paper and equation number) in its
  docstring, and lands with a test that reproduces a number from that paper.
- NumPy-style docstrings, full type annotations (`mypy --strict`).
- Every `.py` file starts with `# SPDX-License-Identifier: GPL-3.0-or-later`.
- One logical change per commit, with a clear message.

By contributing you agree that your contributions are licensed under
GPL-3.0-or-later.
