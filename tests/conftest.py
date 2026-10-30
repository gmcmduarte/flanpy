# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared pytest configuration."""

from hypothesis import settings

# Reproducible property-based tests: the same examples on every run (CI included),
# and no per-example deadline, which is flaky on shared runners.
settings.register_profile("flanpy", derandomize=True, deadline=None)
settings.load_profile("flanpy")
