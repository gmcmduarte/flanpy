# SPDX-License-Identifier: GPL-3.0-or-later
"""Sphinx configuration for the flanpy documentation."""

from __future__ import annotations

import flanpy

# -- Project information -----------------------------------------------------

project = "flanpy"
author = "Guilherme Duarte"
copyright = "2026, Guilherme Duarte"
release = flanpy.__version__
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",  # MyST Markdown pages and executed notebooks (loads myst_parser)
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- MyST / MyST-NB ----------------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
]
myst_dmath_double_inline = True
# Notebooks reproduce the papers' numbers, so they are always executed and a
# failing cell fails the build.
nb_execution_mode = "force"
nb_execution_raise_on_error = True
nb_execution_timeout = 600

# -- Maths -------------------------------------------------------------------

math_numfig = True
math_eqref_format = "({number})"
numfig = True

# -- autodoc / napoleon ------------------------------------------------------

autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sympy": ("https://docs.sympy.org/latest", None),
    "mpmath": ("https://mpmath.org/doc/current", None),
}

# -- Bibliography ------------------------------------------------------------

bibtex_bibfiles = ["refs.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

# -- HTML output -------------------------------------------------------------

html_theme = "shibuya"
html_title = "flanpy"
html_theme_options = {
    "github_url": "https://github.com/gmcmduarte/flanpy",
}
html_context = {
    "source_type": "github",
    "source_user": "gmcmduarte",
    "source_repo": "flanpy",
    "source_version": "main",
    "source_docs_path": "/docs/",
}
