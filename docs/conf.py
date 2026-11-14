"""Sphinx configuration for the torchspin documentation site.

Build locally (from the repository root)::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

The source directory is ``docs/`` itself, so the Markdown pages that live
there (``index.md``, ``pitfalls.md``, ``lft.md``, ...) are the site's pages
and stay readable on GitHub.  Files that live elsewhere in the repository
(``README.md``, ``KNOWN_LIMITATIONS.md``, ``CHANGELOG.md``, the tutorial
notebooks in ``examples/notebooks/``) are pulled in by ``_ext/torchspin_docs.py``.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent

# Document the source tree next to this file, not whatever torchspin happens
# to be installed in the environment.
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DOCS / "_ext"))


def _project_version() -> str:
    """Version from pyproject.toml, the single source of truth.

    Read directly so the build does not depend on installed package
    metadata (an editable install, a PYTHONPATH checkout and a wheel all
    give the same answer).
    """
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if m is None:  # pragma: no cover - pyproject always carries a version
        import torchspin

        return torchspin.__version__
    return m.group(1)


# -- Project information -----------------------------------------------------

project = "torchspin"
author = "Alec Follmer"
copyright = "2026, Alec Follmer"
release = _project_version()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",  # Markdown (MyST) pages and Jupyter notebooks
    "torchspin_docs",  # local: repository links, notebooks, constants, API checks
]

language = "en"
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "_ext",
    "_templates",
    "scripts",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    # Internal notes kept in the repository but not published: planning
    # documents and the docstring inventory (docs/scripts/api_inventory.py).
    "differentiable_roadmap.md",
    "real_pulse_plan.md",
    "v1_acceleration",
    "api_inventory.md",
]
# Pages that other branches add.  Their toctree entries are already in place
# and are dropped by _ext/torchspin_docs.py until the page exists.
torchspin_optional_docs = ["installation", "physics_validation", "chili_empirical_rules"]

# -- MyST (Markdown) and notebooks -------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "amsmath",
]
# GitHub-style heading anchors, so ``page.md#some-heading`` links work both on
# GitHub and in the site.
myst_heading_anchors = 4
# The tutorial notebooks are committed executed; never re-run them in a build.
nb_execution_mode = "off"
nb_merge_streams = True

# -- autodoc / autosummary / napoleon ----------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_class_signature = "mixed"
autodoc_preserve_defaults = True
# Long signatures (SpinSystem, Experiment, Options) flow as a paragraph rather
# than one parameter per line, so the parameter table stays near the top.
# Show ``torchspin.pepper`` rather than ``torchspin.pepper.pepper`` everywhere.
add_module_names = True
python_use_unqualified_type_names = True

napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True
# Keep trailing underscores in parameter names (SpinSystem.D_, A_) visible.
strip_signature_backslash = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}
intersphinx_timeout = 10

# -- Source links (sphinx.ext.linkcode) --------------------------------------

GITHUB = "https://github.com/follmerlab/torchspin"
# CI builds link to the exact commit; local builds link to main.
GITHUB_REF = os.environ.get("GITHUB_SHA") or "main"


def linkcode_resolve(domain, info):
    """GitHub URL of the source lines of a documented object."""
    from torchspin_docs import source_url

    if domain != "py" or not info.get("module"):
        return None
    return source_url(info["module"], info["fullname"], REPO, GITHUB, GITHUB_REF)


# -- HTML output -------------------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_title = f"torchspin {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_copy_source = False
html_theme_options = {
    "github_url": GITHUB,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}
html_context = {
    "github_user": "follmerlab",
    "github_repo": "torchspin",
    "github_version": "main",
    "doc_path": "docs",
}

# Copy buttons: strip prompts so ``>>>`` examples paste cleanly.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False
