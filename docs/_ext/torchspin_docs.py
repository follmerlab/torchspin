"""Local Sphinx extension for the torchspin documentation site.

What it does:

* **Repository links.**  Markdown pages are written with ordinary relative
  links (``../KNOWN_LIMITATIONS.md``, ``../examples/fitting/foo.py``) so they
  work on GitHub and in an editor.  In the built site, links to files that
  are part of the site (Markdown pages in ``docs/``, the included README /
  KNOWN_LIMITATIONS / CHANGELOG, the tutorial notebooks) are pointed at those
  pages; links to any other repository file become GitHub links.  This is the
  Sphinx counterpart of telser-tools' ``docs/_hooks/repo_links.py``.
* **Tutorial notebooks.**  ``examples/notebooks/*.ipynb`` (committed,
  already executed) are copied into ``docs/tutorials/`` at build time
  (git-ignored) so myst-nb can render them without executing anything.
* **Optional pages.**  Toctree entries for pages listed in
  ``torchspin_optional_docs`` (pages that other branches add) are dropped
  while the page does not exist, so the build stays warning-free.
* **Constants.**  ``.. torchspin-constants::`` documents the physical
  constants of ``torchspin.constants`` at their public path
  (``torchspin.BMAGN``), with values and units read from the module source.
* **Public-API check.**  Every name in ``torchspin.__all__`` must be
  documented somewhere in the API reference; a missing one is a build
  warning.
* **Source links.**  :func:`source_url` resolves objects for
  ``sphinx.ext.linkcode``.
"""

from __future__ import annotations

import ast
import inspect
import json
import posixpath
import re
from pathlib import Path

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

GITHUB = "https://github.com/follmerlab/torchspin"
BLOB = f"{GITHUB}/blob/main/"
TREE = f"{GITHUB}/tree/main/"
RAW = "https://raw.githubusercontent.com/follmerlab/torchspin/main/"

# Repository files that are rendered as site pages: repo path -> page source
# path relative to docs/.  Links to them stay inside the site.
INCLUDED_PAGES = {
    "README.md": "getting_started.md",
    "KNOWN_LIMITATIONS.md": "known_limitations.md",
    "CHANGELOG.md": "changelog.md",
}
NOTEBOOK_SRC = "examples/notebooks"
NOTEBOOK_DST = "tutorials"

# [label](target#anchor) and ![alt](target); absolute URLs, in-page anchors,
# mail links and source-root links ("/x.md") are left alone.
_LINK = re.compile(r"(\]\()(?!https?:|mailto:|#|/)([^)\s#]+)(#[^)\s]*)?\)")


# ---------------------------------------------------------------------------
# Repository links
# ---------------------------------------------------------------------------


def rewrite_links(text: str, base: str, repo: Path, in_site=None) -> str:
    """Rewrite relative Markdown links in *text*.

    *base* is the directory, relative to the repository root and in POSIX
    form, of the file the text comes from (``""`` for the root, ``"docs"``
    for ``docs/index.md``).  *in_site(docname)* says whether a page under
    docs/ is part of the build (default: the file exists); pages that are
    not, such as the excluded planning notes, are linked on GitHub.
    """
    if in_site is None:
        def in_site(docname: str) -> bool:
            return any((repo / "docs" / (docname + ext)).is_file() for ext in (".md", ".ipynb", ".rst"))

    def fix(m: re.Match) -> str:
        opener, target, anchor = m.group(1), m.group(2), m.group(3) or ""
        is_image = _preceded_by_image(m)
        path = posixpath.normpath(posixpath.join(base, target))
        if path.startswith(".."):
            return m.group(0)  # points outside the repository: leave it
        if path in INCLUDED_PAGES:
            return f"{opener}/{INCLUDED_PAGES[path]}{anchor})"
        if path.startswith(NOTEBOOK_SRC + "/") and path.endswith(".ipynb"):
            name = posixpath.basename(path)
            return f"{opener}/{NOTEBOOK_DST}/{name}{anchor})"
        if path.startswith("docs/") and path.endswith((".md", ".ipynb", ".rst")):
            if in_site(posixpath.splitext(path[len("docs/"):])[0]):
                return f"{opener}/{path[len('docs/'):]}{anchor})"
        if is_image:
            return f"{opener}{RAW}{path})"
        prefix = TREE if (repo / path).is_dir() else BLOB
        return f"{opener}{prefix}{path}{anchor})"

    return _LINK.sub(fix, text)


def _preceded_by_image(m: re.Match) -> bool:
    """True if the link whose ``](`` starts at ``m.start()`` is ``![...]``."""
    s, i, depth = m.string, m.start() - 1, 0
    while i >= 0:
        c = s[i]
        if c == "]":
            depth += 1
        elif c == "[":
            if depth == 0:
                return i > 0 and s[i - 1] == "!"
            depth -= 1
        elif c == "\n" and i > 0 and s[i - 1] == "\n":
            return False  # paragraph break: give up
        i -= 1
    return False


def _in_site(app: Sphinx):
    return lambda docname: docname in app.env.found_docs


def _drop_missing_optional_entries(app: Sphinx, docname: str, text: str) -> str:
    """Remove toctree entries for optional pages that are not in this build.

    ``torchspin_optional_docs`` lists pages that other branches add
    (``installation``, ``physics_validation``, ...).  Their toctree entries
    are written in the Markdown already; until the page exists the entry is
    dropped instead of producing a "nonexisting document" warning.
    """
    optional = set(app.config.torchspin_optional_docs)
    if not optional or "{toctree}" not in text:
        return text
    here = posixpath.dirname(docname)
    out, in_toc = [], False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```{toctree}"):
            in_toc = True
        elif in_toc and stripped.startswith("```"):
            in_toc = False
        elif in_toc and stripped and not stripped.startswith(":"):
            target = posixpath.normpath(posixpath.join(here, stripped))
            if target in optional and target not in app.env.found_docs:
                continue
        out.append(line)
    return "\n".join(out)


def _on_source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    path = Path(app.env.doc2path(docname))
    if path.suffix != ".md":
        return
    rel = path.resolve().relative_to(_repo(app))
    text = _drop_missing_optional_entries(app, docname, source[0])
    source[0] = rewrite_links(text, posixpath.dirname(rel.as_posix()), _repo(app), _in_site(app))


def _on_include_read(app: Sphinx, relative_path, parent_docname, content: list[str]) -> None:
    path = (Path(app.srcdir) / relative_path).resolve()
    if path.suffix != ".md":
        return
    rel = path.relative_to(_repo(app))
    content[0] = rewrite_links(content[0], posixpath.dirname(rel.as_posix()), _repo(app), _in_site(app))


def _repo(app: Sphinx) -> Path:
    return Path(app.srcdir).resolve().parent


# ---------------------------------------------------------------------------
# Tutorial notebooks
# ---------------------------------------------------------------------------


def _copy_notebooks(app: Sphinx) -> None:
    """Copy the executed tutorial notebooks into docs/tutorials/.

    Markdown-cell links are rewritten relative to the notebook's original
    location.  A file is only rewritten when its content changes, so
    incremental builds stay incremental.
    """
    repo = _repo(app)
    src_dir = repo / NOTEBOOK_SRC
    dst_dir = Path(app.srcdir) / NOTEBOOK_DST
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(exist_ok=True)
    wanted = set()
    for src in sorted(src_dir.glob("*.ipynb")):
        nb = json.loads(src.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "markdown":
                text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                cell["source"] = rewrite_links(text, NOTEBOOK_SRC, repo)
        out = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
        dst = dst_dir / src.name
        wanted.add(dst.name)
        if not dst.exists() or dst.read_text(encoding="utf-8") != out:
            dst.write_text(out, encoding="utf-8")
    for stale in dst_dir.glob("*.ipynb"):
        if stale.name not in wanted:
            stale.unlink()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def read_constants(module) -> list[tuple[str, str, str]]:
    """``(name, value, comment)`` for each public constant of *module*.

    The comment is the block of ``#`` lines directly above the assignment in
    the module source.
    """
    source = Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")
    lines = source.splitlines()
    out = []
    for node in ast.parse(source).body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target = node.targets[0].id
        if not target or target.startswith("_"):
            continue
        comment = []
        i = node.lineno - 2
        while i >= 0 and lines[i].lstrip().startswith("#"):
            comment.insert(0, lines[i].lstrip()[1:].strip())
            i -= 1
        out.append((target, repr(getattr(module, target)), " ".join(comment)))
    return out


class ConstantsDirective(SphinxDirective):
    """Document ``torchspin.constants`` at the package's public path."""

    has_content = False

    def run(self):
        import torchspin

        import torchspin.constants

        path = Path(inspect.getsourcefile(torchspin.constants))
        self.env.note_dependency(str(path))
        public = set(torchspin.__all__)
        rows = [c for c in read_constants(torchspin.constants) if c[0] in public]
        rst = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 18 30 52",
            "",
            "   * - Name",
            "     - Value",
            "     - Meaning (unit)",
        ]
        for name, value, comment in rows:
            rst += [
                f"   * - :data:`~torchspin.{name}`",
                f"     - ``{value}``",
                f"     - {comment}",
            ]
        rst.append("")
        for name, value, comment in rows:
            rst += [
                f".. py:data:: {name}",
                "   :module: torchspin",
                "   :type: float",
                f"   :value: {value}",
                "",
                f"   {comment}" if comment else "",
                "",
            ]
        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(StringList(rst, source=str(path)), self.content_offset, node)
        return node.children


# ---------------------------------------------------------------------------
# Public-API completeness
# ---------------------------------------------------------------------------


def _check_public_api(app: Sphinx, env) -> None:
    import torchspin

    objects = env.get_domain("py").objects
    missing = [n for n in dict.fromkeys(torchspin.__all__) if f"torchspin.{n}" not in objects]
    for name in missing:
        logger.warning(
            "torchspin.%s is in torchspin.__all__ but not documented in the API reference",
            name,
            type="torchspin",
            subtype="undocumented",
        )


# ---------------------------------------------------------------------------
# Source links
# ---------------------------------------------------------------------------


def source_url(module: str, fullname: str, repo: Path, github: str, ref: str) -> str | None:
    """GitHub URL (with line range) of ``module.fullname``, or None."""
    import importlib

    try:
        obj = importlib.import_module(module)
        for part in fullname.split("."):
            obj = getattr(obj, part)
    except Exception:
        return None
    obj = inspect.unwrap(obj) if callable(obj) else obj
    if isinstance(obj, property):
        obj = obj.fget
    try:
        file = inspect.getsourcefile(obj)
        lines, start = inspect.getsourcelines(obj)
    except (TypeError, OSError):
        return None
    if not file:
        return None
    try:
        rel = Path(file).resolve().relative_to(repo.resolve())
    except ValueError:
        return None  # not part of this repository (e.g. torch)
    end = start + len(lines) - 1
    anchor = f"#L{start}-L{end}" if start else ""
    return f"{github}/blob/{ref}/{rel.as_posix()}{anchor}"


# ---------------------------------------------------------------------------


def setup(app: Sphinx):
    app.connect("builder-inited", _copy_notebooks)
    app.connect("source-read", _on_source_read)
    app.connect("include-read", _on_include_read)
    app.connect("env-check-consistency", _check_public_api)
    app.add_directive("torchspin-constants", ConstantsDirective)
    app.add_config_value("torchspin_optional_docs", [], "env")
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
