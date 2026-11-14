"""Write docs/api_inventory.md: every public torchspin name and the state of its docstring.

Usage (from the repository root)::

    python docs/scripts/api_inventory.py                  # writes docs/api_inventory.md
    python docs/scripts/api_inventory.py --log build.log  # also attribute Sphinx docstring warnings
    python docs/scripts/api_inventory.py --check          # exit 1 if a public name has no API page

The group of a name is the API reference page that documents it, read from
docs/api/*.rst, so the inventory and the site cannot disagree.  The docstring
checks are textual (NumPy-style section headers, or Google-style ``Args:``),
so they say whether a section exists, not whether it is complete; the
"undocumented parameters" column compares the signature with the names listed
under Parameters/Attributes.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
REPO = DOCS.parent
sys.path.insert(0, str(REPO))

import torchspin  # noqa: E402

API = DOCS / "api"
OUT = DOCS / "api_inventory.md"

_SECTION = r"^[ \t]*{name}[ \t]*\n[ \t]*-{{3,}}[ \t]*$"
_GOOGLE = r"^[ \t]*{name}:[ \t]*$"


def has_section(doc: str, *names: str) -> bool:
    for name in names:
        if re.search(_SECTION.format(name=name), doc, re.MULTILINE):
            return True
        if re.search(_GOOGLE.format(name=name), doc, re.MULTILINE):
            return True
    return False


def section_body(doc: str, *names: str) -> str:
    """Text of the first NumPy-style section called one of *names*."""
    for name in names:
        m = re.search(_SECTION.format(name=name), doc, re.MULTILINE)
        if m:
            rest = doc[m.end():]
            nxt = re.search(r"^[ \t]*\S[^\n]*\n[ \t]*-{3,}[ \t]*$", rest, re.MULTILINE)
            return rest[: nxt.start()] if nxt else rest
    return ""


def documented_params(doc: str) -> set[str]:
    body = section_body(doc, "Parameters", "Other Parameters", "Attributes")
    names: set[str] = set()
    for line in body.splitlines():
        if not line.strip() or line[:1].isspace():
            continue
        head = line.split(":", 1)[0]
        for part in head.split(","):
            part = part.strip().lstrip("*").strip("`")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
                names.add(part)
    return names


def signature_params(obj) -> list[str]:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    out = []
    for p in sig.parameters.values():
        if p.name in ("self", "cls") or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        out.append(p.name)
    return out


def page_order() -> list[str]:
    """API page stems in the order of the toctrees in docs/api/index.rst."""
    stems, in_toc = [], False
    for line in (API / "index.rst").read_text(encoding="utf-8").splitlines():
        if line.startswith(".. toctree::"):
            in_toc = True
        elif in_toc and line.startswith("   ") and not line.strip().startswith(":"):
            stems.append(line.strip())
        elif in_toc and line.strip() and not line.startswith("   "):
            in_toc = False
    return stems


def groups_from_pages() -> tuple[dict[str, str], list[str]]:
    """Public name -> title of the API page that documents it, and the page titles in order.

    A name's home is the page that renders it: an ``auto*``/``py:data``
    directive, the constants directive, or an autosummary with ``:toctree:``.
    Autosummary tables without ``:toctree:`` only link to the home page.
    """
    import torchspin.constants as constants_module

    group: dict[str, str] = {}
    titles: list[str] = []
    for stem in page_order():
        page = API / f"{stem}.rst"
        lines = page.read_text(encoding="utf-8").splitlines()
        title = lines[0].strip()
        titles.append(title)
        in_summary = toctree = False
        for line in lines:
            if line.startswith(".. autosummary::"):
                in_summary, toctree = True, False
                continue
            if in_summary:
                if line.startswith("   :toctree:"):
                    toctree = True
                    continue
                if line.startswith("   :") or not line.strip():
                    continue
                if line.startswith("   "):
                    if toctree:
                        group.setdefault(line.strip(), title)
                    continue
                in_summary = False
            m = re.match(r"\.\. (?:autofunction|autoclass|autodata|py:data)::\s+(?:torchspin\.)?(\S+)", line)
            if m:
                group.setdefault(m.group(1), title)
            if line.startswith(".. torchspin-constants::"):
                for name in vars(constants_module):
                    if name.isupper():
                        group.setdefault(name, title)
    return group, titles


def kind_of(obj) -> str:
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "data"


def load_warnings(log: Path | None) -> dict[str, list[str]]:
    """Sphinx warnings per fully qualified docstring owner."""
    out: dict[str, list[str]] = defaultdict(list)
    if log is None:
        return out
    pat = re.compile(r"docstring of ([\w.]+):\d+: (WARNING|ERROR): (.*?)(?: \[\w+\])?$")
    for line in log.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)].append(f"{m.group(2)}: {m.group(3)}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--log", type=Path, help="Sphinx build log to attribute docstring warnings from")
    ap.add_argument("--check", action="store_true", help="exit 1 if a public name has no API page")
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args(argv)

    groups, titles = groups_from_pages()
    warnings = load_warnings(args.log)
    names = list(dict.fromkeys(torchspin.__all__))
    rows = []
    for name in names:
        obj = getattr(torchspin, name)
        kind = kind_of(obj)
        if kind == "data":
            doc = ""
            sig = []
        else:
            doc = inspect.getdoc(obj) or ""
            sig = signature_params(obj)
        own = doc.strip() != "" if kind != "data" else None
        if kind == "data":
            params = returns = examples = None
            missing: list[str] = []
        else:
            params = has_section(doc, "Parameters", "Args", "Arguments", "Attributes")
            returns = has_section(doc, "Returns", "Yields") if kind == "function" else None
            examples = has_section(doc, "Examples", "Example")
            listed = documented_params(doc)
            missing = [p for p in sig if p not in listed] if sig else []
        qual = f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', name)}" if kind != "data" else ""
        warn = warnings.get(qual, [])
        rows.append(
            dict(name=name, kind=kind, group=groups.get(name, "**not documented**"),
                 module=getattr(obj, "__module__", "torchspin.constants") if kind != "data" else "",
                 doc=own, params=params, returns=returns, examples=examples,
                 missing=missing, nsig=len(sig), warnings=warn)
        )

    def mark(v):
        return "n/a" if v is None else ("yes" if v else "**no**")

    callables = [r for r in rows if r["kind"] != "data"]
    functions = [r for r in rows if r["kind"] == "function"]
    count = Counter()
    for r in callables:
        count["no_doc"] += not r["doc"]
        count["no_params"] += not r["params"]
        count["no_examples"] += not r["examples"]
        count["missing_params"] += bool(r["missing"])
        count["warned"] += bool(r["warnings"])
    for r in functions:
        count["no_returns"] += not r["returns"]
    undocumented = [r["name"] for r in rows if r["group"] == "**not documented**"]

    lines = [
        "# API docstring inventory",
        "",
        "Generated by `docs/scripts/api_inventory.py` from `torchspin.__all__` and the API",
        "reference pages in `docs/api/`; do not edit by hand.  Regenerate with",
        "",
        "```bash",
        "sphinx-build -b html docs docs/_build/html 2> build.log",
        "python docs/scripts/api_inventory.py --log build.log",
        "```",
        "",
        "The checks are textual: a section counts as present when its NumPy-style",
        "header (`Parameters` / `Returns` / `Examples` underlined with dashes) or",
        "Google-style header (`Args:`) is there.  For classes, a `Parameters` or",
        "`Attributes` section counts as parameters, and Returns does not apply.",
        "*Undocumented parameters* are signature parameters not named in that",
        "section.  *Sphinx warnings* are the docutils warnings this name's docstring",
        "produced in the build passed with `--log`.",
        "",
        "## Summary",
        "",
        f"- Public names in `torchspin.__all__`: {len(rows)} unique "
        f"({len(torchspin.__all__)} entries; `ham_ezho` is listed twice).",
        f"- Functions and classes: {len(callables)} ({len(functions)} functions, "
        f"{len(callables) - len(functions)} classes); data (constants, `__version__`): "
        f"{len(rows) - len(callables)}.",
        f"- Without a docstring: {count['no_doc']}.",
        f"- Without a Parameters (or Attributes) section: {count['no_params']}.",
        f"- Functions without a Returns section: {count['no_returns']} of {len(functions)}.",
        f"- Without an Examples section: {count['no_examples']}.",
        f"- With signature parameters missing from the Parameters section: {count['missing_params']}.",
        f"- With Sphinx docstring warnings: {count['warned']}"
        + ("" if args.log else " (run with `--log` to fill in)") + ".",
        f"- Not documented in the API reference: {len(undocumented)}"
        + (f" ({', '.join(undocumented)})" if undocumented else "") + ".",
        "",
        "## By name",
        "",
        "| Name | Kind | Group | Docstring | Parameters | Returns | Examples | Undocumented parameters | Sphinx warnings |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    order = {t: i for i, t in enumerate(titles)}
    order["**not documented**"] = -1
    for r in sorted(rows, key=lambda r: (order[r["group"]], r["name"].lower())):
        miss = ", ".join(f"`{p}`" for p in r["missing"]) if r["missing"] else ""
        if r["missing"] and len(r["missing"]) == r["nsig"]:
            miss = f"all {r['nsig']}"
        elif len(r["missing"]) > 8:
            miss = ", ".join(f"`{p}`" for p in r["missing"][:8]) + f", ... ({len(r['missing'])})"
        lines.append(
            f"| `{r['name']}` | {r['kind']} | {r['group']} | {mark(r['doc'])} | {mark(r['params'])} | "
            f"{mark(r['returns'])} | {mark(r['examples'])} | {miss} | {len(r['warnings']) or ''} |"
        )

    if args.log:
        per_module = Counter()
        pat = re.compile(r"(torchspin/[\w/]+\.py):docstring of ")
        for line in args.log.read_text(encoding="utf-8").splitlines():
            m = pat.search(line)
            if m and re.search(r"(WARNING|ERROR):", line):
                per_module[m.group(1)] += 1
        lines += [
            "",
            "## Sphinx docstring warnings per module",
            "",
            "Includes modules outside `torchspin.__all__` (`torchspin.lft`, `torchspin.ml`, ...)",
            "that the API reference documents.",
            "",
            "| Module | Warnings |",
            "|---|---|",
        ]
        lines += [f"| `{mod}` | {n} |" for mod, n in per_module.most_common()]
        lines += ["", "### Details for public names", ""]
        for r in rows:
            for w in r["warnings"]:
                lines.append(f"- `{r['name']}`: {w}")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} names)")
    if args.check and undocumented:
        print("not documented in docs/api:", ", ".join(undocumented), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
