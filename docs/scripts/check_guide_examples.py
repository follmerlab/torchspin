"""Run the examples on the user-guide pages (docs/guide/*.md) as doctests.

Usage (from the repository root)::

    python docs/scripts/check_guide_examples.py              # every guide page
    python docs/scripts/check_guide_examples.py fitting.md   # selected pages

Examples are the ```pycon blocks (``>>>`` prompts with expected output).  The
blocks of one page share a namespace and run in order, so a page reads like a
session.  Plain ```python blocks (e.g. GPU-only code) are not run.  Exit status
1 if any example fails.
"""

from __future__ import annotations

import doctest
import re
import sys
import time
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOCS.parent))

_BLOCK = re.compile(r"^```pycon[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
FLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE


def run_page(path: Path) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    blocks = _BLOCK.findall(text)
    if not blocks:
        return 0, 0
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=FLAGS)
    globs: dict = {"__name__": f"guide_{path.stem}"}
    for i, block in enumerate(blocks):
        test = parser.get_doctest(block, globs, f"{path.name}[{i}]", str(path), 0)
        runner.run(test, clear_globs=False)
        globs = test.globs  # DocTest copies its globals: carry them forward
    result = runner.summarize(verbose=False)
    return result.failed, result.attempted


def main(argv: list[str]) -> int:
    pages = [DOCS / "guide" / a for a in argv] if argv else sorted((DOCS / "guide").glob("*.md"))
    failed = attempted = 0
    for page in pages:
        t0 = time.perf_counter()
        f, a = run_page(page)
        failed += f
        attempted += a
        print(f"{page.name}: {a - f}/{a} passed ({time.perf_counter() - t0:.1f} s)")
    print(f"total: {attempted - failed}/{attempted} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
