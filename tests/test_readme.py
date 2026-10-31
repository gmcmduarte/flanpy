# SPDX-License-Identifier: GPL-3.0-or-later
"""The ``pycon`` code blocks of README.md run as doctests, so the example cannot rot.

All blocks share one namespace, in order (a later block may use an earlier one's
names). The fences are stripped before parsing: doctest would read a closing fence
right after an output line as part of the expected output.
"""

import doctest
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"^```pycon\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def readme_blocks(text: str) -> list[tuple[str, int]]:
    """The ``pycon`` blocks of ``text``, each with the 0-based line it starts on."""
    return [(m[1], text.count("\n", 0, m.start(1))) for m in BLOCK.finditer(text)]


def test_readme_has_an_example() -> None:
    assert readme_blocks(README.read_text(encoding="utf-8"))


def test_readme_examples_pass() -> None:
    parser = doctest.DocTestParser()
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    globs: dict[str, object] = {}
    blocks = readme_blocks(README.read_text(encoding="utf-8"))
    for i, (block, line) in enumerate(blocks):
        test = parser.get_doctest(block, globs, f"README.md[{i}]", str(README), line)
        runner.run(test, clear_globs=False)
    results = runner.summarize(verbose=False)
    assert results.failed == 0, f"{results.failed} README example(s) failed (above)"
