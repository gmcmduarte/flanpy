# SPDX-License-Identifier: GPL-3.0-or-later
import re
from importlib.metadata import version

import flanpy


def test_import_and_version() -> None:
    assert isinstance(flanpy.__version__, str)
    assert flanpy.__version__ == version("flanpy")
    # PEP 440 public version, e.g. 0.0.1.dev0
    assert re.fullmatch(
        r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", flanpy.__version__
    )
