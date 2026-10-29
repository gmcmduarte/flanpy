# SPDX-License-Identifier: GPL-3.0-or-later
import re
from importlib.metadata import version

import pyflan


def test_import_and_version() -> None:
    assert isinstance(pyflan.__version__, str)
    assert pyflan.__version__ == version("pyflan")
    # PEP 440 public version, e.g. 0.0.1.dev0
    assert re.fullmatch(
        r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", pyflan.__version__
    )
