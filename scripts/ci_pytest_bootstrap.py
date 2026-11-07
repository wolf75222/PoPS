"""Start pytest in an installed CI prefix: the checkout on ``sys.path``, ``pops`` from the prefix.

``ci_python_dimensions.py`` runs ``python -u scripts/ci_pytest_bootstrap.py <pytest arguments>``
from the checkout, with the interpreter of the prefix it installed from the retained wheel.  This
is a committed file rather than ``python -c`` code so that a qualification receipt can judge the
code that started its process: ``qualification_receipt.py`` refuses code compiled from a string.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(arguments: list[str]) -> int:
    import pops

    package = Path(pops.__file__).resolve()
    if not package.is_relative_to(Path(sys.prefix).resolve()) or package.is_relative_to(ROOT):
        raise SystemExit("pops resolves to %s, not to the installed prefix %s"
                         % (package, sys.prefix))
    import pytest

    return pytest.main(arguments)


if __name__ == "__main__":
    # The path that ``python -c`` gave this bootstrap: the scripts and the checkout first, then
    # the working directory where a script's own directory would be.
    if not sys.flags.safe_path:
        sys.path[0] = ""
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
    raise SystemExit(main(sys.argv[1:]))
