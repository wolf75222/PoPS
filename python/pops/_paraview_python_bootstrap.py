"""Run a PoPS script after the native runtime has initialized ParaView's MPI safely."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("pops ParaView bootstrap requires a Python script")
    script = Path(sys.argv[1]).expanduser().resolve()
    if not script.is_file():
        raise SystemExit("PoPS Catalyst script does not exist: %s" % script)

    # This import must precede Catalyst.  The MPI build requests MPI_THREAD_MULTIPLE while the
    # process is still neutral; pvpython/pvbatch initialize MPI too early with MPI_THREAD_SINGLE.
    from pops import _pops  # noqa: F401

    try:
        __import__("catalyst")
        __import__("catalyst_conduit")
    except (ImportError, ModuleNotFoundError) as error:
        raise SystemExit(
            "ParaView's Catalyst 2 Python modules are unavailable in the neutral host: %s"
            % error
        ) from error

    previous_argv = sys.argv
    previous_path = list(sys.path)
    sys.argv = [str(script), *sys.argv[2:]]
    if sys.path:
        sys.path[0] = str(script.parent)
    else:  # pragma: no cover - CPython normally provides at least one search entry
        sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


if __name__ == "__main__":
    main()
