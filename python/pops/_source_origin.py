"""Where a Python body that PoPS lowers to C++ is written: its file, its line and its function.

An origin is local provenance, read by the source maps and by ``pops.inspect``. It stays outside
every artifact and descriptor identity: moving a case file changes the origin of its bodies, never
the C++ that they lower to.
"""
from __future__ import annotations

import os
from typing import Any

_POPS_ROOT = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT = os.path.join(_POPS_ROOT, "lib")


def code_origin(body: Any) -> dict[str, Any]:
    """The origin of ``body``, a Python function: ``written_in`` is ``pops.lib`` for a body of
    the library, ``pops`` for one of PoPS itself, and ``case`` for any other file; ``path`` is
    relative to the directory that holds the package ``pops`` inside PoPS, absolute elsewhere;
    ``line`` is the first line of the function, its decorator's when it has one."""
    code = getattr(body, "__code__", None)
    if code is None:
        raise TypeError("a source origin is read from a Python function, got %r" % (body,))
    path = os.path.abspath(code.co_filename)
    if path.startswith(_LIB_ROOT + os.sep):
        written_in = "pops.lib"
    elif path.startswith(_POPS_ROOT + os.sep):
        written_in = "pops"
    else:
        written_in = "case"
    if written_in != "case":
        path = os.path.relpath(path, os.path.dirname(_POPS_ROOT))
    return {"written_in": written_in, "path": path, "line": code.co_firstlineno,
            "function": getattr(code, "co_qualname", code.co_name)}


def origin_text(origin: Any) -> str:
    """``pops.lib, <path>:<line>`` or ``the case file, <path>:<line>``."""
    where = {"pops.lib": "pops.lib", "pops": "pops", "case": "the case file"}[origin["written_in"]]
    return "%s, %s:%d" % (where, origin["path"], origin["line"])


__all__ = ["code_origin", "origin_text"]
