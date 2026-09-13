"""Path-independent linker identity for explicitly loaded native plugins."""
from __future__ import annotations

import sys
from typing import Any


def deterministic_program_link_flags(flags: Any) -> list[str]:
    """Return link flags that never encode the caller's output path in a Program plugin.

    Darwin otherwise derives ``LC_ID_DYLIB`` from ``debug.so``/``nodebug.so`` and changes the
    content-derived UUID. Program plugins are loaded explicitly with ``dlopen``; no consumer
    resolves them through this inert install name.
    """
    result = list(flags)
    if sys.platform == "darwin":
        result.append("-Wl,-install_name,@rpath/pops_program.so")
    return result


def deterministic_component_link_flags(flags: Any) -> list[str]:
    """Keep temporary AOT build directories out of authenticated component bytes.

    Components, like Programs, are loaded by their authenticated file path. A stable inert
    install name prevents Darwin's content-derived UUID from varying between MPI ranks.
    """
    result = list(flags)
    if sys.platform == "darwin":
        result.append("-Wl,-install_name,@rpath/pops_component.dylib")
    return result


__all__ = ["deterministic_program_link_flags", "deterministic_component_link_flags"]
