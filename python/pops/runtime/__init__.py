"""Private native execution package used only behind ``pops.bind`` and ``pops.run``.

The package intentionally exports no user API. Internal modules are imported by their exact private
paths, and importing :mod:`pops.runtime` itself is inert.
"""
from __future__ import annotations


__all__: list[str] = []


def __dir__() -> list[str]:
    return []
