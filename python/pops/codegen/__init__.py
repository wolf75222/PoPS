"""Typed compilation authority for the final public lifecycle.

Compilation itself is entered only through :func:`pops.compile`.  The implementation modules under
this package are private engines; they are not alternative user-facing compilers.
"""
from __future__ import annotations

from ._backends import Production
from ._compiler_lowering import CompilerLowerable, CompilerLowering


__all__ = ["Production", "CompilerLowerable", "CompilerLowering"]
