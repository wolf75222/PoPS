"""Compile an authored Program into a shared library for runtime install.

Runtime cadence installers must not import the retired whole-problem driver name.
This module is the codegen-owned seam those installers call.
"""

from __future__ import annotations

from typing import Any


def compile_authored_program(**kwargs: Any) -> Any:
    """Lower one authored Program + model into a production shared library."""
    from pops.codegen._compile_drivers import compile_problem

    return compile_problem(**kwargs)
