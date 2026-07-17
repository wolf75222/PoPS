"""Private native model compiler used by the canonical ``pops.compile`` phase."""
from __future__ import annotations

from pops.codegen.cache import pops_cache_dir  # noqa: F401
from pops.codegen._compile_emit import (  # noqa: F401
    _BACKEND_CAPS,
    model_hash,
    emit_cpp_native_loader,
)
from pops.codegen._compile_drivers import (  # noqa: F401
    compile_native,
    compile_model,
)


__all__ = [
    "compile_model",
    "compile_native",
    "emit_cpp_native_loader",
    "model_hash",
]
