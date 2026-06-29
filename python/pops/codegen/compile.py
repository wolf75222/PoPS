"""Public compile entry point.

The final public route is intentionally singular:

    compiled = pops.compile_problem(...)

Legacy model-level runners such as ``compile_so``, ``compile_aot``,
``compile_native``, ``compile_or_jit`` and ``compile_model`` remain implementation
details of the codegen package while the migration burns them down.  They are
not re-exported here, so importing ``pops.codegen.compile`` no longer gives a
second public orchestration surface beside ``compile_problem``.
"""

from pops.codegen.compile_drivers import compile_problem  # noqa: F401
from pops.codegen.loader import CompiledProblem  # noqa: F401

__all__ = ["compile_problem", "CompiledProblem"]
