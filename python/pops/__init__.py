"""PoPS Python authoring surface.

Python describes typed models/programs and drives code generation. Numerical work runs in
compiled C++/Kokkos/MPI code through :class:`pops.runtime.system.System` and
:class:`pops.runtime.system.AmrSystem`.
"""
# Load the _pops extension (RTLD_GLOBAL so the DSL production .so resolves C++ symbols).
from pops import _bootstrap  # noqa: F401  (import side effect: loads _pops with the right flags)
from pops._bootstrap import abi_key  # noqa: F401
from pops._version import __version__  # noqa: F401

# Runtime layer (the ONLY importer of _pops): systems, parallelism, doctor.
from pops.runtime.system import System, AmrSystem  # noqa: F401
from pops.runtime.threading import set_threads, has_kokkos, parallel_info  # noqa: F401
from pops.runtime.doctor import doctor  # noqa: F401

__all__ = [
    "__version__",
    "System", "AmrSystem",
    "physics", "model", "time", "numerics", "fields", "linalg", "solvers",
    "mesh", "params", "diagnostics", "output", "external", "moments", "lib",
    "codegen", "runtime", "math",
    "abi_key",
    "set_threads", "has_kokkos", "parallel_info", "doctor",
    "compile_problem", "CompiledProblem",
    "compile_library", "read_library_manifest", "LibraryManifest",
    "inspect_capabilities", "CapabilityMatrix", "CapabilityEntry",
]


# Lower / authoring layers. Runtime helpers such as pops.runtime.integrate are not re-exported:
# Python may orchestrate, but it must not advertise a public numerical integration loop.
from . import time  # noqa: E402  (pops.time.Program IR; pure stdlib, no numpy/_pops dependency)
from . import model  # noqa: E402  (pops.model operator-first type system; pure stdlib, Spec 2)
from . import math  # noqa: E402  (pops.math board operators; pure stdlib, Spec 3, dsl lazy)
from . import lib  # noqa: E402  (pops.lib typed-brick descriptor catalog; pure stdlib, Spec 3)
from . import physics  # noqa: E402  (pops.physics board model authoring; numpy-free import, Spec 3)
from . import moments  # noqa: E402  (generic moment authoring tools; ready models live in pops.lib)
from . import numerics  # noqa: E402
from . import fields  # noqa: E402
from . import linalg  # noqa: E402
from . import solvers  # noqa: E402
from . import mesh  # noqa: E402
from . import params  # noqa: E402
from . import diagnostics  # noqa: E402
from . import output  # noqa: E402
from . import external  # noqa: E402
from . import codegen  # noqa: E402
from . import runtime  # noqa: E402
from .codegen.library import (  # noqa: E402,F401  (re-export: brick-library manifest API, Spec 3 section 21)
    LibraryManifest, compile_library, read_library_manifest)


# LAZY pops.compile_problem / pops.CompiledProblem (PEP 562): the codegen engine pulls numpy at
# import (host evaluator of the prototype IR), whereas the native path (System/add_block) and the
# production backend do not need it. Exposing these top-level names LAZILY keeps `import pops`
# numpy-free until the DSL/compile path is first used; numpy's absence then gives a targeted message
# (doctor too).
def __getattr__(name):
    if name == "compile_problem":
        from .codegen.compile import compile_problem
        return compile_problem
    if name == "CompiledProblem":
        from .codegen.loader import CompiledProblem
        return CompiledProblem
    if name == "inspect_capabilities":
        from ._capabilities import inspect_capabilities
        return inspect_capabilities
    if name == "CapabilityMatrix":
        from ._capabilities import CapabilityMatrix
        return CapabilityMatrix
    if name == "CapabilityEntry":
        from ._capabilities import CapabilityEntry
        return CapabilityEntry
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
