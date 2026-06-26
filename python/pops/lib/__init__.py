"""pops.lib -- a catalog of typed brick descriptors and IR macros (Spec 3 / Spec 4).

pops.lib is NOT a Python numerics library. Every entry is one of:

* a NATIVE brick -- a descriptor naming a C++ symbol already in ``include/pops``
  (``pops.lib.riemann.HLLC()`` -> ``pops::HLLCFlux``); a catalogued brick with no native
  symbol yet carries ``available=False`` and an empty ``native_id`` (never a fake id);
* a GENERATED brick -- a descriptor of a DSL-authored brick compiled to C++;
* a MACRO brick -- a Python function that builds Program IR;
* an EXTERNAL C++ brick -- a descriptor of a user ``.so`` registered by id
  (``pops.lib.riemann.User("my_hllc")``).

A descriptor carries metadata only -- a native id, a runtime scheme string,
requirements and capabilities. It computes nothing; the codegen and runtime
consume it. ``pops.lib`` imports only ``pops.ir`` and the stdlib at module scope;
``pops.physics`` and ``pops.time`` are imported lazily, in-function, where needed --
and ``pops.codegen`` / ``_pops`` are never imported (codegen and runtime consume the
descriptors, not the reverse).

This package is the Spec-4 home of the formerly flat ``pops.lib`` module; its public
surface (re-exported below) is the unchanged back-compat guarantee. The moment-model
generator + facade live in :mod:`pops.lib.moments`; the provided models in
:mod:`pops.lib.models`.
"""
from .descriptors import (BrickDescriptor, load_cpp_library, external,
                          _register_manifest, _clear_external_catalog)
from .riemann import riemann
from .reconstruction import reconstruction, limiters
from .spatial import spatial
from .fields import fields
from .operators import projections
from .solvers import (solvers, solver, SolverContext, SolverIR,
                      build_solver_ir, generate_solver_cpp)
from .solvers.preconditioners import preconditioners
from .diagnostics import diagnostics, invariants
# Spec-4 ``lib.time`` is the scheme-builder package (forward_euler / ssprk2 / ...); the
# old Spec-3 MACRO catalog SimpleNamespace now lives in lib.time.macros (internal artifact).
from . import time
from . import moments, models

__all__ = ["BrickDescriptor", "riemann", "reconstruction", "limiters", "spatial",
           "fields", "solvers", "preconditioners", "diagnostics", "projections",
           "invariants", "time", "solver", "build_solver_ir", "generate_solver_cpp",
           "SolverContext", "SolverIR", "load_cpp_library", "external",
           "moments", "models"]
