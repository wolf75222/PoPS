"""pops.codegen.solvers -- the custom-solver generation DSL (Spec 5 criterion 19).

INTERNAL / EXPERIMENTAL -- NOT a stable user API.

This is a solver-GENERATION DSL: it authors a solver IR and lowers it to C++ text. Its surface
(``@solver`` / ``SolverContext`` / ``SolverIR`` / ``build_solver_ir`` / ``generate_solver_cpp``)
is unstable and may change or disappear without notice; it is provided for the project's own
experimentation, not for users to author solvers.

USERS DO NOT AUTHOR SOLVERS. The supported, stable workflow is to CONFIGURE the provided C++
solver descriptors in :mod:`pops.solvers` -- ``pops.solvers.CG()`` / ``GMRES()`` /
``GeometricMG()`` / ``Newton()`` and their typed options -- and let codegen / the
runtime drive the compiled C++ solvers. There is no public ``@solver`` decorator on ``pops`` /
``pops.lib`` / ``pops.solvers``; the only entry to this experimental DSL is this codegen package.

Spec 5 criterion 19 homes a solver-gen DSL, if any, in ``pops.codegen.solvers``: authoring an IR
and lowering it to C++ text is a codegen concern. The DSL was formerly parked under
``pops.lib.solvers``; this package is its canonical home, and the ``pops.lib.solvers`` re-export
shim is removed (``pops.lib`` is presets-only; the solver descriptors live only in
:mod:`pops.solvers`).

The authoring IR (:mod:`.dsl`) imports the heavy :mod:`pops.time` LAZILY (in ``build_solver_ir``),
so this package adds no module-scope ``time`` edge; the C++ lowering (:mod:`.solver_cpp`) is pure
string formatting (no ``_pops``). Importing this package wires the custom-solver registry hooks
(``solvers.custom`` / ``solvers.registered``) onto the shared :data:`pops.solvers.solvers`
namespace -- a downward ``codegen -> solvers`` edge (``pops.solvers`` imports nothing, so no
cycle).
"""
from .dsl import (SolverContext, SolverIR, build_solver_ir, solver,
                  _as_descriptor, _custom_solver, _registered_solvers)
from .solver_cpp import generate_solver_cpp

# This DSL is internal / experimental, not a stable public API (Spec 5 criterion 19).
__experimental__ = True

# Wire the custom-solver registry hooks onto the shared pops.solvers.solvers namespace, where the
# authoring DSL lives. Attaching them here (not in pops.solvers, which must stay free of any DSL
# import) keeps pops.solvers a pure descriptor catalog. Idempotent: re-importing just re-binds the
# same callables. pops.solvers imports nothing, so this codegen -> solvers edge stays acyclic.
from pops.solvers import solvers as solvers  # noqa: E402  (the shared factory namespace)

solvers.custom = _custom_solver
solvers.registered = _registered_solvers

__all__ = ["solver", "build_solver_ir", "generate_solver_cpp",
           "SolverContext", "SolverIR", "solvers"]
