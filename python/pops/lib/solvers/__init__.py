"""pops.lib.solvers -- the linear / nonlinear solver brick catalog (Spec 3).

The matrix-free Krylov solvers are FREE FUNCTIONS in namespace pops (generic_krylov.hpp);
Newton/FixedPoint have no standalone solver type (Newton is the implicit_stepper kernel),
so they are catalogued as planned (available=False).

The custom-solver AUTHORING DSL (``@solver`` / ``SolverContext`` / ``build_solver_ir``)
lives in :mod:`pops.lib.solvers.dsl`; the C++ lowering (``generate_solver_cpp``) lives
in :mod:`pops.lib.solvers.solver_cpp`; the preconditioner catalog is in
:mod:`pops.lib.solvers.preconditioners`. All are re-exported here.

DEFER (no catalogued source): ``solvers.nonlinear`` / ``solvers.local`` -- Newton /
FixedPoint are already ``_planned`` descriptors in this krylov ns; a separate
``nonlinear.py`` / ``local.py`` would invent surface (see the PR-D blueprint DEFER list).
"""
from types import SimpleNamespace

from ..descriptors import _native, _planned
from .dsl import (solver, build_solver_ir, SolverContext, SolverIR,
                  _custom_solver, _registered_solvers, _as_descriptor)
from .solver_cpp import generate_solver_cpp
from .preconditioners import preconditioners


def _solver(name, native_id, **o):
    return _native(name, native_id, name, category="solver", **o)


solvers = SimpleNamespace(
    CG=lambda **o: _solver("cg", "pops::cg_solve", **o),
    BiCGStab=lambda **o: _solver("bicgstab", "pops::bicgstab_solve", **o),
    GMRES=lambda **o: _solver("gmres", "pops::gmres_solve", **o),
    Richardson=lambda **o: _solver("richardson", "pops::richardson_solve", **o),
    Newton=lambda **o: _planned("newton", "newton", category="solver", **o),
    FixedPoint=lambda **o: _planned("fixed_point", "fixed_point", category="solver", **o),
    Schur=lambda **o: _solver("schur", "pops::SchurCondensationOperator", **o),
)

# The custom-solver registry hooks (``@pops.lib.solver`` -> solvers.custom / .registered).
solvers.custom = _custom_solver
solvers.registered = _registered_solvers

__all__ = ["solvers", "solver", "build_solver_ir", "generate_solver_cpp",
           "SolverContext", "SolverIR", "preconditioners"]
