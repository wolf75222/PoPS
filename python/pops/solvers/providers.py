"""Direct native solvers for hierarchy-scoped mathematical problems."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from pops.identity import Identity, make_identity
from pops.solvers._numeric import exact_open_unit_real, optional_positive_int


_HIERARCHY_SOLVER_SCHEMA_VERSION = 1
_DEFAULT_MAX_ITER = 30
_DEFAULT_REL_TOL = 1.0e-9


def _positive_max_iter(value: Any) -> int:
    checked = optional_positive_int(value, where="CompositeTensorFAC(max_iter=)")
    if checked is None:
        raise ValueError("CompositeTensorFAC(max_iter=) must be a positive int")
    return checked


@dataclass(frozen=True, slots=True)
class _PreparedCompositeTensorFAC:
    """Authenticated direct hierarchy solver consumed by :meth:`Program.solve`."""

    tolerance: Any
    max_iterations: int
    identity_data: dict[str, Any]
    identity: Identity

    def build_program_solve(self, *, program: Any, problem: Any,
                            name: Any = None) -> Any:
        build = getattr(program, "_solve_composite_tensor_fac", None)
        if not callable(build):
            raise TypeError("CompositeTensorFAC requires a pops.time.Program")
        return build(problem=problem, prepared=self, name=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class CompositeTensorFAC:
    """Direct scalar tensor-elliptic solver over one AMR hierarchy.

    ``CompositeTensorFAC`` owns the complete solve contract. On a flat hierarchy it runs the
    authenticated tensor apply through BiCGStab; on a refined hierarchy it runs the equivalent
    native composite FAC operator. ``max_iter`` and ``rel_tol`` govern both branches. The FAC-only
    controls shape the refined iteration and are never presented as Krylov options.
    """

    max_iter: int = _DEFAULT_MAX_ITER
    rel_tol: Any = _DEFAULT_REL_TOL
    fine_sweeps: int | None = None
    coarse_rel_tol: Any = None
    coarse_cycles: int | None = None
    verbose: bool | None = None
    solver_id: str = field(init=False, default="composite_tensor_fac")
    capabilities: frozenset[str] = field(
        init=False,
        default_factory=lambda: frozenset(
            {"amr_hierarchy", "flat_bicgstab", "scalar", "tensor_elliptic"}
        ),
    )
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_iter", _positive_max_iter(self.max_iter))
        object.__setattr__(
            self,
            "rel_tol",
            exact_open_unit_real(self.rel_tol, where="CompositeTensorFAC(rel_tol=)"),
        )
        object.__setattr__(
            self,
            "fine_sweeps",
            optional_positive_int(self.fine_sweeps, where="CompositeTensorFAC(fine_sweeps=)"),
        )
        object.__setattr__(
            self,
            "coarse_cycles",
            optional_positive_int(self.coarse_cycles, where="CompositeTensorFAC(coarse_cycles=)"),
        )
        if self.coarse_rel_tol is not None:
            object.__setattr__(
                self,
                "coarse_rel_tol",
                exact_open_unit_real(
                    self.coarse_rel_tol, where="CompositeTensorFAC(coarse_rel_tol=)"
                ),
            )
        if self.verbose is not None and type(self.verbose) is not bool:
            raise TypeError(
                "CompositeTensorFAC(verbose=) must be a Python bool or None (got %r)"
                % (self.verbose,)
            )

    def canonical_identity(self) -> dict[str, Any]:
        # Lazy by design: pops.solvers remains an import-graph sink and does not import pops.ir at
        # module scope merely because the solver catalog is imported.
        from pops.ir.literals import scalar_data

        return {
            "schema_version": _HIERARCHY_SOLVER_SCHEMA_VERSION,
            "solver_id": self.solver_id,
            "capabilities": sorted(self.capabilities),
            "options": {
                "max_iter": self.max_iter,
                "rel_tol": scalar_data(self.rel_tol),
                "fine_sweeps": self.fine_sweeps,
                "coarse_rel_tol": (
                    None if self.coarse_rel_tol is None else scalar_data(self.coarse_rel_tol)
                ),
                "coarse_cycles": self.coarse_cycles,
                "verbose": self.verbose,
            },
        }

    def to_data(self) -> dict[str, Any]:
        return self.canonical_identity()

    @property
    def identity(self) -> Identity:
        return make_identity("hierarchy-solver", self.canonical_identity())

    def prepare_program_solve(self) -> _PreparedCompositeTensorFAC:
        identity_data = self.canonical_identity()
        return _PreparedCompositeTensorFAC(
            tolerance=self.rel_tol,
            max_iterations=self.max_iter,
            identity_data=deepcopy(identity_data),
            identity=make_identity("hierarchy-solver", identity_data),
        )


__all__ = ["CompositeTensorFAC"]
