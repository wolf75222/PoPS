"""pops.solvers.krylov -- the matrix-free Krylov solver catalog (Spec 5 sec.5.7).

The Krylov solvers share the prepared C++ entry point ``pops::solve_prepared_affine``;
each factory here returns an inert :class:`pops.descriptors.BrickDescriptor` naming the real
C++ symbol and the runtime ``scheme`` token. They compute nothing; codegen / the runtime
consume the descriptor. This is the ONE public home of the catalog formerly parked under
``pops.lib.solvers`` (that re-export shim is removed; there is no second public path).

* :func:`CG` -- conjugate gradient (SPD systems);
* :func:`BiCGStab` -- stabilised bi-conjugate gradient (nonsymmetric);
* :func:`GMRES` -- generalised minimal residual (nonsymmetric);
* :func:`Richardson` -- Richardson iteration with explicit relaxation.

Every factory takes a MANDATORY ``max_iter`` (a positive Python int): a dynamic Krylov loop
with no iteration budget is a configuration error. Refusing a missing / non-positive budget
HERE, at descriptor construction, surfaces the same error before the prepared native route is
ever touched (Spec 5 sec.6: a route is refused explainably, pre-compile).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pops.descriptors import BrickDescriptor
from pops.identity import Identity, make_identity
from pops.solvers.requirements import capability_map

# The Krylov solvers are matrix-free free functions over ``pops::MultiFab`` primitives (dot /
# saxpy / the operator apply): the algebra is layout-agnostic, so it runs on a uniform mesh or
# an AMR level, under MPI (the inner-product reductions are collective) and on the GPU (the
# kernels are Kokkos). They therefore declare every route capability (Spec 6 sec.4 / sec.9), so
# a route check can see they are AMR-capable rather than guessing from an empty capability set.
_KRYLOV_CAPABILITIES = capability_map(uniform=True, amr=True, mpi=True, gpu=True)


def _exact_control(value: Any, where: str) -> Any:
    # Lazy to keep the catalog-layer import graph (solvers -> no symbolic implementation module).
    from pops._ir.literals import exact_numeric_scalar
    return exact_numeric_scalar(value, where=where)


def _check_max_iter(name: str, max_iter: Any) -> int:
    """Refuse a missing / non-positive Krylov iteration budget at descriptor construction.

    ``max_iter`` is MANDATORY (a positive Python int): a dynamic solver loop with no budget is a
    configuration error the prepared native route also refuses. Raising here refuses the route
    before the runtime is touched, so a test can assert the refusal without compiling. A bool is
    rejected (``True`` / ``False`` are ints in Python but never a valid budget). @p name is the
    public factory name used in the message.
    """
    if max_iter is None:
        raise ValueError(
            "%s: max_iter is required (dynamic solver loops require max_iter); pass a positive "
            "int, e.g. %s(max_iter=200)" % (name, name))
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError(
            "%s: max_iter must be a positive int (dynamic solver loops require max_iter); got %r"
            % (name, max_iter))
    return int(max_iter)


def _check_rel_tol(name: str, rel_tol: Any) -> Any:
    """Validate an optional relative tolerance as a finite number in ``[0, 1)``.

    ``None`` selects the canonical ``1e-8`` tolerance. An explicit zero selects absolute-only
    stopping and is accepted only when the paired ``abs_tol`` is positive."""
    if rel_tol is None:
        return None
    try:
        value = _exact_control(rel_tol, "%s rel_tol" % name)
        valid = 0 <= value < 1 and math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("%s: rel_tol must be a finite number in [0, 1) or None; got %r"
                         % (name, rel_tol))
    return value


def _check_abs_tol(name: str, abs_tol: Any) -> Any:
    """Return one exact finite non-negative absolute residual threshold."""
    if abs_tol is None:
        return _exact_control(0, "%s abs_tol" % name)
    try:
        value = _exact_control(abs_tol, "%s abs_tol" % name)
        valid = value >= 0 and math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("%s: abs_tol must be a finite number >= 0; got %r" % (name, abs_tol))
    return value


@dataclass(frozen=True, slots=True)
class _PreparedKrylov:
    """Private authenticated provider consumed by ``Program.solve``."""

    method: str
    tolerance: Any
    absolute_tolerance: Any
    max_iterations: int
    restart: int | None
    preconditioner: Any
    omega: Any
    identity: Identity

    def build_program_solve(self, *, program: Any, problem: Any,
                            name: Any = None) -> Any:
        build = getattr(problem, "build_matrix_free_linear", None)
        if not callable(build):
            raise TypeError("Krylov solvers require a pops.linalg.LinearProblem")
        return build(program=program, prepared_solver=self, name=name)


class _KrylovDescriptor(BrickDescriptor):
    """Executable Krylov descriptor; all algorithm controls travel with this object."""

    def prepare_program_solve(self) -> _PreparedKrylov:
        from pops.time._program.solve import _lower_preconditioner

        preconditioner, preconditioner_options = _lower_preconditioner(
            self.options.get("preconditioner"))
        if preconditioner != "identity" and self.scheme not in ("gmres", "bicgstab"):
            raise ValueError(
                "%s does not expose a native preconditioner slot; use GMRES or BiCGStab"
                % self.name)
        tolerance = self.options.get("rel_tol", 1.0e-8)
        restart = self.options.get("restart")
        omega = self.options.get("omega")
        payload = {
            "schema_version": 2,
            "method": self.scheme,
            "tolerance": str(tolerance),
            "absolute_tolerance": str(self.options.get("abs_tol", 0)),
            "max_iterations": self.options["max_iter"],
            "restart": restart,
            "preconditioner": preconditioner,
            "preconditioner_options": preconditioner_options,
            "omega": None if omega is None else str(omega),
        }
        return _PreparedKrylov(
            method=self.scheme,
            tolerance=tolerance,
            absolute_tolerance=self.options.get("abs_tol", 0),
            max_iterations=self.options["max_iter"],
            restart=restart,
            preconditioner=(preconditioner, preconditioner_options),
            omega=omega,
            identity=make_identity("prepared-krylov", payload),
        )


def _solver(name: str, native_id: str, factory: str, max_iter: Any, rel_tol: Any = None,
            *, restart: Any = None, preconditioner: Any = None,
            omega: Any = None, abs_tol: Any = None) -> Any:
    """A native Krylov-solver descriptor in the ``solver`` category (scheme == @p name).

    ``max_iter`` is validated (positive int, mandatory) and folded into the descriptor options so
    the budget travels with the route and is inspectable pre-runtime. ``rel_tol`` (ADC-645) is an
    optional per-descriptor tolerance folded in ONLY when set (omit-when-default), consumed by
    ``P.solve_linear`` when the call-site ``tol`` is left default. @p factory is the public
    factory name used in the refusal message.
    """
    options: dict[str, Any] = {"max_iter": _check_max_iter(factory, max_iter)}
    rel = _check_rel_tol(factory, rel_tol)
    absolute = _check_abs_tol(factory, abs_tol)
    if rel == 0 and absolute == 0:
        raise ValueError(
            "%s: rel_tol and abs_tol cannot both be zero; at least one stopping threshold "
            "must be positive" % factory)
    if rel is not None:
        options["rel_tol"] = rel
    options["abs_tol"] = absolute
    if name == "gmres":
        if restart is None:
            restart = 30
        if isinstance(restart, bool) or not isinstance(restart, int) or restart <= 0:
            raise ValueError("GMRES: restart must be a positive integer")
        options["restart"] = int(restart)
    elif restart is not None:
        raise ValueError("%s: restart is a GMRES-only control" % factory)
    if preconditioner is not None:
        options["preconditioner"] = preconditioner
    if omega is not None:
        options["omega"] = omega
    return _KrylovDescriptor(
        name, "native", category="solver", native_id=native_id, scheme=name,
        capabilities=_KRYLOV_CAPABILITIES, options=options)


def CG(max_iter: Any = None, rel_tol: Any = None, *, preconditioner: Any = None,
       abs_tol: Any = None) -> Any:
    """Conjugate gradient over the prepared affine route (scheme ``"cg"``). Inert.

    ``max_iter`` is a MANDATORY positive int (the iteration budget); a missing / non-positive
    budget is refused at construction (see the module docstring). ``rel_tol`` (ADC-645, optional)
    supplies the ``P.solve_linear`` tolerance when the call-site ``tol`` is left default.
    """
    return _solver("cg", "pops::solve_prepared_affine", "CG", max_iter, rel_tol,
                   preconditioner=preconditioner, abs_tol=abs_tol)


def BiCGStab(max_iter: Any = None, rel_tol: Any = None, *, preconditioner: Any = None,
             abs_tol: Any = None) -> Any:
    """Stabilised bi-CG over the prepared affine route (scheme ``"bicgstab"``).

    ``max_iter`` is a MANDATORY positive int; a missing / non-positive budget is refused.
    ``rel_tol`` (ADC-645, optional) supplies the ``P.solve_linear`` tolerance when the call-site
    ``tol`` is left default.
    """
    return _solver("bicgstab", "pops::solve_prepared_affine", "BiCGStab", max_iter, rel_tol,
                   preconditioner=preconditioner, abs_tol=abs_tol)


def GMRES(max_iter: Any = None, rel_tol: Any = None, *, restart: Any = 30,
          preconditioner: Any = None, abs_tol: Any = None) -> Any:
    """Restarted GMRES over the prepared affine route (scheme ``"gmres"``). Inert.

    ``max_iter`` is a MANDATORY positive int; a missing / non-positive budget is refused.
    ``rel_tol`` (ADC-645, optional) supplies the ``P.solve_linear`` tolerance when the call-site
    ``tol`` is left default.
    """
    return _solver("gmres", "pops::solve_prepared_affine", "GMRES", max_iter, rel_tol,
                   restart=restart, preconditioner=preconditioner, abs_tol=abs_tol)


def Richardson(max_iter: Any = None, rel_tol: Any = None, omega: Any = None, *,
               preconditioner: Any = None, abs_tol: Any = None) -> Any:
    """Richardson iteration over the prepared affine route (scheme ``"richardson"``). Inert.

    ``max_iter`` is a MANDATORY positive int; a missing / non-positive budget is refused.
    ``rel_tol`` (ADC-645, optional) supplies the ``P.solve_linear`` tolerance when the call-site
    ``tol`` is left default. ``omega`` (ADC-645, optional) is the Richardson relaxation factor:
    ``None`` (the default) emits the historical ``omega = 1`` literal byte-identically; a finite
    positive value is baked into the typed prepared controls.
    """
    if omega is not None:
        try:
            omega_value = _exact_control(omega, "Richardson omega")
            valid = omega_value > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Richardson: omega must be a positive number or None; got %r"
                             % (omega,))
    return _solver("richardson", "pops::solve_prepared_affine", "Richardson", max_iter, rel_tol,
                   preconditioner=preconditioner,
                   omega=omega_value if omega is not None else None, abs_tol=abs_tol)


__all__ = ["CG", "BiCGStab", "GMRES", "Richardson"]
