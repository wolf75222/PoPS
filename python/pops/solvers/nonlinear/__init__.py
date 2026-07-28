"""Typed nonlinear solvers used by field residual providers."""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops.descriptors import Availability, Descriptor
from pops.descriptors_report import CapabilitySet
from pops.identity import Identity, make_identity


def _scalar_data(value: Any) -> dict[str, Any]:
    """Serialize an exact scalar without making the solver catalog import the symbolic IR."""
    from pops.identity.scalar import scalar_data

    return scalar_data(value)


def _runtime_number(value: Any) -> float:
    if isinstance(value, Mapping):
        encoded = value.get("value")
        if value.get("kind") != "binary64" or not isinstance(encoded, str):
            raise TypeError("nonlinear runtime scalars require canonical binary64 data")
        return float.fromhex(encoded)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("nonlinear runtime scalars must be numeric or canonical binary64 data")
    return float(value)


@dataclass(frozen=True, slots=True)
class PreparedFieldNonlinear:
    """Authenticated executable prepared by a nonlinear descriptor.

    Resolve validates this small protocol and retains the object beside its canonical manifest.
    Bind invokes :meth:`install`; neither phase switches on a builtin class or algorithm string.
    External providers can implement the same protocol without modifying field installation.
    """

    target: str
    options: Any
    capabilities: frozenset[str]
    identity: Identity

    def __post_init__(self) -> None:
        if self.target not in ("system", "amr_system"):
            raise ValueError("PreparedFieldNonlinear target is unsupported")
        options = MappingProxyType(dict(self.options))
        required = {
            "tolerance", "max_iterations", "linear_tolerance",
            "linear_max_iterations", "restart", "armijo", "minimum_step",
        }
        if set(options) != required:
            raise ValueError("PreparedFieldNonlinear options are incomplete")
        capabilities = frozenset(self.capabilities)
        if not {"residual", "publication_atomic", "reject_attempt"}.issubset(capabilities):
            raise ValueError("PreparedFieldNonlinear omits required solve capabilities")
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "capabilities", capabilities)
        expected = make_identity("prepared-field-nonlinear", self._payload())
        if self.identity != expected:
            raise ValueError("PreparedFieldNonlinear identity is not canonical")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "target": self.target,
            "options": dict(self.options),
            "capabilities": sorted(self.capabilities),
        }

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.token}

    def install(self, runtime: Any, provider_slot: str) -> None:
        setter = getattr(runtime, "set_field_newton_plan", None)
        if not callable(setter):
            raise TypeError(
                "prepared nonlinear provider requires the field nonlinear install protocol")
        o = self.options
        setter(provider_slot, _runtime_number(o["tolerance"]), o["max_iterations"],
               _runtime_number(o["linear_tolerance"]), o["linear_max_iterations"], o["restart"],
               _runtime_number(o["armijo"]), _runtime_number(o["minimum_step"]))


@dataclass(frozen=True, slots=True)
class _PreparedLocalNewton:
    """Immutable controls shared by every form of cell-local nonlinear solve."""

    tolerance: float
    relative_tolerance: float
    step_tolerance: float
    max_iterations: int
    max_evaluations: int
    finite_difference_step: float
    safeguard: str
    damping: float
    max_backtracks: int
    minimum_step: float
    armijo: float
    identity: Identity

    def __post_init__(self) -> None:
        payload = {
            "schema_version": 2,
            "tolerance": _scalar_data(self.tolerance),
            "relative_tolerance": _scalar_data(self.relative_tolerance),
            "step_tolerance": _scalar_data(self.step_tolerance),
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "finite_difference_step": _scalar_data(self.finite_difference_step),
            "safeguard": self.safeguard,
            "damping": _scalar_data(self.damping),
            "max_backtracks": self.max_backtracks,
            "minimum_step": _scalar_data(self.minimum_step),
            "armijo": _scalar_data(self.armijo),
        }
        if self.identity != make_identity("prepared-local-newton", payload):
            raise ValueError("prepared LocalNewton identity is not canonical")

    def build_program_solve(
        self, *, program: Any, problem: Any, name: Any = None,
    ) -> Any:
        """Join two explicit small protocols without inspecting a problem class."""
        build = getattr(problem, "build_with", None)
        if not callable(build):
            raise TypeError("LocalNewton requires a typed Program solve problem")
        return build(program=program, prepared_solver=self, name=name)


_NATIVE_INT_MAX = (1 << 31) - 1


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or value <= 0 or value > _NATIVE_INT_MAX:
        raise ValueError("Newton %s must be a positive Python int" % name)
    return value


def _positive_float(value: Any, name: str, *, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError("Newton %s must be a finite positive scalar" % name)
    result = float(value)
    if upper is not None and result >= upper:
        raise ValueError("Newton %s must be < %s" % (name, upper))
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("Newton %s must be a finite non-negative scalar" % name)
    return float(value)


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < 0 or value > _NATIVE_INT_MAX:
        raise ValueError("Newton %s must be a non-negative Python int" % name)
    return value


class Newton(Descriptor):
    """Global damped Newton-Krylov outer solve for a nonlinear field residual.

    The field discretization's linear solver is used only as the preconditioner.  Residual and JVP
    come from the resolved FieldResidual/Boundary providers; publication occurs only after the
    accepted iterate satisfies the nonlinear tolerance.
    """

    category = "nonlinear_solver"
    native_id = "pops::FieldNewtonSolver"
    scheme = "newton_krylov"

    def __init__(
        self,
        *,
        tolerance: Any = 1.0e-8,
        max_iterations: Any = 20,
        linear_tolerance: Any = 1.0e-3,
        linear_max_iterations: Any = 80,
        restart: Any = 30,
        armijo: Any = 1.0e-4,
        minimum_step: Any = 1.0 / 1024.0,
    ) -> None:
        self.tolerance = _positive_float(tolerance, "tolerance")
        self.max_iterations = _positive_int(max_iterations, "max_iterations")
        self.linear_tolerance = _positive_float(linear_tolerance, "linear_tolerance")
        self.linear_max_iterations = _positive_int(
            linear_max_iterations, "linear_max_iterations")
        self.restart = _positive_int(restart, "restart")
        self.armijo = _positive_float(armijo, "armijo", upper=1.0)
        self.minimum_step = _positive_float(minimum_step, "minimum_step", upper=1.0)

    @property
    def name(self) -> str:
        return "newton_krylov"

    def options(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "linear_tolerance": self.linear_tolerance,
            "linear_max_iterations": self.linear_max_iterations,
            "restart": self.restart,
            "armijo": self.armijo,
            "minimum_step": self.minimum_step,
        }

    def to_data(self) -> dict[str, Any]:
        return {"scheme": self.scheme, **self.options()}

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet({
            "nonlinear_residual": True,
            "jvp": True,
            "line_search": True,
            "publication_atomic": True,
            "uniform": True,
            "amr": True,
        })

    def available(self, context: Any = None) -> Availability:
        del context
        return Availability.yes("native damped Newton-Krylov field outer solve")

    def lower_field_nonlinear(self, *, target: str, layout: Any) -> PreparedFieldNonlinear:
        del layout
        if target not in ("system", "amr_system"):
            raise ValueError("Newton field outer solve requires a uniform or AMR system")
        authored = self.options()
        options = {
            key: value if isinstance(value, int) else _scalar_data(value)
            for key, value in authored.items()
        }
        capabilities = frozenset({
            "residual", "jvp", "line_search", "publication_atomic", "reject_attempt",
            "newton_krylov" if target == "system" else "full_approximation_scheme",
        })
        payload = {
            "schema_version": 1, "target": target, "options": options,
            "capabilities": sorted(capabilities),
        }
        return PreparedFieldNonlinear(
            target, options, capabilities,
            make_identity("prepared-field-nonlinear", payload))


class LocalNewton(Descriptor):
    """Typed controls for the single prepared cell-local nonlinear provider."""

    category = "nonlinear_solver"
    native_id = "pops::LocalNewton"
    scheme = "newton"

    def __init__(
        self,
        *,
        tolerance: Any = 1.0e-12,
        relative_tolerance: Any = 0.0,
        step_tolerance: Any = 0.0,
        max_iterations: Any = 20,
        max_evaluations: Any = 0,
        finite_difference_step: Any = 1.0e-7,
        safeguard: Any = "exact",
        damping: Any = 1.0,
        max_backtracks: Any = 12,
        minimum_step: Any = 1.0 / 4096.0,
        armijo: Any = 1.0e-4,
    ) -> None:
        self.tolerance = _positive_float(tolerance, "tolerance")
        self.relative_tolerance = _nonnegative_float(
            relative_tolerance, "relative_tolerance")
        self.step_tolerance = _nonnegative_float(step_tolerance, "step_tolerance")
        self.max_iterations = _positive_int(max_iterations, "max_iterations")
        self.max_evaluations = _nonnegative_int(max_evaluations, "max_evaluations")
        self.finite_difference_step = _positive_float(
            finite_difference_step, "finite_difference_step")
        if safeguard not in ("exact", "damped", "backtracking"):
            raise ValueError(
                "Newton safeguard must be 'exact', 'damped', or 'backtracking'")
        self.safeguard = safeguard
        self.damping = _positive_float(damping, "damping")
        if self.damping > 1.0:
            raise ValueError("Newton damping must be <= 1")
        if self.safeguard == "exact" and self.damping != 1.0:
            raise ValueError("Newton safeguard='exact' requires damping=1")
        self.max_backtracks = _nonnegative_int(max_backtracks, "max_backtracks")
        self.minimum_step = _positive_float(minimum_step, "minimum_step")
        self.armijo = _positive_float(armijo, "armijo", upper=1.0)
        if self.minimum_step > self.damping:
            raise ValueError("Newton minimum_step must be <= damping")

    def to_data(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "tolerance": self.tolerance,
            "relative_tolerance": self.relative_tolerance,
            "step_tolerance": self.step_tolerance,
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "finite_difference_step": self.finite_difference_step,
            "safeguard": self.safeguard,
            "damping": self.damping,
            "max_backtracks": self.max_backtracks,
            "minimum_step": self.minimum_step,
            "armijo": self.armijo,
        }

    def prepare_program_solve(self) -> _PreparedLocalNewton:
        """Prepare the generic Program solve provider."""
        payload = {
            "schema_version": 2,
            "tolerance": _scalar_data(self.tolerance),
            "relative_tolerance": _scalar_data(self.relative_tolerance),
            "step_tolerance": _scalar_data(self.step_tolerance),
            "max_iterations": self.max_iterations,
            "max_evaluations": self.max_evaluations,
            "finite_difference_step": _scalar_data(self.finite_difference_step),
            "safeguard": self.safeguard,
            "damping": _scalar_data(self.damping),
            "max_backtracks": self.max_backtracks,
            "minimum_step": _scalar_data(self.minimum_step),
            "armijo": _scalar_data(self.armijo),
        }
        return _PreparedLocalNewton(
            tolerance=self.tolerance,
            relative_tolerance=self.relative_tolerance,
            step_tolerance=self.step_tolerance,
            max_iterations=self.max_iterations,
            max_evaluations=self.max_evaluations,
            finite_difference_step=self.finite_difference_step,
            safeguard=self.safeguard,
            damping=self.damping,
            max_backtracks=self.max_backtracks,
            minimum_step=self.minimum_step,
            armijo=self.armijo,
            identity=make_identity("prepared-local-newton", payload),
        )

__all__ = [
    "LocalNewton", "Newton", "PreparedFieldNonlinear",
]
