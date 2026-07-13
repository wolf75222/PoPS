"""Typed nonlinear solvers used by field residual providers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops.descriptors import Availability, Descriptor, _planned
from pops.descriptors_report import CapabilitySet
from pops.identity import Identity, make_identity
from pops.ir.literals import scalar_data


def _runtime_number(value: Any) -> float:
    if isinstance(value, dict) and value.get("kind") == "binary64":
        return float.fromhex(value["value"])
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


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
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
        if self.restart > 50:
            raise ValueError("Newton restart must be <= 50")
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

    def lower_field_nonlinear(self, *, target: str, layout: Any) -> dict[str, Any]:
        del layout
        if target not in ("system", "amr_system"):
            raise ValueError("Newton field outer solve requires a uniform or AMR system")
        authored = self.options()
        options = {
            key: value if isinstance(value, int) else scalar_data(value)
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


def FixedPoint(**options: Any) -> Any:
    """Catalogued fixed-point descriptor; no native field provider claims it yet."""
    return _planned("fixed_point", "fixed_point", category="solver", **options)


__all__ = ["FixedPoint", "Newton", "PreparedFieldNonlinear"]
