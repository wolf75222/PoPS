"""Typed, callback-free initial-condition authoring."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from pops.model import Handle, OwnerPath


def _protocol(value: Any, method: str, *, where: str) -> Any:
    member = getattr(value, method, None)
    if isinstance(value, (str, bytes)) or callable(value) or not callable(member):
        raise TypeError(
            "%s must implement the data-only %s() protocol; strings and callbacks are forbidden"
            % (where, method))
    return member


@dataclass(frozen=True, slots=True)
class InitialCondition:
    """One qualified physical state, one data provider and one projection authority."""

    state: Handle
    value: Any
    projection: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.state, Handle) or self.state.kind != "state" \
                or not self.state.is_instance:
            raise TypeError(
                "InitialCondition.state must be a block-qualified state Handle")
        validate_value = _protocol(
            self.value, "validate_for", where="InitialCondition.value")
        validate_projection = _protocol(
            self.projection, "validate_for", where="InitialCondition.projection")
        _protocol(self.value, "initial_source_options", where="InitialCondition.value")
        _protocol(
            self.projection,
            "initial_projection_options",
            where="InitialCondition.projection",
        )
        validate_value(self.state)
        validate_projection(self.state, self.value)

    def resolve_references(self, resolver: Any) -> InitialCondition:
        if not callable(resolver):
            raise TypeError("InitialCondition resolver must be callable")
        return type(self)(resolver(self.state), self.value, self.projection)

    def canonical_identity(self) -> dict[str, Any]:
        if not self.state.is_resolved:
            raise TypeError(
                "InitialCondition canonical identity requires a resolved qualified state")
        return {
            "schema_version": 1,
            "state": self.state.canonical_identity(),
            "value": self.value.to_data(),
            "projection": self.projection.to_data(),
        }

    @property
    def qualified_id(self) -> str:
        payload = json.dumps(
            self.canonical_identity(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "pops.initial.v1::sha256:%s" % hashlib.sha256(payload.encode()).hexdigest()

    def source(self, owner: Any) -> Any:
        """Lower a resolved declaration to the existing exact AMR source contract."""
        if not self.state.is_resolved:
            raise TypeError("InitialCondition.source requires a resolved qualified state")
        from pops.mesh._amr import CanonicalOptions, InitialConditionSource

        source_options = dict(self.value.initial_source_options())
        projection_options = dict(self.projection.initial_projection_options())
        overlap = set(source_options).intersection(projection_options)
        if overlap:
            raise ValueError(
                "initial value and projection options collide: %s" % sorted(overlap))
        options = {**source_options, **projection_options}
        if not isinstance(options.get("native_route"), str):
            raise TypeError("initial value protocol must declare a native_route")
        provider = Handle(
            "source_%s" % self.qualified_id.rsplit(":", 1)[-1],
            kind="initial_condition_provider",
            owner=OwnerPath.coerce(owner).canonical(),
        )
        return InitialConditionSource(provider, CanonicalOptions(options))

    def bootstrap_method(self) -> Any:
        if getattr(self.value, "reprojectable", None) is True:
            from pops.mesh._amr import AnalyticReprojection

            return AnalyticReprojection()
        if getattr(self.value, "reprojectable", None) is False:
            from pops.mesh._amr import ProlongFromParent

            return ProlongFromParent()
        raise TypeError("initial value protocol must declare exact bool reprojectable")

    @property
    def bootstrap_phases(self) -> tuple[str, ...]:
        phases = getattr(self.projection, "bootstrap_phases", None)
        if not isinstance(phases, tuple):
            raise TypeError("initial projection must declare tuple bootstrap_phases")
        return phases

    def inspect(self) -> dict[str, Any]:
        return {
            "state": self.state.inspect(),
            "value": self.value.to_data(),
            "projection": self.projection.to_data(),
        }


@dataclass(frozen=True, slots=True)
class InitialConditionAuthorities:
    """The exact initial and bootstrap authorities generated from one Case registry."""

    initial_condition_plan: Any
    bootstrap_plan: Any

    def __post_init__(self) -> None:
        from pops.mesh._amr import BootstrapPlan, InitialConditionPlan

        if type(self.initial_condition_plan) is not InitialConditionPlan:
            raise TypeError("initial_condition_plan must be an exact InitialConditionPlan")
        if type(self.bootstrap_plan) is not BootstrapPlan:
            raise TypeError("bootstrap_plan must be an exact BootstrapPlan")


__all__ = ["InitialCondition", "InitialConditionAuthorities"]
