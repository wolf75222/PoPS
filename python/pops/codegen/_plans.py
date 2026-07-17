"""Exact immutable values for the resolve, bind, and install phases.

Every value owns a canonical identity and can re-verify the live values from which that identity
was captured.  Container inputs are recursively frozen.  Runtime arrays and opaque resources are
retained by reference (installation needs the real object), while their content/canonical evidence
is captured so mutation between bind and install is detected.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from types import MappingProxyType
from typing import Any, cast

from pops.identity import Identity, canonical_bytes, make_identity


_TARGETS = frozenset({"system", "amr_system"})
_BIND_RESOURCE_KEYS = frozenset({"execution_context"})
_SEMANTIC_OVERRIDE_KEYS = frozenset({
    "solver", "solvers", "cadence", "layout", "target", "backend", "spatial",
    "outputs", "diagnostics", "program", "algorithm",
})
_ATOMIC = (type(None), bool, int, str, bytes)


def _builtin_restart_authority():
    from pops.output._restart_provider import RestartAuthority

    return RestartAuthority.from_consumer_graph(None)


def _deep_freeze(value: Any) -> Any:
    """Freeze container structure without copying runtime payloads such as arrays/resources."""
    if isinstance(value, Mapping):
        return MappingProxyType({_deep_freeze(key): _deep_freeze(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _string_mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    if any(not isinstance(key, str) or not key for key in value):
        raise TypeError("%s keys must be non-empty strings" % where)
    return _deep_freeze(value)


def _array_evidence(value: Any, *, where: str) -> dict[str, Any] | None:
    """Return content evidence for an array-like value, or ``None`` when it is not array-like."""
    if not (hasattr(value, "__array__") or hasattr(value, "__array_interface__")):
        return None
    import numpy as np

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("%s must not use object dtype" % where)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(canonical_bytes({
        "protocol": "pops.array-evidence.v1",
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }))
    digest.update(memoryview(cast(Any, contiguous)).cast("B"))
    return {
        "kind": "array",
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "content_sha256": digest.hexdigest(),
    }


def _evidence(value: Any, *, where: str) -> Any:
    """Project a resolved value into strict canonical identity data."""
    array = _array_evidence(value, where=where)
    if array is not None:
        return array
    if isinstance(value, _ATOMIC):
        return value
    if isinstance(value, float):
        return {"binary64": value.hex()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, Fraction):
        return {"rational": [value.numerator, value.denominator]}
    if isinstance(value, Enum):
        return {"enum": "%s.%s.%s" % (
            type(value).__module__, type(value).__qualname__, value.name)}
    if isinstance(value, type):
        return {"symbol": "%s.%s" % (value.__module__, value.__qualname__)}
    if type(value) is Identity:
        return {"identity": value.token}
    rows = getattr(value, "rows", None)
    if callable(rows):
        schema = getattr(value, "schema", None)
        schema_hash = getattr(schema, "hash", None)
        return {
            "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
            "schema_hash": schema_hash,
            "rows": _evidence(rows(), where=where),
        }
    if isinstance(value, Mapping):
        rows = [(_evidence(key, where="%s.key" % where),
                 _evidence(item, where="%s.value" % where))
                for key, item in value.items()]
        rows.sort(key=lambda row: canonical_bytes(row[0]))
        if all(isinstance(row[0], str) for row in rows):
            return {row[0]: row[1] for row in rows}
        return {"mapping": [[key, item] for key, item in rows]}
    if isinstance(value, (list, tuple)):
        return [_evidence(item, where=where) for item in value]
    if isinstance(value, (set, frozenset)):
        rows = [_evidence(item, where=where) for item in value]
        return {"set": sorted(rows, key=canonical_bytes)}

    for name in ("artifact_data", "to_data", "to_manifest", "to_dict",
                 "canonical_identity", "options"):
        hook = getattr(value, name, None)
        if callable(hook):
            return {
                "type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                "value": _evidence(hook(), where=where),
            }
    for name in ("artifact_identity", "binary_identity", "definition_identity",
                 "semantic_identity"):
        identity = getattr(value, name, None)
        if type(identity) is Identity:
            return {"type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                    name: identity.token}
        if identity is not None:
            return {"type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                    name: _evidence(identity, where=where)}
    for name in ("module_hash", "_model_hash", "_ir_hash"):
        hook = getattr(value, name, None)
        if callable(hook):
            return {"type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                    name: _evidence(hook(), where=where)}
    module = getattr(value, "module", None)
    module_hash = getattr(module, "module_hash", None)
    if callable(module_hash):
        return {"type": "%s.%s" % (type(value).__module__, type(value).__qualname__),
                "module_hash": _evidence(module_hash(), where=where)}
    raise TypeError(
        "%s contains %s without canonical identity evidence"
        % (where, type(value).__name__))


def _require_identity(value: Any, domain: str, *, where: str) -> Identity:
    if type(value) is not Identity:
        raise TypeError("%s must be an exact pops.identity.Identity" % where)
    if value.domain != domain:
        raise ValueError("%s must have domain %r" % (where, domain))
    return Identity.from_data(value.to_data())


@dataclass(frozen=True, slots=True)
class ResolvedBlock:
    """One fully resolved compiler input block."""

    name: str
    model: Any
    spatial: Any
    backend: str
    state_spaces: tuple[str, ...]
    state_identities: tuple[str, ...] = ()
    numerics: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("ResolvedBlock name must be a non-empty string")
        if not isinstance(self.backend, str) or not self.backend:
            raise TypeError("ResolvedBlock backend must be a resolved non-empty string")
        state_spaces = tuple(self.state_spaces)
        if not state_spaces or any(
                not isinstance(name, str) or not name for name in state_spaces):
            raise TypeError("ResolvedBlock state_spaces must contain non-empty strings")
        if len(set(state_spaces)) != len(state_spaces):
            raise ValueError("ResolvedBlock state_spaces contains a duplicate")
        object.__setattr__(self, "state_spaces", state_spaces)
        state_identities = tuple(self.state_identities)
        if (len(state_identities) != len(state_spaces)
                or any(not isinstance(identity, str) or not identity
                       for identity in state_identities)
                or len(set(state_identities)) != len(state_identities)):
            raise TypeError(
                "ResolvedBlock state_identities must uniquely qualify every state space")
        object.__setattr__(self, "state_identities", state_identities)
        _evidence(self.model, where="ResolvedBlock.model")
        object.__setattr__(self, "spatial", _deep_freeze(self.spatial))
        _evidence(self.spatial, where="ResolvedBlock.spatial")
        object.__setattr__(self, "numerics", _deep_freeze(self.numerics))
        _evidence(self.numerics, where="ResolvedBlock.numerics")


@dataclass(frozen=True, slots=True)
class ResolvedSimulationPlan:
    """The exact, authenticated output of resolve and sole input accepted by compile."""

    snapshot: Any
    target: str
    backend: str
    layout: Any
    layout_plan: Any
    layout_targets: Mapping[str, str]
    time: Any
    blocks: tuple[ResolvedBlock, ...]
    bind_schema: Any
    compile_values: Mapping[Any, Any]
    field_plans: Mapping[str, Any]
    libraries: tuple[Any, ...]
    requirements: Mapping[str, Any]
    capabilities: Mapping[str, Any]
    lowering_coverage: Any
    consumer_graph: Any = None
    restart_authority: Any = field(default_factory=_builtin_restart_authority)
    component_inputs: tuple[Any, ...] = ()
    compile_options: Mapping[str, Any] = field(default_factory=dict)
    resolved_hierarchy: Any = None
    amr_transfer: Any = None
    initial_condition_plan: Any = None
    bootstrap_plan: Any = None
    amr_execution: Any = None
    amr_providers: Mapping[str, Any] = field(default_factory=dict)
    plan_identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        from pops.problem._snapshot import AuthoringSnapshot
        from pops.model.bind_schema import BindSchema
        from pops.mesh import LayoutPlan
        from pops.codegen.lowering_coverage import LoweringCoverageReport

        if type(self.snapshot) is not AuthoringSnapshot:
            raise TypeError("ResolvedSimulationPlan.snapshot must be an AuthoringSnapshot")
        if self.target not in _TARGETS:
            raise ValueError("ResolvedSimulationPlan target must be 'system' or 'amr_system'")
        if not isinstance(self.backend, str) or not self.backend:
            raise TypeError("ResolvedSimulationPlan backend must be a resolved non-empty string")
        if type(self.layout_plan) is not LayoutPlan:
            raise TypeError("ResolvedSimulationPlan.layout_plan must be an exact LayoutPlan")
        targets = _string_mapping(
            self.layout_targets, where="ResolvedSimulationPlan.layout_targets")
        expected_targets = tuple(row.handle.qualified_id for row in self.layout_plan.layouts)
        if tuple(targets) != expected_targets:
            raise ValueError(
                "ResolvedSimulationPlan.layout_targets must match normalized layout order exactly")
        if any(value not in _TARGETS for value in targets.values()):
            raise ValueError("ResolvedSimulationPlan.layout_targets contains an unsupported target")
        object.__setattr__(self, "layout_targets", targets)
        if type(self.lowering_coverage) is not LoweringCoverageReport:
            raise TypeError(
                "ResolvedSimulationPlan.lowering_coverage must be a LoweringCoverageReport")
        if type(self.bind_schema) is not BindSchema:
            raise TypeError("ResolvedSimulationPlan.bind_schema must be an exact BindSchema")
        blocks = tuple(self.blocks)
        if not blocks or any(type(block) is not ResolvedBlock for block in blocks):
            raise TypeError("ResolvedSimulationPlan blocks must contain exact ResolvedBlock values")
        names = [block.name for block in blocks]
        if len(set(names)) != len(names):
            raise ValueError("ResolvedSimulationPlan block names must be unique")
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "layout", _deep_freeze(self.layout))
        object.__setattr__(self, "time", _deep_freeze(self.time))
        if not isinstance(self.compile_values, Mapping):
            raise TypeError("ResolvedSimulationPlan.compile_values must be a mapping")
        object.__setattr__(self, "compile_values", _deep_freeze(self.compile_values))
        expected_compile_values = self.bind_schema.resolve_compile()
        if _evidence(self.compile_values, where="resolved compile values") != _evidence(
                expected_compile_values, where="BindSchema compile values"):
            raise ValueError(
                "ResolvedSimulationPlan.compile_values must exactly match BindSchema.resolve_compile()"
            )
        object.__setattr__(self, "field_plans", _string_mapping(
            self.field_plans, where="ResolvedSimulationPlan.field_plans"))
        for name, registration in self.field_plans.items():
            from pops.codegen.field_install import ResolvedFieldInstallPlan
            if not isinstance(registration, ResolvedFieldInstallPlan):
                raise TypeError(
                    "ResolvedSimulationPlan.field_plans[%r] must be a total resolved install plan"
                    % name)
        for name in ("libraries",):
            object.__setattr__(
                self, name, tuple(_deep_freeze(item) for item in getattr(self, name)))
        if self.libraries:
            raise ValueError(
                "ResolvedSimulationPlan libraries are retired; use authenticated external "
                "component packages through compile_component")
        from pops.external import CompiledComponentArtifact, ExternalComponent
        component_inputs = tuple(self.component_inputs)
        if any(type(item) not in (ExternalComponent, CompiledComponentArtifact)
               for item in component_inputs):
            raise TypeError(
                "ResolvedSimulationPlan.component_inputs must contain exact external component "
                "authoring values or compiled component artifacts")
        component_ids = [
            item.component_id if type(item) is CompiledComponentArtifact
            else item.component_manifest.component_id
            for item in component_inputs
        ]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("ResolvedSimulationPlan.component_inputs contains a duplicate component")
        for item in component_inputs:
            if type(item) is CompiledComponentArtifact:
                item.verify()
            else:
                item.component_type.interface.require_manifest(item.component_manifest)
        for field_plan in self.field_plans.values():
            field_plan.require_component_inputs(component_inputs)
        object.__setattr__(self, "component_inputs", component_inputs)
        for block in self.blocks:
            if block.numerics is None:
                continue
            for boundary in block.numerics.boundaries:
                bindings = tuple(getattr(boundary, "component_bindings", ()))
                if not bindings:
                    continue
                require = getattr(boundary, "require_component_inputs", None)
                if not callable(require):
                    raise TypeError(
                        "boundary component bindings require require_component_inputs(components)"
                    )
                require(component_inputs)
        if self.consumer_graph is not None:
            from pops.output import ConsumerGraph

            if type(self.consumer_graph) is not ConsumerGraph:
                raise TypeError(
                    "ResolvedSimulationPlan.consumer_graph must be an exact ConsumerGraph or None")
            if not self.consumer_graph.is_resolved:
                raise TypeError(
                    "ResolvedSimulationPlan.consumer_graph must be layout-resolved")
        from pops.output._restart_provider import RestartAuthority
        if type(self.restart_authority) is not RestartAuthority:
            raise TypeError(
                "ResolvedSimulationPlan.restart_authority must be an exact RestartAuthority")
        expected_restart = RestartAuthority.from_consumer_graph(self.consumer_graph)
        if self.restart_authority.identity != expected_restart.identity:
            raise ValueError(
                "ResolvedSimulationPlan.restart_authority differs from its ConsumerGraph")
        object.__setattr__(self, "requirements", _string_mapping(
            self.requirements, where="ResolvedSimulationPlan.requirements"))
        object.__setattr__(self, "capabilities", _string_mapping(
            self.capabilities, where="ResolvedSimulationPlan.capabilities"))
        object.__setattr__(self, "compile_options", _string_mapping(
            self.compile_options, where="ResolvedSimulationPlan.compile_options"))
        object.__setattr__(self, "amr_providers", _string_mapping(
            self.amr_providers, where="ResolvedSimulationPlan.amr_providers"))
        self._validate_amr_authorities()
        object.__setattr__(self, "plan_identity", make_identity("resolved-plan", self._payload()))

    def _validate_amr_authorities(self) -> None:
        from pops.codegen._amr_plan_validation import validate_amr_authorities

        validate_amr_authorities(self)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "snapshot_artifact_hash": self.snapshot.artifact_hash,
            "target": self.target,
            "backend": self.backend,
            "bind_schema_artifact_hash": self.bind_schema.artifact_hash,
            "compile_values": _evidence(self.compile_values, where="plan.compile_values"),
            "layout": _evidence(self.layout, where="plan.layout"),
            "layout_plan": _evidence(self.layout_plan, where="plan.layout_plan"),
            "layout_targets": dict(self.layout_targets),
            "time": _evidence(self.time, where="plan.time") if self.time is not None else None,
            "blocks": [{
                "name": block.name,
                "backend": block.backend,
                "state_spaces": block.state_spaces,
                "state_identities": block.state_identities,
                "model": _evidence(block.model, where="plan.block.model"),
                "spatial": _evidence(block.spatial, where="plan.block.spatial"),
                "numerics": _evidence(block.numerics, where="plan.block.numerics"),
            } for block in self.blocks],
            "field_plans": _evidence(self.field_plans, where="plan.field_plans"),
            "consumer_graph": (
                None if self.consumer_graph is None else self.consumer_graph.to_data()
            ),
            "restart_authority": self.restart_authority.to_data(),
            "libraries": _evidence(self.libraries, where="plan.libraries"),
            "component_inputs": _evidence(
                self.component_inputs, where="plan.component_inputs"),
            "requirements": _evidence(self.requirements, where="plan.requirements"),
            "capabilities": _evidence(self.capabilities, where="plan.capabilities"),
            "lowering_coverage": _evidence(
                self.lowering_coverage, where="plan.lowering_coverage"),
            "compile_options": _evidence(self.compile_options, where="plan.compile_options"),
            "resolved_hierarchy": _evidence(
                self.resolved_hierarchy, where="plan.resolved_hierarchy"
            ) if self.resolved_hierarchy is not None else None,
            "amr_transfer": _evidence(
                self.amr_transfer, where="plan.amr_transfer"
            ) if self.amr_transfer is not None else None,
            "initial_condition_plan": _evidence(
                self.initial_condition_plan, where="plan.initial_condition_plan"
            ) if self.initial_condition_plan is not None else None,
            "bootstrap_plan": _evidence(
                self.bootstrap_plan, where="plan.bootstrap_plan"
            ) if self.bootstrap_plan is not None else None,
            "amr_execution": _evidence(
                self.amr_execution, where="plan.amr_execution"
            ) if self.amr_execution is not None else None,
            "amr_providers": _evidence(
                self.amr_providers, where="plan.amr_providers"),
        }

    def verify(self) -> None:
        expected = make_identity("resolved-plan", self._payload())
        if self.plan_identity != expected:
            raise ValueError("ResolvedSimulationPlan identity verification failed")


def _canonicalize_initial_value_mapping(initial_plan: Any, values: Any) -> Mapping[Any, Any]:
    """Resolve only canonical subjects and aliases captured by the originating Case registry."""
    from pops.model import Handle

    if not isinstance(values, Mapping):
        raise TypeError("pops.bind initial_values must be a Handle-keyed mapping")
    if not values:
        return {}
    canonical_subject = getattr(initial_plan, "canonical_subject", None)
    if not callable(canonical_subject):
        raise TypeError(
            "pops.bind initial_values requires an authenticated InitialConditionPlan"
        )
    canonical: dict[Handle, Any] = {}
    for supplied, value in values.items():
        subject = canonical_subject(supplied)
        if not isinstance(subject, Handle) or not subject.is_resolved:
            raise TypeError(
                "InitialConditionPlan.canonical_subject must return a canonical "
                "owner-qualified Handle"
            )
        if subject in canonical:
            raise ValueError(
                "pops.bind initial_values contains multiple aliases for %s"
                % subject.qualified_id
            )
        canonical[subject] = value
    return canonical


def _canonical_initial_values(artifact: Any, values: Any) -> Mapping[Any, Any]:
    """Authenticate public authored/resolved initial-value Handles against one artifact.

    ``BindInputs`` remains an internal canonical record.  The public boundary may receive the live
    block-qualified Handle returned while authoring a Case; resolve it once here and replace it with
    the exact subject retained by the compiled InitialConditionPlan.  A look-alike Handle from
    another Case therefore never becomes a bind alias merely because its local name matches.
    """
    from pops.codegen._compiled_artifact import CompiledSimulationArtifact
    if type(artifact) is not CompiledSimulationArtifact:
        raise TypeError("pops.bind requires an exact CompiledSimulationArtifact")
    if not isinstance(values, Mapping):
        raise TypeError("pops.bind initial_values must be a Handle-keyed mapping")
    if not values:
        return {}
    initial_plan = artifact.plan.initial_condition_plan
    if initial_plan is None:
        raise ValueError("pops.bind initial_values requires a resolved InitialConditionPlan")
    return _canonicalize_initial_value_mapping(initial_plan, values)


@dataclass(frozen=True, slots=True)
class BindInputs:
    """Concrete values/resources accepted by bind, with reference-preserving evidence."""

    initial_state: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[Any, Any] = field(default_factory=dict)
    aux: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    initial_values: Mapping[Any, Any] = field(default_factory=dict)
    inputs_identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "initial_state", _string_mapping(
            self.initial_state, where="BindInputs.initial_state"))
        if not isinstance(self.params, Mapping):
            raise TypeError("BindInputs.params must be a mapping")
        object.__setattr__(self, "params", _deep_freeze(self.params))
        object.__setattr__(self, "aux", _string_mapping(self.aux, where="BindInputs.aux"))
        object.__setattr__(self, "resources", _string_mapping(
            self.resources, where="BindInputs.resources"))
        if not isinstance(self.initial_values, Mapping):
            raise TypeError("BindInputs.initial_values must be a Handle-keyed mapping")
        from pops.model import Handle
        if any(not isinstance(key, Handle) or not key.is_resolved for key in self.initial_values):
            raise TypeError("BindInputs.initial_values keys must be canonical owner-qualified Handles")
        object.__setattr__(self, "initial_values", _deep_freeze(self.initial_values))
        forbidden = set(self.resources) & _SEMANTIC_OVERRIDE_KEYS
        unknown = set(self.resources) - _BIND_RESOURCE_KEYS
        if forbidden:
            raise TypeError(
                "BindInputs resources cannot override resolved semantics: %s"
                % sorted(forbidden))
        if unknown:
            raise TypeError(
                "BindInputs resources support only %s (got %s)"
                % (sorted(_BIND_RESOURCE_KEYS), sorted(unknown)))
        object.__setattr__(self, "inputs_identity", make_identity("bind-inputs", self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "initial_state": _evidence(self.initial_state, where="bind.initial_state"),
            "params": _evidence(self.params, where="bind.params"),
            "aux": _evidence(self.aux, where="bind.aux"),
            "resources": _evidence(self.resources, where="bind.resources"),
            "initial_values": _evidence(
                self.initial_values, where="bind.initial_values"
            ),
        }

    def verify(self) -> None:
        expected = make_identity("bind-inputs", self._payload())
        if self.inputs_identity != expected:
            raise ValueError(
                "BindInputs identity verification failed; a value/resource was mutated")


@dataclass(frozen=True, slots=True)
class InstallPlan:
    """Final bind-created value and the only input accepted by runtime installation."""

    artifact: Any
    bind_inputs: BindInputs
    instances: Mapping[str, Any]
    params: Any
    aux: Mapping[str, Any]
    resources: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, Any] = field(default_factory=dict)
    execution_context: Any = None
    bind_identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        from pops.codegen._compiled_artifact import CompiledSimulationArtifact
        from pops.model.resolved_bindings import ResolvedBindings

        if type(self.artifact) is not CompiledSimulationArtifact:
            raise TypeError("InstallPlan.artifact must be an exact CompiledSimulationArtifact")
        if type(self.bind_inputs) is not BindInputs:
            raise TypeError("InstallPlan.bind_inputs must be an exact BindInputs")
        object.__setattr__(self, "instances", _string_mapping(
            self.instances, where="InstallPlan.instances"))
        if type(self.params) is not ResolvedBindings:
            raise TypeError("InstallPlan.params must be exact resolved BindSchema values")
        if self.params.schema.hash != self.artifact.bind_schema.hash:
            raise ValueError("InstallPlan.params were resolved from a different BindSchema")
        object.__setattr__(self, "aux", _string_mapping(self.aux, where="InstallPlan.aux"))
        object.__setattr__(self, "resources", _string_mapping(
            self.resources, where="InstallPlan.resources"))
        object.__setattr__(self, "components", _string_mapping(
            self.components, where="InstallPlan.components"))
        from pops.external import InstalledComponent
        expected_components = tuple(
            item.component_id for item in self.artifact.component_artifacts)
        if tuple(self.components) != expected_components:
            raise ValueError(
                "InstallPlan components must match compiled component artifact order exactly")
        for artifact in self.artifact.component_artifacts:
            installed = self.components[artifact.component_id]
            if type(installed) is not InstalledComponent:
                raise TypeError("InstallPlan components must contain exact InstalledComponent values")
            if installed.artifact_identity != artifact.artifact_identity:
                raise ValueError("InstallPlan component changed the compiled artifact identity")
            if installed.native_handle is None:
                raise ValueError("InstallPlan components must be loaded before runtime installation")
            installed.verify()
        from pops._platform_contracts import (
            ExecutionContext,
            serial_execution_context,
            validate_component_launch,
            validate_launch,
        )
        context = self.execution_context
        if context is None:
            context = serial_execution_context(self.artifact.platform_manifest)
        if type(context) is not ExecutionContext:
            raise TypeError("InstallPlan.execution_context must be an exact ExecutionContext")
        supplied_context = self.resources.get("execution_context")
        if supplied_context is not None and supplied_context is not context:
            raise ValueError(
                "InstallPlan execution_context must be the exact BindInputs resource")
        validate_launch(self.artifact.platform_manifest, context, ())
        for component in self.artifact.component_artifacts:
            validate_component_launch(component.platform_manifest, context, ())
        object.__setattr__(self, "execution_context", context)
        expected_names = tuple(block.name for block in self.artifact.blocks)
        if tuple(self.instances) != expected_names:
            raise ValueError("InstallPlan instances must match compiled block order exactly")
        for block in self.artifact.blocks:
            instance = self.instances[block.name]
            if not isinstance(instance, Mapping) or not set(instance).issubset(
                    {"model", "spatial", "initial"}):
                raise TypeError(
                    "InstallPlan instance %r must contain only model/spatial/initial" % block.name)
            if instance.get("model") is not block.model:
                raise ValueError(
                    "InstallPlan instance %r changed the compiled model" % block.name)
            if _evidence(instance.get("spatial"), where="install spatial") != _evidence(
                    block.spatial, where="artifact spatial"):
                raise ValueError(
                    "InstallPlan instance %r changed the resolved spatial descriptor" % block.name)
            expected_initial = self.bind_inputs.initial_state.get(block.name)
            if ("initial" in instance) != (block.name in self.bind_inputs.initial_state) \
                    or ("initial" in instance and instance["initial"] is not expected_initial):
                raise ValueError(
                    "InstallPlan instance %r initial state does not come from BindInputs"
                    % block.name)
        if _evidence(self.aux, where="InstallPlan.aux") != _evidence(
                self.bind_inputs.aux, where="BindInputs.aux"):
            raise ValueError("InstallPlan aux values must come from BindInputs")
        if _evidence(self.resources, where="InstallPlan.resources") != _evidence(
                self.bind_inputs.resources, where="BindInputs.resources"):
            raise ValueError("InstallPlan resources must come from BindInputs")
        object.__setattr__(self, "bind_identity", make_identity("bind", self._payload()))

    @property
    def target(self) -> str:
        return self.artifact.plan.target

    @property
    def layout(self) -> Any:
        return self.artifact.plan.layout

    @property
    def capabilities(self) -> Mapping[str, Any]:
        return self.artifact.plan.capabilities

    @property
    def resolved_hierarchy(self) -> Any:
        return self.artifact.plan.resolved_hierarchy

    @property
    def amr_transfer(self) -> Any:
        return self.artifact.plan.amr_transfer

    @property
    def initial_condition_plan(self) -> Any:
        return self.artifact.plan.initial_condition_plan

    @property
    def bootstrap_plan(self) -> Any:
        return self.artifact.plan.bootstrap_plan

    @property
    def amr_execution(self) -> Any:
        return self.artifact.plan.amr_execution

    @property
    def amr_providers(self) -> Mapping[str, Any]:
        return self.artifact.plan.amr_providers

    @property
    def initial_values(self) -> Mapping[Any, Any]:
        return self.bind_inputs.initial_values

    @property
    def n_blocks(self) -> int:
        return len(self.artifact.blocks)

    @property
    def block_models(self) -> Mapping[str, Any]:
        return MappingProxyType({block.name: block.model for block in self.artifact.blocks})

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_identity": self.artifact.artifact_identity.to_data(),
            "inputs_identity": self.bind_inputs.inputs_identity.to_data(),
            "target": self.target,
            "capabilities": _evidence(self.capabilities, where="install.capabilities"),
            "instances": _evidence(self.instances, where="install.instances"),
            "params": _evidence(self.params, where="install.params"),
            "aux": _evidence(self.aux, where="install.aux"),
            "resources": _evidence(self.resources, where="install.resources"),
            "components": _evidence(self.components, where="install.components"),
            "initial_values": _evidence(
                self.initial_values, where="install.initial_values"
            ),
            "execution_context": _evidence(
                self.execution_context, where="install.execution_context"),
        }

    def verify(self) -> None:
        self.artifact.verify()
        self.bind_inputs.verify()
        expected = make_identity("bind", self._payload())
        if self.bind_identity != expected:
            raise ValueError("InstallPlan bind identity verification failed")


def require_install_plan(value: Any) -> InstallPlan:
    """Reject every wrong-phase value; installation accepts no structural lookalikes."""
    if type(value) is not InstallPlan:
        raise TypeError("runtime installation requires an exact InstallPlan")
    value.verify()
    return value


__all__ = [
    "BindInputs", "InstallPlan", "ResolvedBlock", "ResolvedSimulationPlan",
    "require_install_plan",
]
