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
from urllib.parse import unquote

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
    if contiguous.size:
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


def canonical_block_instance_owner(*, case: Any, block: Any, model_owner: Any) -> Any:
    """Return the official canonical OwnerPath of one Case-block model instance."""
    from pops.model.ownership import OwnerKind, OwnerPath
    from pops.problem.handles import BlockHandle

    if isinstance(block, BlockHandle):
        return block.instance_owner_path.canonical()
    if not isinstance(block, str) or not block:
        raise TypeError("block instance identity requires a BlockHandle or non-empty block name")
    if isinstance(case, str):
        case_owner = OwnerPath.case(case)
    else:
        case_owner = OwnerPath.coerce(case)
        if case_owner.kind is not OwnerKind.CASE:
            raise ValueError("block instance identity requires a Case owner")
        if case_owner.is_authoring:
            case_owner = case_owner.canonical()
    return case_owner.child(OwnerKind.BLOCK, block).instance_of(model_owner).canonical()


def _owner_path_of(model: Any) -> Any:
    """Return the canonical model-definition OwnerPath, or None when the model has none."""
    from pops.model.ownership import OwnerKind, OwnerPath, UnresolvedOwnershipError

    owner = getattr(model, "owner_path", None)
    if owner is None:
        owner = getattr(getattr(model, "_m", None), "owner_path", None)
    if owner is None:
        return None
    try:
        path = OwnerPath.coerce(owner)
    except TypeError:
        return None
    if path.is_authoring:
        try:
            path = path.canonical()
        except UnresolvedOwnershipError:
            return None
    if path.kind is not OwnerKind.MODEL_DEFINITION and not path.contains(OwnerKind.MODEL_DEFINITION):
        return None
    return path


def _model_definition_owner(path: Any) -> Any:
    """Return the model-definition suffix of a Case-block instance OwnerPath."""
    from pops.model.ownership import OwnerKind, OwnerPath

    start = next(
        index for index, node in enumerate(path.nodes)
        if node.kind is OwnerKind.MODEL_DEFINITION
    )
    return OwnerPath(
        path.nodes[start:],
        _definition_fingerprint=path.definition_fingerprint,
    )


def _canonical_owner_path_from_qid(qid: str) -> Any:
    """Parse and authenticate the sole accepted string owner representation.

    Resolved-plan records persist ``str(OwnerPath)`` rather than a live owner
    object.  Accepting that representation is safe only when it parses back to
    the exact canonical ``OwnerPath`` spelling; a display string, an authoring
    capability, or a merely similar path must not become identity authority.
    """
    from pops.model.ownership import OwnerKind, OwnerPath, OwnerSegment

    if not qid or "#authoring=" in qid:
        raise ValueError("block instance owner qid must be a canonical OwnerPath string")
    try:
        nodes = []
        fingerprint = None
        for raw_segment in qid.split("/"):
            kind_text, separator, encoded_name = raw_segment.partition(":")
            if not separator or not kind_text or not encoded_name:
                raise ValueError("owner qid segment is malformed")
            kind = OwnerKind(kind_text)
            if kind is OwnerKind.MODEL_DEFINITION:
                encoded_name, marker, encoded_fingerprint = encoded_name.partition("@")
                if not marker or not encoded_name or not encoded_fingerprint:
                    raise ValueError("model-definition qid segment is malformed")
                if fingerprint is not None:
                    raise ValueError("owner qid contains multiple definition fingerprints")
                fingerprint = unquote(encoded_fingerprint)
            elif "@" in encoded_name:
                raise ValueError("owner qid contains an unexpected fingerprint marker")
            nodes.append((kind, unquote(encoded_name)))
        path = OwnerPath(
            tuple(OwnerSegment(kind, name) for kind, name in nodes),
            _definition_fingerprint=fingerprint,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "block instance owner qid must be a canonical OwnerPath string"
        ) from exc
    if str(path) != qid:
        raise ValueError("block instance owner qid must be a canonical OwnerPath string")
    return path


def authenticate_block_instance_owner(
    value: Any,
    *,
    block_name: str | None = None,
    model_owner: Any = None,
    allow_unscoped: bool = False,
) -> tuple[str, Any, Any]:
    """Return ``(qid, instance_owner_data, model_owner_data)`` for one Case block."""
    if value is None or value == "":
        if allow_unscoped:
            return "", None, None
        raise ValueError(
            "public resolved/compiled plans require a canonical Case-block instance owner"
        )
    from pops.model.ownership import OwnerKind, OwnerPath, UnresolvedOwnershipError
    from pops.problem.handles import BlockHandle

    if isinstance(value, BlockHandle):
        path = value.instance_owner_path.canonical()
    elif isinstance(value, OwnerPath):
        path = value.canonical()
    elif isinstance(value, str):
        path = _canonical_owner_path_from_qid(value)
    elif isinstance(value, Mapping):
        path = OwnerPath.from_data(value)
        if path.is_authoring:
            raise UnresolvedOwnershipError(
                "block instance owner must be canonical before compilation"
            )
    else:
        raise TypeError(
            "block instance owner must be a BlockHandle, OwnerPath, or OwnerPath.to_data()"
        )
    kinds = tuple(node.kind for node in path.nodes)
    if OwnerKind.CASE not in kinds or OwnerKind.BLOCK not in kinds:
        raise ValueError("block instance owner must be Case-block qualified")
    if OwnerKind.MODEL_DEFINITION not in kinds:
        raise ValueError("block instance owner must instantiate a model definition")
    if path.is_authoring:
        raise UnresolvedOwnershipError(
            "block instance owner must be canonical before compilation"
        )
    block_nodes = [node for node in path.nodes if node.kind is OwnerKind.BLOCK]
    if len(block_nodes) != 1:
        raise ValueError("block instance owner must contain exactly one Case-block node")
    if block_name is not None and block_nodes[0].name != block_name:
        raise ValueError(
            "block instance owner block node %r does not match resolved block name %r"
            % (block_nodes[0].name, block_name)
        )
    instance_model = _model_definition_owner(path)
    expected_model = model_owner
    if expected_model is not None and not isinstance(expected_model, OwnerPath):
        expected_model = _owner_path_of(expected_model)
    if isinstance(expected_model, OwnerPath):
        if expected_model.is_authoring:
            expected_model = expected_model.canonical()
        if instance_model != expected_model:
            raise ValueError(
                "block instance owner model definition %s does not match resolved model owner %s"
                % (instance_model, expected_model)
            )
        model_data = expected_model.to_data()
    else:
        model_data = instance_model.to_data()
    return str(path), path.to_data(), model_data


def authenticate_block_instance_owner_qid(
    value: Any, *, allow_unscoped: bool = False, block_name: str | None = None,
    model_owner: Any = None,
) -> str:
    """Return one official Case-block instance qid, or empty on the internal unscoped path."""
    qid, _owner, _model = authenticate_block_instance_owner(
        value, block_name=block_name, model_owner=model_owner, allow_unscoped=allow_unscoped,
    )
    return qid


def attest_precompiled_consumer_owner(
    model: Any,
    requested: Any,
    *,
    declare_auxiliary_providers: bool | None = None,
) -> None:
    """Refuse a precompiled binary whose baked consumer owner is not the requested instance."""
    requested_qid = requested if isinstance(requested, str) and requested else (
        authenticate_block_instance_owner_qid(requested, allow_unscoped=False)
    )
    if not isinstance(requested_qid, str) or not requested_qid:
        raise ValueError(
            "public compilation requires a canonical Case-block instance owner"
        )
    observed = getattr(model, "consumer_owner_qid", None)
    if observed != requested_qid:
        raise ValueError(
            "precompiled model consumer owner %r was not baked for Case-block instance %r; "
            "recompile the artifact for this owner"
            % (observed, requested_qid)
        )
    if declare_auxiliary_providers is not None:
        observed_declaration = getattr(model, "declares_auxiliary_providers", None)
        if observed_declaration != bool(declare_auxiliary_providers):
            raise ValueError(
                "precompiled model provider-declaration role %r was not baked for plan role %r; "
                "recompile the artifact for this owner"
                % (observed_declaration, bool(declare_auxiliary_providers))
            )


def _provider_graph_evidence(model: Any) -> Any:
    pack = getattr(model, "_auxiliary_provider_metadata", None)
    if pack is None:
        module = getattr(model, "module", None)
        pack = getattr(module, "provider_pack", None) if module is not None else None
        to_data = getattr(pack, "to_data", None)
        if callable(to_data):
            pack = to_data()
    if pack is None:
        return None
    return _evidence(pack, where="declaration.providers")


def provider_declaration_contract(block: Any) -> bytes:
    """Return the authenticated model/provider contract used to pick one declarer."""
    payload = {
        "model_owner": getattr(block, "model_owner", None),
        "model": _evidence(block.model, where="declaration.model"),
        "providers": _provider_graph_evidence(block.model),
    }
    return canonical_bytes(payload)


def assign_provider_declaration_roles(blocks: tuple[Any, ...]) -> None:
    """Mark exactly one block per identical model/provider contract as the declarer."""
    seen: dict[bytes, str] = {}
    for block in blocks:
        contract = provider_declaration_contract(block)
        if contract not in seen:
            seen[contract] = block.name
            object.__setattr__(block, "declares_auxiliary_providers", True)
        else:
            object.__setattr__(block, "declares_auxiliary_providers", False)


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
    instance_owner_qid: Any = ""
    numerics: Any = None
    instance_owner: Any = field(init=False, default=None)
    model_owner: Any = field(init=False, default=None)
    declares_auxiliary_providers: bool = False

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
        qid, owner_data, model_data = authenticate_block_instance_owner(
            self.instance_owner_qid,
            block_name=self.name,
            model_owner=_owner_path_of(self.model),
            allow_unscoped=True,
        )
        object.__setattr__(self, "instance_owner_qid", qid)
        object.__setattr__(self, "instance_owner", owner_data)
        object.__setattr__(self, "model_owner", model_data)
        object.__setattr__(
            self, "declares_auxiliary_providers", bool(self.declares_auxiliary_providers))
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
    native_layouts: Mapping[str, Any] = field(default_factory=dict)
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
    resolved_dimension: int = field(init=False)
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
        from pops.codegen._native_spatial_layout import (
            native_spatial_layouts,
            resolved_dimension,
        )

        expected_native_layouts = native_spatial_layouts(self.layout_plan)
        supplied_native_layouts = self.native_layouts or expected_native_layouts
        if not isinstance(supplied_native_layouts, Mapping) \
                or tuple(supplied_native_layouts) != tuple(expected_native_layouts):
            raise ValueError(
                "ResolvedSimulationPlan.native_layouts must match normalized layout order exactly")
        for layout_id, expected in expected_native_layouts.items():
            actual = supplied_native_layouts[layout_id]
            if type(actual) is not type(expected) or actual.to_data() != expected.to_data():
                raise ValueError(
                    "ResolvedSimulationPlan.native_layouts differs from LayoutPlan normalization")
        object.__setattr__(self, "native_layouts", _deep_freeze(supplied_native_layouts))
        object.__setattr__(self, "resolved_dimension", resolved_dimension(self.native_layouts))
        from pops.time import Program
        if type(self.time) is not Program:
            raise TypeError(
                "ResolvedSimulationPlan.time must be the exact whole-system pops.Program "
                "accepted by pops.resolve"
            )
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
        missing = [block.name for block in blocks if not block.instance_owner_qid]
        if missing:
            raise ValueError(
                "ResolvedSimulationPlan blocks %s have no canonical Case-block instance owner"
                % missing
            )
        assign_provider_declaration_roles(blocks)
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
            "native_layouts": _evidence(self.native_layouts, where="plan.native_layouts"),
            "resolved_dimension": self.resolved_dimension,
            "layout_targets": dict(self.layout_targets),
            "time": _evidence(self.time, where="plan.time"),
            "blocks": [{
                "name": block.name,
                "backend": block.backend,
                "state_spaces": block.state_spaces,
                "state_identities": block.state_identities,
                "instance_owner_qid": block.instance_owner_qid,
                "instance_owner": block.instance_owner,
                "model_owner": block.model_owner,
                "declares_auxiliary_providers": block.declares_auxiliary_providers,
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
    "authenticate_block_instance_owner", "authenticate_block_instance_owner_qid",
    "assign_provider_declaration_roles", "attest_precompiled_consumer_owner",
    "canonical_block_instance_owner", "provider_declaration_contract",
    "require_install_plan",
]
