"""Strict decoder for immutable runtime planning contracts.

The encoder is each contract's ``to_data`` projection.  Decoding accepts only that exact current
shape, reconstructs every nested value, and authenticates every supplied identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from pops._frozen_data import freeze_data, thaw_data
from pops.identity import Identity, make_identity


def canonical_text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def nonnegative_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError("%s must be an integer >= 0" % where)
    return value


def positive_integer(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TypeError("%s must be an integer >= 1" % where)
    return value


def frozen_rows(value: Any, where: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise TypeError("%s must be a tuple" % where)
    return tuple(freeze_data(item, "%s[]" % where) for item in value)


def string_tuple(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError("%s must be a tuple" % where)
    result = tuple(canonical_text(item, "%s[]" % where) for item in value)
    if len(result) != len(set(result)):
        raise ValueError("%s contains duplicate values" % where)
    return result


class RuntimePlanningError(ValueError):
    """Structured fail-closed refusal at the runtime-plan trust boundary."""

    def __init__(self, code: str, path: str, message: str, *, evidence: Any = None) -> None:
        super().__init__(message)
        self.code = canonical_text(code, "RuntimePlanningError.code")
        self.path = canonical_text(path, "RuntimePlanningError.path")
        self.evidence = freeze_data(evidence, "RuntimePlanningError.evidence")

    def to_data(self) -> dict[str, Any]:
        return {"code": self.code, "path": self.path, "message": str(self),
                "evidence": thaw_data(self.evidence)}


def refuse(code: str, path: str, message: str, *, evidence: Any = None) -> None:
    raise RuntimePlanningError(code, path, message, evidence=evidence)


class DataContract:
    _domain: ClassVar[str]
    identity: Identity

    def _seal(self, payload: Mapping[str, Any]) -> None:
        object.__setattr__(self, "identity", make_identity(self._domain, dict(payload)))

    def to_data(self) -> dict[str, Any]:
        return {**self._payload(), "identity": self.identity.to_data()}

    @classmethod
    def from_data(cls, data: Any) -> Any:
        return decode_runtime_value(cls, data)


def proved_platform(plan: Any) -> tuple[Any, Any, tuple[str, ...], dict[str, Any]]:
    from pops._platform_contracts import ExecutionContext, PlatformManifest, validate_launch

    platform, context = plan.artifact.platform_manifest, plan.execution_context
    if type(platform) is not PlatformManifest:
        raise TypeError("runtime planning requires an exact PlatformManifest")
    if type(context) is not ExecutionContext:
        raise TypeError("runtime planning requires an exact ExecutionContext")
    try:
        validate_launch(platform, context, ())
        facts = {
            "backend": platform.backend.require("platform.backend"),
            "target": platform.target.require("platform.target"),
            "abi": platform.abi.require("platform.abi"),
            "device": platform.device.require("platform.device"),
            "communicator": platform.communicator.require("platform.communicator"),
            "storage": platform.precision.storage.require("platform.precision.storage"),
            "compute": platform.precision.compute.require("platform.precision.compute"),
            "accumulation": platform.precision.accumulation.require("platform.precision.accumulation"),
            "reduction": platform.precision.reduction.require("platform.precision.reduction"),
            "dimensions": platform.capabilities["dimensions"].require("platform.capabilities.dimensions"),
        }
        spaces = platform.memory_spaces.require("platform.memory_spaces")
    except (KeyError, TypeError, ValueError) as exc:
        refuse("unknown_platform_evidence", "platform",
               "runtime planning requires complete selected-platform proof: %s" % exc)
    if not isinstance(spaces, tuple) or not spaces or any(not isinstance(item, str) or not item for item in spaces) or len(spaces) != len(set(spaces)):
        refuse("invalid_memory_spaces", "platform.memory_spaces",
               "platform memory spaces must be a unique non-empty tuple", evidence=spaces)
    dimensions = facts["dimensions"]
    if not isinstance(dimensions, tuple) or len(dimensions) != 1 or isinstance(dimensions[0], bool) or not isinstance(dimensions[0], int):
        refuse("ambiguous_platform_dimension", "platform.capabilities.dimensions",
               "runtime planning requires exactly one selected dimension", evidence=dimensions)
    facts["dimension"] = dimensions[0]
    return platform, context, spaces, facts


def component_features(platform: Any, manifests: Mapping[str, Any]) -> tuple[str, ...]:
    proof = platform.capabilities.get("features")
    if proof is None or not proof.known:
        required = sorted({feature for manifest in manifests.values()
                           for variant in manifest.target["variants"]
                           for feature in variant["features"]})
        if required:
            refuse("unknown_platform_features", "platform.capabilities.features",
                   "component target features lack platform proof", evidence=required)
        return ()
    values = proof.require("platform.capabilities.features")
    if not isinstance(values, tuple) or any(not isinstance(item, str) for item in values):
        refuse("invalid_platform_features", "platform.capabilities.features",
               "platform features must be a tuple of strings", evidence=values)
    return values


def validate_component_platform(manifest: Any, facts: Mapping[str, Any],
                                features: tuple[str, ...]) -> None:
    from pops.model import ComponentManifestError

    try:
        manifest.require_target({"dimension": facts["dimension"], "scalar": facts["compute"],
                                 "device": facts["device"], "features": features})
    except ComponentManifestError as exc:
        refuse("component_target_refused", "component[%s].target" % manifest.component_id,
               str(exc), evidence=exc.to_data())
    if manifest.reads and facts["compute"] not in manifest.precision["inputs"]:
        refuse("component_input_precision_mismatch",
               "component[%s].precision.inputs" % manifest.component_id,
               "component does not accept platform compute precision",
               evidence={"platform": facts["compute"],
                         "component": list(manifest.precision["inputs"])})
    if manifest.writes and facts["storage"] not in manifest.precision["outputs"]:
        refuse("component_output_precision_mismatch",
               "component[%s].precision.outputs" % manifest.component_id,
               "component does not produce platform storage precision",
               evidence={"platform": facts["storage"],
                         "component": list(manifest.precision["outputs"])})
    if manifest.precision["accumulation"] != facts["accumulation"]:
        refuse("component_accumulation_precision_mismatch",
               "component[%s].precision.accumulation" % manifest.component_id,
               "component accumulation precision differs from selected platform",
               evidence={"platform": facts["accumulation"],
                         "component": manifest.precision["accumulation"]})


def component_map(plan: Any, value: Any) -> Mapping[str, Any]:
    from pops.model import ComponentManifest

    if not isinstance(value, Mapping):
        raise TypeError("component_manifests must map block names to ComponentManifest values")
    expected = tuple(block.name for block in plan.artifact.blocks)
    missing, unknown = sorted(set(expected) - set(value)), sorted(set(value) - set(expected))
    if missing or unknown:
        refuse("component_set_mismatch", "component_manifests",
               "component manifest set must match compiled blocks exactly",
               evidence={"missing": missing, "unknown": unknown})
    if any(type(value[name]) is not ComponentManifest for name in expected):
        raise TypeError("every component manifest must be an exact ComponentManifest")
    return {name: value[name] for name in expected}


def _exact(value: Any, fields: set[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    missing, unknown = sorted(fields - set(value)), sorted(set(value) - fields)
    if missing or unknown:
        raise ValueError("%s fields mismatch: missing=%s, unknown=%s" % (where, missing, unknown))
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("%s must be a list" % where)
    return value


def _identity(value: Any, where: str) -> Identity:
    try:
        return Identity.from_data(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s is not an exact Identity: %s" % (where, exc)) from exc


def _authenticate(result: Any, row: Mapping[str, Any], where: str) -> Any:
    supplied = _identity(row["identity"], "%s.identity" % where)
    if supplied != result.identity:
        raise ValueError("%s identity does not authenticate its complete payload" % where)
    if result.to_data() != dict(row):
        raise ValueError("%s data is not in canonical form" % where)
    return result


def _access(value: Any, where: str) -> Any:
    from ._runtime_plan_contracts import FieldAccess

    row = _exact(value, {"resource", "mode", "memory_space"}, where)
    return FieldAccess(row["resource"], row["mode"], row["memory_space"])


def _use(value: Any, where: str) -> Any:
    from ._runtime_plan_contracts import ResourceUse

    row = _exact(value, {"resource", "memory_space", "first_call", "last_call", "modes"}, where)
    return ResourceUse(
        row["resource"],
        row["memory_space"],
        row["first_call"],
        row["last_call"],
        tuple(_list(row["modes"], "%s.modes" % where)),
    )


def _buffer(value: Any, where: str) -> Any:
    from ._runtime_plan_contracts import BufferAllocation

    row = _exact(value, {"resource", "memory_space", "size_bytes", "first_call", "last_call"}, where)
    return BufferAllocation(row["resource"], row["memory_space"], row["size_bytes"], row["first_call"], row["last_call"])


def decode_runtime_value(cls: type[Any], data: Any) -> Any:
    """Decode one exact contract class; structural lookalikes and legacy shapes are refused."""
    from ._runtime_plan_contracts import (
        ClockJoin,
        Collective,
        CommunicationPlan,
        DeterminismGuarantee,
        Fence,
        HaloExchange,
        LayoutTransfer,
        ResourcePlan,
        RuntimeCall,
        RuntimePlanBundle,
        RUNTIME_PLAN_SCHEMA_VERSION,
    )

    where = cls.__name__
    if cls is RuntimeCall:
        fields = {
            "ordinal",
            "block_id",
            "component_id",
            "component_type",
            "component_manifest_identity",
            "layout_id",
            "entry_point",
            "reads",
            "writes",
            "requirements",
            "effects",
            "clocks",
            "identity",
        }
        row = _exact(data, fields, where)
        result = RuntimeCall(
            ordinal=row["ordinal"],
            block_id=row["block_id"],
            component_id=row["component_id"],
            component_type=row["component_type"],
            component_manifest_identity=_identity(row["component_manifest_identity"], "RuntimeCall.component_manifest_identity"),
            layout_id=row["layout_id"],
            entry_point=row["entry_point"],
            reads=tuple(_access(item, "RuntimeCall.reads[]") for item in _list(row["reads"], "RuntimeCall.reads")),
            writes=tuple(_access(item, "RuntimeCall.writes[]") for item in _list(row["writes"], "RuntimeCall.writes")),
            requirements=tuple(_list(row["requirements"], "RuntimeCall.requirements")),
            effects=tuple(_list(row["effects"], "RuntimeCall.effects")),
            clocks=tuple(_list(row["clocks"], "RuntimeCall.clocks")),
        )
        return _authenticate(result, row, where)
    simple: dict[type[Any], tuple[set[str], Any]] = {
        HaloExchange: (
            {"call_id", "resource", "layout_id", "depth", "identity"},
            lambda r: HaloExchange(r["call_id"], r["resource"], r["layout_id"], r["depth"]),
        ),
        LayoutTransfer: (
            {
                "mapping_id",
                "provider_id",
                "source_layout_id",
                "target_layout_id",
                "channel",
                "identity",
            },
            lambda r: LayoutTransfer(
                r["mapping_id"],
                r["provider_id"],
                r["source_layout_id"],
                r["target_layout_id"],
                r["channel"],
            ),
        ),
        Collective: (
            {
                "call_id",
                "resource",
                "operation",
                "strategy",
                "communicator_id",
                "sequence",
                "identity",
            },
            lambda r: Collective(
                r["call_id"],
                r["resource"],
                r["operation"],
                r["strategy"],
                r["communicator_id"],
                r["sequence"],
            ),
        ),
        Fence: (
            {
                "resource",
                "before_call_id",
                "after_call_id",
                "source_space",
                "target_space",
                "identity",
            },
            lambda r: Fence(
                r["resource"],
                r["before_call_id"],
                r["after_call_id"],
                r["source_space"],
                r["target_space"],
            ),
        ),
        ClockJoin: (
            {"call_id", "source_clock", "target_clock", "policy", "identity"},
            lambda r: ClockJoin(r["call_id"], r["source_clock"], r["target_clock"], r["policy"]),
        ),
    }
    if cls in simple:
        fields, factory = simple[cls]
        row = _exact(data, fields, where)
        return _authenticate(factory(row), row, where)
    if cls is CommunicationPlan:
        fields = {
            "layout_plan_id",
            "communicator_id",
            "halos",
            "transfers",
            "collectives",
            "fences",
            "clock_joins",
            "identity",
        }
        row = _exact(data, fields, where)
        nested = (
            ("halos", HaloExchange),
            ("transfers", LayoutTransfer),
            ("collectives", Collective),
            ("fences", Fence),
            ("clock_joins", ClockJoin),
        )
        values = {name: tuple(decode_runtime_value(kind, item) for item in _list(row[name], "%s.%s" % (where, name))) for name, kind in nested}
        result = CommunicationPlan(row["layout_plan_id"], row["communicator_id"], **values)
        return _authenticate(result, row, where)
    if cls is ResourcePlan:
        fields = {
            "layout_plan_id",
            "execution_context_identity",
            "memory_spaces",
            "uses",
            "buffers",
            "mapping_provider_ids",
            "fence_ids",
            "declared_requirements",
            "identity",
        }
        row = _exact(data, fields, where)
        result = ResourcePlan(
            layout_plan_id=row["layout_plan_id"],
            execution_context_identity=_identity(row["execution_context_identity"], "ResourcePlan.execution_context_identity"),
            memory_spaces=tuple(_list(row["memory_spaces"], "ResourcePlan.memory_spaces")),
            uses=tuple(_use(item, "ResourcePlan.uses[]") for item in _list(row["uses"], "ResourcePlan.uses")),
            buffers=tuple(_buffer(item, "ResourcePlan.buffers[]") for item in _list(row["buffers"], "ResourcePlan.buffers")),
            mapping_provider_ids=tuple(_list(row["mapping_provider_ids"], "ResourcePlan.mapping_provider_ids")),
            fence_ids=tuple(_list(row["fence_ids"], "ResourcePlan.fence_ids")),
            declared_requirements=tuple(_list(row["declared_requirements"], "ResourcePlan.declared_requirements")),
        )
        return _authenticate(result, row, where)
    if cls is DeterminismGuarantee:
        fields = {
            "classification",
            "scope",
            "assumptions",
            "component_evidence",
            "execution_context_identity",
            "identity",
        }
        row = _exact(data, fields, where)
        result = DeterminismGuarantee(
            classification=row["classification"],
            scope=tuple(_list(row["scope"], "DeterminismGuarantee.scope")),
            assumptions=row["assumptions"],
            component_evidence=row["component_evidence"],
            execution_context_identity=_identity(row["execution_context_identity"], "DeterminismGuarantee.execution_context_identity"),
        )
        return _authenticate(result, row, where)
    if cls is RuntimePlanBundle:
        fields = {
            "schema_version",
            "install_identity",
            "platform_identity",
            "execution_context_identity",
            "layout_plan_id",
            "calls",
            "communication",
            "resources",
            "determinism",
            "identity",
        }
        row = _exact(data, fields, where)
        if row["schema_version"] != RUNTIME_PLAN_SCHEMA_VERSION or isinstance(row["schema_version"], bool):
            raise ValueError("unsupported RuntimePlanBundle schema_version")
        result = RuntimePlanBundle(
            install_identity=_identity(row["install_identity"], "bundle.install_identity"),
            platform_identity=_identity(row["platform_identity"], "bundle.platform_identity"),
            execution_context_identity=_identity(row["execution_context_identity"], "bundle.execution_context_identity"),
            layout_plan_id=row["layout_plan_id"],
            calls=tuple(decode_runtime_value(RuntimeCall, item) for item in _list(row["calls"], "bundle.calls")),
            communication=decode_runtime_value(CommunicationPlan, row["communication"]),
            resources=decode_runtime_value(ResourcePlan, row["resources"]),
            determinism=decode_runtime_value(DeterminismGuarantee, row["determinism"]),
        )
        return _authenticate(result, row, where)
    raise TypeError("unsupported runtime contract decoder %r" % cls)


__all__ = ["decode_runtime_value"]
