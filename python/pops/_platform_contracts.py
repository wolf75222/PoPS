"""Strict platform, backend, execution and field-view contracts.

The values in this module are metadata only.  They never initialize MPI/Kokkos and never obtain a
communicator, datatype, or device from process-global state.  A fact is usable only when carried by
an explicit :class:`CapabilityProof`; ``CapabilityProof.unknown()`` means absence of proof.
"""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, overload

from pops.identity import Identity, make_identity
from pops._platform_manifest_io import (
    manifest_from_data,
    manifest_to_data,
    precision_from_data,
    precision_to_data,
    proof_from_data,
    proof_to_data,
)


PLATFORM_CONTRACT_SCHEMA_VERSION = 1
_CENTERINGS = frozenset({"cell", "node", "face_x", "face_y", "face_z"})
_LAYOUTS = frozenset({"right", "left", "strided"})
_OWNERSHIP = frozenset({"borrowed", "owned", "shared"})
_STD_YEARS = {"11": "201103", "14": "201402", "17": "201703",
              "20": "202002", "23": "202302"}


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise TypeError("%s must be non-empty canonical text" % where)
    return value


def _freeze_value(value: Any, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item, where) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return MappingProxyType({key: _freeze_value(item, where)
                                 for key, item in sorted(value.items())})
    raise TypeError("%s contains non-canonical %s" % (where, type(value).__name__))


@dataclass(frozen=True, slots=True)
class CapabilityProof:
    """One value plus its evidence; missing evidence is strictly unknown."""

    value: Any
    evidence: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_value(self.value, "CapabilityProof.value"))
        if self.evidence is not None:
            _text(self.evidence, "CapabilityProof.evidence")
        elif self.value is not None:
            raise ValueError("CapabilityProof without evidence must have value=None")

    @classmethod
    def proven(cls, value: Any, evidence: str) -> CapabilityProof:
        return cls(value, evidence)

    @classmethod
    def unknown(cls) -> CapabilityProof:
        return cls(None, None)

    @classmethod
    def from_data(cls, data: Any) -> CapabilityProof:
        return proof_from_data(cls, data)

    @property
    def known(self) -> bool:
        return self.evidence is not None

    def require(self, where: str) -> Any:
        if not self.known:
            raise PlatformContractError(
                "%s has no capability proof; unknown is absence of proof" % where,
                field=where, expected="explicit proof", actual=None)
        return self.value

    def to_data(self) -> dict[str, Any]:
        return proof_to_data(self)


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Independent storage, compute, accumulation and reduction scalar proofs."""

    storage: CapabilityProof
    compute: CapabilityProof
    accumulation: CapabilityProof
    reduction: CapabilityProof

    def __post_init__(self) -> None:
        for name in ("storage", "compute", "accumulation", "reduction"):
            if type(getattr(self, name)) is not CapabilityProof:
                raise TypeError("PrecisionPolicy.%s must be a CapabilityProof" % name)

    def to_data(self) -> dict[str, Any]:
        return precision_to_data(self)

    @classmethod
    def from_data(cls, data: Any) -> PrecisionPolicy:
        return precision_from_data(cls, CapabilityProof, data)


def _proof_mapping(value: Any, where: str) -> Mapping[str, CapabilityProof]:
    if not isinstance(value, Mapping):
        raise TypeError("%s must be a mapping" % where)
    if any(not isinstance(key, str) or not key or type(proof) is not CapabilityProof
           for key, proof in value.items()):
        raise TypeError("%s must map non-empty names to exact CapabilityProof values" % where)
    return MappingProxyType(dict(sorted(value.items())))


@dataclass(frozen=True, slots=True)
class PlatformManifest:
    """Selected artifact platform.  Every compatibility-bearing fact is proved explicitly."""

    backend: CapabilityProof
    target: CapabilityProof
    abi: CapabilityProof
    precision: PrecisionPolicy
    device: CapabilityProof
    memory_spaces: CapabilityProof
    communicator: CapabilityProof
    capabilities: Mapping[str, CapabilityProof]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _validate_manifest_fields(self, "PlatformManifest")
        object.__setattr__(self, "capabilities", _proof_mapping(
            self.capabilities, "PlatformManifest.capabilities"))
        object.__setattr__(self, "identity", make_identity("platform-manifest", self.to_data()))

    def to_data(self) -> dict[str, Any]:
        return manifest_to_data(self, PLATFORM_CONTRACT_SCHEMA_VERSION)

    @classmethod
    def from_data(cls, data: Any) -> PlatformManifest:
        return manifest_from_data(
            cls, CapabilityProof, PrecisionPolicy, data,
            schema_version=PLATFORM_CONTRACT_SCHEMA_VERSION, where="PlatformManifest")


@dataclass(frozen=True, slots=True)
class RuntimeBackendManifest:
    """Actual runtime backend contract consumed at bind/install and before every generic launch."""

    backend: CapabilityProof
    target: CapabilityProof
    abi: CapabilityProof
    precision: PrecisionPolicy
    device: CapabilityProof
    memory_spaces: CapabilityProof
    communicator: CapabilityProof
    capabilities: Mapping[str, CapabilityProof]
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        _validate_manifest_fields(self, "RuntimeBackendManifest")
        object.__setattr__(self, "capabilities", _proof_mapping(
            self.capabilities, "RuntimeBackendManifest.capabilities"))
        object.__setattr__(self, "identity", make_identity(
            "runtime-backend-manifest", self.to_data()))

    def to_data(self) -> dict[str, Any]:
        return manifest_to_data(self, PLATFORM_CONTRACT_SCHEMA_VERSION)

    @classmethod
    def from_data(cls, data: Any) -> RuntimeBackendManifest:
        return manifest_from_data(
            cls, CapabilityProof, PrecisionPolicy, data,
            schema_version=PLATFORM_CONTRACT_SCHEMA_VERSION, where="RuntimeBackendManifest")


def _validate_manifest_fields(value: Any, where: str) -> None:
    for name in ("backend", "target", "abi", "device", "memory_spaces", "communicator"):
        if type(getattr(value, name)) is not CapabilityProof:
            raise TypeError("%s.%s must be a CapabilityProof" % (where, name))
    if type(value.precision) is not PrecisionPolicy:
        raise TypeError("%s.precision must be an exact PrecisionPolicy" % where)


@dataclass(frozen=True, slots=True)
class ExecutionResource:
    """Identity-bearing explicit runtime resource; opaque handle is never inferred or serialized."""

    kind: str
    identity: str
    handle: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _text(self.kind, "ExecutionResource.kind")
        _text(self.identity, "ExecutionResource.identity")

    def to_data(self) -> dict[str, str]:
        return {"kind": self.kind, "identity": self.identity}


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """All launch resources, including communicator and datatype, with no global defaults."""

    backend: RuntimeBackendManifest
    communicator: ExecutionResource
    datatype: ExecutionResource
    device: ExecutionResource
    identity: Identity = field(init=False)

    def __post_init__(self) -> None:
        if type(self.backend) is not RuntimeBackendManifest:
            raise TypeError("ExecutionContext.backend must be a RuntimeBackendManifest")
        for name in ("communicator", "datatype", "device"):
            resource = getattr(self, name)
            if type(resource) is not ExecutionResource or resource.kind != name:
                raise TypeError("ExecutionContext.%s must be an ExecutionResource(kind=%r)"
                                % (name, name))
        expected_comm = self.backend.communicator.require("backend.communicator")
        expected_device = self.backend.device.require("backend.device")
        if self.communicator.identity != expected_comm:
            raise PlatformContractError(
                "execution communicator does not match the runtime backend manifest",
                field="communicator", expected=expected_comm, actual=self.communicator.identity)
        if self.device.identity != expected_device:
            raise PlatformContractError(
                "execution device does not match the runtime backend manifest",
                field="device", expected=expected_device, actual=self.device.identity)
        if self.communicator.identity != "serial" and self.communicator.handle is None:
            raise PlatformContractError(
                "a non-serial ExecutionContext requires an explicit communicator handle",
                field="communicator", expected="explicit handle", actual=None)
        if self.device.identity not in ("host", "cpu") and self.device.handle is None:
            raise PlatformContractError(
                "a non-host ExecutionContext requires an explicit device handle",
                field="device", expected="explicit handle", actual=None)
        object.__setattr__(self, "identity", make_identity("execution-context", self.to_data()))

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": PLATFORM_CONTRACT_SCHEMA_VERSION,
                "backend_identity": self.backend.identity.to_data(),
                "communicator": self.communicator.to_data(),
                "datatype": self.datatype.to_data(), "device": self.device.to_data()}

    @classmethod
    def mpi_world(cls, artifact: Any, communicator: Any) -> ExecutionContext:
        """Build the exact native ``MPI_COMM_WORLD`` launch context for one artifact.

        The compiled artifact must itself have been authenticated for the world-communicator
        route.  A duplicated, split or otherwise custom communicator is intentionally rejected:
        the current native engines consume ``MPI_COMM_WORLD`` and do not expose a communicator
        injection ABI.
        """
        from pops.codegen._compiled_artifact import CompiledSimulationArtifact

        if type(artifact) is not CompiledSimulationArtifact:
            raise TypeError(
                "ExecutionContext.mpi_world requires the exact artifact returned by pops.compile"
            )
        try:
            MPI = importlib.import_module("mpi4py.MPI")
        except ImportError as exc:
            raise RuntimeError(
                "ExecutionContext.mpi_world requires mpi4py and an MPI-enabled PoPS module"
            ) from exc
        if not isinstance(communicator, MPI.Comm) or MPI.Comm.Compare(
                communicator, MPI.COMM_WORLD) != MPI.IDENT:
            raise ValueError(
                "ExecutionContext.mpi_world accepts only mpi4py.MPI.COMM_WORLD; custom MPI "
                "communicators are not consumed by the native provider"
            )
        from pops.runtime._platform_manifest import native_runtime_backend

        backend = native_runtime_backend(artifact.platform_manifest)
        expected = backend.communicator.require("runtime.communicator")
        if expected != "MPI_COMM_WORLD":
            raise PlatformContractError(
                "the compiled artifact/runtime pair does not prove MPI_COMM_WORLD",
                field="communicator", expected="MPI_COMM_WORLD", actual=expected,
            )
        return cls(
            backend=backend,
            communicator=ExecutionResource(
                "communicator", "MPI_COMM_WORLD", handle=communicator
            ),
            datatype=ExecutionResource("datatype", "float64", handle=MPI.DOUBLE),
            device=ExecutionResource("device", "host"),
        )


@dataclass(frozen=True, slots=True)
class FieldViewDescriptor:
    """Complete rank-preserving field-view ABI.  Three-dimensional requests stay representable."""

    name: str
    dimension: int
    extents: tuple[int, ...]
    strides: tuple[int, ...]
    centering: str
    ghosts: tuple[tuple[int, int], ...]
    scalar: str
    memory_space: str
    patch: str
    layout: str
    ownership: str

    def __post_init__(self) -> None:
        _text(self.name, "FieldViewDescriptor.name")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int) \
                or self.dimension < 1:
            raise TypeError("FieldViewDescriptor.dimension must be an integer >= 1")
        for name in ("extents", "strides"):
            values = tuple(getattr(self, name))
            if len(values) != self.dimension or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0
                    for item in values):
                raise ValueError("FieldViewDescriptor.%s must contain %d positive integers"
                                 % (name, self.dimension))
            object.__setattr__(self, name, values)
        ghosts = tuple(tuple(pair) for pair in self.ghosts)
        if len(ghosts) != self.dimension or any(
                len(pair) != 2 or any(isinstance(item, bool) or not isinstance(item, int)
                                      or item < 0 for item in pair) for pair in ghosts):
            raise ValueError("FieldViewDescriptor.ghosts must contain one non-negative pair per axis")
        object.__setattr__(self, "ghosts", ghosts)
        if self.centering not in _CENTERINGS:
            raise ValueError("unsupported field centering %r" % self.centering)
        if self.layout not in _LAYOUTS:
            raise ValueError("unsupported field layout %r" % self.layout)
        if self.ownership not in _OWNERSHIP:
            raise ValueError("unsupported field ownership %r" % self.ownership)
        for name in ("scalar", "memory_space", "patch"):
            _text(getattr(self, name), "FieldViewDescriptor.%s" % name)

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension,
                "extents": list(self.extents), "strides": list(self.strides),
                "centering": self.centering, "ghosts": [list(pair) for pair in self.ghosts],
                "scalar": self.scalar, "memory_space": self.memory_space,
                "patch": self.patch, "layout": self.layout, "ownership": self.ownership}


class PlatformContractError(ValueError):
    """Fail-closed platform/field refusal raised before a kernel is called."""

    def __init__(self, message: str, *, field: str, expected: Any, actual: Any) -> None:
        super().__init__(message)
        self.field, self.expected, self.actual = field, expected, actual


def _require_same(field_name: str, required: CapabilityProof,
                  provided: CapabilityProof) -> None:
    expected = required.require("artifact.%s" % field_name)
    actual = provided.require("runtime.%s" % field_name)
    if expected != actual:
        raise PlatformContractError(
            "%s mismatch: artifact requires %r, runtime proves %r"
            % (field_name, expected, actual), field=field_name, expected=expected, actual=actual)


def _normalized_std(text: str) -> str | None:
    found = re.search(r"(?:^|;)\s*std=(\d{6})L?(?:;|$)", text)
    if found:
        return found.group(1)
    parts = text.split("|")
    if len(parts) > 2:
        flag = re.fullmatch(r"(?:c|gnu)\+\+(\d{2})", parts[2].strip().lower())
        return _STD_YEARS.get(flag.group(1)) if flag else None
    return None


def _abi_parts(value: Any) -> tuple[str | None, str | None]:
    text = str(value)
    found = re.search(r"(?:^|;)\s*headers=([^;]+)", text)
    headers = found.group(1).strip() if found else (text.split("|", 1)[0].strip() or None)
    return headers, _normalized_std(text)


def _require_abi(required: CapabilityProof, provided: CapabilityProof) -> None:
    expected = required.require("artifact.abi")
    actual = provided.require("runtime.abi")
    expected_parts, actual_parts = _abi_parts(expected), _abi_parts(actual)
    if expected_parts[0] != actual_parts[0] or (
            expected_parts[1] is not None and actual_parts[1] is not None
            and expected_parts[1] != actual_parts[1]):
        raise PlatformContractError(
            "ABI mismatch between artifact and runtime backend", field="abi",
            expected=expected, actual=actual)


def _validate_launch_facts(platform: PlatformManifest, context: ExecutionContext,
                           fields: Sequence[FieldViewDescriptor],
                           expected_fields: Sequence[FieldViewDescriptor],
                           *, compare_route: bool) -> None:
    backend = context.backend
    route_fields = ("backend", "target") if compare_route else ()
    for name in (*route_fields, "device", "memory_spaces", "communicator"):
        _require_same(name, getattr(platform, name), getattr(backend, name))
    _require_abi(platform.abi, backend.abi)
    for name in ("storage", "compute", "accumulation", "reduction"):
        _require_same("precision.%s" % name, getattr(platform.precision, name),
                      getattr(backend.precision, name))
    supported_dimensions = tuple(backend.capabilities["dimensions"].require(
        "runtime.capabilities.dimensions"))
    supported_centerings = tuple(backend.capabilities["centerings"].require(
        "runtime.capabilities.centerings"))
    supported_scalars = tuple(backend.capabilities["scalars"].require(
        "runtime.capabilities.scalars"))
    supported_memory = tuple(backend.memory_spaces.require("runtime.memory_spaces"))
    actual = tuple(fields)
    expected = {item.name: item for item in expected_fields}
    if len(expected) != len(tuple(expected_fields)):
        raise ValueError("expected field names must be unique")
    for view in actual:
        if type(view) is not FieldViewDescriptor:
            raise TypeError("fields must contain exact FieldViewDescriptor values")
        _require_field_capability(view, "dimension", view.dimension, supported_dimensions)
        _require_field_capability(view, "centering", view.centering, supported_centerings)
        _require_field_capability(view, "scalar", view.scalar, supported_scalars)
        _require_field_capability(view, "memory_space", view.memory_space, supported_memory)
        if view.scalar != context.datatype.identity:
            raise PlatformContractError(
                "field scalar does not match ExecutionContext datatype", field="datatype",
                expected=view.scalar, actual=context.datatype.identity)
        requirement = expected.get(view.name)
        if requirement is not None:
            for name in ("dimension", "extents", "centering", "scalar", "memory_space"):
                if getattr(view, name) != getattr(requirement, name):
                    raise PlatformContractError(
                        "field %r %s mismatch" % (view.name, name), field=name,
                        expected=getattr(requirement, name), actual=getattr(view, name))
    missing = sorted(set(expected) - {item.name for item in actual})
    if missing:
        raise PlatformContractError("required field view(s) are missing: %s" % missing,
                                    field="fields", expected=missing, actual=None)


def validate_launch(platform: PlatformManifest, context: ExecutionContext,
                    fields: Sequence[FieldViewDescriptor],
                    expected_fields: Sequence[FieldViewDescriptor] = ()) -> None:
    """Validate every platform and field proof before the caller may enter a kernel."""
    if type(platform) is not PlatformManifest or type(context) is not ExecutionContext:
        raise TypeError("validate_launch requires exact PlatformManifest and ExecutionContext values")
    _validate_launch_facts(platform, context, fields, expected_fields, compare_route=True)


def validate_component_launch(platform: PlatformManifest, context: ExecutionContext,
                              fields: Sequence[FieldViewDescriptor],
                              expected_fields: Sequence[FieldViewDescriptor] = ()) -> None:
    """Validate an authenticated AOT component against its host simulation runtime.

    ``aot-component/component`` identifies the component artifact's build role; it is intentionally
    distinct from the enclosing simulation's ``production/system`` or ``production/amr_system``
    route.  Only that exact component route receives this comparison rule.  Every execution-bearing
    fact (ABI, precision, device, memory spaces and communicator) remains fail-closed.
    """
    if type(platform) is not PlatformManifest or type(context) is not ExecutionContext:
        raise TypeError(
            "validate_component_launch requires exact PlatformManifest and ExecutionContext values")
    route = (
        platform.backend.require("component.backend"),
        platform.target.require("component.target"),
    )
    expected_route = ("aot-component", "component")
    if route != expected_route:
        raise PlatformContractError(
            "component artifact route must be exactly %r" % (expected_route,),
            field="component_route", expected=expected_route, actual=route)
    _validate_launch_facts(platform, context, fields, expected_fields, compare_route=False)


def _require_field_capability(view: FieldViewDescriptor, field_name: str,
                              value: Any, supported: tuple[Any, ...]) -> None:
    if value not in supported:
        raise PlatformContractError(
            "field %r requests unsupported %s=%r; runtime proves only %r"
            % (view.name, field_name, value, supported), field=field_name,
            expected=supported, actual=value)


def launch_checked(platform: PlatformManifest, context: ExecutionContext,
                   fields: Sequence[FieldViewDescriptor], kernel: Callable[..., Any],
                   *, expected_fields: Sequence[FieldViewDescriptor] = ()) -> Any:
    """Run ``kernel`` only after the complete generic descriptor gate succeeds."""
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    validate_launch(platform, context, fields, expected_fields)
    return kernel(context, tuple(fields))


@overload
def proven_serial_manifest(*, backend: str, target: str, abi: str,
                           runtime: Literal[False] = False) -> PlatformManifest: ...


@overload
def proven_serial_manifest(*, backend: str, target: str, abi: str,
                           runtime: Literal[True]) -> RuntimeBackendManifest: ...


@overload
def proven_serial_manifest(*, backend: str, target: str, abi: str,
                           runtime: bool) -> PlatformManifest | RuntimeBackendManifest: ...


def proven_serial_manifest(*, backend: str, target: str, abi: str,
                           runtime: bool = False) -> PlatformManifest | RuntimeBackendManifest:
    """Exact supported 2D/float64/host route used by the current generic runtime."""
    evidence = "pops.native.2d-float64-host.v1"
    proof = lambda value: CapabilityProof.proven(value, evidence)  # noqa: E731
    cls = RuntimeBackendManifest if runtime else PlatformManifest
    return cls(backend=proof(_text(backend, "backend")), target=proof(_text(target, "target")),
               abi=proof(_text(abi, "abi")),
               precision=PrecisionPolicy(*(proof("float64") for _ in range(4))),
               device=proof("host"), memory_spaces=proof(("host",)),
               communicator=proof("serial"), capabilities={
                   "dimensions": proof((2,)), "centerings": proof(("cell",)),
                   "scalars": proof(("float64",)),
                   "layouts": proof(("right", "left", "strided")),
                   "ownership": proof(("borrowed", "owned", "shared")),
                   "generic_field_view": proof(True),
               })


def artifact_platform_manifest(
    *, backend: str, target: str, component: Any, communicator: str | None = None
) -> PlatformManifest:
    """Build the selected platform identity from emitted component facts, preserving unknowns."""
    evidence = "pops.compiled-component-metadata.v1"
    proof = lambda value: CapabilityProof.proven(value, evidence)  # noqa: E731
    unknown = CapabilityProof.unknown
    abi = getattr(component, "abi_key", None)
    precision = getattr(component, "precision_policy", None)
    if precision is None:
        precision = PrecisionPolicy(*(proof("float64") for _ in range(4)))
    if type(precision) is not PrecisionPolicy:
        raise TypeError("compiled component precision_policy must be an exact PrecisionPolicy")
    # This is the SELECTED baseline route, not a claim that the component lacks MPI/GPU variants.
    # A non-host/non-serial selection must name itself on the compiled component explicitly.
    device_value = getattr(component, "platform_device", None) or "host"
    component_communicator = getattr(component, "communicator", None)
    if communicator is not None and component_communicator not in (None, communicator):
        raise PlatformContractError(
            "compiled component communicator differs from the selected artifact route",
            field="communicator", expected=communicator, actual=component_communicator,
        )
    communicator_value = communicator or component_communicator or "serial"
    spaces = getattr(component, "memory_spaces", None)
    if spaces is None and device_value == "host":
        spaces = ("host",)
    return PlatformManifest(
        backend=proof(_text(backend, "backend")), target=proof(_text(target, "target")),
        abi=proof(str(abi)) if abi else unknown(), precision=precision,
        device=proof(device_value) if device_value else unknown(),
        memory_spaces=proof(tuple(spaces)) if spaces else unknown(),
        communicator=proof(communicator_value) if communicator_value else unknown(),
        capabilities={
            "dimensions": proof((2,)), "centerings": proof(("cell",)),
            "scalars": proof(("float64",)),
            "layouts": proof(("right", "left", "strided")),
            "ownership": proof(("borrowed", "owned", "shared")),
            "generic_field_view": proof(True),
        })


def serial_execution_context(platform: PlatformManifest) -> ExecutionContext:
    """Materialize explicit serial resources from a proved serial artifact platform."""
    backend = proven_serial_manifest(
        backend=platform.backend.require("platform.backend"),
        target=platform.target.require("platform.target"),
        abi=platform.abi.require("platform.abi"), runtime=True)
    return ExecutionContext(
        backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"))


__all__ = [
    "PLATFORM_CONTRACT_SCHEMA_VERSION", "CapabilityProof", "PrecisionPolicy",
    "PlatformManifest", "RuntimeBackendManifest", "ExecutionResource", "ExecutionContext",
    "FieldViewDescriptor", "PlatformContractError", "validate_launch",
    "validate_component_launch", "launch_checked",
    "proven_serial_manifest", "artifact_platform_manifest", "serial_execution_context",
]
