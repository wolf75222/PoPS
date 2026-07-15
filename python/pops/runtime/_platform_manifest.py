"""Private bind-time adapter for exact platform and execution-context contracts."""
from collections.abc import Mapping
from pops._platform_contracts import (
    PLATFORM_CONTRACT_SCHEMA_VERSION,
    CapabilityProof,
    ExecutionContext,
    ExecutionResource,
    FieldViewDescriptor,
    PlatformContractError,
    PlatformManifest,
    PrecisionPolicy,
    RuntimeBackendManifest,
    launch_checked,
    proven_serial_manifest,
    serial_execution_context,
    validate_launch,
)


def execution_context_for_bind(platform, resources):
    """Resolve the sole explicit context against authenticated native runtime facts."""
    exact = dict(resources or {})
    supplied = exact.pop("execution_context", None)
    runtime = native_runtime_backend(platform)
    if supplied is not None:
        if exact:
            raise TypeError(
                "execution_context is the sole runtime resource authority; competing keys %s"
                % sorted(exact))
        if type(supplied) is not ExecutionContext:
            raise TypeError("resources['execution_context'] must be an exact ExecutionContext")
        if supplied.backend != runtime:
            raise PlatformContractError(
                "ExecutionContext backend proof does not match the loaded native runtime",
                field="runtime_backend", expected=runtime.identity.token,
                actual=supplied.backend.identity.token)
        validate_launch(platform, supplied, ())
        return supplied
    if exact:
        raise TypeError(
            "communicator/device/datatype must be carried by resources['execution_context']; "
            "standalone runtime resource keys are not a launch contract: %s" % sorted(exact))
    if runtime.communicator.require("runtime.communicator") != "serial":
        raise PlatformContractError(
            "a non-serial compiled artifact requires an explicit ExecutionContext at pops.bind",
            field="communicator", expected=runtime.communicator.require("runtime.communicator"),
            actual=None,
        )
    context = ExecutionContext(
        backend=runtime,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"))
    validate_launch(platform, context, ())
    return context


def native_runtime_backend(platform):
    from pops import _pops
    fn = getattr(_pops, "runtime_backend_manifest", None)
    if not callable(fn):
        raise RuntimeError(
            "loaded _pops exposes no runtime_backend_manifest; rebuild/install this exact PoPS tree")
    raw_data = fn(
        platform.backend.require("platform.backend"),
        platform.target.require("platform.target"),
        platform.communicator.require("platform.communicator"),
    )
    if not isinstance(raw_data, Mapping):
        raise TypeError("native runtime_backend_manifest() must return a mapping")
    data = dict(raw_data)
    expected = {"schema_version", "backend", "target", "abi", "precision", "device",
                "memory_spaces", "communicator", "capabilities", "evidence", "identity"}
    if set(data) != expected or data["schema_version"] != PLATFORM_CONTRACT_SCHEMA_VERSION:
        raise ValueError("native RuntimeBackendManifest has an incompatible exact schema")
    evidence = data["evidence"]
    if not isinstance(evidence, str) or not evidence:
        raise TypeError("native runtime backend evidence must be non-empty text")
    proof = lambda value: CapabilityProof.proven(value, evidence)  # noqa: E731
    precision = data["precision"]
    if not isinstance(precision, Mapping) or set(precision) != {
            "storage", "compute", "accumulation", "reduction"}:
        raise ValueError("native precision policy must name all four independent stages")
    capabilities = data["capabilities"]
    if not isinstance(capabilities, Mapping):
        raise TypeError("native runtime capabilities must be a mapping")
    memory_spaces = data["memory_spaces"]
    if not isinstance(memory_spaces, (list, tuple)):
        raise TypeError("native runtime memory_spaces must be a sequence")
    result = RuntimeBackendManifest(
        backend=proof(data["backend"]), target=proof(data["target"]), abi=proof(data["abi"]),
        precision=PrecisionPolicy(**{name: proof(value) for name, value in precision.items()}),
        device=proof(data["device"]), memory_spaces=proof(tuple(memory_spaces)),
        communicator=proof(data["communicator"]),
        capabilities={name: proof(tuple(value) if isinstance(value, list) else value)
                      for name, value in capabilities.items()})
    if result.identity.token != data["identity"]:
        raise ValueError("native RuntimeBackendManifest identity does not match its exact payload")
    return result

__all__ = [
    "PLATFORM_CONTRACT_SCHEMA_VERSION", "CapabilityProof", "PrecisionPolicy",
    "PlatformManifest", "RuntimeBackendManifest", "ExecutionResource", "ExecutionContext",
    "FieldViewDescriptor", "PlatformContractError", "validate_launch", "launch_checked",
    "proven_serial_manifest", "serial_execution_context", "execution_context_for_bind",
    "native_runtime_backend",
]
