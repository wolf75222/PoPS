"""Private bind-time adapter for exact platform and execution-context contracts."""
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
    """Resolve the one explicit context carried into InstallPlan; never inspect global MPI/device."""
    exact = dict(resources or {})
    supplied = exact.pop("execution_context", None)
    runtime = _native_runtime_backend(platform)
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
    context = ExecutionContext(
        backend=runtime,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "host"))
    validate_launch(platform, context, ())
    return context


def _native_runtime_backend(platform):
    from pops import _pops
    fn = getattr(_pops, "runtime_backend_manifest", None)
    if not callable(fn):
        raise RuntimeError(
            "loaded _pops exposes no runtime_backend_manifest; rebuild/install this exact PoPS tree")
    data = dict(fn(platform.backend.require("platform.backend"),
                   platform.target.require("platform.target")))
    expected = {"schema_version", "backend", "target", "abi", "precision", "device",
                "memory_spaces", "communicator", "capabilities", "evidence", "identity"}
    if set(data) != expected or data["schema_version"] != PLATFORM_CONTRACT_SCHEMA_VERSION:
        raise ValueError("native RuntimeBackendManifest has an incompatible exact schema")
    evidence = data["evidence"]
    proof = lambda value: CapabilityProof.proven(value, evidence)  # noqa: E731
    precision = data["precision"]
    if set(precision) != {"storage", "compute", "accumulation", "reduction"}:
        raise ValueError("native precision policy must name all four independent stages")
    result = RuntimeBackendManifest(
        backend=proof(data["backend"]), target=proof(data["target"]), abi=proof(data["abi"]),
        precision=PrecisionPolicy(**{name: proof(value) for name, value in precision.items()}),
        device=proof(data["device"]), memory_spaces=proof(tuple(data["memory_spaces"])),
        communicator=proof(data["communicator"]),
        capabilities={name: proof(tuple(value) if isinstance(value, list) else value)
                      for name, value in data["capabilities"].items()})
    if result.identity.token != data["identity"]:
        raise ValueError("native RuntimeBackendManifest identity does not match its exact payload")
    return result

__all__ = [
    "PLATFORM_CONTRACT_SCHEMA_VERSION", "CapabilityProof", "PrecisionPolicy",
    "PlatformManifest", "RuntimeBackendManifest", "ExecutionResource", "ExecutionContext",
    "FieldViewDescriptor", "PlatformContractError", "validate_launch", "launch_checked",
    "proven_serial_manifest", "serial_execution_context", "execution_context_for_bind",
]
