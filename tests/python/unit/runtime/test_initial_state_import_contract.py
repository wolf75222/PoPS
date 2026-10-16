"""Input transfer authority is distinct from device kernel view authority."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pops._platform_contracts import (
    CapabilityProof, ExecutionContext, ExecutionResource, PlatformContractError,
    proven_serial_manifest, validate_initial_state_import, validate_launch,
)
from pops.runtime._platform_validation import _initial_field, validate_platform_bind


def proof(value):
    return CapabilityProof.proven(value, "test-native-import-authority")


def contracts():
    platform = proven_serial_manifest(backend="production", target="system", abi="test")
    backend = proven_serial_manifest(backend="production", target="system", abi="test", runtime=True)
    capabilities = dict(backend.capabilities)
    capabilities.update(execution_backend=proof("Cuda"), shared_space=proof("CudaUVM"),
                        stream_identity=proof("Cuda::default"))
    facts = dict(device=proof("cuda"), memory_spaces=proof(("device",)), capabilities=capabilities)
    platform, backend = replace(platform, **facts), replace(backend, **facts)
    context = ExecutionContext(backend=backend,
        communicator=ExecutionResource("communicator", "serial"),
        datatype=ExecutionResource("datatype", "float64"),
        device=ExecutionResource("device", "cuda", handle=object()))
    return platform, context


def field(array=None):
    if array is None:
        array = np.ones((2, 3, 5), dtype=np.float64)
    return _initial_field("state", array, 2, ((5, 3),))


def replace_capability(platform, context, name, value):
    capabilities = dict(platform.capabilities)
    if value is None:
        capabilities.pop(name)
    else:
        capabilities[name] = value
    return replace(platform, capabilities=capabilities), replace(context,
        backend=replace(context.backend, capabilities=capabilities))


def test_host_input_can_be_copied_without_becoming_a_device_kernel_view():
    platform, context = contracts()
    view = field()
    validate_initial_state_import(platform, context, (view,))
    assert view.memory_space == "host"
    assert context.backend.memory_spaces.require("memory") == ("device",)
    with pytest.raises(PlatformContractError, match="memory_space"):
        validate_launch(platform, context, (view,))


@pytest.mark.parametrize("array", (
    np.ones((2, 3, 5)), np.asfortranarray(np.ones((2, 3, 5))),
    np.ones((2, 3, 10))[:, :, ::2],
))
def test_bind_validates_copy_input_shapes_and_layouts(array):
    platform, context = contracts()
    plan = SimpleNamespace(resolved_dimension=2,
        native_layouts={"grid": SimpleNamespace(shape=(5, 3))})
    assert validate_platform_bind(platform, context, {"state": array}, plan) == []


@pytest.mark.parametrize("name", ("initial_state_import_mode", "initial_state_import_memory_spaces"))
@pytest.mark.parametrize("value", (None, CapabilityProof.unknown()))
def test_import_requires_both_native_proofs(name, value):
    platform, context = replace_capability(*contracts(), name, value)
    with pytest.raises(PlatformContractError):
        validate_initial_state_import(platform, context, (field(),))
    validate_launch(platform, context, ())


def test_import_rejects_a_mode_that_has_no_native_copy_implementation():
    platform, context = replace_capability(*contracts(), "initial_state_import_mode", proof("borrow"))
    with pytest.raises(PlatformContractError, match="copy contract"):
        validate_initial_state_import(platform, context, (field(),))


def test_import_rejects_artifact_runtime_transfer_mismatch():
    platform, context = contracts()
    capabilities = dict(context.backend.capabilities)
    capabilities["initial_state_import_memory_spaces"] = proof(("device",))
    context = replace(context, backend=replace(context.backend, capabilities=capabilities))
    with pytest.raises(PlatformContractError, match="mismatch"):
        validate_initial_state_import(platform, context, (field(),))


@pytest.mark.parametrize("changes", ({"scalar": "float32"}, {"memory_space": "device"},
                                     {"dimension": 1, "extents": (15,), "strides": (1,), "ghosts": ((0, 0),)}))
def test_import_preserves_scalar_memory_and_dimension_checks(changes):
    platform, context = contracts()
    if "dimension" in changes:
        capabilities = dict(platform.capabilities)
        capabilities["supported_dimensions"] = proof((2,))
        platform = replace(platform, capabilities=capabilities)
        context = replace(context, backend=replace(context.backend, capabilities=capabilities))
    with pytest.raises(PlatformContractError):
        validate_initial_state_import(platform, context, (replace(field(), **changes),))


def test_import_still_checks_platform_abi_and_unique_field_names():
    platform, context = contracts()
    with pytest.raises(PlatformContractError, match="ABI"):
        validate_initial_state_import(replace(platform, abi=proof("foreign")), context, (field(),))
    with pytest.raises(PlatformContractError, match="duplicate"):
        validate_initial_state_import(platform, context, (field(), field()))
