"""A third-party field method can own a distinct target/operator contract."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from pops.fields import (
    PreparedFieldLoweringEvidence,
    PreparedFieldLoweringProvider,
    PreparedFieldLoweringRequest,
    PreparedFieldLoweringResolution,
    PreparedFieldMethod,
    PreparedFieldRuntimeInstallContext,
    register_prepared_field_lowering_provider,
)
from pops.fields._prepared_field_lowering_registry import (
    prepared_field_lowering_binding_from_data,
    prepared_field_lowering_binding_from_descriptor,
)
from pops.fields._prepared_field_nullspace_registry import PreparedFieldNullspaceFacts
from pops.fields._prepared_field_solver_registry import PreparedFieldSolverFacts


def _resolve(options, request, where):
    if dict(options) != {"block_size": 5}:
        raise ValueError("%s external method requires block_size=5" % where)
    if request.target != "external.runtime.block-field@1":
        raise ValueError("external provider received a foreign target")
    if request.output_components != ("q0", "q1", "q2", "q3", "q4"):
        raise ValueError("external provider received a foreign output space")
    topology = "test-owned-block-topology"
    return PreparedFieldLoweringResolution(
        {
            "external_native_contract": {
                "method": "test-owned-block-field",
                "block_size": 5,
            },
        },
        PreparedFieldSolverFacts(
            target=request.target,
            operator={
                "principal": "test-owned-block-vector-diffusion",
                "block_size": 5,
            },
            layout={
                "kind": "test-owned-layout",
                "topology_identity": topology,
            },
            hierarchy={"mode": "test-owned-hierarchy"},
            boundary={"provider": "test-owned-boundary-contract"},
            nonlinear=False,
        ),
        PreparedFieldNullspaceFacts(
            topology,
            0,
            {"principal": "test-owned-block-vector-diffusion"},
        ),
        (PreparedFieldLoweringEvidence(
            "test:external-field-method",
            "lowered",
            ("test-owned-native-contract",),
        ),),
    )


def _validate(binding, where):
    del where
    assert binding.resolution.solver_facts.operator["block_size"] == 5
    assert binding.resolution.solver_facts.target == "external.runtime.block-field@1"


def _parameter_handles(binding, operator, discretization):
    del binding, operator, discretization
    return {"test-owned-consumer": ()}


def _bind_native_options(binding, operator, discretization, params):
    del binding, operator, discretization
    if not isinstance(params, Mapping):
        raise TypeError("external bind inputs must be a mapping")
    return {}


def _install_bound_options(binding, context, bound_options):
    context.engine.install_test_block_field_options(
        context.slot, binding.identity, dict(bound_options)
    )


def _prepare_output(binding, context, operator, discretization):
    del operator, discretization
    if context.resources.get("reject_output") is True:
        raise ValueError("external output preflight rejected its resources")
    return {
        "binding_identity": binding.identity,
        "block_size": binding.resolution.native_options[
            "external_native_contract"
        ]["block_size"],
    }


def _install_output(binding, context, output_payload):
    assert output_payload["binding_identity"] == binding.identity
    context.engine.register_test_block_field_output(
        context.slot,
        output_payload["binding_identity"],
        output_payload["block_size"],
    )


_PROVIDER = register_prepared_field_lowering_provider(PreparedFieldLoweringProvider(
    provider_id="tests.external.block-field-lowering",
    version=1,
    resolver_id="tests.external.block-field-lowering.resolve@1",
    resolution_validator_id="tests.external.block-field-lowering.validate@1",
    runtime_binder_id="tests.external.block-field-lowering.bind@1",
    output_preparer_id="tests.external.block-field-lowering.prepare-output@1",
    bound_options_installer_id=(
        "tests.external.block-field-lowering.install-options@1"
    ),
    output_installer_id="tests.external.block-field-lowering.install-output@1",
    capabilities={
        "target": "external.runtime.block-field@1",
        "operator": "test-owned-block-vector-diffusion",
        "output_rank": 5,
        # This provider installs an authenticated contract on its own target engine.  It makes no
        # numerical-solve claim because this test deliberately contributes no native compute.
        "runtime_installation": {
            "contract": "install-only-no-solve",
            "preflight": "engine-free-canonical@1",
            "commit": "provider-owned@1",
        },
    },
    resolver=_resolve,
    resolution_validator=_validate,
    parameter_handles=_parameter_handles,
    bind_native_options=_bind_native_options,
    prepare_output=_prepare_output,
    install_bound_options=_install_bound_options,
    install_output=_install_output,
))


def test_external_method_resolves_distinct_target_operator_and_output_facts() -> None:
    method = PreparedFieldMethod(_PROVIDER, block_size=5)
    request = PreparedFieldLoweringRequest(
        "block_field",
        object(),
        object(),
        "external.runtime.block-field@1",
        ("q0", "q1", "q2", "q3", "q4"),
        object(),
    )

    binding = prepared_field_lowering_binding_from_descriptor(
        method, request=request, where="test external field method"
    )
    restored = prepared_field_lowering_binding_from_data(binding.to_data())

    assert restored == binding
    assert restored.resolution.solver_facts.target == request.target
    assert restored.resolution.solver_facts.operator == {
        "principal": "test-owned-block-vector-diffusion",
        "block_size": 5,
    }
    assert restored.resolution.native_options["external_native_contract"][
        "block_size"
    ] == 5
    assert _PROVIDER.parameter_handles(restored, object(), object()) == {
        "test-owned-consumer": ()
    }
    assert _PROVIDER.bind_native_options(restored, object(), object(), {}) == {}


def test_external_method_uses_the_same_generic_runtime_install_protocol() -> None:
    method = PreparedFieldMethod(_PROVIDER, block_size=5)
    request = PreparedFieldLoweringRequest(
        "block_field",
        object(),
        object(),
        "external.runtime.block-field@1",
        ("q0", "q1", "q2", "q3", "q4"),
        object(),
    )
    binding = prepared_field_lowering_binding_from_descriptor(
        method, request=request, where="test external runtime field method"
    )
    calls = []

    class ExternalEngine:
        def register_test_block_field_output(self, slot, identity, block_size):
            calls.append(("output", slot, identity, block_size))

        def install_test_block_field_options(self, slot, identity, params):
            calls.append(("options", slot, identity, params))

    context = PreparedFieldRuntimeInstallContext(
        target=request.target,
        engine=ExternalEngine(),
        resources={},
        slot="external-slot",
    )
    _PROVIDER.install_runtime(binding, context, object(), object(), {})

    assert calls == [
        ("output", "external-slot", binding.identity, 5),
        ("options", "external-slot", binding.identity, {}),
    ]


def test_external_method_preflight_failure_performs_no_engine_mutation() -> None:
    method = PreparedFieldMethod(_PROVIDER, block_size=5)
    request = PreparedFieldLoweringRequest(
        "block_field",
        object(),
        object(),
        "external.runtime.block-field@1",
        ("q0", "q1", "q2", "q3", "q4"),
        object(),
    )
    binding = prepared_field_lowering_binding_from_descriptor(
        method, request=request, where="test rejected external runtime field method"
    )
    calls = []

    class ExternalEngine:
        def register_test_block_field_output(self, *args):
            calls.append(("output", args))

        def install_test_block_field_options(self, *args):
            calls.append(("options", args))

    context = PreparedFieldRuntimeInstallContext(
        target=request.target,
        engine=ExternalEngine(),
        resources={"reject_output": True},
        slot="external-slot",
    )
    with pytest.raises(ValueError, match="output preflight rejected"):
        _PROVIDER.install_runtime(binding, context, object(), object(), {})
    assert calls == []
