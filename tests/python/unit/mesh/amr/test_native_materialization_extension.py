"""Open AMR actions lower through one strict immutable native descriptor."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pops.amr import (
    CanonicalOptions,
    NativeAMRActionKind,
    NativeAMRMaterializationCapabilities,
    NativeAMRMaterializationDescriptor,
    NativeAMRMaterializationKind,
    TransferCapabilities,
)
from pops.codegen._amr_plan_validation import _validated_native_materialization
from pops.mesh import LayoutPlanBuilder
from pops.mesh._amr.transfer import (
    AccuracyRequirement,
    CELL_CENTERED,
    CELL_SPACE,
    CONSERVATIVE_REPRESENTATION,
    DENSE_STORAGE,
    PHYSICAL,
    PROLONGATION,
    RESTRICTION,
    ResolvedTransfer,
    TransferKey,
    TransferRequirement,
)
from pops.model import Handle, OwnerPath
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout


OWNER = OwnerPath.case("third-party-amr-materialization")


def _key(operation=PROLONGATION):
    return TransferKey(
        CELL_SPACE,
        CELL_CENTERED,
        CONSERVATIVE_REPRESENTATION,
        DENSE_STORAGE,
        operation,
    )


def _transfer_capabilities() -> TransferCapabilities:
    return TransferCapabilities(
        order=2,
        ghost_depth=(1,),
        dimensions=(2,),
        conservative=True,
        refinement_ratios=(2,),
    )


def _descriptor(
    key: TransferKey,
    *,
    route: str = "conservative_linear",
    schema_version: int = 1,
    capability_ids: tuple[str, ...] | None = None,
    capabilities: object = ...,
) -> NativeAMRMaterializationDescriptor:
    if capability_ids is None:
        native_capabilities = NativeAMRMaterializationCapabilities.for_materialization(
            NativeAMRMaterializationKind.PHYSICAL,
            key.operation,
            transfer=_transfer_capabilities(),
        )
    else:
        native_capabilities = NativeAMRMaterializationCapabilities(
            capability_ids,
            _transfer_capabilities(),
        )
    if capabilities is not ...:
        native_capabilities = capabilities
    provider_id = "test.amr.provider.v1::third_party_conservative"
    return NativeAMRMaterializationDescriptor(
        schema_version=schema_version,
        action=NativeAMRActionKind.APPLY_TRANSFER_PROVIDER,
        materialization=NativeAMRMaterializationKind.PHYSICAL,
        operation=key.operation,
        transfer_key_identity=key.identity,
        provider_qualified_id=provider_id,
        provider_identity=CanonicalOptions({
            "qualified_id": provider_id,
            "authority": "test.third_party",
            "options": {"implementation": "external-package"},
        }),
        options=CanonicalOptions({"native_route": route}),
        native_route=route,
        capabilities=native_capabilities,  # type: ignore[arg-type]
    )


@dataclass(frozen=True, slots=True)
class ThirdPartyPhysicalAction:
    """No PoPS base class: only the native materialization protocol is implemented."""

    route: str = "conservative_linear"

    def native_amr_materialization(
        self, *, key: TransferKey,
    ) -> NativeAMRMaterializationDescriptor:
        return _descriptor(key, route=self.route)


def _prepared_entry(action=ThirdPartyPhysicalAction()):
    state = Handle("U", kind="state", owner=OwnerPath.model("third-party-state"))
    builder = LayoutPlanBuilder(OWNER)
    layout = builder.layout("adaptive", final_amr_layout(cartesian_grid(n=8)))
    builder.assign_state(state, layout)
    plan = builder.resolve(states=(state,))
    key = _key()
    requirement = TransferRequirement(
        state,
        layout,
        key,
        PHYSICAL,
        AccuracyRequirement(
            order=2,
            ghost_depth=(1,),
            dimension=2,
            refinement_ratio=(2, 2),
            conservative=True,
        ),
    )
    return requirement, ResolvedTransfer(key, (requirement,), action)


def test_unrelated_third_party_action_crosses_validation_and_closed_preparation():
    _, entry = _prepared_entry()

    native = _validated_native_materialization(entry)
    assert type(entry.action) is ThirdPartyPhysicalAction
    assert type(native) is NativeAMRMaterializationDescriptor
    assert native.schema_version == 1
    assert native.native_route == "conservative_linear"
    assert native.capabilities.transfer == _transfer_capabilities()
    prepared_data = entry.to_data()["action"]
    assert prepared_data["descriptor_type"] == "pops.amr.native_materialization"
    assert prepared_data["provider"]["qualified_id"] == native.provider_qualified_id


class MissingProtocolAction:
    pass


class MalformedDescriptorAction:
    def native_amr_materialization(self, *, key):
        del key
        return {"schema_version": 1}


class NonDeterministicAction:
    def __init__(self):
        self.calls = 0

    def native_amr_materialization(self, *, key):
        self.calls += 1
        route = "conservative_linear" if self.calls % 2 else "different_valid_route"
        return _descriptor(key, route=route)


class WrongKeyAction:
    def native_amr_materialization(self, *, key):
        del key
        return _descriptor(_key(RESTRICTION), route="volume_average")


class WrongVersionAction:
    def native_amr_materialization(self, *, key):
        return _descriptor(key, schema_version=2)


class MissingCapabilityAction:
    def native_amr_materialization(self, *, key):
        return _descriptor(
            key,
            capability_ids=("pops.amr.materialization.physical.v1",),
        )


class MissingCapabilityPayloadAction:
    def native_amr_materialization(self, *, key):
        return _descriptor(key, capabilities=None)


@pytest.mark.parametrize(
    ("action", "error", "message"),
    (
        (MissingProtocolAction(), TypeError, "must implement native_amr_materialization"),
        (MalformedDescriptorAction(), TypeError, "must return an exact"),
        (NonDeterministicAction(), ValueError, "non-deterministic"),
        (WrongKeyAction(), ValueError, "another transfer key"),
        (WrongVersionAction(), ValueError, "schema_version must be exactly 1"),
        (MissingCapabilityAction(), ValueError, "missing capabilities"),
        (MissingCapabilityPayloadAction(), TypeError, "exact capability evidence"),
    ),
)
def test_action_protocol_fails_closed(action, error, message):
    with pytest.raises(error, match=message):
        _prepared_entry(action)
