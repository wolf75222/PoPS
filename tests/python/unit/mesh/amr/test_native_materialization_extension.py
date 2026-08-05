"""Open AMR actions lower through one strict immutable native descriptor."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pops.amr import (
    CanonicalOptions,
    NativeAMRActionKind,
    NativeAMRMaterializationCapabilities,
    NativeAMRMaterializationDescriptor,
    NativeAMRMaterializationKind,
    TransferCapabilities,
)
from pops.codegen._amr_plan_validation import (
    _physical_axis_contract,
    _ranked_ghost_depth,
    _validated_native_materialization,
)
from pops.domain import CartesianDomain
from pops.mesh import CartesianGrid, LayoutPlanBuilder, PeriodicAxes
from pops.mesh._amr.transfer import (
    AccuracyRequirement,
    CELL_CENTERED,
    CELL_SPACE,
    CONSERVATIVE_REPRESENTATION,
    DENSE_STORAGE,
    FACE_CENTERED,
    FACE_SPACE,
    FACE_Z_CENTERED,
    ORIENTED_FACE_CENTERINGS,
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


def _ranked_face_requirement(dimension, centering):
    frame = CartesianDomain(
        "ranked-native-materialization-%d" % dimension,
        tuple(0.0 for _ in range(dimension)),
        tuple(1.0 for _ in range(dimension)),
    ).frame()
    grid = CartesianGrid(
        frame=frame,
        cells=tuple(8 for _ in range(dimension)),
        periodic=PeriodicAxes(frame.axes),
    )
    state = Handle(
        "face_z", kind="state", owner=OwnerPath.model("ranked-native-materialization")
    )
    builder = LayoutPlanBuilder(OWNER)
    layout = builder.layout("ranked_%d" % dimension, final_amr_layout(grid))
    builder.assign_state(state, layout)
    layout_plan = builder.resolve(states=(state,))
    requirement = TransferRequirement(
        state,
        layout,
        TransferKey(
            FACE_SPACE,
            centering,
            CONSERVATIVE_REPRESENTATION,
            DENSE_STORAGE,
            PROLONGATION,
        ),
        PHYSICAL,
        AccuracyRequirement(
            order=2,
            ghost_depth=tuple(1 for _ in range(dimension)),
            dimension=dimension,
            refinement_ratio=tuple(2 for _ in range(dimension)),
            conservative=True,
        ),
    )
    return SimpleNamespace(layout_plan=layout_plan), requirement


def test_face_centering_contract_exports_z_and_uses_layout_axis_names() -> None:
    assert tuple(axis.name for axis in ORIENTED_FACE_CENTERINGS) == (
        "face_x",
        "face_y",
        "face_z",
    )
    plan, requirement = _ranked_face_requirement(3, FACE_Z_CENTERED)

    axis, axis_kind, dimension = _physical_axis_contract(plan, requirement)

    assert axis == (FACE_SPACE.qualified_id, FACE_Z_CENTERED.qualified_id)
    assert axis_kind == "face"
    assert dimension == 3
    assert _ranked_ghost_depth((2,), dimension=dimension, where="test") == (2, 2, 2)
    assert _ranked_ghost_depth((1, 2, 3), dimension=dimension, where="test") == (1, 2, 3)


def test_face_centering_must_belong_to_the_exact_layout_axes() -> None:
    plan, requirement = _ranked_face_requirement(2, FACE_Z_CENTERED)

    _, axis_kind, dimension = _physical_axis_contract(plan, requirement)

    assert axis_kind is None
    assert dimension == 2
    with pytest.raises(ValueError, match="face_x/face_y/face_z"):
        _ranked_face_requirement(3, FACE_CENTERED)


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
