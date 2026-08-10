import json
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from pops import _pops
from pops._generated_release_contract import UNIFORM_CHECKPOINT_PAYLOAD_VERSION
from tests.python.unit.codegen._typed_artifact_fixture import artifact_fixture
from tests.python.support.native_execution_context import artifact_execution_context
from pops.codegen._plans import BindInputs, InstallPlan
from pops.identity import make_identity
from pops.mesh._layout_plan_contracts import NativeSpatialLayout
from pops.runtime._bound_snapshot import (
    BoundSnapshot,
    _require_exact_install_inputs,
    build_uniform_snapshot,
)
from pops.runtime._checkpoint_manifest import (
    IDENTITY_KEY,
    MANIFEST_KEY,
    authenticate_checkpoint_payload,
    inspect_checkpoint_payload_integrity,
    seal_checkpoint_payload,
)
from pops.runtime._checkpoint_spatial import (
    CheckpointSpatialContract,
    add_checkpoint_spatial_contract,
    authenticate_checkpoint_spatial_contract,
    cell_count,
    install_checkpoint_spatial_contract,
    inspect_checkpoint_spatial_contract,
)
from pops.runtime._run_manifest import RunManifest
from pops.runtime._step_strategy import run_control_payload
from pops.runtime._amr_system import AmrSystem
from pops.runtime._system import System
from pops.time import AdaptiveCFL, FixedDt, StepTransactionPlan


def _native_spatial_layout(dimension, shape):
    axes = tuple("xyz"[:dimension])
    return NativeSpatialLayout(
        layout_id="case:test/layout:grid",
        coordinate_system="pops://coordinates/cartesian-nd@1",
        cell_measure="pops://measures/cartesian-cell@1",
        axis_names=axes,
        shape=shape,
        lower=tuple(-float(axis + 1) for axis in range(dimension)),
        upper=tuple(float(extent - axis - 1) for axis, extent in enumerate(shape)),
        periodicity=tuple(axis % 2 == 0 for axis in range(dimension)),
        centering="cell",
        decomposition={"kind": "single_box", "shape": list(shape)},
    )


def _run_control(cfl=0.4):
    return run_control_payload(AdaptiveCFL(cfl))


def _bound_snapshot():
    return BoundSnapshot(
        semantic_identity=make_identity("semantic", {"problem": "advection"}),
        artifact_identity=make_identity("artifact", {"binary": "abc"}),
        layout={"kind": "uniform"},
        blocks=[
            {
                "name": "tracer",
                "definition_identity": {"model": "m"},
                "spatial": {"flux": "hll"},
                "evolve": True,
            }
        ],
        field_plans={},
        step_transaction=StepTransactionPlan(FixedDt(0.1)).to_data(),
        params=[],
        aux_evidence={},
        initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", {"slots": []}),
    )


def _exact_install_plan():
    fixture = artifact_fixture()
    native_abi = _pops.abi_key()
    if not isinstance(native_abi, str) or not native_abi:
        raise RuntimeError("loaded native runtime exposes no authenticated ABI key")
    for block in fixture.blocks:
        block.model.definition_identity = {
            "protocol": "pops.test.compiled-model-definition.v1",
            "block": block.name,
        }
        block.model.abi_key = native_abi
    for row in fixture.layout_programs:
        row.program.abi_key = native_abi
    artifact = type(fixture)(
        plan=fixture.plan,
        program=fixture.program,
        blocks=fixture.blocks,
    )
    bind_inputs = BindInputs()
    return InstallPlan(
        artifact=artifact,
        bind_inputs=bind_inputs,
        instances={
            block.name: {"model": block.model, "spatial": block.spatial}
            for block in artifact.blocks
        },
        params=artifact.bind_schema.resolve_bind({}, compile_values=artifact.plan.compile_values),
        aux={},
        execution_context=artifact_execution_context(artifact),
    )


def test_bound_snapshot_has_domain_separated_bind_identity_and_json_view():
    snapshot = _bound_snapshot()
    assert snapshot.bind_identity.domain == "bind"
    payload = snapshot.to_dict()
    assert payload["schema_version"] == 7
    assert payload["field_plans"] == {}
    assert "solvers" not in payload and not hasattr(snapshot, "solvers")
    assert payload["bind_identity"]["hexdigest"] == snapshot.bind_identity.hexdigest
    assert "outputs" not in payload and "diagnostics" not in payload
    assert not hasattr(snapshot, "outputs") and not hasattr(snapshot, "diagnostics")
    json.dumps(payload, allow_nan=False)


def test_bound_snapshot_projects_execution_context_identity_digest_without_loss():
    identity = make_identity("runtime-backend-manifest", {"backend": "serial"})
    snapshot = BoundSnapshot(
        semantic_identity=make_identity("semantic", {}),
        artifact_identity=make_identity("artifact", {}),
        layout={"kind": "uniform"},
        blocks=[],
        field_plans={},
        step_transaction={},
        params=[],
        aux_evidence={},
        initial_evidence={},
        bind_schema_identity=make_identity("bind-schema", {}),
        execution_context={"backend_identity": identity.to_data()},
    )
    assert snapshot.to_dict()["execution_context"]["backend_identity"]["digest"] == {
        "bytes_hex": identity.hexdigest,
    }


def test_bound_snapshot_refuses_a_bare_bind_identity():
    with pytest.raises(TypeError, match="bind_identity"):
        BoundSnapshot(
            semantic_identity=make_identity("semantic", {}),
            artifact_identity=make_identity("artifact", {}),
            layout={"kind": "uniform"},
            blocks=[],
            field_plans={},
            step_transaction={},
            params=[],
            aux_evidence={},
            initial_evidence={},
            bind_schema_identity=make_identity("bind-schema", {}),
            bind_identity=make_identity("bind", {"spoofed": True}),
        )


def test_final_bound_snapshot_uses_only_the_verified_install_plan_authority():
    plan = _exact_install_plan()
    engine = SimpleNamespace(
        _execution_context=plan.execution_context,
        _lower_spatial=lambda value: value,
    )
    resolved_models = {name: spec["model"] for name, spec in plan.instances.items()}

    snapshot = build_uniform_snapshot(
        engine,
        plan.artifact,
        resolved_models,
        plan.instances,
        plan.artifact.plan.field_plans,
        plan.aux,
        plan.params,
        install_plan=plan,
    )

    assert snapshot.bind_identity == plan.bind_identity


def test_final_bound_snapshot_refuses_a_structural_install_plan_lookalike():
    plan = _exact_install_plan()
    engine = SimpleNamespace(_execution_context=plan.execution_context)
    lookalike = SimpleNamespace(
        artifact=plan.artifact,
        instances=plan.instances,
        params=plan.params,
        aux=plan.aux,
        bind_identity=plan.bind_identity,
    )

    with pytest.raises(TypeError, match="exact InstallPlan"):
        _require_exact_install_inputs(
            engine,
            plan.artifact,
            plan.instances,
            plan.artifact.plan.field_plans,
            plan.aux,
            plan.params,
            lookalike,
        )


@pytest.mark.parametrize(
    "changed",
    ("artifact", "instances", "params", "aux", "field_plans", "execution_context"),
)
def test_final_bound_snapshot_refuses_every_install_plan_input_alias(changed):
    plan = _exact_install_plan()
    engine = SimpleNamespace(_execution_context=plan.execution_context)
    values = {
        "compiled": plan.artifact,
        "instances": plan.instances,
        "field_plans": plan.artifact.plan.field_plans,
        "aux": plan.aux,
        "params": plan.params,
    }
    if changed == "artifact":
        values["compiled"] = object()
    elif changed == "instances":
        values["instances"] = dict(plan.instances)
    elif changed == "params":
        values["params"] = plan.artifact.bind_schema.resolve_bind(
            {}, compile_values=plan.artifact.plan.compile_values
        )
    elif changed == "aux":
        values["aux"] = dict(plan.aux)
    elif changed == "field_plans":
        values["field_plans"] = dict(plan.artifact.plan.field_plans)
    else:
        engine._execution_context = None

    with pytest.raises(ValueError, match="exact value from the InstallPlan"):
        _require_exact_install_inputs(engine, install_plan=plan, **values)


def test_bound_snapshot_refuses_repr_based_extension():
    with pytest.raises(TypeError, match="cannot enter bind identity"):
        BoundSnapshot(
            semantic_identity=make_identity("semantic", {}),
            artifact_identity=make_identity("artifact", {}),
            layout={"kind": "uniform"},
            blocks=[],
            field_plans={"phi": object()},
            step_transaction={},
            params=[],
            aux_evidence={},
            initial_evidence={},
            bind_schema_identity=make_identity("bind-schema", {}),
        )


def test_run_identity_changes_only_with_effective_controls():
    bind = _bound_snapshot().bind_identity
    first = RunManifest(
        bind_identity=bind,
        start_time=0.0,
        start_macro_step=0,
        controls={
            "t_end": 1.0,
            "step_transaction": _run_control(0.4),
            "max_steps": 10,
            "output_mode": "current-directory",
        },
    )
    same = RunManifest(
        bind_identity=bind,
        start_time=0.0,
        start_macro_step=0,
        controls={
            "t_end": 1.0,
            "step_transaction": _run_control(0.4),
            "max_steps": 10,
            "output_mode": "current-directory",
        },
    )
    changed = RunManifest(
        bind_identity=bind,
        start_time=0.0,
        start_macro_step=0,
        controls={
            "t_end": 1.0,
            "step_transaction": _run_control(0.2),
            "max_steps": 10,
            "output_mode": "current-directory",
        },
    )
    assert first.run_identity == same.run_identity
    assert first.run_identity != changed.run_identity


def test_run_identity_authenticates_restart_continuation_lineage():
    bind = _bound_snapshot().bind_identity
    controls = {
        "t_end": 1.0,
        "step_transaction": _run_control(0.4),
        "max_steps": 10,
        "output_mode": "current-directory",
    }
    recorded = make_identity("run", {"continuation": "recorded"})
    regridded = make_identity("run", {"continuation": "regridded"})

    exact = RunManifest(
        bind_identity=bind,
        continuation_identity=recorded,
        start_time=0.5,
        start_macro_step=5,
        controls=controls,
    )
    transformed = RunManifest(
        bind_identity=bind,
        continuation_identity=regridded,
        start_time=0.5,
        start_macro_step=5,
        controls=controls,
    )

    assert exact.run_identity != transformed.run_identity
    assert exact.to_dict()["payload"]["continuation_identity"] == recorded.token
    assert RunManifest.from_dict(exact.to_dict()).to_dict() == exact.to_dict()
    with pytest.raises(TypeError, match="continuation_identity"):
        RunManifest(
            bind_identity=bind,
            continuation_identity=make_identity("restart", {}),
            start_time=0.5,
            start_macro_step=5,
            controls=controls,
        )


def test_run_manifest_strict_round_trip_and_no_numeric_coercion():
    bind = _bound_snapshot().bind_identity
    manifest = RunManifest(
        bind_identity=bind,
        start_time=0.0,
        start_macro_step=0,
        controls={
            "t_end": 1.0,
            "step_transaction": _run_control(),
            "max_steps": 10,
            "output_mode": "current-directory",
        },
    )
    assert RunManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
    with pytest.raises(TypeError, match="max_steps"):
        RunManifest(
            bind_identity=bind,
            start_time=0.0,
            start_macro_step=0,
            controls={
                "t_end": 1.0,
                "step_transaction": _run_control(),
                "max_steps": True,
                "output_mode": "current-directory",
            },
        )
    with pytest.raises(ValueError, match="finite"):
        RunManifest(
            bind_identity=bind,
            start_time=0.0,
            start_macro_step=0,
            controls={
                "t_end": float("nan"),
                "step_transaction": _run_control(),
                "max_steps": 10,
                "output_mode": "current-directory",
            },
        )


def test_internal_engines_do_not_reintroduce_public_strategy_controls():
    for runtime in (System, AmrSystem):
        signature = inspect.signature(runtime.run)
        assert "strategy" not in signature.parameters
        assert "cfl" not in signature.parameters
        assert signature.parameters["controls"].default is None
        assert signature.parameters["max_steps"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("shape", "ratios", "expected_cells"),
    [
        ((7,), ((3,),), 7),
        ((3, 4), ((2, 3),), 12),
        ((2, 3, 4), ((2, 1, 4),), 24),
    ],
)
def test_checkpoint_spatial_contract_is_exact_and_rank_generic(shape, ratios, expected_cells):
    native = _native_spatial_layout(len(shape), shape)
    contract = CheckpointSpatialContract.from_native_layout(
        native, transition_ratios=ratios
    )

    assert contract.dimension == len(shape)
    assert contract.shape == shape
    assert all(len(row) == contract.dimension for row in contract.refinement_ratios)
    assert cell_count(contract.shape) == expected_cells
    assert contract.cells_at_level(1) == expected_cells * cell_count(ratios[0])
    assert CheckpointSpatialContract.from_data(contract.to_data()) == contract


def test_checkpoint_spatial_dimension_mismatch_is_refused_before_restart_mutation():
    current_owner = SimpleNamespace()
    recorded_owner = SimpleNamespace()
    install_checkpoint_spatial_contract(
        current_owner, _native_spatial_layout(2, (3, 4)), transition_ratios=((2, 3),)
    )
    recorded = install_checkpoint_spatial_contract(
        recorded_owner,
        _native_spatial_layout(3, (3, 4, 5)),
        transition_ratios=((2, 3, 4),),
    )
    payload = {}
    add_checkpoint_spatial_contract(payload, recorded)
    mutation_started = False

    with pytest.raises(ValueError, match="dimension 3 does not match native dimension 2"):
        authenticate_checkpoint_spatial_contract(current_owner, payload)
        mutation_started = True
    assert mutation_started is False


def test_checkpoint_spatial_schema_refuses_padding_and_parallel_2d_keys():
    native = _native_spatial_layout(3, (2, 3, 4))
    contract = CheckpointSpatialContract.from_native_layout(
        native, transition_ratios=((2, 3, 4),)
    )
    forged_schema = contract.to_data()
    forged_schema["schema_version"] = True
    with pytest.raises(TypeError, match="schema_version must be an exact integer"):
        CheckpointSpatialContract.from_data(forged_schema)

    data = contract.to_data()
    data["shape"].append(1)
    with pytest.raises(ValueError, match="shape length"):
        CheckpointSpatialContract.from_data(data)

    payload = {}
    add_checkpoint_spatial_contract(payload, contract)
    payload["nx"] = 2
    with pytest.raises(ValueError, match="forbidden legacy spatial keys"):
        inspect_checkpoint_spatial_contract(payload)


def test_checkpoint_install_refuses_unresolved_scalar_ratio_metadata():
    native = _native_spatial_layout(3, (2, 3, 4))
    with pytest.raises(TypeError, match="already be an exact-rank axis vector"):
        CheckpointSpatialContract.from_native_layout(
            native, transition_ratios=(2, 3)
        )


def test_checkpoint_install_requires_native_products_to_match_before_publishing_authority():
    calls = []

    class Native:
        def _prepare_checkpoint_spatial_contract(self, data):
            calls.append(data)
            return [24, 192]

    owner = SimpleNamespace(_s=Native())
    contract = install_checkpoint_spatial_contract(
        owner,
        _native_spatial_layout(3, (2, 3, 4)),
        transition_ratios=((2, 1, 4),),
    )
    assert calls == [contract.to_data()]

    owner = SimpleNamespace(
        _s=SimpleNamespace(_prepare_checkpoint_spatial_contract=lambda _data: [24, 193])
    )
    with pytest.raises(RuntimeError, match="products differ"):
        install_checkpoint_spatial_contract(
            owner,
            _native_spatial_layout(3, (2, 3, 4)),
            transition_ratios=((2, 1, 4),),
        )
    assert not hasattr(owner, "_checkpoint_spatial_contract")


def test_checkpoint_manifest_authenticates_exact_payload_and_runtime_identities(monkeypatch):
    snapshot = _bound_snapshot()
    run = RunManifest(
        bind_identity=snapshot.bind_identity,
        start_time=0.0,
        start_macro_step=0,
        controls={
            "t_end": 1.0,
            "step_transaction": _run_control(),
            "max_steps": 10,
            "output_mode": "current-directory",
        },
    )
    owner = SimpleNamespace(
        _checkpoint_identities=lambda: (
            snapshot.semantic_identity,
            snapshot.artifact_identity,
            snapshot.bind_identity,
        ),
        last_run_identity=run.run_identity,
    )
    payload = {
        "pops_checkpoint_version": UNIFORM_CHECKPOINT_PAYLOAD_VERSION,
        "t": 0.5,
        "macro_step": 2,
        "abi_key": "test-abi",
        "state_tracer": np.arange(4, dtype=np.float64),
        "inactive_level": np.empty((0, 4), dtype=np.float64),
    }
    monkeypatch.setattr("pops.runtime._engine_descriptors.abi_key", lambda: "test-abi")
    restart = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")

    class PayloadView:
        files = list(payload)

        def __getitem__(self, key):
            return payload[key]

        def __contains__(self, key):
            return key in payload

    inspected_manifest, inspected_restart = inspect_checkpoint_payload_integrity(
        PayloadView(),
        runtime_kind="uniform",
    )
    assert inspected_restart == restart
    assert inspected_manifest["runtime_kind"] == "uniform"
    assert authenticate_checkpoint_payload(owner, PayloadView(), runtime_kind="uniform") == restart
    assert str(payload[IDENTITY_KEY]) == restart.token
    manifest = json.loads(payload[MANIFEST_KEY])
    assert manifest["runtime_kind"] == "uniform"
    assert manifest["arrays"]["inactive_level"]["shape"] == [0, 4]

    payload["state_tracer"] = np.arange(4, dtype=np.float64) + 1.0
    with pytest.raises(ValueError, match="digest mismatch"):
        authenticate_checkpoint_payload(owner, PayloadView(), runtime_kind="uniform")


def test_checkpoint_without_current_manifest_is_refused(monkeypatch):
    monkeypatch.setattr("pops.runtime._engine_descriptors.abi_key", lambda: "test-abi")
    owner = SimpleNamespace(bound_snapshot=_bound_snapshot())

    class Historical:
        files = ["pops_checkpoint_version"]

        def __getitem__(self, key):
            return 1

    with pytest.raises(ValueError, match="historical formats are refused"):
        authenticate_checkpoint_payload(owner, Historical(), runtime_kind="uniform")
