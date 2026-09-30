"""Exact checkpoint lifecycle envelopes without a native build or fictitious run.

The storage/ABI probes are explicit unit seams. Public native fixtures are unchanged.
"""
from copy import deepcopy
from io import BytesIO
import json
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from pops.identity import make_identity
from pops.output._checkpoint_collective import _require_manifest_restart_identity, _manifest_character_budget
from pops.runtime._checkpoint_manifest import (
    IDENTITY_KEY, MANIFEST_KEY, _seal_checkpoint_payload_with_identities,
    CHECKPOINT_SCHEMA_VERSION, BOUND_INITIAL_CHECKPOINT_SCHEMA_VERSION, RUN_ORIGIN_CHECKPOINT_SCHEMA_VERSION,
    authenticate_checkpoint_payload, checkpoint_lifecycle_evidence, checkpoint_run_identity,
    inspect_checkpoint_payload_integrity, seal_checkpoint_payload, require_bound_initial_children,
)
from pops.runtime._lifecycle import _LifecycleMixin
from pops.runtime._multi_layout_executor import _CompositeTemporalRestartState, _MultiLayoutUniformExecutor
from pops.runtime._run_manifest import begin_run
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import AdaptiveCFL, Program


class _Owner(_LifecycleMixin):
    def __init__(self):
        identities = tuple(make_identity(domain, {"fixture": "bound-initial"})
                           for domain in ("semantic", "artifact", "bind"))
        self._bound_snapshot = SimpleNamespace(**dict(zip(
            ("semantic_identity", "artifact_identity", "bind_identity"), identities, strict=True)))
        self._last_run_identity = self._last_run_manifest = self._restart_lineage_identity = None
        self._time, self._step = 0., 0
        self._temporal_restart_state = TemporalRestartState()
        policy = AdaptiveCFL(.4)
        program = Program("bound_initial_checkpoint")
        program.step_strategy(policy)
        self._temporal_restart_state.configure_program(program.temporal_manifest(),
                                                       time=0., macro_step=0, strategy=policy)
    def time(self): return self._time
    def macro_step(self): return self._step


def _payload(owner):
    return {"t": np.asarray(owner.time()), "macro_step": np.asarray(owner.macro_step()),
            "abi_key": np.asarray("unit-abi"), "state": np.array([1., 2., 3.]),
            "temporal_restart_state": np.asarray(owner._temporal_restart_state.checkpoint_json(
                time=owner.time(), macro_step=owner.macro_step()))}


def _wire(payload):
    stream = BytesIO()
    np.savez(stream, **payload)
    return stream.getvalue()


@pytest.mark.parametrize("kind", ("uniform", "amr"))
def test_initial_wire_authenticates_bound_origin_and_restart_then_real_run(monkeypatch, kind):
    owner = _Owner()
    payload = _payload(owner)
    before = deepcopy(owner._temporal_restart_state.__dict__)
    identity = seal_checkpoint_payload(owner, payload, runtime_kind=kind)
    manifest, inspected = inspect_checkpoint_payload_integrity(payload, runtime_kind=kind)
    assert identity == inspected
    assert manifest["schema_version"] == 2
    assert manifest["origin"] == {"schema_version": 1, "kind": "bound_initial"}
    assert manifest["run_identity"] is None and owner.last_run_identity is None
    assert owner._temporal_restart_state.__dict__ == before
    _require_manifest_restart_identity(manifest, identity.token)
    assert len(payload[MANIFEST_KEY]) <= _manifest_character_budget(tuple(manifest["arrays"]))
    descriptors = ModuleType("pops.runtime._engine_descriptors")
    descriptors.abi_key = lambda: "unit-abi"
    monkeypatch.setitem(sys.modules, descriptors.__name__, descriptors)
    with np.load(BytesIO(_wire(payload)), allow_pickle=False) as stored:
        assert authenticate_checkpoint_payload(owner, stored, runtime_kind=kind) == identity
        source = checkpoint_run_identity(stored)
        assert source is None
        restored = _Owner()
        restored._temporal_restart_state = TemporalRestartState.from_json(
            stored["temporal_restart_state"], time=0., macro_step=0)
        restored._restore_checkpoint_run_identity(source)
    assert restored.last_run_identity is None and restored._restart_lineage_identity is None
    assert restored._temporal_restart_state._restored_pending
    restored._temporal_restart_state.begin_run(restored._temporal_restart_state.strategy,
                                               time=0., macro_step=0)
    request = begin_run(restored, t_end=.1, step_transaction=restored._temporal_restart_state.strategy,
                        max_steps=1, output_dir=None)
    assert request.start_time == 0. and request.start_macro_step == 0
    assert restored.last_run_identity == request.run_identity
    after = _payload(restored)
    seal_checkpoint_payload(restored, after, runtime_kind=kind)
    assert json.loads(after[MANIFEST_KEY])["schema_version"] == 1
    assert "origin" not in json.loads(after[MANIFEST_KEY])


def test_legacy_run_envelope_and_identity_remain_exactly_unchanged():
    owner = _Owner()
    request = begin_run(owner, t_end=.1, step_transaction=owner._temporal_restart_state.strategy,
                        max_steps=1, output_dir=None)
    payload = _payload(owner)
    old_route = deepcopy(payload)
    semantic, artifact, bind = owner._checkpoint_identities()
    old_identity = _seal_checkpoint_payload_with_identities(old_route, runtime_kind="uniform",
        semantic=semantic, artifact=artifact, bind=bind, run=request.run_identity)
    identity = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
    assert identity == old_identity and payload[MANIFEST_KEY] == old_route[MANIFEST_KEY]
    assert checkpoint_run_identity(payload) == request.run_identity
    manifest = json.loads(payload[MANIFEST_KEY])
    _require_manifest_restart_identity(manifest, identity.token)


@pytest.mark.parametrize("fault", ("native_clock", "negative_zero", "native_step", "temporal_clock", "transactions"))
def test_missing_run_alone_never_authenticates_initial_origin(fault):
    owner = _Owner()
    if fault == "native_clock":
        owner._time = .1
    elif fault == "negative_zero":
        owner._time = -0.
    elif fault == "native_step":
        owner._step = 1
    elif fault == "temporal_clock":
        owner._temporal_restart_state.time_hex = (.1).hex()
    else:
        owner._temporal_restart_state.transaction_stats["rejected"] = 1
    with pytest.raises(RuntimeError):
        checkpoint_lifecycle_evidence(owner)
    assert owner.last_run_identity is None


@pytest.mark.parametrize("fault", ("boolean_origin_version", "run_on_initial", "noninitial_clock", "negative_zero"))
def test_rehashed_forged_initial_origin_is_refused_by_both_validators(fault):
    owner = _Owner()
    payload = _payload(owner)
    seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
    manifest = json.loads(payload[MANIFEST_KEY])
    if fault == "boolean_origin_version":
        manifest["origin"]["schema_version"] = True
    elif fault == "run_on_initial":
        from pops.runtime._checkpoint_manifest import _identity_json
        manifest["run_identity"] = _identity_json(make_identity("run", {"forged": 1}))
    elif fault == "noninitial_clock":
        manifest["clock"] = {"time": (.1).hex(), "macro_step": 1}
    else:
        manifest["clock"] = {"time": (-0.).hex(), "macro_step": 0}
    restart = make_identity("restart", {key: value for key, value in manifest.items() if key != "restart_identity"})
    from pops.runtime._checkpoint_manifest import _identity_json
    manifest["restart_identity"] = _identity_json(restart)
    payload[MANIFEST_KEY] = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    payload[IDENTITY_KEY] = restart.token
    with pytest.raises(ValueError):
        _require_manifest_restart_identity(manifest, restart.token)
    with pytest.raises(ValueError):
        inspect_checkpoint_payload_integrity(payload, runtime_kind="uniform")


def test_composite_initial_origin_authenticates_every_child_and_clears_lineage():
    parent = _Owner()
    children = (_Owner(), _Owner())
    parent._temporal_restart_state = _CompositeTemporalRestartState(
        {name: owner._temporal_restart_state for name, owner in zip(("first", "second"), children, strict=True)})
    payload = {"t": np.asarray(0.), "macro_step": np.asarray(0), "abi_key": np.asarray("unit-abi"),
               "layout_ids": np.asarray(("first", "second"))}
    for index, child in enumerate(children):
        image = _payload(child)
        seal_checkpoint_payload(child, image, runtime_kind="uniform")
        payload["layout_checkpoint_%d" % index] = np.frombuffer(_wire(image), dtype=np.uint8).copy()
    seal_checkpoint_payload(parent, payload, runtime_kind="multi_layout_uniform")
    inspect_checkpoint_payload_integrity(payload, runtime_kind="multi_layout_uniform")
    prepared = tuple(SimpleNamespace(payload=_payload(child)) for child in children)
    for child in prepared:
        seal_checkpoint_payload(parent, child.payload, runtime_kind="uniform")
    require_bound_initial_children(prepared, parent._temporal_restart_state)
    probe = SimpleNamespace(_engines={str(index): child for index, child in enumerate(children)})
    _MultiLayoutUniformExecutor._restore_checkpoint_run_identity(probe, None)
    assert probe._last_run_identity is None and probe._restart_lineage_identity is None
    assert all(child.last_run_identity is None for child in children)


def test_supported_metadata_does_not_change_legacy_emission_version():
    from pops._generated_release_contract import CHECKPOINT_ENVELOPE_SCHEMA_VERSION
    assert CHECKPOINT_SCHEMA_VERSION == CHECKPOINT_ENVELOPE_SCHEMA_VERSION
    assert RUN_ORIGIN_CHECKPOINT_SCHEMA_VERSION == 1
    assert BOUND_INITIAL_CHECKPOINT_SCHEMA_VERSION == 2


def test_bound_initial_composite_refuses_admitted_run_origin_child():
    parent, child = _Owner(), _Owner()
    begin_run(child, t_end=.1, step_transaction=child._temporal_restart_state.strategy,
              max_steps=1, output_dir=None)
    payload = _payload(child)
    seal_checkpoint_payload(child, payload, runtime_kind="uniform")
    with pytest.raises(ValueError, match="noninitial child"):
        require_bound_initial_children((SimpleNamespace(payload=payload),), parent._temporal_restart_state)


def test_initial_seal_refuses_forged_temporal_image_before_payload_publication():
    owner = _Owner()
    payload = _payload(owner)
    data = json.loads(str(payload["temporal_restart_state"]))
    data["transaction_stats"]["accepted"] = 1
    payload["temporal_restart_state"] = np.asarray(json.dumps(data))
    with pytest.raises(RuntimeError, match="zero attempted transactions"):
        seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
    assert MANIFEST_KEY not in payload and IDENTITY_KEY not in payload
    assert owner.last_run_identity is None


def test_initial_seal_requires_authenticated_bind_and_declared_controller():
    owner = _Owner()
    owner._bound_snapshot = None
    with pytest.raises(RuntimeError, match="completed pops.bind"):
        checkpoint_lifecycle_evidence(owner)
    owner = _Owner()
    owner._temporal_restart_state = TemporalRestartState()
    with pytest.raises(RuntimeError, match="declared step strategy"):
        checkpoint_lifecycle_evidence(owner)
