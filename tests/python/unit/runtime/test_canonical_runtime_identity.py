import json
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from pops.identity import make_identity
from pops.runtime._bound_snapshot import BoundSnapshot
from pops.runtime._checkpoint_manifest import (
    IDENTITY_KEY,
    MANIFEST_KEY,
    authenticate_checkpoint_payload,
    seal_checkpoint_payload,
)
from pops.runtime._run_manifest import RunManifest
from pops.runtime.amr_system import AmrSystem
from pops.runtime.system import System


def _bound_snapshot():
    return BoundSnapshot(
        semantic_identity=make_identity("semantic", {"problem": "advection"}),
        artifact_identity=make_identity("artifact", {"binary": "abc"}),
        layout={"kind": "uniform"},
        blocks=[{"name": "tracer", "definition_identity": {"model": "m"},
                 "spatial": {"flux": "hll"}, "evolve": True}],
        solvers={},
        cadence={"kind": "compiled-time", "substeps": 1, "stride": 1, "cfl": "default"},
        params=[], aux_evidence={}, initial_evidence={}, outputs=[], diagnostics=[],
        bind_schema_identity=make_identity("bind-schema", {"slots": []}),
    )


def test_bound_snapshot_has_domain_separated_bind_identity_and_json_view():
    snapshot = _bound_snapshot()
    assert snapshot.bind_identity.domain == "bind"
    assert snapshot.to_dict()["bind_identity"]["hexdigest"] == snapshot.bind_identity.hexdigest
    json.dumps(snapshot.to_dict(), allow_nan=False)


def test_bound_snapshot_refuses_repr_based_extension():
    with pytest.raises(TypeError, match="cannot enter bind identity"):
        BoundSnapshot(
            semantic_identity=make_identity("semantic", {}),
            artifact_identity=make_identity("artifact", {}),
            layout={"kind": "uniform"}, blocks=[], solvers={"phi": object()},
            cadence={"kind": "default"}, params=[], aux_evidence={}, initial_evidence={},
            outputs=[], diagnostics=[],
            bind_schema_identity=make_identity("bind-schema", {}),
        )


def test_run_identity_changes_only_with_effective_controls():
    bind = _bound_snapshot().bind_identity
    first = RunManifest(
        bind_identity=bind, start_time=0.0, start_macro_step=0,
        controls={"t_end": 1.0, "cfl": 0.4, "max_steps": 10, "output_mode": "current-directory"})
    same = RunManifest(
        bind_identity=bind, start_time=0.0, start_macro_step=0,
        controls={"t_end": 1.0, "cfl": 0.4, "max_steps": 10, "output_mode": "current-directory"})
    changed = RunManifest(
        bind_identity=bind, start_time=0.0, start_macro_step=0,
        controls={"t_end": 1.0, "cfl": 0.2, "max_steps": 10, "output_mode": "current-directory"})
    assert first.run_identity == same.run_identity
    assert first.run_identity != changed.run_identity


def test_run_manifest_strict_round_trip_and_no_numeric_coercion():
    bind = _bound_snapshot().bind_identity
    manifest = RunManifest(
        bind_identity=bind, start_time=0.0, start_macro_step=0,
        controls={"t_end": 1.0, "cfl": 0.4, "max_steps": 10,
                  "output_mode": "current-directory"})
    assert RunManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
    with pytest.raises(TypeError, match="max_steps"):
        RunManifest(
            bind_identity=bind, start_time=0.0, start_macro_step=0,
            controls={"t_end": 1.0, "cfl": 0.4, "max_steps": True,
                      "output_mode": "current-directory"})
    with pytest.raises(ValueError, match="finite"):
        RunManifest(
            bind_identity=bind, start_time=0.0, start_macro_step=0,
            controls={"t_end": float("nan"), "cfl": 0.4, "max_steps": 10,
                      "output_mode": "current-directory"})


def test_uniform_and_amr_run_share_the_cfl_resolution_contract():
    assert inspect.signature(System.run).parameters["cfl"].default is None
    assert inspect.signature(AmrSystem.run).parameters["cfl"].default is None


def test_checkpoint_manifest_authenticates_exact_payload_and_runtime_identities(monkeypatch):
    snapshot = _bound_snapshot()
    run = RunManifest(
        bind_identity=snapshot.bind_identity, start_time=0.0, start_macro_step=0,
        controls={"t_end": 1.0, "cfl": 0.4, "max_steps": 10, "output_mode": "current-directory"})
    owner = SimpleNamespace(
        bound_snapshot=snapshot, last_run_identity=run.run_identity)
    payload = {
        "pops_checkpoint_version": 3, "t": 0.5, "macro_step": 2,
        "abi_key": "test-abi", "state_tracer": np.arange(4, dtype=np.float64),
    }
    monkeypatch.setattr("pops.runtime.bricks.abi_key", lambda: "test-abi")
    restart = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")

    class PayloadView:
        files = list(payload)

        def __getitem__(self, key):
            return payload[key]

        def __contains__(self, key):
            return key in payload

    assert authenticate_checkpoint_payload(owner, PayloadView(), runtime_kind="uniform") == restart
    assert str(payload[IDENTITY_KEY]) == restart.token
    assert json.loads(payload[MANIFEST_KEY])["runtime_kind"] == "uniform"

    payload["state_tracer"] = np.arange(4, dtype=np.float64) + 1.0
    with pytest.raises(ValueError, match="digest mismatch"):
        authenticate_checkpoint_payload(owner, PayloadView(), runtime_kind="uniform")


def test_checkpoint_without_current_manifest_is_refused(monkeypatch):
    monkeypatch.setattr("pops.runtime.bricks.abi_key", lambda: "test-abi")
    owner = SimpleNamespace(bound_snapshot=_bound_snapshot())

    class Historical:
        files = ["pops_checkpoint_version"]

        def __getitem__(self, key):
            return 1

    with pytest.raises(ValueError, match="historical formats are refused"):
        authenticate_checkpoint_payload(owner, Historical(), runtime_kind="uniform")
