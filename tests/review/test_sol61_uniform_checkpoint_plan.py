"""Two-rank host protocol: real Uniform prepare, consensus and manifest sealing.

Only the ABI and transport are explicit host seams; no native catalogue is replaced.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from types import ModuleType, SimpleNamespace
import ast
import importlib
import subprocess
import sys

import numpy as np
import pytest

from pops.identity import make_identity
from pops.output import _checkpoint_collective as protocol
from pops.runtime._checkpoint_manifest import seal_checkpoint_payload
from pops.runtime._checkpoint_spatial import CheckpointSpatialContract
from pops.runtime._lifecycle import _LifecycleMixin
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import FixedDt, Program


@pytest.fixture
def uniform_io(monkeypatch):
    descriptor = ModuleType("pops.runtime._engine_descriptors")
    descriptor.abi_key = lambda: "host-explicit-abi"
    monkeypatch.setitem(sys.modules, descriptor.__name__, descriptor)
    # Load the exact source adapter without asking the native selector for a DSO.
    name = "pops.runtime._system_io"
    previous = sys.modules.pop(name, None)
    module = importlib.import_module(name)
    yield module._SystemIO
    sys.modules.pop(name, None)
    if previous is not None:
        sys.modules[name] = previous


class NativeMetadata:
    """Explicit native metadata seam, including a legitimately local shard."""
    def __init__(self, rank):
        self.rank = rank
        self.local_state = np.arange(rank * 2, (rank + 1) * 2, dtype=np.float64)
        self.ncomp = 1
        self.stride = 1
    def time(self): return 0.
    def macro_step(self): return 0
    def block_names(self): return ("u",)
    def n_vars(self, block): return self.ncomp
    def variable_names(self, block, kind): return tuple("q%d" % i for i in range(self.ncomp))
    def state_global(self, block): return np.tile(np.arange(4, dtype=np.float64), (self.ncomp, 1))
    def capture_auxiliary_checkpoint_accepted_state(self): return b"POPSAUX2"
    def field_provider_slots(self): return ()
    def installed_program_hash(self): return "7" * 64
    def history_names(self): return ()
    def program_cache_nodes(self): return ()
    def program_substeps(self): return 1
    def program_stride(self): return self.stride
    def program_cadence_window_steps(self): return 0
    def program_cadence_window_dt(self): return 0.
    def program_cadence_window_start_time(self): return 0.
    def program_last_dt(self): return 0.
    def effective_options_report(self):
        return {"topology": {"dimension": 1, "periodicity": [True]}, "eb": {
            "enabled": False, "geometry_mode": "none", "kappa_min": 0.,
            "face_open_eps": 0., "cut_theta_min": 0., "semantic_digest": "",
            "materialization_digest": "", "generation": 0}}


def owners(uniform_io):
    class Owner(uniform_io, _LifecycleMixin):
        def __init__(self, rank):
            self.rank = rank
            self._s = NativeMetadata(rank)
            self._last_run_identity = self._last_run_manifest = None
            self._bound_snapshot = SimpleNamespace(**{
                name + "_identity": make_identity(name, {"host": "shared"})
                for name in ("semantic", "artifact", "bind")})
            self._checkpoint_spatial_contract = CheckpointSpatialContract(
                1, (4,), (0.,), (1.,), (True,), (),
                make_identity("native-spatial-layout", {"shared": True}).token)
            policy = FixedDt(.01)
            program = Program("host_checkpoint")
            program.step_strategy(policy)
            self._temporal_restart_state = TemporalRestartState()
            self._temporal_restart_state.configure_program(
                program.temporal_manifest(), time=0., macro_step=0, strategy=policy)
        def time(self): return self._s.time()
        def macro_step(self): return self._s.macro_step()
    return [Owner(rank) for rank in range(2)]


def run_protocol(monkeypatch, pair, target, *, bad_state=False):
    """Run real consensus over an ordered, barrier-bounded two-rank transport."""
    barrier = Barrier(2, timeout=5)
    rows = {}
    calls = []
    handles = [SimpleNamespace(rank=i, round=0) for i in range(2)]
    def gather(handle, value):
        key = handle.round
        rows[key, handle.rank] = value
        barrier.wait()
        result = tuple(rows[key, rank] for rank in range(2))
        barrier.wait()
        handle.round += 1
        return result
    def broadcast(handle, value, root):
        return gather(handle, value)[root]
    monkeypatch.setattr(protocol, "checkpoint_topology", lambda owner:
        protocol.CheckpointTopology(owner.rank, 2, handles[owner.rank]))
    monkeypatch.setattr(protocol, "allgather_value", gather)
    monkeypatch.setattr(protocol, "broadcast_value", broadcast)
    def one(owner):
        def prepare():
            plan = owner._prepare_checkpoint_capture(target)
            calls.append((owner.rank, "prepare"))
            return plan, plan.capture_identity
        def capture(plan):
            calls.append((owner.rank, "capture"))
            payload = dict(plan.payload)
            payload["state_u"] = owner._s.state_global("u").copy()
            if bad_state and owner.rank == 1:
                payload["state_u"][0, 0] += 1.
            identity = seal_checkpoint_payload(owner, payload, runtime_kind="uniform")
            return payload, identity.token
        def publish(payload):
            calls.append((owner.rank, "publish"))
            return str(target)
        try:
            return protocol.collective_checkpoint_capture(owner, "host Uniform", prepare, capture, publish)
        except Exception as error:
            return error
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(one, pair))
    return results, calls


@pytest.mark.parametrize("empty_peer", (False, True))
@pytest.mark.parametrize("width", (1, 3))
def test_distinct_local_shards_share_prepare_and_only_root_publishes(uniform_io, monkeypatch, tmp_path, empty_peer, width):
    pair = owners(uniform_io)
    for owner in pair:
        owner._s.ncomp = width
    if empty_peer:
        pair[1]._s.local_state = np.empty((0,), dtype=np.float64)
    assert pair[0]._s.local_state.tobytes() != pair[1]._s.local_state.tobytes()
    results, calls = run_protocol(monkeypatch, pair, tmp_path / "checkpoint.npz")
    assert results == [str(tmp_path / "checkpoint.npz")] * 2
    assert sorted(calls) == [(0, "capture"), (0, "prepare"), (0, "publish"),
                             (1, "capture"), (1, "prepare")]


@pytest.mark.parametrize("fault", ("metadata", "geometry", "cadence", "bind"))
def test_genuine_plan_mismatch_stops_every_rank_before_capture(uniform_io, monkeypatch, tmp_path, fault):
    pair = owners(uniform_io)
    if fault == "metadata":
        pair[1]._s.ncomp = 2
    if fault == "cadence":
        pair[1]._s.stride = 2
    if fault == "geometry":
        pair[1]._checkpoint_spatial_contract = replace(pair[1]._checkpoint_spatial_contract, upper=(2.,))
    if fault == "bind":
        pair[1]._bound_snapshot.bind_identity = make_identity("bind", {"host": "foreign"})
    results, calls = run_protocol(monkeypatch, pair, tmp_path / "checkpoint.npz")
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert all("capture plans differ across ranks" in str(result) for result in results)
    assert sorted(calls) == [(0, "prepare"), (1, "prepare")]


def test_genuine_state_mismatch_stops_every_rank_before_publication(uniform_io, monkeypatch, tmp_path):
    results, calls = run_protocol(monkeypatch, owners(uniform_io), tmp_path / "checkpoint.npz", bad_state=True)
    assert all(isinstance(result, RuntimeError) for result in results), results
    assert all("sealed payloads differ across ranks" in str(result) for result in results)
    assert all(phase != "publish" for _, phase in calls)


def installed_component(path):
    """Exact metadata object; no fake binary is installed or loaded by this probe."""
    from pops import interfaces
    from pops._platform_contracts import proven_serial_manifest
    from pops.external import InstalledComponent
    from pops.external.artifacts import ComponentRuntimeContract
    from pops.model import ComponentManifest
    interface = interfaces.Transfer
    manifest = ComponentManifest(
        uri="pops://review/checkpoint/transfer", component_type="transfer", version="1.0.0",
        facets=interface.facets,
        signature={"generic": True, "native_interface": interface.signature_declaration()},
        interfaces=interface.manifest_declarations(),
        target={"variants": [{"dimension": 1, "scalar": "float64", "device": "cpu", "features": []}]},
        entry_points={"interface_table": "pops_component_interface_v1"})
    return InstalledComponent(
        manifest.component_id, manifest.manifest_digest,
        ComponentRuntimeContract.from_manifest(manifest), interface,
        proven_serial_manifest(backend="host", target="system", abi="host-metadata"),
        manifest.entry_points, make_identity("binary", {"metadata-only": True}),
        make_identity("artifact", {"metadata-only": True}), path, "source", object())


def test_same_component_residence_keeps_bind_identity_and_local_provenance(tmp_path):
    """Historical projection differed; the versioned content projection agrees."""
    from pops.codegen._plans import _evidence
    first = installed_component(tmp_path / "rank0" / "component.so")
    second = replace(first, path=tmp_path / "rank1" / "component.so")
    assert first.to_data()["path"] != second.to_data()["path"]
    assert first.binary_identity == second.binary_identity
    assert first.artifact_identity == second.artifact_identity
    assert make_identity("bind", first.to_data()) != make_identity("bind", second.to_data())
    assert first.bind_identity_data()["schema_version"] == 2
    assert make_identity("bind", _evidence({"map": first}, where="install.components")) == \
        make_identity("bind", _evidence({"map": second}, where="install.components"))


@pytest.mark.parametrize("fault", ("binary", "artifact", "entries", "origin", "loaded", "runtime"))
def test_component_content_and_execution_metadata_remain_exact(tmp_path, fault):
    from pops.codegen._plans import _evidence
    first = installed_component(tmp_path / "rank0" / "component.so")
    changes = {
        "binary": {"binary_identity": make_identity("binary", {"foreign": True})},
        "artifact": {"artifact_identity": make_identity("artifact", {"foreign": True})},
        "entries": {"entry_symbols": {"interface_table": "foreign_entry"}},
        "origin": {"origin": "fixed"},
        "loaded": {"native_handle": None},
        "runtime": {"runtime_contract": replace(first.runtime_contract, component_type="foreign")},
    }
    second = replace(first, **changes[fault])
    assert _evidence(first, where="install.components") != _evidence(second, where="install.components")


def test_relocated_component_flows_through_both_strict_checkpoint_votes(uniform_io, monkeypatch, tmp_path):
    from pops.codegen._plans import _evidence
    pair = owners(uniform_io)
    first = installed_component(tmp_path / "rank0" / "component.so")
    components = (first, replace(first, path=tmp_path / "rank1" / "component.so"))
    for owner, component in zip(pair, components, strict=True):
        owner._bound_snapshot.bind_identity = make_identity(
            "bind", _evidence({"map": component}, where="install.components"))
    results, calls = run_protocol(monkeypatch, pair, tmp_path / "checkpoint.npz")
    assert results == [str(tmp_path / "checkpoint.npz")] * 2
    assert [(rank, phase) for rank, phase in calls if phase == "publish"] == [(0, "publish")]


def test_actual_resident_bytes_are_still_authenticated_at_both_paths(monkeypatch, tmp_path):
    from pops.external import ComponentPackageError
    from pops.external import artifacts
    first_path, second_path = tmp_path / "rank0.dat", tmp_path / "rank1.dat"
    image = b"explicit opaque host input; not an executable component"
    first_path.write_bytes(image)
    second_path.write_bytes(image)
    first = replace(installed_component(first_path), native_handle=None,
                    binary_identity=artifacts._binary_identity(image))
    second = replace(first, path=second_path)
    # Only symbol inspection is an explicit unit seam; digest and files are real.
    monkeypatch.setattr(artifacts, "inspect_exported_symbols", lambda path: set(first.entry_symbols.values()))
    first.verify()
    second.verify()
    assert first.bind_identity_data() == second.bind_identity_data()
    second_path.write_bytes(image + b" changed")
    with pytest.raises(ComponentPackageError, match="installed binary identity mismatch"):
        second.verify()
    first.verify()


def test_component_free_bind_payload_and_digest_are_byte_exact_to_parent():
    from pops.codegen import _plans
    from pops.identity import canonical_bytes
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["rtk", "proxy", "git", "show", "bfae73f34174079bc6f2d8f4d50aa11d08eca5b5:python/pops/codegen/_plans.py"],
        cwd=root, text=True)
    tree = ast.parse(source)
    evidence = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_evidence")
    install = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "InstallPlan")
    payload = next(node for node in install.body if isinstance(node, ast.FunctionDef) and node.name == "_payload")
    namespace = dict(vars(_plans))
    exec(compile(ast.Module(body=[evidence, payload], type_ignores=[]), "historical-bfae-projection", "exec"), namespace)
    plan = SimpleNamespace(
        artifact=SimpleNamespace(artifact_identity=make_identity("artifact", {"shared": 1})),
        bind_inputs=SimpleNamespace(inputs_identity=make_identity("bind-inputs", {"shared": 1})),
        target="system", capabilities={"dimension": 2},
        instances={"u": {"initial": np.arange(6, dtype=np.float64).reshape(2, 3)}},
        params={"alpha": .125}, aux={}, resources={}, components={},
        initial_values={}, execution_context={"communicator": "serial"})
    old = namespace["_payload"](plan)
    new = _plans.InstallPlan._payload(plan)
    assert canonical_bytes(old) == canonical_bytes(new)
    assert make_identity("bind", old) == make_identity("bind", new)
