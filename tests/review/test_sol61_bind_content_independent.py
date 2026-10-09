"""Independent source/host reception, never native execution or author test helpers.

Real InstalledComponent/CompiledComponentArtifact and InstallPlan payload code.
Synthetic binary bytes have an explicit symbol-inspection seam; the InstallPlan
payload receiver is a controlled record, so full bind/loader admission is pending.
"""

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pops._platform_contracts import proven_serial_manifest
from pops.codegen import _plans
from pops.external import artifacts
from pops.external._package_data import ComponentPackageError
from pops.identity import make_identity, canonical_bytes
from pops.interfaces import Transfer
from pops.model import ComponentManifest

CANDIDATE = "29ce923e008d8506b9e1e28ea5aa00ea74d1d668"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def compiled(monkeypatch):
    # Exact Python authority classes; only nm, not identities, is substituted.
    monkeypatch.setattr(
        artifacts,
        "inspect_exported_symbols",
        lambda path: frozenset({"pops_component_interface_v1"}),
    )
    monkeypatch.setattr(
        artifacts,
        "inspect_symbol_bytes",
        lambda data, suffix: frozenset({"pops_component_interface_v1"}),
    )
    manifest = ComponentManifest(
        uri="pops://review/independent/residence",
        component_type="transfer",
        version="1.0.0",
        facets=Transfer.facets,
        signature={"native_interface": Transfer.to_data()},
        interfaces=Transfer.manifest_declarations(),
        entry_points={"interface_table": "pops_component_interface_v1"},
    )
    Transfer.require_manifest(manifest)
    binary = b"SYNTHETIC PROTOCOL BYTES ONLY; NEVER A NATIVE DSO"
    return artifacts.CompiledComponentArtifact(
        manifest.component_id,
        manifest.manifest_digest,
        artifacts.ComponentRuntimeContract.from_manifest(manifest),
        Transfer,
        proven_serial_manifest(backend="system", target="cpu", abi="host-review-not-native"),
        manifest.entry_points,
        artifacts._binary_identity(binary),
        binary,
        source_package=make_identity("component-package", {"independent": True}),
    )


def payload(component):
    # Execute the complete true InstallPlan payload function, not a duplicate
    # hand-written component serializer. Constructor/loader are out of scope.
    plan = SimpleNamespace(
        artifact=SimpleNamespace(artifact_identity=make_identity("artifact", {"shared": True})),
        bind_inputs=SimpleNamespace(inputs_identity=make_identity("bind-inputs", {"shared": True})),
        target="system",
        capabilities={},
        instances={"distribution": {"initial": np.arange(6.0, dtype=np.float64)}},
        params={},
        aux={},
        resources={},
        components={component.component_id: component},
        initial_values={},
        execution_context=None,
    )
    return _plans.InstallPlan._payload(plan)


def test_relocated_component_changes_only_residence_and_keeps_full_bind(compiled, tmp_path):
    ranks = [compiled.install(tmp_path / str(rank)) for rank in range(3)]
    assert len({r.path for r in ranks}) == 3
    for rank in ranks:
        rank.verify()
    rows = [payload(rank) for rank in ranks]
    assert all(canonical_bytes(row) == canonical_bytes(rows[0]) for row in rows)
    assert len({make_identity("bind", row).token for row in rows}) == 1
    # Independent metadata expectation: no projection derived from a fake hook.
    for rank in ranks:
        evidence = rank.bind_identity_data()
        assert evidence["schema_version"] == 2
        assert "path" not in evidence["component"]
        assert evidence["component"]["binary_identity"] == compiled.binary_identity.to_data()
        assert evidence["component"]["artifact_identity"] == compiled.artifact_identity.to_data()
        assert rank.to_data()["path"] == str(rank.path)
        assert evidence["component"]["runtime_contract"]["component_id"] == compiled.component_id


@pytest.mark.parametrize(
    "field",
    (
        "component_manifest",
        "binary_identity",
        "artifact_identity",
        "entry_symbols",
        "origin",
        "runtime_contract",
        "platform_manifest",
        "loaded",
    ),
)
def test_changed_authority_metadata_cannot_share_bind_identity(compiled, tmp_path, field):
    first = compiled.install(tmp_path / "first")
    edits = {
        "component_manifest": make_identity("component-manifest", {"foreign": True}),
        "binary_identity": artifacts._binary_identity(b"other synthetic bytes"),
        "artifact_identity": make_identity("component-artifact", {"foreign": True}),
        "entry_symbols": {"interface_table": "foreign_symbol"},
        "origin": "fixed",
        "runtime_contract": replace(first.runtime_contract, component_type="foreign"),
        "platform_manifest": proven_serial_manifest(
            backend="system", target="cpu", abi="foreign-host-abi"
        ),
        "loaded": object(),
    }
    key = "native_handle" if field == "loaded" else field
    foreign = replace(first, **{key: edits[field]})
    assert make_identity("bind", payload(first)) != make_identity("bind", payload(foreign))


@pytest.mark.parametrize("fault", ("changed", "missing", "foreign_relocation"))
def test_actual_file_authentication_before_load(compiled, tmp_path, fault):
    installed = compiled.install(tmp_path / "rank0")
    if fault == "changed":
        installed.path.write_bytes(b"changed synthetic bytes")
    elif fault == "missing":
        installed.path.unlink()
    else:
        other = tmp_path / "foreign.so"
        other.write_bytes(b"foreign synthetic bytes")
        installed = replace(installed, path=other)
    with pytest.raises(ComponentPackageError, match="installed binary identity mismatch"):
        installed.verify()


def test_fake_projection_protocol_does_not_receive_installed_normalization():
    class Counterfeit:
        def bind_identity_data(self):
            raise AssertionError("unissued protocol must never be trusted")

        def to_data(self):
            return {"path": "/rank-local/foreign", "binary_identity": "invented"}

    data = _plans._evidence(Counterfeit(), where="review.fake")
    assert data["value"]["path"] == "/rank-local/foreign"
    assert "schema_version" not in data["value"]


def test_parent_component_free_payload_bytes_exact():
    # Full old _evidence function, independently loaded from the frozen parent;
    # no author fixtures or normalized/tolerant comparison.
    text = subprocess.check_output(
        ["git", "show", CANDIDATE + "^:python/pops/codegen/_plans.py"], cwd=ROOT, text=True
    )
    node = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_evidence"
    )
    old = dict(vars(_plans))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "frozen-parent-evidence", "exec"), old)
    values = {
        "array": np.arange(12.0, dtype=np.float64).reshape(3, 4),
        "tuple": (1, None, b"exact"),
        "float": -0.0,
        "identity": make_identity("independent", {"payload": True}),
    }
    assert canonical_bytes(_plans._evidence(values, where="review.legacy")) == canonical_bytes(
        old["_evidence"](values, where="review.legacy")
    )


def test_only_true_installed_type_receives_new_projection(compiled, tmp_path):
    class ForeignInstalled(artifacts.InstalledComponent):
        pass

    real = compiled.install(tmp_path)
    inherited = ForeignInstalled(
        **{name: getattr(real, name) for name in real.__dataclass_fields__}
    )
    assert "path" not in _plans._evidence(real, where="real")["value"]["component"]
    assert _plans._evidence(inherited, where="subclass")["value"]["path"] == str(real.path)


def uniform_capture_identity(bind):
    # Evaluate the true complete native-free capture-identity projection. The
    # remaining agreed runtime metadata are explicit inert host seams.
    text = (ROOT / "python/pops/runtime/_system_io.py").read_text()
    tree = ast.parse(text)
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_prepare_checkpoint_capture"
    )
    node = next(
        n
        for n in method.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "capture_identity" for t in n.targets)
    )
    empty = SimpleNamespace(to_data=lambda: {})
    namespace = dict(
        make_identity=make_identity,
        target=Path("/shared/checkpoint.npz"),
        time=0.0,
        macro_step=0,
        spatial=empty,
        embedded_boundary=empty,
        out={"abi_key": "host-review"},
        block_evidence=[{"block": "distribution"}],
        field_slots=(),
        prog_hash="f" * 64,
        cadence=empty,
        history_plan=empty,
        cache_evidence=[],
        runtime_identities=[
            make_identity("semantic", {"same": True}).to_data(),
            make_identity("artifact", {"same": True}).to_data(),
            bind.to_data(),
        ],
        lifecycle={},
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), "actual Uniform capture projection", "exec"
        ),
        namespace,
    )
    return namespace["capture_identity"]


@pytest.mark.parametrize("foreign", (False, True))
def test_actual_collective_preflight_with_relocated_bind_only(
    compiled, tmp_path, monkeypatch, foreign
):
    from pops.output import _checkpoint_collective as cp

    components = [compiled.install(tmp_path / str(rank)) for rank in range(3)]
    if foreign:
        components[2] = replace(components[2], origin="fixed")
    tokens = [uniform_capture_identity(make_identity("bind", payload(c))) for c in components]
    assert (len(set(tokens)) == 1) is not foreign
    entered = []

    def capture(plan):
        entered.append(plan)
        raise RuntimeError("HOST STOP BEFORE ANY NATIVE CALL")

    for rank in range(3):
        monkeypatch.setattr(
            cp,
            "checkpoint_topology",
            lambda owner, rank=rank: cp.CheckpointTopology(rank, 3, object()),
        )

        def gather(world, envelope):
            # Only transport is a seam. All projected evidence and protocol
            # validation come from actual source methods.
            return tuple(
                {
                    "rank": i,
                    "value": tokens[i] if envelope["error"] is None else None,
                    "error": envelope["error"],
                }
                for i in range(3)
            )

        monkeypatch.setattr(cp, "allgather_value", gather)
        before = len(entered)
        message = (
            "capture plans differ across ranks" if foreign else "HOST STOP BEFORE ANY NATIVE CALL"
        )
        with pytest.raises(RuntimeError, match=message):
            cp.collective_checkpoint_capture(
                object(),
                "Uniform",
                lambda rank=rank: (components[rank], tokens[rank]),
                capture,
                lambda artifact: pytest.fail("publication forbidden in host probe"),
            )
        assert len(entered) == before + (0 if foreign else 1)


@pytest.fixture
def uniform_capture_source():
    text = subprocess.check_output(
        [
            "git",
            "show",
            "01cc51b6127d20ab37c551ca14f4e38611b95add:tests/python/integration/runtime/test_public_captured_diffusion.py",
        ],
        cwd=ROOT,
        text=True,
    )
    nodes = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name in ("capture", "same_images")
    ]
    import contextlib
    import struct

    namespace = dict(
        np=np,
        struct=struct,
        DT=0.01,
        HISTORY_MAX_LAG=1,
        collective_call=lambda world, fn: fn(),
        collective_check=lambda world: contextlib.nullcontext(),
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "Root frozen Uniform fixture", "exec"),
        namespace,
    )
    return namespace


class HostUniformObservation:
    """Explicit observation seam; synthetic bytes never claimed as native CP."""

    def __init__(self, step, diagnostics=None):
        self.step = step
        self._executor = self
        self.calls = []
        self.diagnostics = diagnostics or {}
        self.aux = b"POPSAUX2 HOST TEST ONLY"

    def state_global(self, name):
        return np.zeros((1 if name == "material" else 3, 2, 2))

    def history_global(self, name, slot):
        self.calls.append((name, slot))
        return np.full(
            (2, 2), float(int(name[1:]) + 10 + (10 if self.step == 2 and slot == 1 else 0))
        )

    def history_depth(self, name):
        return 2

    def history_initialized(self, name):
        return True

    def history_fill_count(self, name):
        return self.step

    def history_slot_dt(self, name, slot):
        return 0.01

    def history_sample_identity(self, name):
        import struct

        return (
            b"POPSHID1"
            + struct.pack("<Q", len(name))
            + name.encode()
            + struct.pack("<qQ", -1, 2)
            + b"".join(
                struct.pack(
                    "<QQQQ",
                    2,
                    int.from_bytes(struct.pack("<d", start), "little"),
                    int.from_bytes(struct.pack("<d", 0.01), "little"),
                    1,
                )
                for start in (0.0, (self.step - 1) * 0.01)
            )
        )

    def capture_auxiliary_checkpoint_accepted_state(self):
        return self.aux

    def program_diagnostics(self):
        return self.diagnostics

    def time(self):
        return self.step * 0.01

    def macro_step(self):
        return self.step


def test_root_uniform_fixture_reads_actual_slots_and_accepted_aux_accessor(uniform_capture_source):
    code = uniform_capture_source
    one = HostUniformObservation(1)
    two = HostUniformObservation(2)
    first = code["capture"](object(), one, 2, 3, step=1)
    second = code["capture"](object(), two, 2, 3, step=2, first_solution=first[0]["solution"])
    assert first[1][0] is one.aux and second[1][0] is two.aux
    assert one.calls == two.calls == [(f"q{i}", slot) for i in range(3) for slot in (0, 1)]
    for i in range(3):
        np.testing.assert_array_equal(second[0]["solution"][i], np.full((2, 2), 20.0 + i))
    code["same_images"](
        second,
        code["capture"](
            object(), HostUniformObservation(2), 2, 3, step=2, first_solution=first[0]["solution"]
        ),
    )


def test_observation_diagnostics_are_not_silently_continuation_state(uniform_capture_source):
    code = uniform_capture_source
    before = code["capture"](
        object(), HostUniformObservation(1, {"host.diagnostic": 17.0}), 2, 3, step=1
    )
    after = code["capture"](object(), HostUniformObservation(1), 2, 3, step=1)
    assert before[1][0] == after[1][0] and before[1][2:] == after[1][2:]
    for name in before[0]:
        np.testing.assert_array_equal(before[0][name], after[0][name])
    # Preserve the actual fixture failure rather than accepting a missing metric.
    with pytest.raises(AssertionError):
        code["same_images"](before, after)
