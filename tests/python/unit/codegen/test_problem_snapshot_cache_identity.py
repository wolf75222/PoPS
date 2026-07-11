"""AuthoringSnapshot participates in the real compiled-Program cache identity."""
from __future__ import annotations

from pathlib import Path

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.codegen.compile_provenance import read_artifact_sidecar  # noqa: E402
from pops.identity import Identity  # noqa: E402
from pops.model import Module  # noqa: E402
class _Program:
    name = "same-program"
    source = 'extern "C" void pops_install_program(void*) {}\n'

    def emit_cpp_program(self, *, model, target):
        del model, target
        return self.source


class _Compiled:
    def __init__(self, so_path, program, model, abi_key, cxx, std, **metadata):
        self.so_path = so_path
        self.program = program
        self.model = model
        self.abi_key = abi_key
        self.cxx = cxx
        self.std = std
        self.problem_hash = metadata["problem_hash"]
        self.cache_key = metadata["cache_key"]
        self._problem_snapshot = metadata.get("problem_snapshot")

    @property
    def authoring_snapshot(self):
        return self._problem_snapshot


def _install_fake_toolchain(monkeypatch, tmp_path):
    """Exercise the production driver and real sidecar writer without invoking a compiler."""
    import pops.codegen.compile_drivers as drivers
    import pops.codegen.cache as cache
    import pops.codegen.loader as loader
    import pops.codegen.module_lowering as lowering
    import pops.codegen.toolchain as toolchain
    import pops.model.manifest as manifest

    monkeypatch.setattr(lowering, "lower_and_validate", lambda model, facade: (model, None))
    monkeypatch.setattr(manifest, "module_manifest_of", lambda model: None)
    monkeypatch.setattr(loader, "CompiledProblem", _Compiled)
    monkeypatch.setattr(drivers, "pops_include", lambda: str(tmp_path))
    monkeypatch.setattr(drivers, "pops_header_signature", lambda include: "HEADER")
    monkeypatch.setattr(
        drivers, "pops_loader_build_flags", lambda cxx: ("fake-c++", [], []))
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cxx, std: "c++23")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "kokkos=fake;mpi=off")
    monkeypatch.setattr(
        cache, "_precision_cache_key", lambda: "precision=double;real_bytes=8")
    monkeypatch.setattr(cache, "_registry_cache_key", lambda: "routes=fake;capvocab=1")
    monkeypatch.setattr(drivers, "_registry_cache_key", lambda: "routes=fake;capvocab=1")
    monkeypatch.setattr(drivers, "_dsl_optflags", lambda: [])
    monkeypatch.setattr(drivers, "deterministic_program_link_flags", lambda flags: list(flags))
    monkeypatch.setattr(
        drivers,
        "_identity_cache_so_path",
        lambda spec_identity: str(tmp_path / (spec_identity.hexdigest + ".so")),
    )

    compiled_paths = []

    def fake_compile(command, context):
        del context
        out = Path(command[command.index("-o") + 1])
        out.write_bytes(b"fake shared object")
        compiled_paths.append(str(out))

    monkeypatch.setattr(drivers, "_run_compile", fake_compile)
    for name in (
        "POPS_CODEGEN_DIR", "POPS_DUMP_IR", "POPS_DUMP_CPP", "POPS_KEEP_GENERATED",
        "POPS_CODEGEN_LOG", "POPS_LOG",
    ):
        monkeypatch.delenv(name, raising=False)
    return drivers, compiled_paths


def test_distinct_problem_snapshots_get_distinct_paths_keys_and_matching_sidecars(
        monkeypatch, tmp_path):
    drivers, compiled_paths = _install_fake_toolchain(monkeypatch, tmp_path)
    model = Module("same-model")
    program = _Program()
    first_problem = pops.Problem(name="first-problem").block("u", physics=model)
    second_problem = pops.Problem(name="second-problem").block("u", physics=model)
    first_snapshot = first_problem.freeze()
    second_snapshot = second_problem.freeze()

    assert first_snapshot.hash != second_snapshot.hash
    first = drivers.compile_problem(
        time=program, model=model, problem_snapshot=first_snapshot)
    second = drivers.compile_problem(
        time=program, model=model, problem_snapshot=second_snapshot)

    assert isinstance(first.semantic_identity, Identity)
    assert isinstance(second.semantic_identity, Identity)
    assert first.problem_hash == first.semantic_identity.hexdigest
    assert second.problem_hash == second.semantic_identity.hexdigest
    assert first.so_path != second.so_path
    assert first.cache_key != second.cache_key
    assert first.authoring_snapshot is first_snapshot
    assert second.authoring_snapshot is second_snapshot
    assert read_artifact_sidecar(first.so_path)[
        "artifact_spec_identity"] == first.artifact_spec_identity.token
    assert read_artifact_sidecar(second.so_path)[
        "artifact_spec_identity"] == second.artifact_spec_identity.token
    assert compiled_paths == [first.so_path, second.so_path]

    hit = drivers.compile_problem(time=program, model=model, problem_snapshot=first_snapshot)
    assert hit.so_path == first.so_path and hit.cache_key == first.cache_key
    assert compiled_paths == [first.so_path, second.so_path]


def test_advanced_compile_problem_without_semantic_authority_is_rejected(monkeypatch, tmp_path):
    drivers, _ = _install_fake_toolchain(monkeypatch, tmp_path)
    program = _Program()
    with pytest.raises(TypeError, match="semantic program identity requires"):
        drivers.compile_problem(time=program, model=Module("same-model"))


def test_different_semantic_snapshots_never_alias_one_artifact_identity(
        monkeypatch, tmp_path):
    from pops.problem._snapshot import AuthoringSnapshot

    class RuntimeSetting:
        def __init__(self, default):
            self.default = default

        def to_data(self):
            return {
                "kind": "runtime",
                "dtype": "Real",
                "storage": "runtime_slot",
                "default": self.default,
            }

        def artifact_data(self):
            return {
                "kind": "runtime",
                "dtype": "Real",
                "storage": "runtime_slot",
            }

    drivers, compiled_paths = _install_fake_toolchain(monkeypatch, tmp_path)
    first_snapshot = AuthoringSnapshot({"parameter": RuntimeSetting(1.0)})
    second_snapshot = AuthoringSnapshot({"parameter": RuntimeSetting(2.0)})
    assert first_snapshot.hash != second_snapshot.hash
    assert first_snapshot.artifact_hash == second_snapshot.artifact_hash

    model = Module("same-model")
    program = _Program()
    first = drivers.compile_problem(
        time=program, model=model, problem_snapshot=first_snapshot)
    second = drivers.compile_problem(
        time=program, model=model, problem_snapshot=second_snapshot)

    assert first.problem_hash != second.problem_hash
    assert first.cache_key != second.cache_key
    assert first.so_path != second.so_path
    assert first.authoring_snapshot is first_snapshot
    assert second.authoring_snapshot is second_snapshot
    assert compiled_paths == [first.so_path, second.so_path]


def test_compile_authenticates_full_snapshot_before_using_artifact_hash(monkeypatch, tmp_path):
    from pops.problem._snapshot import AuthoringSnapshot

    drivers, compiled_paths = _install_fake_toolchain(monkeypatch, tmp_path)
    snapshot = AuthoringSnapshot({"problem": "strict-full-snapshot"})
    object.__setattr__(snapshot, "_hash", "a" * 64)

    with pytest.raises(ValueError, match="canonical payload"):
        drivers.compile_problem(
            time=_Program(), model=Module("same-model"), problem_snapshot=snapshot)
    assert compiled_paths == []
