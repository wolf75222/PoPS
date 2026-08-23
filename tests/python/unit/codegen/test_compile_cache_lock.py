"""The content-addressed native cache has one publication authority across processes."""

from __future__ import annotations

import multiprocessing
import os
from pathlib import Path

import pytest

from pops.codegen.cache import (
    _artifact_cache_lock,
    _artifact_cache_staging_path,
    compile_cached_artifact,
)
from pops.codegen.compile_provenance import (
    artifact_sidecar_path,
    read_artifact_sidecar,
    verify_cached_artifact,
    write_artifact_sidecar,
)
from pops.identity import Identity, make_identity


def _hold_cache_lock(path, attempting, entered, release):
    attempting.set()
    with _artifact_cache_lock(path):
        entered.set()
        release.wait(10)


def _publish_one_cached_artifact(
    destination,
    semantic_token,
    spec_token,
    compiler_count,
    first_compile_started,
    release_first_compile,
    attempting,
    second_compile_entered,
    result_queue,
):
    """Spawn-safe worker for the end-to-end cache-publication race contract."""
    semantic = Identity.from_token(semantic_token)
    spec = Identity.from_token(spec_token)
    attempting.set()

    def compile_staged(staging):
        with compiler_count.get_lock():
            compiler_count.value += 1
            ordinal = compiler_count.value
        if ordinal == 1:
            first_compile_started.set()
            release_first_compile.wait(10)
        else:
            second_compile_entered.set()
        Path(staging).write_bytes(b"sealed native artifact")
        return staging

    published, binary, artifact = compile_cached_artifact(
        destination,
        semantic_identity=semantic,
        spec_identity=spec,
        compile_staged=compile_staged,
    )
    result_queue.put((published, binary.token, artifact.token))


def test_artifact_cache_lock_serializes_independent_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    artifact = str(tmp_path / "content-addressed.so")
    first_attempting, first_entered, release_first = (
        context.Event(), context.Event(), context.Event()
    )
    second_attempting, second_entered, release_second = (
        context.Event(), context.Event(), context.Event()
    )
    first = context.Process(
        target=_hold_cache_lock,
        args=(artifact, first_attempting, first_entered, release_first),
    )
    second = context.Process(
        target=_hold_cache_lock,
        args=(artifact, second_attempting, second_entered, release_second),
    )
    try:
        first.start()
        assert first_attempting.wait(5)
        assert first_entered.wait(5)
        second.start()
        assert second_attempting.wait(5)
        assert not second_entered.wait(0.2)
        release_first.set()
        assert second_entered.wait(5)
        release_second.set()
        first.join(5)
        second.join(5)
        assert first.exitcode == 0
        assert second.exitcode == 0
        assert Path(artifact + ".pops-cache.lock").is_file()
    finally:
        release_first.set()
        release_second.set()
        for process in (first, second):
            if process.is_alive():
                process.terminate()
            process.join(5)


def test_cached_publication_serializes_compile_hash_and_sidecar_commit(tmp_path):
    """Two MPI-like processes publish one binary; the follower only authenticates it."""
    context = multiprocessing.get_context("spawn")
    destination = str(tmp_path / "content-addressed.so")
    semantic = make_identity("semantic", {"program": "same"})
    spec = make_identity("artifact-spec", {"target": "system"})
    compiler_count = context.Value("i", 0)
    first_compile_started = context.Event()
    release_first_compile = context.Event()
    first_attempting = context.Event()
    second_attempting = context.Event()
    second_compile_entered = context.Event()
    results = context.Queue()
    workers = [
        context.Process(
            target=_publish_one_cached_artifact,
            args=(
                destination,
                semantic.token,
                spec.token,
                compiler_count,
                first_compile_started,
                release_first_compile,
                first_attempting,
                second_compile_entered,
                results,
            ),
        ),
        context.Process(
            target=_publish_one_cached_artifact,
            args=(
                destination,
                semantic.token,
                spec.token,
                compiler_count,
                first_compile_started,
                release_first_compile,
                second_attempting,
                second_compile_entered,
                results,
            ),
        ),
    ]
    try:
        workers[0].start()
        assert first_attempting.wait(5)
        assert first_compile_started.wait(5)
        workers[1].start()
        assert second_attempting.wait(5)
        assert not second_compile_entered.wait(0.2), (
            "a follower must wait for the sealed cache entry, never compile over it"
        )
        release_first_compile.set()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0

        published = [results.get(timeout=2) for _ in workers]
        assert compiler_count.value == 1
        assert {row[0] for row in published} == {destination}
        assert len({row[1] for row in published}) == 1
        assert len({row[2] for row in published}) == 1
        verify_cached_artifact(destination, semantic_identity=semantic, spec_identity=spec)
        assert not tuple(tmp_path.glob(".*.pops-stage-*.so"))
    finally:
        release_first_compile.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(5)


def test_cached_publication_replaces_inner_staging_identity_with_outer_authority(tmp_path):
    destination = str(tmp_path / "facade-cache.so")
    inner_semantic = make_identity("semantic", {"authority": "inner-model"})
    inner_spec = make_identity("artifact-spec", {"authority": "inner-model"})
    outer_semantic = make_identity("semantic", {"authority": "facade"})
    outer_spec = make_identity("artifact-spec", {"authority": "facade"})

    def compile_staged(staging):
        Path(staging).write_bytes(b"facade native artifact")
        write_artifact_sidecar(
            staging,
            semantic_identity=inner_semantic,
            spec_identity=inner_spec,
        )
        return staging

    compile_cached_artifact(
        destination,
        semantic_identity=outer_semantic,
        spec_identity=outer_spec,
        compile_staged=compile_staged,
    )

    published = read_artifact_sidecar(destination)
    assert published["semantic_identity"] == outer_semantic.token
    assert published["artifact_spec_identity"] == outer_spec.token
    verify_cached_artifact(
        destination,
        semantic_identity=outer_semantic,
        spec_identity=outer_spec,
    )


def test_cached_publication_refuses_an_unreserved_compiler_output(tmp_path):
    destination = str(tmp_path / "content-addressed.so")
    semantic = make_identity("semantic", {"program": "same"})
    spec = make_identity("artifact-spec", {"target": "system"})

    def compile_elsewhere(staging):
        Path(staging).write_bytes(b"private output")
        wrong = tmp_path / "wrong.so"
        wrong.write_bytes(b"foreign output")
        return str(wrong)

    with pytest.raises(RuntimeError, match="reserved staging path"):
        compile_cached_artifact(
            destination,
            semantic_identity=semantic,
            spec_identity=spec,
            compile_staged=compile_elsewhere,
        )
    assert not Path(destination).exists()
    assert not tuple(tmp_path.glob(".*.pops-stage-*.so"))


def test_cached_publication_records_final_identity_for_later_explicit_siblings(
    tmp_path, monkeypatch
):
    from pops.codegen import _artifact_identity as artifact_identity
    from pops.codegen import _compile_drivers as drivers

    destination = str(tmp_path / "content-addressed.so")
    semantic = make_identity("semantic", {"program": "same"})
    cached_spec = make_identity("artifact-spec", {"target": "system", "revision": 1})
    other_spec = make_identity("artifact-spec", {"target": "system", "revision": 2})

    def compile_staged(staging):
        Path(staging).write_bytes(b"sealed native artifact")
        return staging

    compile_cached_artifact(
        destination,
        semantic_identity=semantic,
        spec_identity=cached_spec,
        compile_staged=compile_staged,
    )
    compile_cached_artifact(
        destination,
        semantic_identity=semantic,
        spec_identity=cached_spec,
        compile_staged=lambda staging: pytest.fail("verified cache hit must not recompile"),
    )

    class Model:
        _elliptic_fields = {}

        @staticmethod
        def _check_require_metadata(require_metadata, backend):
            del require_metadata, backend

    def compile_explicit(model, path, *args, **kwargs):
        del model, args, kwargs
        Path(path).write_bytes(b"explicit different artifact")
        return path

    monkeypatch.setattr(drivers, "_native_kokkos_compiler", lambda cxx: cxx)
    monkeypatch.setattr(drivers, "_abi_key_python", lambda include, cxx, std: "test-abi")
    monkeypatch.setattr(
        artifact_identity, "model_artifact_spec", lambda model, **kwargs: (semantic, other_spec)
    )
    monkeypatch.setattr(drivers, "compile_native", compile_explicit)
    sibling = drivers.compile_model(
        Model(),
        so_path=destination,
        include="headers",
        cxx="c++",
        std="c++23",
    )

    assert sibling != destination
    assert Path(sibling).read_bytes() == b"explicit different artifact"
    assert Path(destination).read_bytes() == b"sealed native artifact"


def test_cached_publication_refuses_an_orphaned_identity_sidecar(tmp_path):
    destination = str(tmp_path / "content-addressed.so")
    semantic = make_identity("semantic", {"program": "same"})
    spec = make_identity("artifact-spec", {"target": "system"})
    Path(artifact_sidecar_path(destination)).write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="identity sidecar.*orphaned"):
        compile_cached_artifact(
            destination,
            semantic_identity=semantic,
            spec_identity=spec,
            compile_staged=lambda staging: pytest.fail("must not compile over orphan provenance"),
        )


@pytest.mark.parametrize("kind", ("directory", "symlink"))
def test_cached_publication_refuses_nonregular_or_symbolic_artifact_entries(tmp_path, kind):
    destination = tmp_path / "content-addressed.so"
    semantic = make_identity("semantic", {"program": "same"})
    spec = make_identity("artifact-spec", {"target": "system"})
    Path(artifact_sidecar_path(str(destination))).write_text("{}\n", encoding="utf-8")
    if kind == "directory":
        destination.mkdir()
    else:
        if os.name == "nt":
            pytest.skip("Windows symlink creation requires external privilege")
        destination.symlink_to(tmp_path / "missing-target.so")

    with pytest.raises(RuntimeError, match="regular non-symbolic-link"):
        compile_cached_artifact(
            str(destination),
            semantic_identity=semantic,
            spec_identity=spec,
            compile_staged=lambda staging: pytest.fail("must not compile over residual entry"),
        )


def test_implicit_model_cache_refuses_orphaned_sidecar_before_compilation(tmp_path, monkeypatch):
    from pops.codegen import _artifact_identity as artifact_identity
    from pops.codegen import _compile_drivers as drivers

    destination = str(tmp_path / "model-cache.so")
    semantic = make_identity("semantic", {"model": "same"})
    spec = make_identity("artifact-spec", {"model": "same"})
    Path(artifact_sidecar_path(destination)).write_text("{}\n", encoding="utf-8")

    class Model:
        _elliptic_fields = {}

        @staticmethod
        def _check_require_metadata(require_metadata, backend):
            del require_metadata, backend

    monkeypatch.setattr(drivers, "_native_kokkos_compiler", lambda cxx: cxx)
    monkeypatch.setattr(drivers, "_abi_key_python", lambda include, cxx, std: "test-abi")
    monkeypatch.setattr(drivers, "_identity_cache_so_path", lambda identity: destination)
    monkeypatch.setattr(
        artifact_identity, "model_artifact_spec", lambda model, **kwargs: (semantic, spec)
    )
    monkeypatch.setattr(
        drivers, "compile_native", lambda *args, **kwargs: pytest.fail("must not compile")
    )

    with pytest.raises(RuntimeError, match="identity sidecar.*orphaned"):
        drivers.compile_model(Model(), include="headers", cxx="c++", std="c++23")


def test_implicit_program_cache_force_refuses_orphaned_sidecar_before_compilation(
    tmp_path, monkeypatch
):
    from pops.codegen import _artifact_identity as artifact_identity
    from pops.codegen import _compile_drivers as drivers
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    destination = str(tmp_path / "program-cache.so")
    semantic = make_identity("semantic", {"program": "same"})
    spec = make_identity("artifact-spec", {"program": "same"})
    Path(artifact_sidecar_path(destination)).write_text("{}\n", encoding="utf-8")
    program, module = _program_fixture("cache-orphan")
    monkeypatch.delenv("POPS_CODEGEN_DIR", raising=False)
    monkeypatch.setattr(drivers, "_identity_cache_so_path", lambda identity: destination)
    monkeypatch.setattr(
        artifact_identity, "program_artifact_spec", lambda **kwargs: (semantic, spec)
    )
    monkeypatch.setattr(
        drivers, "_run_compile", lambda *args, **kwargs: pytest.fail("must not compile")
    )

    with pytest.raises(RuntimeError, match="identity sidecar.*orphaned"):
        drivers.compile_problem(
            model=module,
            time=program,
            force=True,
            include=INCLUDE,
            native_dimension=3,
        )


def test_implicit_program_cache_force_authenticates_before_recompilation(tmp_path, monkeypatch):
    from pops.codegen import _artifact_identity as artifact_identity
    from pops.codegen import _compile_drivers as drivers
    from pops.codegen.compile_provenance import write_artifact_sidecar
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    destination = str(tmp_path / "program-cache.so")
    semantic = make_identity("semantic", {"program": "same"})
    spec = make_identity("artifact-spec", {"program": "same"})
    Path(destination).write_bytes(b"authenticated cached binary")
    write_artifact_sidecar(destination, semantic_identity=semantic, spec_identity=spec)
    program, module = _program_fixture("cache-force")
    monkeypatch.delenv("POPS_CODEGEN_DIR", raising=False)
    monkeypatch.setattr(drivers, "_identity_cache_so_path", lambda identity: destination)
    monkeypatch.setattr(
        artifact_identity, "program_artifact_spec", lambda **kwargs: (semantic, spec)
    )
    monkeypatch.setattr(
        drivers, "_run_compile", lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("force reached compiler after strict cache authentication")
        )
    )

    with pytest.raises(RuntimeError, match="force reached compiler after strict cache authentication"):
        drivers.compile_problem(
            model=module,
            time=program,
            force=True,
            include=INCLUDE,
            native_dimension=3,
        )


def test_cache_staging_path_is_uniquely_reserved_in_the_destination_directory(tmp_path):
    destination = tmp_path / "content-addressed.so"
    first = Path(_artifact_cache_staging_path(destination))
    second = Path(_artifact_cache_staging_path(destination))
    assert first.parent == destination.parent
    assert second.parent == destination.parent
    assert first != second
    assert first.suffix == destination.suffix
    assert first.is_file()
    assert second.is_file()


def test_amr_cache_miss_projects_field_role_coefficients_only_for_identity(tmp_path, monkeypatch):
    from pops.codegen import _artifact_identity as artifact_identity
    from pops.codegen import _compile_drivers as drivers
    from pops.identity import canonical_bytes

    destination = str(tmp_path / "amr-field-roles.so")
    roles = (
        {
            "kind": "rhs",
            "field": "tests.electrostatic.slot",
            "block": "material",
            "binding_ordinal": 0,
            "binding_identity": "tests.electrostatic.binding.0",
            "provider_key": "electrostatic",
            "coefficient": 0.1,
        },
    )
    observed = {}

    class Model:
        _elliptic_fields = {}

        @staticmethod
        def _check_require_metadata(require_metadata, backend):
            del require_metadata, backend

    monkeypatch.setattr(drivers, "_native_kokkos_compiler", lambda cxx: cxx)
    monkeypatch.setattr(drivers, "_abi_key_python", lambda include, cxx, std: "test-abi")
    monkeypatch.setattr(drivers, "_identity_cache_so_path", lambda spec: destination)
    monkeypatch.setattr(drivers, "_record_artifact_identity", lambda *args: None)

    def fake_artifact_spec(model, **kwargs):
        del model
        observed["identity_name"] = kwargs["name"]
        return object(), object()

    def fake_compile_native(model, path, *args, native_field_roles, **kwargs):
        del model, args, kwargs
        observed["runtime_roles"] = native_field_roles
        return path

    monkeypatch.setattr(artifact_identity, "model_artifact_spec", fake_artifact_spec)
    monkeypatch.setattr(drivers, "compile_native", fake_compile_native)
    monkeypatch.setattr(drivers, "publish_staged_artifact", lambda *args, **kwargs: None)

    assert drivers.compile_model(
        Model(),
        include="test-headers",
        cxx="test-c++",
        std="c++23",
        target="amr_system",
        name="field-package",
        _native_field_roles=roles,
    ) == destination

    identity_roles = ({**roles[0], "coefficient": (0.1).hex()},)
    assert observed["identity_name"] == "field-package#amr-field-roles:%s" % canonical_bytes(
        identity_roles
    ).hex()
    assert observed["runtime_roles"][0]["coefficient"] == 0.1
    assert type(observed["runtime_roles"][0]["coefficient"]) is float


def test_failed_program_compile_leaves_no_partial_final_or_staging_binary(
    tmp_path, monkeypatch
):
    from pops.codegen import _compile_drivers as drivers
    from tests.python.unit.runtime.test_pops_env import INCLUDE, _program_fixture

    monkeypatch.setenv("POPS_CODEGEN_DIR", str(tmp_path))
    monkeypatch.setattr(drivers, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(drivers, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(drivers, "_probe_cxx_std", lambda cc, std: std or "c++23")

    def failed_compile(command, _where):
        output = command[command.index("-o") + 1]
        Path(output).write_bytes(b"partial compiler output")
        raise RuntimeError("simulated compiler crash")

    monkeypatch.setattr(drivers, "_run_compile", failed_compile)
    program, module = _program_fixture("crashed-publication")
    with pytest.raises(RuntimeError, match="simulated compiler crash"):
        drivers.compile_problem(
            model=module,
            time=program,
            force=True,
            include=INCLUDE,
            native_dimension=3,
        )

    assert not tuple(tmp_path.glob("*.so"))
    assert not tuple(tmp_path.glob(".*.pops-stage-*.so"))
    assert not tuple(tmp_path.glob("*.pops-artifact.json"))
    assert tuple(tmp_path.glob("*.failed.cpp"))
