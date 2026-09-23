"""Public Model lowering must publish one sealed cache entry across processes.

The compiler stand-in writes byte fixtures, not executable simulation results. Authoring,
artifact identities, the facade, inter-process locks and sidecar authentication are real.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest


def _model(cache, monkeypatch, compiler):
    from pops.codegen import abi, cache as cache_module, compile_link_flags, toolchain
    from pops.physics._facade import Model

    monkeypatch.setenv("POPS_CACHE_DIR", str(cache))
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(toolchain, "loader_cxx_std", lambda: "c++23")
    monkeypatch.setattr(toolchain, "_native_kokkos_compiler", lambda cxx: "test-c++")
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "test-features")
    monkeypatch.setattr(abi, "_abi_key_python", lambda *args: "test-abi")
    monkeypatch.setattr(cache_module, "_precision_cache_key", lambda: "double")
    monkeypatch.setattr(compile_link_flags, "deterministic_component_link_flags", lambda _: [])
    model = Model("facade-cache-publication")
    (rho,) = model.conservative_vars("rho")
    model.flux(x=[rho], y=[rho])
    model.eigenvalues(x=[0.0 * rho + 1.0], y=[0.0 * rho + 1.0])
    model.primitive_vars(rho)
    model.conservative_from([rho])
    monkeypatch.setattr(model._m, "compile", compiler)
    return model


def _concurrent_compile(cache, target, started, compiling, release, finished, results):
    with pytest.MonkeyPatch.context() as monkeypatch:
        def compiler(path, *args, **kwargs):
            del args, kwargs
            Path(path).write_bytes(b"partial native compiler output")
            compiling.set()
            if not release.wait(10):
                raise RuntimeError("test publisher was not released")
            Path(path).write_bytes(b"complete native compiler output")
            return path

        model = _model(cache, monkeypatch, compiler)
        started.set()
        try:
            artifact = model.compile(include="test-headers", target=target)
            results.put(("ok", artifact.so_path, artifact.artifact_identity.token))
        except Exception as exc:
            results.put(("error", type(exc).__name__, str(exc)))
        finally:
            finished.set()


@pytest.mark.parametrize("target", ("system", "amr_system"))
def test_concurrent_facade_cache_callers_never_observe_partial_publication(tmp_path, target):
    context = multiprocessing.get_context("spawn")
    release, results = context.Event(), context.Queue()
    started = [context.Event(), context.Event()]
    compiling = [context.Event(), context.Event()]
    finished = [context.Event(), context.Event()]
    workers = [context.Process(
        target=_concurrent_compile,
        args=(str(tmp_path), target, started[i], compiling[i], release, finished[i], results),
    ) for i in range(2)]
    try:
        workers[0].start()
        assert compiling[0].wait(10)
        workers[1].start()
        assert started[1].wait(10)
        peer_finished_before_publication = finished[1].wait(0.3)
        visible_before_publication = tuple(
            path for path in tmp_path.glob("*.so") if not path.name.startswith(".")
        )
        release.set()
        for worker in workers:
            worker.join(10)
            assert worker.exitcode == 0
        observations = [results.get(timeout=2) for _ in workers]
        assert all(row[0] == "ok" for row in observations), observations
        assert not peer_finished_before_publication
        assert not visible_before_publication
        assert not compiling[1].is_set(), "the waiting caller must reuse the sealed artifact"
        assert observations[0] == observations[1]
        assert len(tuple(tmp_path.glob("*.so"))) == 1
        assert len(tuple(tmp_path.glob("*.pops-cache.lock"))) == 1
        assert not tuple(tmp_path.glob(".*.pops-stage-*"))
    finally:
        release.set()
        for worker in workers:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(5)
        results.close()


def test_failed_facade_compile_does_not_publish_partial_output(tmp_path, monkeypatch):
    def compiler(path, *args, **kwargs):
        del args, kwargs
        Path(path).write_bytes(b"incomplete compiler output")
        raise RuntimeError("compiler interrupted")

    model = _model(tmp_path, monkeypatch, compiler)
    with pytest.raises(RuntimeError, match="compiler interrupted"):
        model.compile(include="test-headers")
    assert not tuple(tmp_path.glob("*.so"))
    assert not tuple(tmp_path.glob("*.pops-artifact.json"))
    assert not tuple(tmp_path.glob(".*.pops-stage-*"))


def test_facade_cache_refuses_changed_bytes_without_recompiling(tmp_path, monkeypatch):
    from pops.codegen.compile_provenance import StaleArtifactError

    calls = []

    def compiler(path, *args, **kwargs):
        del args, kwargs
        calls.append(path)
        Path(path).write_bytes(b"complete compiler output")
        return path

    model = _model(tmp_path, monkeypatch, compiler)
    artifact = model.compile(include="test-headers")
    Path(artifact.so_path).write_bytes(b"changed after authentication")
    with pytest.raises(StaleArtifactError):
        model.compile(include="test-headers")
    assert len(calls) == 1


def test_explicit_facade_destination_still_forces_compilation(tmp_path, monkeypatch):
    calls = []

    def compiler(path, *args, **kwargs):
        del args, kwargs
        calls.append(path)
        Path(path).write_bytes(b"complete compiler output")
        return path

    model = _model(tmp_path, monkeypatch, compiler)
    destination = str(tmp_path / "explicit-model.so")
    for _ in range(2):
        assert model.compile(destination, include="test-headers").so_path == destination
    assert calls == [destination, destination]
    assert not tuple(tmp_path.glob("*.pops-cache.lock"))
