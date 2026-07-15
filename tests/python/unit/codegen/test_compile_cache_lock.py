"""The content-addressed native cache has one publication authority across processes."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest

from pops.codegen.cache import _artifact_cache_lock, _artifact_cache_staging_path


def _hold_cache_lock(path, attempting, entered, release):
    attempting.set()
    with _artifact_cache_lock(path):
        entered.set()
        release.wait(10)


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
        drivers.compile_problem(model=module, time=program, include=INCLUDE)

    assert not tuple(tmp_path.glob("*.so"))
    assert not tuple(tmp_path.glob(".*.pops-stage-*.so"))
    assert not tuple(tmp_path.glob("*.pops-artifact.json"))
    assert tuple(tmp_path.glob("*.failed.cpp"))
