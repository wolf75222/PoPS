"""Python control-flow checks for exact MPI test artifact publication; no native execution."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.python.integration.mpi import _compile_once as helper


class _FileArtifact:
    def __init__(self, cache: Path):
        self.cache = cache
        self.artifact_identity = SimpleNamespace(hexdigest=self._digest())

    def _digest(self):
        digest = hashlib.sha256()
        for name in ("model.so", "program.so", "sidecar.json"):
            digest.update((self.cache / name).read_bytes())
        return digest.hexdigest()

    def verify(self):
        if self._digest() != self.artifact_identity.hexdigest:
            raise ValueError("artifact bytes or sidecar changed")


def _published_files(path, *, model=b"one exact model"):
    path.mkdir()
    (path / "model.so").write_bytes(model)
    (path / "program.so").write_bytes(b"one exact Program")
    (path / "sidecar.json").write_text('{"native_abi": "qualified-test-abi"}')
    return _FileArtifact(path)


def _world(monkeypatch, *, rank, publication=None):
    comm = SimpleNamespace(rank=rank, size=2)
    gathers = []
    broadcasts = []
    monkeypatch.setattr(helper, "require_world", lambda value: comm)

    def gather(_comm, value):
        gathers.append(value)
        return (value, value) if len(gathers) == 1 else ("", value)

    def broadcast(_comm, value, *, root):
        assert root == 0
        broadcasts.append(value)
        return value if rank == 0 else publication

    monkeypatch.setattr(helper, "allgather_value", gather)
    monkeypatch.setattr(helper, "broadcast_value", broadcast)
    return comm, gathers, broadcasts


_PLAN = SimpleNamespace(plan_identity=SimpleNamespace(hexdigest="same-resolved-plan"))


def test_root_publishes_its_isolated_cache_and_verified_complete_artifact(tmp_path, monkeypatch):
    artifact = _published_files(tmp_path / "rank0-test-cache")
    monkeypatch.setenv("POPS_CACHE_DIR", str(artifact.cache))
    comm, gathers, broadcasts = _world(monkeypatch, rank=0)
    calls = []

    def compile_artifact(plan):
        assert plan is _PLAN
        calls.append(os.environ["POPS_CACHE_DIR"])
        return artifact

    result = helper.compile_resolved_plan_once(
        comm, _PLAN, route="full-original-case", compile_artifact=compile_artifact)
    assert result is artifact
    assert calls == [str(artifact.cache)]
    assert broadcasts == [(True, str(artifact.cache.resolve()), artifact.artifact_identity.hexdigest)]
    assert gathers == ["same-resolved-plan", ""]
    assert os.environ["POPS_CACHE_DIR"] == str(artifact.cache)


@pytest.mark.parametrize("previous", (None, "rank1-independent-test-cache"))
def test_peer_loads_publisher_resource_and_restores_its_fixture_cache(tmp_path, monkeypatch, previous):
    artifact = _published_files(tmp_path / "rank0-test-cache")
    if previous is None:
        monkeypatch.delenv("POPS_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("POPS_CACHE_DIR", previous)
    comm, gathers, _ = _world(monkeypatch, rank=1, publication=(
        True, str(artifact.cache), artifact.artifact_identity.hexdigest))
    seen = []

    def load(plan):
        assert plan is _PLAN
        seen.append(Path(os.environ["POPS_CACHE_DIR"]))
        return _FileArtifact(seen[-1])

    result = helper.compile_resolved_plan_once(comm, _PLAN, route="full-case", compile_artifact=load)
    assert result.artifact_identity.hexdigest == artifact.artifact_identity.hexdigest
    assert seen == [artifact.cache]
    assert gathers == ["same-resolved-plan", ""]
    assert os.environ.get("POPS_CACHE_DIR") == previous


@pytest.mark.parametrize("failure", ("load", "different-model", "tampered-sidecar", "missing-cache"))
def test_peer_failures_are_collective_and_restore_fixture_environment(tmp_path, monkeypatch, failure):
    artifact = _published_files(tmp_path / "rank0-test-cache")
    other = _published_files(tmp_path / "different-cache", model=b"different compiled model")
    previous = str(tmp_path / "rank1-isolated-cache")
    monkeypatch.setenv("POPS_CACHE_DIR", previous)
    cache = tmp_path / "absent-publication" if failure == "missing-cache" else artifact.cache
    comm, gathers, _ = _world(monkeypatch, rank=1, publication=(
        True, str(cache), artifact.artifact_identity.hexdigest))
    calls = []

    def load(_plan):
        calls.append(os.environ["POPS_CACHE_DIR"])
        if failure == "load":
            raise ValueError("peer cache authentication failed")
        if failure == "different-model":
            return other
        if failure == "tampered-sidecar":
            (artifact.cache / "sidecar.json").write_text('{"native_abi":"changed"}')
        return artifact

    with pytest.raises(RuntimeError, match="authenticated artifact cache load failed: rank 1"):
        helper.compile_resolved_plan_once(comm, _PLAN, route="full-case", compile_artifact=load)
    assert len(gathers) == 2 and gathers[1]
    if failure == "different-model":
        assert "differs from rank zero's exact publication" in gathers[1]
    if failure == "tampered-sidecar":
        assert "artifact bytes or sidecar changed" in gathers[1]
    assert calls == ([] if failure == "missing-cache" else [str(artifact.cache)])
    assert os.environ["POPS_CACHE_DIR"] == previous


def test_root_verification_failure_is_published_before_peer_load(tmp_path, monkeypatch):
    artifact = _published_files(tmp_path / "rank0-test-cache")
    monkeypatch.setenv("POPS_CACHE_DIR", str(artifact.cache))
    comm, gathers, broadcasts = _world(monkeypatch, rank=0)

    def compile_artifact(_plan):
        (artifact.cache / "model.so").write_bytes(b"changed after preparation")
        return artifact

    with pytest.raises(RuntimeError, match="rank 0 artifact publication failed"):
        helper.compile_resolved_plan_once(
            comm, _PLAN, route="full-case", compile_artifact=compile_artifact)
    assert broadcasts[0][0] is False
    assert "artifact bytes or sidecar changed" in broadcasts[0][1]
    assert gathers == ["same-resolved-plan"]
