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


def _cached_plan(**options):
    return SimpleNamespace(plan_identity=_PLAN.plan_identity, compile_options=options)


def _cache_marker(cache):
    marker = cache / ".unit-publication"
    marker.write_text("unique-publisher-token", encoding="ascii")
    return (True, str(cache), marker.name, "unique-publisher-token")


def _cached_world(monkeypatch, *, rank, publication=None, identity=None,
                  gather_hook=None, broadcast_hook=None):
    comm = SimpleNamespace(rank=rank, size=2)
    events = []
    counts = {"gather": 0, "broadcast": 0}
    monkeypatch.setattr(helper, "require_world", lambda value: comm)
    monkeypatch.delenv("POPS_CODEGEN_DIR", raising=False)

    def gather(_comm, value):
        counts["gather"] += 1
        index = counts["gather"]
        events.append(("gather", index, value))
        if gather_hook is not None:
            replacement = gather_hook(index, value)
            if replacement is not None:
                return replacement
        if index <= 2:
            return (value, value)
        return tuple(value if peer == rank else "" for peer in range(2))

    def broadcast(_comm, value, *, root):
        assert root == 0
        counts["broadcast"] += 1
        index = counts["broadcast"]
        events.append(("broadcast", index, value))
        if broadcast_hook is not None:
            broadcast_hook(index, value)
        return value if rank == 0 else (publication if index == 1 else identity)

    monkeypatch.setattr(helper, "allgather_value", gather)
    monkeypatch.setattr(helper, "broadcast_value", broadcast)
    return comm, events


@pytest.mark.parametrize("options", ({}, {"force": False, "so_path": None},
                                    {"model_source_policy": "require"}))
def test_cached_root_publishes_visible_resource_before_compiling_and_identity_after_errors(
        tmp_path, monkeypatch, options):
    import pops

    artifact = _published_files(tmp_path / "rank0-cache")
    monkeypatch.setenv("POPS_CACHE_DIR", str(artifact.cache))
    comm, events = _cached_world(monkeypatch, rank=0)
    plan = _cached_plan(**options)
    verifies = []
    original_verify = artifact.verify

    def verify():
        verifies.append(os.environ["POPS_CACHE_DIR"])
        original_verify()

    monkeypatch.setattr(artifact, "verify", verify)

    def compile_artifact(value):
        assert value is plan
        assert [(kind, index) for kind, index, _ in events] == [
            ("gather", 1), ("gather", 2), ("broadcast", 1), ("gather", 3)]
        publication = events[2][2]
        assert publication[0] is True
        assert (Path(publication[1]) / publication[2]).read_text() == publication[3]
        assert Path(os.environ["POPS_CACHE_DIR"]) == artifact.cache
        return artifact

    monkeypatch.setattr(pops, "compile", compile_artifact)
    result = helper.compile_resolved_plan_cached(
        comm, plan, route="full-case", compile_artifact=compile_artifact)
    assert result is artifact
    assert [(kind, index) for kind, index, _ in events] == [
        ("gather", 1), ("gather", 2), ("broadcast", 1), ("gather", 3),
        ("gather", 4), ("broadcast", 2), ("gather", 5)]
    assert events[5][2] == artifact.artifact_identity.hexdigest
    assert verifies == [str(artifact.cache)] * 2
    assert os.environ["POPS_CACHE_DIR"] == str(artifact.cache)


@pytest.mark.parametrize("previous", (None, "independent-peer-cache"))
def test_cached_peer_uses_published_resource_and_restores_environment(tmp_path, monkeypatch, previous):
    import pops

    artifact = _published_files(tmp_path / "publisher")
    publication = _cache_marker(artifact.cache)
    if previous is None:
        monkeypatch.delenv("POPS_CACHE_DIR", raising=False)
    else:
        monkeypatch.setenv("POPS_CACHE_DIR", previous)
    comm, events = _cached_world(monkeypatch, rank=1, publication=publication,
                                identity=artifact.artifact_identity.hexdigest)
    seen = []

    def compile_artifact(_plan):
        seen.append(os.environ["POPS_CACHE_DIR"])
        assert len(events) == 4  # no wait for a root artifact identity
        return artifact

    monkeypatch.setattr(pops, "compile", compile_artifact)
    assert helper.compile_resolved_plan_cached(
        comm, _cached_plan(), route="full-case", compile_artifact=compile_artifact) is artifact
    assert seen == [str(artifact.cache)]
    assert os.environ.get("POPS_CACHE_DIR") == previous


@pytest.mark.parametrize("reason", ("callback", "codegen-dir", "so-path", "force", "recompile", "peer"))
def test_cached_ineligible_rank_selects_historical_protocol_collectively(monkeypatch, reason):
    import pops

    options = {"so-path": {"so_path": "explicit.so"}, "force": {"force": True},
               "recompile": {"model_source_policy": "recompile"}}.get(reason, {})
    plan = _cached_plan(**options)
    comm, events = _cached_world(
        monkeypatch, rank=0,
        gather_hook=lambda index, value: ((value, ("", False))
                                         if index == 1 and reason == "peer" else None))
    if reason == "codegen-dir":
        monkeypatch.setenv("POPS_CODEGEN_DIR", "explicit-generated-directory")

    def compile_artifact(_plan):
        raise AssertionError("fallback must own the callback schedule")

    monkeypatch.setattr(pops, "compile", compile_artifact if reason != "callback" else lambda _: None)
    sentinel = object()
    calls = []

    def historical(world, value, *, route, compile_artifact):
        calls.append((world, value, route, compile_artifact))
        return sentinel

    monkeypatch.setattr(helper, "compile_resolved_plan_once", historical)
    assert helper.compile_resolved_plan_cached(
        comm, plan, route="unchanged-route", compile_artifact=compile_artifact) is sentinel
    assert calls == [(comm, plan, "unchanged-route", compile_artifact)]
    assert len(events) == 1 and events[0][:2] == ("gather", 1)


@pytest.mark.parametrize("failure", ("mismatch", "local-getter", "peer-getter", "empty"))
def test_cached_plan_identity_failures_converge_before_resource_publication(monkeypatch, failure):
    import pops

    class BrokenPlan:
        compile_options = {}

        @property
        def plan_identity(self):
            raise ValueError("identity unavailable")

    plan = BrokenPlan() if failure == "local-getter" else _cached_plan()
    if failure == "empty":
        plan.plan_identity = SimpleNamespace(hexdigest="")

    def gather_hook(index, value):
        if index == 2 and failure == "mismatch":
            return (value, ("", "different-resolved-plan"))
        if index == 2 and failure == "peer-getter":
            return (value, ("ValueError: peer identity unavailable", ""))
        return None

    comm, events = _cached_world(monkeypatch, rank=0, gather_hook=gather_hook)

    def compile_artifact(_plan):
        raise AssertionError("plan failure must precede compilation")

    monkeypatch.setattr(pops, "compile", compile_artifact)
    with pytest.raises(RuntimeError, match="resolved .*plan"):
        helper.compile_resolved_plan_cached(comm, plan, route="full-case", compile_artifact=compile_artifact)
    assert [kind for kind, _, _ in events] == ["gather", "gather"]


@pytest.mark.parametrize("failure", ("root-setup", "nonshared", "changed-marker"))
def test_cached_resource_failure_is_collective_before_any_compilation(tmp_path, monkeypatch, failure):
    import pops
    from pops.codegen import cache

    artifact = _published_files(tmp_path / "publisher")
    publication = _cache_marker(artifact.cache)
    if failure == "nonshared":
        publication = (True, str(tmp_path / "absent"), publication[2], publication[3])
    elif failure == "changed-marker":
        (artifact.cache / publication[2]).write_text("not-the-publisher-token")
    else:
        def fail_cache():
            raise OSError("cache directory creation failed")
        monkeypatch.setattr(cache, "pops_cache_dir", fail_cache)
    comm, events = _cached_world(monkeypatch, rank=0 if failure == "root-setup" else 1,
                                publication=publication)
    previous = "rank-specific-cache"
    monkeypatch.setenv("POPS_CACHE_DIR", previous)

    def compile_artifact(_plan):
        raise AssertionError("invisible cache must not be compiled")

    monkeypatch.setattr(pops, "compile", compile_artifact)
    with pytest.raises(RuntimeError, match="shared MPI cache publication failed"):
        helper.compile_resolved_plan_cached(
            comm, _cached_plan(), route="full-case", compile_artifact=compile_artifact)
    assert [(kind, index) for kind, index, _ in events] == [
        ("gather", 1), ("gather", 2), ("broadcast", 1), ("gather", 3)]
    assert os.environ["POPS_CACHE_DIR"] == previous


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("failure", ("compile", "none", "verify", "identity-getter"))
def test_cached_local_failure_converges_before_identity_broadcast(tmp_path, monkeypatch, rank, failure):
    import pops

    artifact = _published_files(tmp_path / "publisher")
    publication = _cache_marker(artifact.cache)
    comm, events = _cached_world(monkeypatch, rank=rank, publication=publication)
    previous = str(artifact.cache) if rank == 0 else "peer-private-cache"
    monkeypatch.setenv("POPS_CACHE_DIR", previous)

    class BrokenIdentity:
        def verify(self):
            pass

        @property
        def artifact_identity(self):
            raise ValueError("artifact identity unavailable")

    def compile_artifact(_plan):
        assert os.environ["POPS_CACHE_DIR"] == str(artifact.cache)
        if failure == "compile":
            raise ValueError("compiler failure")
        if failure == "none":
            return None
        if failure == "identity-getter":
            return BrokenIdentity()
        (artifact.cache / "sidecar.json").write_text("changed during compile")
        return artifact

    monkeypatch.setattr(pops, "compile", compile_artifact)
    with pytest.raises(RuntimeError, match="concurrent cached artifact preparation failed: rank %d" % rank):
        helper.compile_resolved_plan_cached(
            comm, _cached_plan(), route="full-case", compile_artifact=compile_artifact)
    assert [index for kind, index, _ in events if kind == "broadcast"] == [1]
    assert events[-1][:2] == ("gather", 4)
    assert os.environ["POPS_CACHE_DIR"] == previous


def test_cached_remote_compile_failure_prevents_root_identity_publication(tmp_path, monkeypatch):
    import pops

    artifact = _published_files(tmp_path / "publisher")
    monkeypatch.setenv("POPS_CACHE_DIR", str(artifact.cache))
    comm, events = _cached_world(
        monkeypatch, rank=0,
        gather_hook=lambda index, value: ((value, "ValueError: peer compile failed")
                                         if index == 4 else None))
    monkeypatch.setattr(pops, "compile", lambda _: artifact)
    with pytest.raises(RuntimeError, match="preparation failed: rank 1: ValueError: peer compile failed"):
        helper.compile_resolved_plan_cached(
            comm, _cached_plan(), route="full-case", compile_artifact=pops.compile)
    assert events[-1][:2] == ("gather", 4)
    assert [index for kind, index, _ in events if kind == "broadcast"] == [1]


@pytest.mark.parametrize("rank", (0, 1))
@pytest.mark.parametrize("failure", ("identity", "sidecar"))
def test_cached_final_verification_rejects_changed_identity_or_evidence(tmp_path, monkeypatch, rank, failure):
    import pops

    artifact = _published_files(tmp_path / "publisher")
    initial_identity = artifact.artifact_identity.hexdigest
    publication = _cache_marker(artifact.cache)
    previous = str(artifact.cache) if rank == 0 else "peer-private-cache"
    monkeypatch.setenv("POPS_CACHE_DIR", previous)

    def broadcast_hook(index, _value):
        if index == 2:
            if failure == "identity":
                # Self-consistent different bytes still must match rank zero's
                # already authenticated identity, not merely their own checksum.
                (artifact.cache / "model.so").write_bytes(b"different compiled model")
                artifact.artifact_identity = SimpleNamespace(hexdigest=artifact._digest())
            else:
                (artifact.cache / "sidecar.json").write_text("late tampering")

    comm, events = _cached_world(monkeypatch, rank=rank, publication=publication,
                                identity=initial_identity, broadcast_hook=broadcast_hook)
    monkeypatch.setattr(pops, "compile", lambda _: artifact)
    with pytest.raises(RuntimeError, match="authenticated concurrent artifact publication failed: rank %d" % rank):
        helper.compile_resolved_plan_cached(
            comm, _cached_plan(), route="full-case", compile_artifact=pops.compile)
    assert events[-1][:2] == ("gather", 5)
    diagnostic = events[-1][2]
    assert ("differs from rank zero's exact publication" if failure == "identity"
            else "artifact bytes or sidecar changed") in diagnostic
    assert os.environ["POPS_CACHE_DIR"] == previous


def test_public_profile_requests_cached_protocol_without_changing_serial_path(monkeypatch):
    from tests.python.support.public_native_profile import compile_profile_plan_once

    artifact = SimpleNamespace(verify=lambda: None)
    calls = []

    def cached(world, plan, *, route, compile_artifact):
        calls.append((world, plan, route, compile_artifact))
        return artifact

    monkeypatch.setattr(helper, "compile_resolved_plan_cached", cached)
    world = object()
    callback = lambda _: artifact
    assert compile_profile_plan_once(world, _PLAN, route="full-case", compile_artifact=callback) is artifact
    assert calls == [(world, _PLAN, "full-case", callback)]
    assert compile_profile_plan_once(None, _PLAN, route="full-case", compile_artifact=callback) is artifact
    assert len(calls) == 1
