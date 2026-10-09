"""Real fixture/helper source protocol with substituted communication, never native/JIT."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from pops import _native_collectives
from tests.python.integration.mpi import _compile_once
from tests.python.integration.runtime import test_m19_product_support_runtime as fixture


def test_true_published_source_load_preserves_plan_and_does_not_rewrite(tmp_path):
    plan, _ = fixture.resolve_product_chain(tmp_path, 4, 3, 3)
    files = {p: (p.stat().st_mtime_ns, sha256(p.read_bytes()).hexdigest()) for p in tmp_path.rglob("*") if p.is_file()}
    peer, _ = fixture.resolve_product_chain(tmp_path, 4, 3, 3, provider_factory=fixture._load_published_provider)
    assert plan.plan_identity == peer.plan_identity
    assert files == {p: (p.stat().st_mtime_ns, sha256(p.read_bytes()).hexdigest()) for p in files}
    # Resealing a component elsewhere cannot bypass the expected common source body.
    source = next(tmp_path.rglob("physical_map.cpp"))
    source.write_bytes(source.read_bytes() + b"// changed body\n")
    with pytest.raises((ValueError, RuntimeError), match="(?:digest|source|hash|payload|sha256)"):
        fixture.resolve_product_chain(tmp_path, 4, 3, 3, provider_factory=fixture._load_published_provider)


class Artifact:
    artifact_identity = SimpleNamespace(hexdigest="same-authenticated-artifact")

    def verify(self):
        return None


def test_actual_prepare_uses_publisher_paths_and_real_compile_once(tmp_path, monkeypatch):
    root_dir, peer_dir = tmp_path / "rank0", tmp_path / "rank1"
    root_cache, peer_cache = tmp_path / "root-cache", tmp_path / "peer-cache"
    root_cache.mkdir()
    peer_cache.mkdir()
    resolved = SimpleNamespace(plan_identity=SimpleNamespace(hexdigest="exact-plan"))
    state = {"rank": 0, "path_index": 0, "published": None}
    world = SimpleNamespace(rank=0, size=2)
    seen = []

    def broadcast(_world, value, *, root):
        assert root == 0
        if isinstance(value, tuple):
            if state["rank"] == 0:
                state["published"] = value
            return state["published"]
        paths = (root_dir, root_dir / "m19-product-receipts")
        result = str(paths[state["path_index"]])
        state["path_index"] += 1
        return result

    monkeypatch.setattr(_native_collectives, "broadcast_value", broadcast)
    monkeypatch.setattr(_native_collectives, "allgather_value", lambda _w, value: (value, value))
    monkeypatch.setattr(_compile_once, "require_world", lambda _w: world)
    def artifact_broadcast(_world, value, *, root):
        assert root == 0
        if state["rank"] == 0:
            state["published"] = value
        return state["published"]

    monkeypatch.setattr(_compile_once, "broadcast_value", artifact_broadcast)
    monkeypatch.setattr(_compile_once, "allgather_value", lambda _w, value: (value, value))

    def resolve(directory, *args, **kwargs):
        seen.append((state["rank"], "resolve", directory, kwargs.get("provider_factory")))
        return resolved, ()

    def compile_artifact(plan):
        import os
        assert plan is resolved
        seen.append((state["rank"], "compile_or_verified_load", Path(os.environ["POPS_CACHE_DIR"])))
        return Artifact()

    monkeypatch.setattr(fixture, "resolve_product_chain", resolve)
    monkeypatch.setattr(fixture.pops, "compile", compile_artifact)
    monkeypatch.setattr(fixture, "artifact_execution_context", lambda _a: SimpleNamespace(
        communicator=SimpleNamespace(identity="MPI_COMM_WORLD", handle=world)))
    monkeypatch.setenv("POPS_CACHE_DIR", str(root_cache))
    _, _, root_receipts = fixture._prepare_artifact(world, root_dir, 4, 3, 3, False)
    world.rank = state["rank"] = 1
    state["path_index"] = 0
    monkeypatch.setenv("POPS_CACHE_DIR", str(peer_cache))
    _, _, peer_receipts = fixture._prepare_artifact(world, peer_dir, 4, 3, 3, False)
    assert root_receipts == peer_receipts == root_dir / "m19-product-receipts"
    assert seen == [(0, "resolve", root_dir / "providers", None),
                    (0, "compile_or_verified_load", root_cache),
                    (1, "resolve", root_dir / "providers", fixture._load_published_provider),
                    (1, "compile_or_verified_load", root_cache)]
    import os
    assert os.environ["POPS_CACHE_DIR"] == str(peer_cache)


@pytest.mark.parametrize("rank", (0, 1, 2))
def test_exact_checkpoint_path_on_each_rank_including_empty_peer(tmp_path, monkeypatch, rank):
    world = SimpleNamespace(rank=rank, size=3)
    monkeypatch.setattr(_native_collectives, "allgather_value", lambda _w, value: (value,) * 3)
    assert fixture._checkpoint_target(world, tmp_path, "accepted") == (tmp_path / "accepted").resolve()


def test_divergent_checkpoint_refuses_before_even_reading_native_state(tmp_path, monkeypatch):
    world = SimpleNamespace(rank=1, size=2)
    def gather(_w, value):
        return (value, str(tmp_path / "rank0" / "accepted")) if isinstance(value, str) else (value, value)
    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    monkeypatch.setattr(fixture, "state_snapshots", lambda *_: pytest.fail("native state read before path refusal"))
    with pytest.raises(AssertionError, match="product checkpoint path differs across MPI ranks"):
        fixture._save(world, None, None, tmp_path, "accepted")


@pytest.mark.parametrize("phase", ("../escape", "", "/absolute"))
def test_checkpoint_phase_is_an_exact_basename(tmp_path, phase):
    with pytest.raises(AssertionError, match="exact basename"):
        fixture._checkpoint_target(None, tmp_path, phase)
