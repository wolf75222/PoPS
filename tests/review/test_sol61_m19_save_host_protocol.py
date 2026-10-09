"""Execute authentic save/check helpers with substituted collectives/runtime.

No PoPS import, native checkpoint, file publication, MPI or saved-state positive.
Every admitted path stops at an explicit checkpoint boundary before file writes.
"""

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/python/integration/runtime/test_m19_product_support_runtime.py"


class CheckpointBoundary(RuntimeError):
    pass


def actual_save_scope(monkeypatch, rank, *, divergent=False):
    world = SimpleNamespace(rank=rank, size=3)
    events = []
    collective_module = ModuleType("pops._native_collectives")

    def gather(_world, value):
        assert _world is world
        if isinstance(value, str):
            events.append(("path_consensus", value))
            return (value, value + "-foreign", value) if divergent else (value,) * 3
        if isinstance(value, dict):
            events.append(("owner_consensus", value))
            return (value, {}, {})
        return (value,) * 3

    collective_module.allgather_value = gather
    monkeypatch.setitem(__import__("sys").modules, "pops._native_collectives", collective_module)
    scope = {"Path": Path, "np": np}
    checks = ROOT / "tests/python/support/collective_checks.py"
    exec(compile(ast.parse(checks.read_text()), str(checks), "exec"), scope)
    tree = ast.parse(FIXTURE.read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_save", "_checkpoint_target")
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "NAMES" for target in node.targets)
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(FIXTURE), "exec"), scope)

    def state(name):
        events.append(("state", name))
        # Empty peers still enter every gather. These values never reach a file.
        return np.ones((3, 1, 1)) if rank == 0 else None

    def boxes(name):
        events.append(("boxes", name))
        return ((0, 0),) if rank == 0 else ()

    def checkpoint(target):
        events.append(("checkpoint", target))
        raise CheckpointBoundary("independent review checkpoint boundary")

    runtime = SimpleNamespace(state_global=state, local_boxes=boxes, checkpoint=checkpoint)
    return scope, world, runtime, events


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("phase", ["initial", "accepted", "restored", "replayed"])
def test_actual_save_authenticates_common_target_then_reads_every_state_before_checkpoint(
    tmp_path, monkeypatch, rank, phase
):
    scope, world, runtime, events = actual_save_scope(monkeypatch, rank)
    with pytest.raises(AssertionError, match="independent review checkpoint boundary"):
        scope["_save"](world, runtime, object(), tmp_path, phase)
    expected = (tmp_path / phase).resolve()
    assert events[0] == ("path_consensus", str(expected))
    assert [name for kind, name in events if kind == "state"] == list(scope["NAMES"])
    assert [name for kind, name in events if kind == "boxes"] == list(scope["NAMES"])
    assert events[-1] == ("checkpoint", expected)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_actual_save_divergent_peer_path_refuses_before_any_runtime_call(
    tmp_path, monkeypatch, rank
):
    scope, world, runtime, events = actual_save_scope(monkeypatch, rank, divergent=True)
    with pytest.raises(AssertionError, match="product checkpoint path differs across MPI ranks"):
        scope["_save"](world, runtime, object(), tmp_path, "accepted")
    assert events == [("path_consensus", str((tmp_path / "accepted").resolve()))]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("phase", ["/absolute", "../accepted", "accepted/sub", ".", "..", "", None])
def test_actual_save_invalid_phase_refuses_before_native_state_or_files(
    tmp_path, monkeypatch, phase
):
    scope, world, runtime, events = actual_save_scope(monkeypatch, 0)
    with pytest.raises(AssertionError, match="product checkpoint (phase|escapes)"):
        scope["_save"](world, runtime, object(), tmp_path, phase)
    assert events == []
    assert list(tmp_path.iterdir()) == []
