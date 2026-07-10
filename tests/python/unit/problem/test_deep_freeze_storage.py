"""Deep freeze detaches stale Problem registry references and seals storage."""
import json
from types import MappingProxyType

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.descriptors import BrickDescriptor  # noqa: E402
from pops.problem._snapshot import build_problem_snapshot  # noqa: E402


def test_problem_freeze_detaches_stale_registry_views_and_keeps_hash_stable():
    problem = pops.Problem(name="deep-storage")
    problem.add_block("u", BrickDescriptor("model", "native"))
    problem.param("alpha", {"weights": [1, 2]})

    stale_block = problem._block_registry.spec("u")
    stale_param = problem._param_registry.get("alpha")
    stale_params_view = problem._params
    before = build_problem_snapshot(problem).hash

    snapshot = problem.freeze()
    assert snapshot.hash == before
    assert isinstance(problem._block_registry._blocks, MappingProxyType)
    assert isinstance(problem._block_registry.spec("u"), MappingProxyType)
    assert isinstance(problem._param_registry.get("alpha"), MappingProxyType)

    stale_block["model"] = "detached mutation"
    stale_param["default"]["weights"].append(3)
    stale_params_view["alpha"]["default"]["weights"].append(4)

    live_param = problem._param_registry.get("alpha")
    assert live_param["default"]["weights"] == (1, 2)
    assert build_problem_snapshot(problem).hash == snapshot.hash
    json.dumps(problem.to_dict(), sort_keys=True)

    with pytest.raises(TypeError):
        problem._block_registry.spec("u")["model"] = object()
    with pytest.raises(TypeError):
        live_param["default"]["weights"] += (3,)
    with pytest.raises(RuntimeError, match="frozen"):
        problem._block_registry._blocks = {}
    with pytest.raises(RuntimeError, match="frozen"):
        del problem._block_registry._blocks
    with pytest.raises(AttributeError, match="identity"):
        del problem._name

