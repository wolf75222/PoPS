"""Deep freeze detaches stale Problem registry references and seals storage."""
import json
from types import MappingProxyType

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.model import Module  # noqa: E402
from pops.params import ParamProvenance, RuntimeParam  # noqa: E402
from pops.problem._snapshot import build_problem_snapshot  # noqa: E402


def test_problem_freeze_detaches_stale_registry_views_and_keeps_hash_stable():
    problem = pops.Problem(name="deep-storage")
    problem.block("u", Module("model"))
    declaration = RuntimeParam(
        "alpha", default=1.0,
        provenance=ParamProvenance("test", metadata={"weights": [1, 2]}),
    )
    handle = problem.param(declaration)

    stale_block = problem._block_registry.spec("u")
    stale_param = problem._param_registry.get(handle)
    stale_params_view = problem._params
    before = build_problem_snapshot(problem).hash

    snapshot = problem.freeze()
    assert snapshot.hash == before
    assert isinstance(problem._block_registry._blocks, MappingProxyType)
    assert isinstance(problem._block_registry.spec("u"), MappingProxyType)
    assert isinstance(problem._param_registry._declarations, MappingProxyType)
    assert problem._param_registry.get(handle) is declaration

    stale_block["model"] = "detached mutation"
    stale_params_view.clear()
    with pytest.raises(AttributeError, match="immutable"):
        stale_param.default = 2.0

    live_param = problem._param_registry.get(handle)
    assert live_param.provenance.to_data()["metadata"]["weights"] == [1, 2]
    assert build_problem_snapshot(problem).hash == snapshot.hash
    json.dumps(problem.to_dict(), sort_keys=True)

    with pytest.raises(TypeError):
        problem._block_registry.spec("u")["model"] = object()
    with pytest.raises(TypeError):
        problem._param_registry._declarations["alpha"] = RuntimeParam("alpha", default=2.0)
    with pytest.raises(RuntimeError, match="frozen"):
        problem._block_registry._blocks = {}
    with pytest.raises(RuntimeError, match="frozen"):
        del problem._block_registry._blocks
    with pytest.raises(AttributeError, match="identity"):
        del problem._name
