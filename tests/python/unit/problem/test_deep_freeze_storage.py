"""Deep freeze detaches stale Case registry references and seals storage."""
import json
from types import MappingProxyType

import pytest

pops = pytest.importorskip("pops", exc_type=ImportError)

from pops.model import Module  # noqa: E402
from pops.params import ParamProvenance, RuntimeParam  # noqa: E402
from pops.problem._detached import detached_frozen  # noqa: E402
from pops.problem._snapshot import build_problem_snapshot  # noqa: E402


def test_case_freeze_detaches_stale_registry_views_and_keeps_hash_stable():
    case = pops.Case(name="deep-storage")
    case.block("u", Module("model"))
    declaration = RuntimeParam(
        "alpha", default=1.0,
        provenance=ParamProvenance("test", metadata={"weights": [1, 2]}),
    )
    handle = case.param(declaration)

    stale_block = case._block_registry.spec("u")
    stale_param = case._param_registry.get(handle)
    stale_params_view = case._params
    before = build_problem_snapshot(case).hash

    snapshot = case.freeze()
    assert snapshot.hash == before
    assert isinstance(case._block_registry._blocks, MappingProxyType)
    assert isinstance(case._block_registry.spec("u"), MappingProxyType)
    assert isinstance(case._param_registry._declarations, MappingProxyType)
    assert case._param_registry.get(handle) is declaration

    stale_block["model"] = "detached mutation"
    stale_params_view.clear()
    with pytest.raises(AttributeError, match="immutable"):
        stale_param.default = 2.0

    live_param = case._param_registry.get(handle)
    assert live_param.provenance.to_data()["metadata"]["weights"] == [1, 2]
    assert build_problem_snapshot(case).hash == snapshot.hash
    json.dumps(case.to_dict(), sort_keys=True)

    with pytest.raises(TypeError):
        case._block_registry.spec("u")["model"] = object()
    with pytest.raises(TypeError):
        case._param_registry._declarations["alpha"] = RuntimeParam("alpha", default=2.0)
    with pytest.raises(RuntimeError, match="frozen"):
        case._block_registry._blocks = {}
    with pytest.raises(RuntimeError, match="frozen"):
        del case._block_registry._blocks
    with pytest.raises(AttributeError, match="identity"):
        del case._name


def test_plain_mutable_extension_record_cannot_cross_compiled_boundary():
    """A copied-but-still-mutable foreign record is not an immutable protocol."""

    class MutableExtension:
        def __init__(self):
            self.options = {"order": 2}

    with pytest.raises(TypeError, match="retained extension values must implement freeze"):
        detached_frozen(MutableExtension())
