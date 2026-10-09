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


def _validated_transport():
    from pops.lib.time import SSPRK2
    from pops.numerics import DiscretizationPlan
    from tests.python.unit.numerics.test_discretization_plan import _declarations

    _, model, state, _, rate, method = _declarations()
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, method)
    case = pops.Case("immutable-numerical-method")
    block = case.block("tracer", model)
    case.numerics(numerics, block=block)
    case.program(SSPRK2(block[state], rate=rate))
    return pops.validate(case), numerics, method


def _resolve_transport(case):
    from pops.layouts import Uniform
    from tests.python.support.layout_plan import cartesian_grid

    return pops.resolve(case, layout=Uniform(cartesian_grid(n=8, periodic=True)))


@pytest.mark.parametrize("attribute", ("reconstruction", "positivity_floor", "_frozen"))
def test_validated_method_rejects_deletion_before_resolve(attribute):
    case, _, method = _validated_transport()
    before = _resolve_transport(case)
    snapshot_hash = case.snapshot.hash

    with pytest.raises(RuntimeError, match="frozen.*cannot delete"):
        delattr(method, attribute)

    after = _resolve_transport(case)
    assert method.formal_order == 2
    assert case.snapshot.hash == snapshot_hash
    assert after.plan_identity == before.plan_identity
    before.verify()
    after.verify()


def test_validated_numerical_plan_rejects_family_deletion():
    case, numerics, _ = _validated_transport()
    with pytest.raises(RuntimeError, match="frozen.*cannot delete"):
        del numerics.rates
    _resolve_transport(case).verify()


def test_frozen_method_copy_remains_a_detached_mutable_authoring_value():
    from copy import copy
    from pops.numerics.reconstruction import FirstOrder

    case, _, method = _validated_transport()
    before = _resolve_transport(case)
    editable = copy(method)
    del editable.positivity_floor
    editable.positivity_floor = 0.01
    editable.reconstruction = FirstOrder()
    assert editable.formal_order == 1
    assert method.formal_order == 2
    assert _resolve_transport(case).plan_identity == before.plan_identity


def test_compile_rejects_tampered_resolved_method_before_native_selection(monkeypatch):
    from pops import _native_selector

    case, _, _ = _validated_transport()
    resolved = _resolve_transport(case)

    def native_selection_must_not_run(*args, **kwargs):
        pytest.fail("modified plan crossed the native selection boundary")

    monkeypatch.setattr(_native_selector, "select_native_dimension", native_selection_must_not_run)
    # Deliberately bypass the normal Python guard: compile must independently
    # authenticate the captured plan, including numerical-method parameters.
    object.__setattr__(resolved.blocks[0].spatial, "positivity_floor", 0.01)
    with pytest.raises(ValueError, match="identity verification failed"):
        pops.compile(resolved)
