"""Owner-total ProgramModelGraph coverage for ADC-662."""
from __future__ import annotations

import hashlib

import pytest

from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
from pops.codegen.program_models import ProgramModelGraph
from pops.model import OwnerKind, OwnerPath
from pops.problem.handles import BlockHandle


class _Model:
    def __init__(self, name, owner=None):
        self.name = name
        self.owner_path = owner or OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        if self.owner_path.is_authoring:
            digest = hashlib.sha256(name.encode()).hexdigest()
            self.owner_path._bind_definition_fingerprint(
                "test-model:sha256:%s" % digest)

    def to_data(self):
        return {"name": self.name, "owner": self.owner_path.presentation().to_data()}


class _EmitModel:
    def __init__(self, source):
        self.source = source
        self.owner_path = source.owner_path


def _block(name, model):
    return ResolvedBlock(name, model, None, "production")


@pytest.fixture
def lowered(monkeypatch):
    calls = []

    def lower(model, *, facade):
        assert facade is model
        calls.append(model)
        return _EmitModel(model), {"module": model.name}

    monkeypatch.setattr("pops.codegen.module_lowering.lower_and_validate", lower)
    return calls


def test_graph_covers_every_block_and_dispatches_by_exact_owner(lowered):
    ions = _Model("ions")
    electrons = _Model("electrons")
    graph = ProgramModelGraph.from_resolved_blocks((
        _block("ion_fluid", ions),
        _block("electron_fluid", electrons),
    ))

    assert graph.model_for_owner(ions.owner_path).source is ions
    assert graph.model_for_owner(electrons.owner_path).source is electrons
    assert graph.model_for_block("ion_fluid").source is ions
    assert graph.model_for_block("electron_fluid").source is electrons
    assert tuple(lowered) == (ions, electrons)

    case = OwnerPath.fresh(OwnerKind.CASE, "plasma")
    block = BlockHandle(
        "electron_fluid", owner=case, model_owner=electrons.owner_path,
    )
    assert graph.model_for_block(block).source is electrons

    with pytest.raises(KeyError, match="no route for block"):
        graph.model_for_block("missing")
    with pytest.raises(KeyError, match="no model for owner"):
        graph.model_for_owner(OwnerPath.model("missing"))


def test_repeated_instances_of_one_model_lower_once_and_keep_two_routes(lowered):
    transport = _Model("transport")
    graph = ProgramModelGraph.from_resolved_blocks((
        _block("left", transport),
        _block("right", transport),
    ))

    assert tuple(graph.owners_by_block) == ("left", "right")
    assert graph.owners_by_block["left"] == graph.owners_by_block["right"]
    assert graph.model_for_block("left") is graph.model_for_block("right")
    assert lowered == [transport]


def test_graph_rejects_canonical_owner_collision_without_first_wins(lowered):
    first = _Model("same")
    second = _Model("same")
    assert first.owner_path != second.owner_path
    assert first.owner_path.canonical() == second.owner_path.canonical()

    with pytest.raises(ValueError, match="distinct authoring model authorities collide"):
        ProgramModelGraph.from_resolved_blocks((
            _block("first", first),
            _block("second", second),
        ))


def test_block_handle_owner_mismatch_is_rejected(lowered):
    first = _Model("first")
    second = _Model("second")
    graph = ProgramModelGraph.from_resolved_blocks((_block("fluid", first),))
    block = BlockHandle(
        "fluid",
        owner=OwnerPath.fresh(OwnerKind.CASE, "case"),
        model_owner=second.owner_path,
    )

    with pytest.raises(ValueError, match="carries model owner"):
        graph.model_for_block(block)


def test_graph_routes_are_immutable_and_plan_has_no_first_model_escape(lowered):
    model = _Model("transport")
    graph = ProgramModelGraph.from_resolved_blocks((_block("fluid", model),))

    with pytest.raises(TypeError):
        graph.owners_by_block["other"] = model.owner_path
    with pytest.raises(TypeError):
        graph.models_by_owner[model.owner_path.canonical()] = object()
    assert not hasattr(ResolvedSimulationPlan, "first_model")
