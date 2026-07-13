"""ADC-652 follow-up: physics models deep-freeze with their Case."""
from __future__ import annotations

from types import MappingProxyType

import pytest

import pops
from pops.math import ddt, div
from pops.physics import Model as BoardModel
from pops.physics.facade import Model as DslModel
from pops.problem._snapshot import build_problem_snapshot


def _dsl_advection(name="advection"):
    model = DslModel(name)
    (u,) = model.conservative_vars("u")
    model.flux(x=[u], y=[u])
    one = 0 * u + 1
    model.eigenvalues(x=[one], y=[one])
    return model, u


def _board_advection(name="board_advection"):
    from pops.frames import Cartesian2D

    frame = Cartesian2D()
    x_axis, y_axis = frame.axes
    model = BoardModel(name, frame=frame)
    state = model.state("U", components=["u"])
    flux = model.flux(
        "F", frame=frame, state=state,
        components={x_axis: (state[0],), y_axis: (state[0],)},
        waves={x_axis: (1,), y_axis: (1,)},
    )
    model.rate("A", equation=ddt(state) == -div(flux))
    return model, state


def test_dsl_freeze_is_idempotent_deep_and_codegen_readable():
    model, u = _dsl_advection()
    cpp_before = model._m.emit_cpp()
    hash_before = model._m._model_hash()

    assert model.freeze() is model
    assert model.freeze() is model
    assert model.frozen and model._m.frozen
    assert isinstance(model.params, MappingProxyType)
    assert isinstance(model._m.prim_defs, MappingProxyType)
    assert isinstance(model._m.cons_names, tuple)

    with pytest.raises(RuntimeError, match="frozen"):
        model.aux("B_z")
    with pytest.raises(RuntimeError, match="frozen"):
        model._m.aux("B_z")
    with pytest.raises(TypeError):
        model._m.prim_defs["late"] = u
    with pytest.raises(AttributeError):
        model._m.cons_names.append("late")
    with pytest.raises(RuntimeError, match="frozen"):
        model.params = {}
    with pytest.raises(RuntimeError, match="frozen"):
        model._frozen = False
    with pytest.raises(RuntimeError, match="frozen"):
        del model.params
    with pytest.raises(RuntimeError, match="frozen"):
        del model._m.cons_names

    assert model._m.emit_cpp() == cpp_before
    assert model._m._model_hash() == hash_before
    assert model.check() is True
    assert model.list_operators()


def test_case_freezes_external_board_reference_and_keeps_snapshot_stable():
    model, _state = _board_advection()
    case = pops.Case(name="frozen-physics")
    case.block("transport", model)
    module_hash = model.module.module_hash()

    snapshot = case.freeze()

    assert model.frozen and model.dsl.frozen and model.dsl._m.frozen
    with pytest.raises(RuntimeError, match="frozen"):
        model.field("late")
    with pytest.raises(RuntimeError, match="frozen"):
        model.dsl.primitive("late", 1)
    with pytest.raises(RuntimeError, match="frozen"):
        model.dsl._m.set_flux([1], [1])
    with pytest.raises(TypeError):
        model._states["late"] = object()
    with pytest.raises(RuntimeError, match="frozen"):
        model.name = "changed"
    with pytest.raises(RuntimeError, match="frozen"):
        del model.name

    assert model.module.module_hash() == module_hash
    assert build_problem_snapshot(case).hash == snapshot.hash
    assert "U" in model.inspect()["states"]
    assert "A" in model.module.list_operators()
    assert "void" in model.dsl._m.emit_cpp()


def test_multispecies_owned_module_and_registry_are_sealed():
    model = BoardModel("two_fluid_freeze")
    electrons = model.species("electrons", state=["ne"])
    ions = model.species("ions", state=["ni"])
    model.coupled_rate(
        "collision", inputs=[electrons, ions],
        outputs={electrons: [ions["ni"]], ions: [electrons["ne"]]})
    module = model.module
    operator = module.operator_registry().get("collision")
    hash_before = module.module_hash()

    case = pops.Case(name="multi-freeze")
    case.block("plasma", model)
    case.freeze()

    assert module.frozen and module.operator_registry().frozen
    with pytest.raises((TypeError, RuntimeError)):
        module.state_space("late", ("x",))
    with pytest.raises(RuntimeError, match="frozen"):
        module._state_spaces = {}
    with pytest.raises(RuntimeError, match="frozen"):
        del module._state_spaces
    with pytest.raises(TypeError):
        module._state_spaces["late"] = object()
    with pytest.raises(RuntimeError, match="frozen"):
        del module.operator_registry()._by_name
    with pytest.raises(RuntimeError, match="frozen"):
        operator.kind = "local_rate"
    with pytest.raises(RuntimeError, match="frozen"):
        del operator.kind
    with pytest.raises(TypeError):
        operator.capabilities["late"] = True

    assert module.module_hash() == hash_before
    assert "collision" in model.dump_module_ir()


def test_failed_standalone_freeze_rolls_back_the_whole_cascade():
    class FailingDescriptor:
        def __init__(self):
            self.frozen = False

        def freeze(self):
            self.frozen = True
            raise RuntimeError("injected physics freeze failure")

    model, _state = _board_advection("freeze_rollback")
    failing = FailingDescriptor()
    model._test_descriptors = {"bad": failing}

    with pytest.raises(RuntimeError, match="injected"):
        model.freeze()

    assert not model.frozen and not model.dsl.frozen and not model.dsl._m.frozen
    assert not failing.frozen
    assert isinstance(model._test_descriptors, dict)
    model._test_descriptors.pop("bad")
    assert model.freeze() is model and model.frozen


def test_failed_descriptor_freeze_restores_nested_mutables_in_place():
    class Box:
        def __init__(self):
            shared = ["original"]
            self.items = shared
            self.metadata = {"shared": shared, "nested": {"status": "original"}}

    class FailingNestedDescriptor:
        def __init__(self, payload):
            self.payload = payload
            self.frozen = False

        def freeze(self):
            self.payload.items.append("leaked")
            self.payload.metadata["shared"].append("also leaked")
            self.payload.metadata["nested"]["status"] = "mutated"
            self.payload.metadata["new"] = {"leaked": [True]}
            self.frozen = True
            raise RuntimeError("injected nested freeze failure")

    model, _state = _board_advection("nested_freeze_rollback")
    box = Box()
    items = box.items
    metadata = box.metadata
    nested = metadata["nested"]
    failing = FailingNestedDescriptor(box)
    model._test_descriptors = {"bad": failing}

    with pytest.raises(RuntimeError, match="injected nested"):
        model.freeze()

    assert not model.frozen and not model.dsl.frozen and not model.dsl._m.frozen
    assert not failing.frozen
    assert failing.payload is box
    assert box.items is items and items == ["original"]
    assert box.metadata is metadata
    assert metadata == {"shared": items, "nested": {"status": "original"}}
    assert metadata["shared"] is items
    assert metadata["nested"] is nested

    model._test_descriptors.pop("bad")
    assert model.freeze() is model and model.frozen
