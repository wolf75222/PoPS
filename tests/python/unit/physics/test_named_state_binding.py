"""Declared state names survive public model lowering and block qualification."""
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.lib import time as methods
from pops.math import ddt
from pops.model import StateHandle


@pytest.mark.parametrize("name", ("U", "inventory", "population-density"))
def test_named_state_has_one_typed_identity_in_each_program_instance(name):
    frame = Rectangle("named_domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("named_model", frame=frame)
    state = model.state(name, components=("first", "second"))
    first, second = state
    source = model.source("production", on=state, value=(first, -second))
    rate = model.rate("evolution", equation=ddt(state) == source)
    case = pops.Case("named_case")
    left = case.block("left", model=model, states=(state,))
    right = case.block("right", model=model, states=(state,))

    assert tuple(model.module.state_spaces()) == (name,)
    declared = model.module.state_handle(model.module.state_spaces()[name])
    assert declared.space == state.space
    for block in (left, right):
        instance = block[state]
        assert isinstance(instance, StateHandle)
        assert instance.space.name == name
        assert instance.declaration_ref == declared
        assert instance.block_ref == block
        assert block[state] is instance
        assert methods.ForwardEuler(instance, rate=rate).validate()
        assert methods.SSPRK2(instance, rate=rate).validate()
    assert left[state] != right[state]


def test_legacy_dsl_retains_its_default_state_name():
    from pops.physics._facade import Model

    model = Model("legacy_dsl")
    model.conservative_vars("amount")
    assert model.state_space().name == "U"
    assert tuple(model.module.state_spaces()) == ("U",)
