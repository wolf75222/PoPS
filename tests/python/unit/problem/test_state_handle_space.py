"""A typed state Handle is sufficient even when one Module declares several states."""
from __future__ import annotations

import pytest

from pops.model import Module, StateHandle
from pops.problem import Case
from pops.time import Program
from pops.time.program_detach import detach_compiled_program


def test_state_handle_carries_its_authoritative_space_without_live_registry_lookup():
    module = Module("two-states")
    electrons = module.state_space("electrons", ("ne", "ue"))
    ions = module.state_space("ions", ("ni", "ui", "energy"))
    electron_handle = module.state_handle(electrons)
    ion_handle = module.state_handle(ions)
    block = Case(name="plasma").block("fluid", module)
    program = Program("multi-state")

    electron_time = program.state(block, electron_handle)
    ion_time = program.state(block, ion_handle)

    assert isinstance(electron_handle, StateHandle)
    assert electron_time.space is electrons
    assert electron_time.n.space is electrons
    assert ion_time.space is ions
    assert ion_time.n.space is ions
    assert block[electron_handle].space is electrons
    assert program._state_spaces[electron_time.state] is electrons
    assert program._state_spaces[ion_time.state] is ions
    with pytest.raises(AttributeError, match="immutable"):
        electron_handle.space = ions


def test_multi_state_space_identity_survives_serialization_and_detachment():
    module = Module("two-states-detached")
    electrons = module.state_space("electrons", ("ne", "ue"))
    ions = module.state_space("ions", ("ni", "ui", "energy"))
    block = Case(name="plasma-detached").block("fluid", module)
    program = Program("multi-state-detached")
    electron_time = program.state(block, module.state_handle(electrons))
    ion_time = program.state(block, module.state_handle(ions))
    _ = electron_time.n, ion_time.n

    serialized = program._serialize()
    states = {node["state"]["local_id"]: node for node in serialized["nodes"]}
    assert states["electrons"]["space"]["name"] == "electrons"
    assert states["ions"]["space"]["name"] == "ions"

    detached = detach_compiled_program(program)
    detached_spaces = {
        state_ref.local_id: space for state_ref, space in detached._state_spaces.items()
    }
    assert detached_spaces == {"electrons": electrons, "ions": ions}
    assert detached._ir_hash() == program._ir_hash()
