"""Coexisting issued storage routes; authoring only, without Native reception."""

import pytest

from pops.fields import CellMidpoint, CellVolumeMeasure, DirectSpatialInteraction, SpatialInteractionKernel
from pops.time._program.detach import detach_compiled_program
from tests.review.test_sol61_closed_original_interaction import witness


@pytest.mark.parametrize("history_first", (False, True))
def test_global_history_and_completed_source_share_an_unread_storage_owner(history_first):
    case, _layout, program, source, first = witness(width=3, selected=2, order=(2, 0, 1))
    owner = first.attrs["closed_field_source"]["owner_block"]
    prior_routes = program._block_indices()
    model = owner._instance_registry._blocks[owner.local_id]["model"]
    storage = case.block("both-issued-storage-routes", model)
    declaration = next(value.state for value in program._time_states.values() if value.block is owner)
    time = program.state(storage[declaration.declaration_ref])

    def history():
        program.store_history("accepted-global-component", source, depth=3, owner_block=storage)

    def interaction():
        program.spatial_interaction(
            source, SpatialInteractionKernel(2, lambda x, y: 1 + x[0] * y[1]),
            output_space=first.space, measure=CellVolumeMeasure(), quadrature=CellMidpoint(),
            realization=DirectSpatialInteraction(2**24),
            source_scope="completed_original", owner_block=storage,
        )

    (history if history_first else interaction)()
    (interaction if history_first else history)()
    routes = program._block_indices()
    assert {block: routes[block] for block in prior_routes} == prior_routes
    assert routes[storage] == len(prior_routes)
    assert len(routes) == len(prior_routes) + 1
    assert not any(value.op == "state" and value.block is storage for value in program._values)
    assert not any(state.block_ref is storage for state in program._commits)
    assert time not in program._time_current_values
    assert source.space is source.block is source.state_ref is None
    image = program._serialize(include_provenance=False)
    assert image["version"] == 19
    clone = detach_compiled_program(program)
    assert clone._serialize(include_provenance=False) == image
    assert clone._block_indices() == routes
    program.freeze()
    assert program._serialize(include_provenance=False) == image
    assert program._block_indices() == routes
