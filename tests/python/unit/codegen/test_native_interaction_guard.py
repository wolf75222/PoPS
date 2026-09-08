"""Guard boundaries execute only the selected imported interaction contexts."""
import ctypes
import numpy as np
import pops
import pytest
from pops.layouts import Uniform
from test_native_interaction_matrix import _imported_factory, _initial, _expected
from test_interaction_inventory_quadrature import interaction_case
from interaction_test_layout import interaction_grid, require_two_rank_partition
from tests.python.support.native_execution_context import artifact_execution_context


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    factory = _imported_factory(tmp_path_factory.mktemp("guarded") / "external")
    case, _program, _model, _maps = interaction_case(
        guarded=True, inventories=False, native_function=factory)
    resolved = pops.resolve(pops.validate(case), layout=Uniform(interaction_grid(n=16)))
    artifact = pops.compile(resolved)
    native = ctypes.CDLL(str(artifact.program.so_path))
    native.pops_interaction_calls.restype = ctypes.c_long
    return artifact, native


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
@pytest.mark.parametrize("active", (False, True), ids=("inactive-invalid-input","active-valid-input"))
def test_native_interaction_guard_prevents_eager_fallible_calls(compiled, active, record_property):
    artifact, native = compiled
    initial = _initial(16)
    if not active:
        initial["left"][:] = 0
        initial["right"][2] = 0  # imported law divides by this mass if evaluated
    simulation = pops.bind(artifact, initial_state=initial,
        resources={"execution_context": artifact_execution_context(artifact)})
    local_cells = sum(int(np.prod(np.asarray(hi)-lo)) for lo,hi in simulation.local_boxes("left"))
    native.pops_interaction_reset()
    report = pops.run(simulation,t_end=.1,max_steps=100)
    assert report.accepted_steps == 100 and report.rejected_steps == 0
    assert native.pops_interaction_calls() == (200*local_cells if active else 0)
    for name, expected in zip(("left","right"),_expected(100,1),strict=True):
        actual = simulation.state_global(name)
        if active:
            np.testing.assert_allclose(actual,expected[:,None,None]*np.ones((1,16,16)),rtol=0,atol=1e-11)
        else:
            np.testing.assert_array_equal(actual,initial[name])
        assert np.all(np.isfinite(actual))
    assert simulation._executor_for_block("left")._program_exchange_records() == []
    record_property("selected_guard",active)
    record_property("actual_native_calls_local",native.pops_interaction_calls())
    record_property("accepted_steps",report.accepted_steps)
    record_property("guard_contexts", "two recipient branches retain distinct selected evaluation contexts")


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
def test_rank_local_native_failure_in_selected_arm_rolls_back(compiled,record_property):
    artifact,native = compiled
    initial = _initial(16)
    initial["right"][2,-1,-1] = 0
    simulation = pops.bind(artifact,initial_state=initial,
        resources={"execution_context":artifact_execution_context(artifact)})
    from pops import _pops
    from pops._native_collectives import allgather_value, require_world

    world = require_world(_pops.mpi_world())
    ranks = int(world.size)
    assert ranks in (1, 2), "selected-arm rollback qualification declares one or two ranks"
    if ranks == 2:
        require_two_rank_partition(simulation, n=16)
    boxes = simulation.local_boxes("left")
    cells = sum(int(np.prod(np.asarray(hi) - lo)) for lo, hi in boxes)
    owns_bad = any(all(int(lo[d]) <= 15 < int(hi[d]) for d in (0, 1))
                   for lo, hi in boxes)
    # A serial run owns both declared bands; a two-rank run places the sole
    # invalid last cell on rank one. Verify the actual native ownership in both modes.
    expected_cells = (256,) if ranks == 1 else (128, 128)
    expected_owners = (True,) if ranks == 1 else (False, True)
    cells_by_rank = allgather_value(world, cells)
    owners_by_rank = allgather_value(world, owns_bad)
    assert cells_by_rank == expected_cells
    assert owners_by_rank == expected_owners
    native.pops_interaction_reset()
    from pops._bootstrap import StepAttemptRejected
    with pytest.raises(StepAttemptRejected) as failure:
        pops.run(simulation,t_end=.001,max_steps=1)
    assert allgather_value(world,type(failure.value).__name__) == ("StepAttemptRejected",)*int(world.size)
    for name in ("left","right"):
        np.testing.assert_array_equal(simulation.state_global(name),initial[name])
    assert simulation._executor_for_block("left")._program_exchange_records() == []
    calls = native.pops_interaction_calls()
    expected_calls = tuple(count - int(owner)
                           for count, owner in zip(expected_cells, expected_owners, strict=True))
    assert calls == cells - int(owns_bad)
    calls_by_rank = allgather_value(world, calls)
    assert calls_by_rank == expected_calls
    record_property("native_mpi_ranks", ranks)
    record_property("owned_cells_by_rank", cells_by_rank)
    record_property("invalid_last_cell_owner_by_rank", owners_by_rank)
    record_property("actual_rejected_attempt_calls_local", calls)
    record_property("actual_rejected_attempt_calls_by_rank", calls_by_rank)
    record_property("later_selected_recipient_calls",0)
