"""Repeated accepted mathematical occurrences survive sharing the native application."""
import ctypes
from fractions import Fraction
import numpy as np
import pops
import pytest
from pops.layouts import Uniform
from test_native_interaction_matrix import _imported_factory, _initial, _expected
from test_interaction_inventory_quadrature import interaction_case
from interaction_test_layout import interaction_grid
from tests.python.support.native_execution_context import artifact_execution_context


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
@pytest.mark.parametrize("n", (16,32,64))
def test_repeated_native_projection_and_compensating_accepted_weight(n,tmp_path,record_property):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    factory = _imported_factory(tmp_path / "external")
    case, _program, _model, _maps = interaction_case(
        n=n,repeated=True,left_weight=Fraction(1,2),native_function=factory)
    resolved = pops.resolve(pops.validate(case),layout=Uniform(interaction_grid(n=n)))
    artifact = pops.compile(resolved)
    native = ctypes.CDLL(str(artifact.program.so_path))
    native.pops_interaction_calls.restype = ctypes.c_long
    simulation = pops.bind(artifact,initial_state=_initial(n),
        resources={"execution_context":artifact_execution_context(artifact)})
    cells = sum(int(np.prod(np.asarray(hi)-lo)) for lo,hi in simulation.local_boxes("left"))
    native.pops_interaction_reset()
    report = pops.run(simulation,t_end=.1,max_steps=100)
    assert report.accepted_steps == 100 and report.rejected_steps == 0
    for name,expected in zip(("left","right"),_expected(100,1),strict=True):
        np.testing.assert_allclose(simulation.state_global(name),
            expected[:,None,None]*np.ones((1,n,n)),rtol=0,atol=1e-11)
    assert native.pops_interaction_calls() == 100*cells
    rows = simulation._executor_for_block("left")._program_exchange_records()
    assert len(rows) == 6
    for inventory in ("momentum","total_energy"):
        selected = [row for row in rows if row["occurrence_identity"].endswith(":"+inventory)]
        assert len(selected) == 3 and len({row["occurrence_identity"] for row in selected}) == 3
        assert abs(sum(row["integrated_amount"] for row in selected)) < 1e-11
    record_property("n",n)
    record_property("actual_joint_calls_local",native.pops_interaction_calls())
    record_property("left_mathematical_occurrences",2)
    record_property("left_accepted_weight","1/2")
    record_property("accepted_records_last_step",len(rows))
    record_property("accepted_steps",report.accepted_steps)
