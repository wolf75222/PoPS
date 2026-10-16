import numpy as np
import pops
import pytest
from examples.api_v02.method_relaxed_cutcell import build_t10_cutcell
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_lowerability import all_ops
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.cutcell_composite_oracle import pair, factor, _dense


@pytest.mark.parametrize("ratio", (1, 2))
def test_cutcell_equations_use_both_generic_measure_paths(ratio):
    spec = build_t10_cutcell(temporal_ratio=ratio)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    values = list(all_ops(plan.time))
    assert sum(value.op == "composite_pair" for value in values) == 3
    assert sum(value.op == "composite_integral" for value in values) == 2
    cpp = emit_cpp_program(plan.time, target="amr_system",
                           model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "ctx.accept_increment(" in cpp and "ctx.composite_integral(" in cpp
    assert "PyObject" not in cpp


def test_cut_fraction_changes_the_root_and_unweighted_root_fails_physical_energy():
    old = [np.asarray([[[1., 1.]], [[0., 0.]], [[.5, 1.5]]])]
    candidate = [old[0].copy()]
    omega = old[0][2]
    candidate[0][0] = 1-.5*omega**2
    candidate[0][1] = omega
    physical = [np.asarray([[.05, .5]])]
    unit = [np.asarray([[.5, .5]])]
    gamma = factor(old, candidate, physical)
    wrong = factor(old, candidate, unit)
    assert abs(gamma-wrong) > .01
    accepted = [old[0]+gamma*(candidate[0]-old[0])]
    inaccurate = [old[0]+wrong*(candidate[0]-old[0])]
    assert abs(pair(accepted, accepted, physical)-pair(old, old, physical)) < 1.e-14
    assert abs(pair(inaccurate, inaccurate, physical)-pair(old, old, physical)) > 1.e-3


def test_cutcell_observer_nonsquare_offsets_and_replicas_keep_numpy_axis_order():
    values = np.arange(12.).reshape(3, 4)
    piece = {"lower": (1, 2), "upper": (4, 6), "values": values[None, ...]}
    result = _dense(({"fraction": (piece,)}, {"fraction": (piece,)}), "fraction", (5, 7))
    np.testing.assert_array_equal(result[1:4, 2:6], values)
    assert np.isnan(result[0]).all() and np.isnan(result[:, 0]).all()
    conflicting = dict(piece, values=(values+1)[None, ...])
    with pytest.raises(AssertionError):
        _dense(({"fraction": (piece,)}, {"fraction": (conflicting,)}), "fraction", (5, 7))
