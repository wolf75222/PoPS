"""Real resolve/lower/emit witnesses for complete principal finite volumes."""
import pytest
import pops

from tests.python.unit.numerics.test_principal_grouping_contract import _principal_case


def _lower(case, model, program):
    from pops.codegen.program_models import ProgramModelGraph
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.time import AdaptiveCFL
    program.step_strategy(AdaptiveCFL(.3))
    layout = Uniform(CartesianGrid(frame=model.frame, cells=(8, 8),
                                   periodic=PeriodicAxes(model.frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    return resolved, ProgramModelGraph.from_resolved_blocks(resolved.blocks)


@pytest.mark.parametrize("size", (2, 3, 5))
def test_scalar_storage_emits_one_complete_native_principal_face_evaluation(size):
    from pops.codegen.program_codegen import emit_cpp_program
    case, model, _, _, _, program, _ = _principal_case(size)
    resolved, graph = _lower(case, model, program)
    source = emit_cpp_program(resolved.time, model=graph)
    assert "PreparedPrincipalFlux<pops::kNativeDimension,%d>" % size in source
    assert source.count("resource.evaluate(") == 1
    assert source.count("_resource.publish(") == 1
    assert "using State = pops::StateVec<n_vars>" in source
    assert "explicit_frequency()" in source
    assert 'pops_program_has_dt_bound() { return true; }' in source
    for component in range(size):
        assert "result[%d] =" % component in source


def test_group_parameter_tables_keep_exact_slots_for_every_sampled_block():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_emit_params import program_param_entries
    case, model, _, _, _, program, _ = _principal_case(2, parameter=True)
    resolved, graph = _lower(case, model, program)
    entries = program_param_entries(resolved.time, graph)
    assert entries == [(0, "speed", 0, 1.), (1, "speed", 0, 1.)]
    source = emit_cpp_program(resolved.time, model=graph)
    assert "params.get(0)" in source
    assert "parameter_sets[0]" in source and "parameter_sets[1]" in source


def test_group_user_reconstruction_uses_its_authored_native_stencil():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.numerics.reconstruction import User
    recipe = User(lambda sample: sample(0) + .25 * (sample(1) - sample(-1)), formal_order=2)
    case, model, _, _, _, program, _ = _principal_case(2, reconstruction=recipe)
    resolved, graph = _lower(case, model, program)
    source = emit_cpp_program(resolved.time, model=graph)
    assert recipe.options["source_identity"] in source
    assert "sample(-1)" in source and "sample(1)" in source
