"""Path suspension and accepted-flux proofs cross real structured regions."""
import pytest
import pops
from examples.api_v02.method_path_hierarchy_composition import build_path_hierarchy_composition
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_lowerability import _check_amr_flux_weights
from pops.numerics.terms import Flux
from typed_program_support import typed_state


@pytest.mark.parametrize("mode", ("straight", "branch", "fold", "unrolled"))
def test_complete_path_region_and_composite_solve_are_lowered(mode):
    spec = build_path_hierarchy_composition(mode=mode)
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    model = ProgramModelGraph.from_resolved_blocks(plan.blocks, scalar_layout_plan=plan.layout_plan)
    source = emit_cpp_program(plan.time, model_graph=model, target="amr_system")
    count = 2 if mode == "unrolled" else 1
    assert source.count("ctx.stage_path_rhs(") == count
    assert source.count("ctx.publish_staged_path_rhs(") == count
    assert source.count("ctx.capture_rhs_input_trace(") == count
    assert source.count("ctx.prepare_spatial_hierarchy(") == 1
    assert "HierarchyBarrierKind::nonlinear_solve" in source
    assert "requires the unique synchronized hierarchy barrier" not in source
    if mode == "fold":
        assert "hierarchy_fold_done_" in source and "->repeat();" in source
    if mode == "branch":
        assert "hierarchy_join_" in source


def _folded_flux(power):
    program = pops.Program("folded_exact_flux")
    state = typed_state(program, "fluid", state_name="U")
    initial = program.value("initial", 1*state.n, at=state.next.point)
    coefficient = (1, program.dt, program.dt*program.dt)[power]
    def advance(body, value, index):
        del index
        rate = body.rhs(state=value, terms=[Flux()])
        return body.value("next", value + coefficient*rate, at=value.point)
    result = program.fold(initial, 3, advance)
    program.commit(state.next, result)
    return program


def test_fold_carry_preserves_exact_integrated_flux_proof():
    _check_amr_flux_weights(_folded_flux(1))


@pytest.mark.parametrize("power", (0, 2))
def test_fold_cannot_hide_unintegrated_or_twice_integrated_flux(power):
    with pytest.raises(ValueError, match="dt powers"):
        _check_amr_flux_weights(_folded_flux(power))
