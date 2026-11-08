"""Source-only closed equation/provenance checks, never a native receipt."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_stencil_extensions import build_stencil_extensions, outflow_ssprk2
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


@pytest.mark.parametrize("sample", ("state", "cell", "field"))
def test_partial_amr_stencil_calls_its_exact_source_extension(sample):
    spec = build_stencil_extensions(sample=sample)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time, target="amr_system",
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.attach_generated_face_basis(") == 2
    assert "periodic()" not in source
    if sample == "field":
        assert source.count("ctx.prepare_stencil_field_solution(") == 2
        assert source.count(".sample_extension(") == 2
        assert "ctx.solve_hierarchy_field(" in source
    elif sample == "cell":
        assert source.count(".sample_extension(") == 2
        assert "cell_sample_scope_" in source
        assert "computed cell sample extension" in source
        assert ".sample_uniform(" not in source


def test_nonperiodic_midpoint_sampling_keeps_two_stage_extensions():
    spec = build_stencil_extensions(amr=False, periodic=False)
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    assert source.count("ctx.prepare_generated_state(") == 4
    assert source.count(".sample_extension(") == 2
    old = np.array([[[1., 2., 3., 4.]]])
    correct = outflow_ssprk2(old, dt=.01, steps=2)
    # A periodic fallback transports the last cell into the inlet and disagrees.
    def periodic_rhs(q):
        return -(q-np.roll(q, 1, axis=-1))*4
    wrong = old.copy()
    for _ in range(2):
        stage = wrong+.01*periodic_rhs(wrong)
        wrong = .5*wrong+.5*(stage+.01*periodic_rhs(stage))
    assert np.max(abs(correct-wrong)) > .1


def test_fresh_residual_cannot_borrow_delivery_from_its_predictor():
    from pops.codegen.typed_delivery import delivered_frontiers
    spec = build_stencil_extensions(sample="state")
    resolved = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    first, second = [v for v in resolved.time._values if v.op == "stencil_flux_divergence"]
    frontier = lambda value: value.op == "stencil_flux_divergence"
    assert delivered_frontiers(second, lambda value: value.id == first.id,
                               stop_at=frontier) == (frozenset(),)
    delivered = delivered_frontiers(second, lambda value: value.id == second.id, stop_at=frontier)
    assert len(delivered) == 1 and len(delivered[0]) == 1


def test_typed_global_field_history_retains_its_own_space_through_freeze():
    from pops.time._program.detach import detach_compiled_program
    from pops.time._program.global_history_storage import descriptor
    spec = build_stencil_extensions(sample="field")
    program = spec.case._time
    original = descriptor(program, "start_potential")
    assert original and '"field_space"' in original
    clone = detach_compiled_program(program)
    assert descriptor(clone, "start_potential") == original
    source = next(v for v in program._values if v.op == "field_component")
    assert source.block is None and source.state_ref is None and source.space.kind == "field"
    program.freeze()
    assert descriptor(program, "start_potential") == original
