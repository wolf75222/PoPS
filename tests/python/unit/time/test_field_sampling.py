"""Public typed scalar samples retain native provider, storage and clock authority."""
from __future__ import annotations

import pytest

import pops
from pops.analytic import coordinate, cos, sin, where
from pops.domain import Rectangle
from pops.fields import AnalyticAux, FieldStorageBinding
from pops.frames import Cartesian2D
from pops.mesh import CartesianGrid, LayoutPlanBuilder, PeriodicAxes
from pops.layouts import Uniform
from pops.model import PhysicalDimension, PhysicalSupport


def sample_case(*, periodic=True, field_representation="field", field_sampling="cell",
                with_expected_phi=False):
    frame = Rectangle("sample domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    unit = PhysicalDimension()
    support = PhysicalSupport((("x", "sample_domain"), ("y", "sample_domain")))
    model = pops.Model("sample carrier", frame=frame)
    state = model.state("Q", components=("q",), units=(unit,), support=support, sampling="cell")
    model.auxiliary("geometry", unit=unit, frame=frame.canonical_id)
    if with_expected_phi:
        # The independent reference is a second global auxiliary, consumed alone at compact slot 0.
        model.auxiliary("zz_expected_phi", unit=unit, frame=frame.canonical_id)
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    flux = model.flux("stationary flux", frame=frame, state=state,
                      components={axis: (0*state[0],) for axis in frame.axes},
                      waves={axis: (0*state[0],) for axis in frame.axes})
    rate = model.rate("stationary", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                      reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    module = model.module
    aux = module.aux_handle(module.aux()["geometry"])
    x, y = (coordinate(frame, axis) for axis in frame.axes)
    module.aux_provider(AnalyticAux(aux, where((x-.5)**2+(y-.5)**2 < .2**2, 1., 0.), frame=frame))
    if with_expected_phi:
        expected_aux = module.aux_handle(module.aux()["zz_expected_phi"])
        expected = where((x-.5)**2+(y-.5)**2 < .2**2,
                         sin(3*x+2*y)+.1*cos(7*x-5*y), 0.)
        module.aux_provider(AnalyticAux(expected_aux, expected, frame=frame))
    spaces = tuple(module.field_space(name, (name,), units=(unit,), support=support,
                    frame=frame.canonical_id, sampling=field_sampling,
                    representation=field_representation) for name in ("mask", "phi"))
    case = pops.Case("scalar samples")
    block = case.block("carrier", model)
    case.numerics(numerics, block=block)
    grid = CartesianGrid(frame=frame, cells=(16,16),
                         periodic=PeriodicAxes(frame.axes) if periodic else None)
    builder = LayoutPlanBuilder(case.owner_path.canonical())
    layout = builder.layout("uniform", Uniform(grid))
    builder.assign_state(case.resolve(block[state]), layout)
    builder.assign_block(case.resolve(block), layout)
    bindings = tuple(FieldStorageBinding((module.field_handle(space),), layout, (0,1))
                     for space in spaces)
    program = pops.Program("samples")
    temporal = program.state(block[state])
    return case, program, temporal, aux, spaces, bindings, (builder, {layout: Uniform(grid)})


def projected_case(*, periodic=True, with_expected_phi=False):
    case, program, temporal, aux, spaces, bindings, builder = sample_case(
        periodic=periodic, with_expected_phi=with_expected_phi)
    mask = program.field_sample(aux, template=temporal.n, storage=bindings[0], space=spaces[0])
    operator = program.matrix_free_operator("projected", template=temporal.n,
                                            storage=bindings[1], space=spaces[1])

    def apply(p, out, phi):
        zero = p.scalar_field("zero", like=phi)
        inside = p.where(mask, phi, zero)
        lap = p.laplacian(p.scalar_field("lap", like=phi), inside)
        negative = p.value("negative", -lap)
        return p.where(mask, negative, phi)

    bound = program.set_apply(operator, apply)
    return case, program, temporal, mask, bound, builder


def test_typed_aux_sample_and_projected_operator_author_without_state_conversion():
    _, program, temporal, mask, operator, _ = projected_case()
    assert mask.block == temporal.n.block and mask.state_ref is None
    assert mask.attrs["sample_kind"] == "aux"
    assert operator.attrs["apply_in"].space == operator.space
    assert operator.attrs["apply_in"].state_ref is None
    assert operator.attrs["stencil_access"].required_ghost_depth == 1
    assert program._ir_hash()


def test_state_component_sample_uses_the_same_explicit_scalar_storage():
    _, program, temporal, _, spaces, bindings, _ = sample_case()
    sample = program.field_sample(temporal.n, component="q", template=temporal.n,
                                  storage=bindings[1], space=spaces[1])
    assert sample.attrs["sample_kind"] == "state"
    assert sample.attrs["component"] == 0 and sample.state_ref is None


def test_scalar_sample_refuses_foreign_owner_even_with_identical_names():
    _, program, temporal, _, spaces, bindings, _ = sample_case()
    _, _, _, foreign_aux, _, _, _ = sample_case()
    with pytest.raises((ValueError, TypeError), match="owner|declaration|registered|foreign|authentic"):
        program.field_sample(foreign_aux, template=temporal.n, storage=bindings[0], space=spaces[0])


def test_bound_operator_refuses_untyped_mask_capture():
    _, program, temporal, _, spaces, bindings, _ = sample_case()
    mask = program.cell_gt(temporal.n, .5)
    operator = program.matrix_free_operator("typed", template=temporal.n,
                                            storage=bindings[1], space=spaces[1])
    with pytest.raises(ValueError, match="storage authority"):
        program.set_apply(operator, lambda p, out, value:
                          p.where(mask, value, p.scalar_field(like=value)))


def test_typed_scratch_cannot_change_component_width():
    _, program, _, _, operator, _ = projected_case()
    with pytest.raises(ValueError, match="one component"):
        program.scalar_field(ncomp=2, like=operator.attrs["apply_in"])


def test_typed_comparison_refuses_to_drop_storage_or_physical_authority():
    _, program, _, mask, _, _ = projected_case()
    before = program._ir_hash()
    with pytest.raises(NotImplementedError, match="qualified dimensionless mask space"):
        program.cell_lt(mask, .5)
    assert program._ir_hash() == before


def test_typed_sample_refuses_point_values_without_a_qualified_transform():
    _, program, temporal, aux, spaces, bindings, _ = sample_case(field_sampling="cell_value")
    template = temporal.n
    before = program._ir_hash()
    with pytest.raises(NotImplementedError, match="qualified cell sampling"):
        program.field_sample(aux, template=template, storage=bindings[0], space=spaces[0])
    assert program._ir_hash() == before


def test_typed_sample_refuses_a_non_field_representation():
    _, program, temporal, aux, spaces, bindings, _ = sample_case(field_representation="conservative")
    template = temporal.n
    before = program._ir_hash()
    with pytest.raises(NotImplementedError, match="qualified field representation"):
        program.field_sample(aux, template=template, storage=bindings[0], space=spaces[0])
    assert program._ir_hash() == before


@pytest.mark.parametrize("periodic", (True, False))
def test_typed_operator_refuses_legacy_gradient_divergence_boundary_bypass(periodic):
    _, program, temporal, _, spaces, bindings, _ = sample_case(periodic=periodic)
    operator = program.matrix_free_operator("typed gradient-divergence", template=temporal.n,
                                            storage=bindings[1], space=spaces[1])
    before = program._ir_hash()

    def mixed_apply(p, out, phi):
        gradient = p.gradient(p.scalar_field(ncomp=2), phi)
        return p.divergence(p.scalar_field(), gradient)

    with pytest.raises(ValueError, match="exact storage/space/owner authority"):
        program.set_apply(operator, mixed_apply)
    assert program._ir_hash() == before


def test_nonperiodic_typed_operator_requires_boundary_authority_for_its_stencil_contract():
    _, program, temporal, _, spaces, bindings, authority = sample_case(periodic=False)
    operator = program.matrix_free_operator("declared stencil", stencil_depth=1,
                                            template=temporal.n, storage=bindings[1],
                                            space=spaces[1])
    # No Laplacian opcode exists in this apply: the operator's typed capability is authoritative.
    program.set_apply(operator, lambda _program, _out, phi: phi)
    builder, _providers = authority
    from pops.time.references import canonical_handle
    layout = builder.resolve(states=(canonical_handle(temporal.n.state_ref),),
                              blocks=(canonical_handle(temporal.n.block),))
    from pops.time.field_sampling import validate_resolved_scalar_storage
    with pytest.raises(NotImplementedError, match="conducting field boundary authority"):
        validate_resolved_scalar_storage(program, layout, target="system")


def manufactured_program(*, periodic=True):
    case, program, temporal, mask, operator, authority = projected_case(
        periodic=periodic, with_expected_phi=True)
    builder, providers = authority
    space = operator.space
    storage = operator.attrs["scalar_storage"]
    load = program.field_sample(temporal.n, component="q", template=temporal.n,
                                storage=storage, space=space, name="scalar_load")
    zero = program.scalar_field("rhs_zero", like=load)
    rhs = program.where(mask, load, zero)
    from pops.linalg import LinearProblem, LinearOperatorProperties
    from pops.solvers import CG
    from pops.time import FailRun, FixedDt
    solution = program.solve(LinearProblem(operator, rhs, nullspace=None,
                    properties=LinearOperatorProperties.symmetric_positive_definite()),
                    solver=CG(max_iter=400, rel_tol=1e-11, abs_tol=1e-12)).consume(action=FailRun())
    registry = temporal.n.block._instance_registry
    model = registry.spec(temporal.n.block.local_id)["model"]
    module = model.module
    expected_aux = module.aux_handle(module.aux()["zz_expected_phi"])
    expected_phi = program.field_sample(expected_aux, template=temporal.n,
                                        storage=storage, space=space, name="expected_phi")
    error = program.value("solution_error", solution-expected_phi)
    program.record_scalar("scalar_solution_error_inf", program.norm_inf(error))
    program.record_scalar("scalar_solution_norm", program.norm_inf(solution))
    program.record_scalar("scalar_solution_l2", program.norm2(solution))
    program.record_scalar("sampled_mask_cells", program.sum(mask))
    program.commit(temporal.next, program.value("unchanged_carrier", temporal.n,
                                              at=temporal.next.point))
    program.step_strategy(FixedDt(.01))
    case.program(program)
    plan = builder.resolve(states=(case.resolve(temporal.n.state_ref),),
                           blocks=(case.resolve(temporal.n.block),))
    return case, plan, providers


def test_public_projected_scalar_program_resolves_its_exact_storage():
    case, layout, providers = manufactured_program()
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    assert any(node.op == "field_sample" for node in resolved.time._values)


def test_projected_scalar_codegen_uses_prepared_native_sample_and_private_capture():
    case, layout, providers = manufactured_program()
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks,
                                                   scalar_layout_plan=resolved.layout_plan)
    code = emit_cpp_program(resolved.time, model_graph=graph, target="system")
    assert "ctx.prepare_provider_values(" in code
    assert "provider_values_view<1>" in code and "providers(index, 0)" in code
    assert "ctx.alloc_shared_scalar_field_for_block(0, 1, 1)" in code
    assert "frozen_field_A" in code and "copy_allocated" in code
    assert "maskA(index, mc)" in code
    clone_phase = code.index("ctx_owner->prepare_operator_session_storage([&]()")
    boundary_phase = code.index("ctx_owner->prepare_mesh_boundary_session(", clone_phase)
    assert clone_phase < code.index("= std::make_shared<pops::Real>", clone_phase) < boundary_phase
    assert code.index("= std::make_shared<pops::MultiFab", clone_phase) < boundary_phase


def test_manufactured_reference_samples_a_nonzero_global_auxiliary_as_compact_slot_zero():
    case, layout, providers = manufactured_program()
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    reference = next(node for node in resolved.time._values
                     if node.op == "field_sample" and node.name == "expected_phi")
    from pops.model.provider_pack import ComponentKey, build_provider_pack
    model = reference.block._instance_registry.spec(reference.block.local_id)["model"]
    module = model.module
    auxiliary = reference.attrs["source"]
    declaration = auxiliary.declaration_ref or auxiliary
    key = ComponentKey(str(declaration.owner_path.canonical()), "aux",
                       declaration.local_id, declaration.local_id)
    pack = build_provider_pack(module).select((key,))
    assert len(pack) == 1 and pack.lookup(key).slot == 1
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks,
                                                   scalar_layout_plan=resolved.layout_plan)
    code = emit_cpp_program(resolved.time, model_graph=graph, target="system")
    sample_view = "sample%d" % reference.id
    sample_start = code.index("auto sample_stage_%d" % reference.id)
    sample_end = code.index("output(index, 0) = providers(index, 0);", sample_start)
    assert "provider_values_view<1>" in code[sample_start:sample_end]
    assert "const auto output = %s->fab(li).view();" % sample_view in code[sample_start:sample_end]


def test_compiled_detached_graph_emits_samples_without_live_authoring_registries():
    case, layout, providers = manufactured_program()
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_graph_lowering import emit_program_graph
    graph = build_program_model_graph(resolved)
    detached = detach_compiled_program(resolved.time)
    samples = [node for node in detached._values if node.op == "field_sample"]
    assert samples and all(node.block._instance_registry is None for node in samples)
    assert detached._ir_hash() == resolved.time._ir_hash()
    code = emit_program_graph(detached.to_graph(), lowering_program=detached,
                              model_graph=graph, target="system")
    assert "providers(index, 0)" in code and "scalar_solution_error_inf" in code


def test_compiled_sample_requires_resolved_layout_proof_not_just_matching_attributes():
    case, layout, providers = manufactured_program()
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    from pops.codegen.program_models import ProgramModelGraph
    from pops.time._program.detach import detach_compiled_program
    from pops.time.field_sampling import emitted_sample_contract
    detached = detach_compiled_program(resolved.time)
    sample = next(node for node in detached._values if node.op == "field_sample")
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    with pytest.raises(TypeError, match="resolved model graph and LayoutPlan"):
        emitted_sample_contract(sample, graph)


def test_manufactured_bootstrap_initializes_exact_declared_fields_without_pruning_inputs():
    case, layout, providers = manufactured_program()
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    import numpy as np
    from tests.python.integration.runtime.test_public_scalar_field_sample import initial_declared_fields
    mask = np.arange(16*16).reshape(16,16) % 3 == 0
    potential = np.zeros((16,16), dtype=np.float64)
    inputs = initial_declared_fields(resolved, mask, potential)
    assert len(inputs) == 2
    assert {key.space_kind for key in inputs} == {"field"}
    assert all(value.flags.c_contiguous and value.dtype == np.float64 for value in inputs.values())
    mask_input = next(value for key, value in inputs.items() if key.space_name == "mask")
    phi_input = next(value for key, value in inputs.items() if key.space_name == "phi")
    assert np.array_equal(mask_input, mask.astype(np.float64))
    assert np.array_equal(phi_input, potential)


def test_nonperiodic_projection_refuses_unqualified_field_boundary_authority():
    case, layout, providers = manufactured_program(periodic=False)
    with pytest.raises(NotImplementedError, match="conducting field boundary authority"):
        pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)


def test_scalar_sampling_refuses_a_same_shaped_foreign_layout():
    case, program, temporal, aux, spaces, bindings, authority = sample_case()
    builder, _ = authority
    # Shape equality is insufficient; even a same-shaped official LayoutHandle differs.
    other = builder.layout("other", authority[1][bindings[0].layout])
    wrong = FieldStorageBinding(bindings[0].unknowns, other, (0,1))
    sample = program.field_sample(aux, template=temporal.n, storage=wrong, space=spaces[0])
    assert sample.attrs["scalar_storage"].layout == other
    plan = builder.resolve(states=(case.resolve(temporal.n.state_ref),),
                           blocks=(case.resolve(temporal.n.block),))
    from pops.time.field_sampling import validate_resolved_scalar_storage
    with pytest.raises(ValueError, match="layout differs"):
        validate_resolved_scalar_storage(program, plan, target="system")


def test_scalar_sampling_refuses_wrong_support_axis_binding():
    _, program, temporal, aux, spaces, bindings, authority = sample_case()
    wrong = FieldStorageBinding(bindings[0].unknowns, bindings[0].layout, (1,0))
    program.field_sample(aux, template=temporal.n, storage=wrong, space=spaces[0])
    builder, _ = authority
    from pops.time.references import canonical_handle
    plan = builder.resolve(states=(canonical_handle(temporal.n.state_ref),),
                           blocks=(canonical_handle(temporal.n.block),))
    from pops.time.field_sampling import validate_resolved_scalar_storage
    with pytest.raises(ValueError, match="support-axis"):
        validate_resolved_scalar_storage(program, plan, target="system")


def test_bound_selection_refuses_a_mask_from_another_evaluation_point():
    _, program, _, mask, operator, _ = projected_case()
    later = program.value("later_mask", mask, at=program.stage("later", c=.5))
    with pytest.raises(ValueError, match="evaluation points"):
        program.where(later, operator.attrs["apply_in"], operator.attrs["apply_out"])
