"""A solved-field observation authenticates its exact context and physical carrier."""
import pytest

import pops
from pops.domain import Rectangle
from pops.fields import CellCenteredSecondOrder, FieldDiscretization, FieldOutput, FieldStorageBinding, GradientOutput
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Dirichlet
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div, laplacian
from pops.mesh import CartesianGrid, LayoutPlanBuilder
from pops.model import FieldSpace, PhysicalDimension, PhysicalSupport
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.solvers.elliptic import CartesianCG
from pops.solvers.tolerances import Relative
from pops.time import FailRun


def field_sample_case(*, adaptive=False):
    frame = Rectangle("typed square", (0., 0.), (1., 1.)).frame(Cartesian2D())
    unit = PhysicalDimension()
    support = PhysicalSupport((("x", "typed_square"), ("y", "typed_square")))
    model = pops.Model("typed solved carrier", frame=frame)
    state = model.state("Q", components=("q",), units=(unit,), support=support, sampling="cell")
    field_space = FieldSpace("potential", ("potential", "grad_x", "grad_y"),
        units=(unit, unit, unit), support=support, sampling="cell", frame=frame.canonical_id)
    potential = model.field("potential", space=field_space)
    field_operator = model.field_operator("Gauss", unknown=potential,
        equation=-laplacian(potential) == state[0],
        outputs=(FieldOutput("potential", potential), GradientOutput("grad", potential)))
    flux = model.flux("stationary", frame=frame, state=state,
        components={axis: (0 * state[0],) for axis in frame.axes},
        waves={axis: (0 * state[0],) for axis in frame.axes})
    rate = model.rate("stationary", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    module = model.module
    observed = module.field_space("phi_copy", ("copy",), units=(unit,), support=support,
                                 sampling="cell", frame=frame.canonical_id)
    case = pops.Case("typed solve sample")
    block = case.block("carrier", model)
    case.numerics(numerics, block=block)
    from pops.solvers.elliptic import GeometricMG, RedBlackVcycle
    field = case.field(field_operator, FieldDiscretization(method=CellCenteredSecondOrder(),
        solver=GeometricMG(cycle=RedBlackVcycle(),tolerance=Relative(1e-10)) if adaptive else
            CartesianCG(tolerance=Relative(1e-10)),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.)),)))
    if adaptive:
        from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
            Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
        from pops.layouts import AMR
        from pops.params import RuntimeParam
        from pops.math import ValueExpr
        from pops.lib.amr import ConservativeLinear, StateTransfer
        threshold = case.param(RuntimeParam("refine density", default=.5))
        tagging = AMRTagging(rules=(Tag(ValueExpr(case.resolve(block[state]))["q"] >
            ValueExpr(case.resolve(threshold))), Buffer(cells=2, selection="refine-requests")),
            hysteresis=Hysteresis(0,EqualityPolicy.HOLD),conflict_policy=ConflictPolicy.REFINE_WINS)
        transfer = AMRTransfer()
        transfer.state(case.resolve(block[state]), StateTransfer(prolongation=ConservativeLinear(physical_faces="one-sided")))
        grid = AMR(grid=CartesianGrid(frame=frame,cells=(8,8)),
            hierarchy=AMRHierarchy(max_levels=2,ratios=(2,)),tagging=tagging,
            regrid=AMRRegrid.frozen(),transfer=transfer,execution=AMRExecution.synchronous())
    else:
        grid = Uniform(CartesianGrid(frame=frame, cells=(8,8)))
    builder = LayoutPlanBuilder(case.owner_path.canonical())
    layout = builder.layout("uniform", grid)
    builder.assign_state(case.resolve(block[state]), layout)
    builder.assign_block(case.resolve(block), layout)
    builder.assign_field(case.resolve(field), layout)
    storage = FieldStorageBinding((module.field_handle(observed),), layout, (0,1))
    program = pops.Program("typed solve sampling")
    temporal = program.state(block[state])
    fields = field(temporal.n).consume(action=FailRun())
    return case, program, temporal, fields, potential, observed, storage, (builder, {layout:grid})


def test_amr_solved_scalar_sample_uses_active_owner_scratch_and_qualified_native_ghosts():
    case,program,temporal,fields,potential,observed,storage,authority=field_sample_case(adaptive=True)
    sample=program.field_sample(fields,template=temporal.n,component=(potential,"potential"),
                                 storage=storage,space=observed)
    program.record_scalar("amr_field_sample_norm",program.norm_inf(sample))
    program.commit(temporal.next,program.value("unchanged",temporal.n,at=temporal.next.point))
    from pops.time import FixedDt
    program.step_strategy(FixedDt(.01))
    case.program(program)
    from pops.initial import InitialCondition
    from pops.lib.initial import Constant
    from pops.projection import ConservativeCellAverage
    case.initials.add(InitialCondition(state=temporal.n.state_ref,value=Constant((1.,)),
                                      projection=ConservativeCellAverage()))
    builder,providers=authority
    layout=builder.resolve(states=(case.resolve(temporal.n.state_ref),),
        fields=(case.resolve(fields.field_context.field),),blocks=(case.resolve(temporal.n.block),))
    resolved=pops.resolve(pops.validate(case),layout=layout,layout_providers=providers)
    from pops.codegen._orchestration_compile import build_program_model_graph
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_graph_lowering import emit_program_graph
    graph=build_program_model_graph(resolved)
    detached=detach_compiled_program(resolved.time)
    code=emit_program_graph(detached.to_graph(),lowering_program=detached,model_graph=graph,
                           target="amr_system",field_plans=resolved.field_plans)
    assert "ctx.scalar_scratch(" in code
    assert "ctx.copy_solved_field_scalar_sample(" in code
    assert "ctx.alloc_shared_scalar_field_for_block(" not in code


def test_solved_sample_retains_context_state_and_exact_component_key():
    _case, program, temporal, fields, potential, observed, storage, _ = field_sample_case()
    sample = program.field_sample(fields, template=temporal.n, component=(potential,"potential"),
                                  storage=storage, space=observed)
    assert sample.inputs == (temporal.n, fields)
    assert sample.field_context == fields.field_context
    assert sample.attrs["sample_kind"] == "field"
    assert sample.attrs["source"] == potential
    assert sample.state_ref is None
    assert sample.point == temporal.n.point


def test_solved_sample_refuses_another_ssa_state_at_the_same_point_atomically():
    _case, program, temporal, fields, potential, observed, storage, _ = field_sample_case()
    other = program.value("other state", temporal.n)
    before = program._ir_hash()
    with pytest.raises(ValueError, match="incompatible field context"):
        program.field_sample(fields, template=other, component=(potential,"potential"),
                             storage=storage, space=observed)
    assert program._ir_hash() == before


def test_solved_sample_refuses_a_homonymous_foreign_declaration():
    _case, program, temporal, fields, _potential, observed, storage, _ = field_sample_case()
    foreign = field_sample_case()[4]
    before = program._ir_hash()
    with pytest.raises((TypeError,ValueError), match="authentic|owner|declaration|registered|foreign"):
        program.field_sample(fields, template=temporal.n, component=(foreign,"potential"),
                             storage=storage, space=observed)
    assert program._ir_hash() == before


def test_typed_board_field_preserves_provider_and_carrier_metadata():
    case, _program, temporal, _fields, potential, observed, _storage, _ = field_sample_case()
    model = case.blocks()["carrier"]._instance_registry.spec("carrier")["model"]
    declared = model.field_spaces()[potential.local_id]
    carrier = model.module.field_spaces()["fields"]
    provider = model.module.operator_registry().get("Gauss").signature.output
    assert provider.components == declared.components == carrier.components
    for attribute in ("units", "support", "sampling", "frame", "clock"):
        assert getattr(declared, attribute) == getattr(provider, attribute)
        assert getattr(declared, attribute) == getattr(carrier, attribute)
    assert observed.frame == temporal.n.space.frame == carrier.frame


def test_solved_sample_resolves_and_emits_its_compact_provider_slot():
    case, program, temporal, fields, potential, observed, storage, authority = field_sample_case()
    sample = program.field_sample(fields, template=temporal.n, component=(potential,"potential"),
                                  storage=storage, space=observed)
    program.record_scalar("sample_norm", program.norm_inf(sample))
    program.commit(temporal.next, program.value("unchanged", temporal.n, at=temporal.next.point))
    from pops.time import FixedDt
    program.step_strategy(FixedDt(.01))
    case.program(program)
    builder, providers = authority
    layout = builder.resolve(states=(case.resolve(temporal.n.state_ref),),
                             fields=(case.resolve(fields.field_context.field),),
                             blocks=(case.resolve(temporal.n.block),))
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks,
                                                   scalar_layout_plan=resolved.layout_plan)
    from pops.time._program.detach import detach_compiled_program
    detached = detach_compiled_program(resolved.time)
    assert detached._ir_hash() == resolved.time._ir_hash()
    assert all(node.block._instance_registry is None for node in detached._values if node.block)
    code = emit_cpp_program(detached, model_graph=graph, target="system",
                            field_plans=resolved.field_plans)
    assert "ctx.solve_fields_from_state_at(" in code
    assert "ctx.prepare_provider_values(" in code
    assert "provider_values_view<1>" in code and "providers(index, 0)" in code


def test_declared_field_component_projection_needs_no_second_physical_field():
    _case, program, temporal, fields, potential, observed, storage, _ = field_sample_case()
    projected = FieldSpace("potential", ("potential",), units=observed.units,
        support=observed.support, sampling=observed.sampling, frame=observed.frame,
        clock=observed.clock)
    binding = FieldStorageBinding((potential,), storage.layout, storage.observation_axes)
    sample = program.field_sample(fields, template=temporal.n, component=(potential,"potential"),
                                  storage=binding, space=projected)
    assert sample.attrs["scalar_storage_component"] == (potential, "potential")
    scratch = program.scalar_field(like=sample)
    assert scratch.attrs["scalar_storage_component"] == (potential, "potential")


def test_consumed_scalar_solve_can_seed_exact_native_field_without_output_context():
    case, program, temporal, fields, potential, observed, storage, authority = field_sample_case()
    projected = FieldSpace("potential", ("potential",), units=observed.units,
        support=observed.support, sampling=observed.sampling, frame=observed.frame,
        clock=observed.clock)
    binding = FieldStorageBinding((potential,), storage.layout, storage.observation_axes)
    sample = program.field_sample(fields, template=temporal.n, component=(potential,"potential"),
                                  storage=binding, space=projected)
    operator = program.matrix_free_operator("identity", template=temporal.n,
                                             storage=binding, space=projected)
    program.set_apply(operator, lambda _p, _out, phi: phi)
    from pops.linalg import LinearProblem, LinearOperatorProperties
    from pops.solvers import CG
    solution = program.solve(LinearProblem(operator, sample, nullspace=None,
        properties=LinearOperatorProperties.symmetric_positive_definite()),
        solver=CG(max_iter=20)).consume(action=FailRun())
    extrapolated = program.value("extrapolated", 2*solution-sample)
    target = program.value("after source", temporal.n)
    program.publish_field_warm_start(fields.field_context.field, extrapolated,
                                     fields=fields, template=temporal.n, state=target)
    warm = program._values[-1]
    assert warm.op == "field_warm_start" and warm.field_context is None
    assert warm.inputs == (extrapolated, fields, temporal.n, target)
    before = program._ir_hash()
    with pytest.raises(ValueError, match="incompatible field context"):
        program.publish_field_warm_start(fields.field_context.field, extrapolated,
                                         fields=fields, template=target, state=target)
    assert program._ir_hash() == before
    program.record_scalar("sample_norm", program.norm_inf(sample))
    program.commit(temporal.next, program.value("unchanged", temporal.n, at=temporal.next.point))
    from pops.time import FixedDt
    program.step_strategy(FixedDt(.01))
    case.program(program)
    builder, providers = authority
    layout = builder.resolve(states=(case.resolve(temporal.n.state_ref),),
                             fields=(case.resolve(fields.field_context.field),),
                             blocks=(case.resolve(temporal.n.block),))
    resolved = pops.resolve(pops.validate(case), layout=layout, layout_providers=providers)
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks,
                                                   scalar_layout_plan=resolved.layout_plan)
    from pops.time._program.detach import detach_compiled_program
    detached = detach_compiled_program(resolved.time)
    assert detached._ir_hash() == resolved.time._ir_hash()
    code = emit_cpp_program(detached, model_graph=graph, target="system",
                            field_plans=resolved.field_plans)
    assert "ctx.publish_field_warm_start(" in code
    solve_call = code.index("ctx.solve_prepared_linear(")
    failure_guard = code.index("pops::SolveConsumption::kFailRun", solve_call)
    assert solve_call < failure_guard < code.index("ctx.publish_field_warm_start(")
