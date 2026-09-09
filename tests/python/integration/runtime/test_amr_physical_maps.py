"""Public AMR moment/pullback binding with exact independent hierarchy authorities."""
from dataclasses import replace
from pathlib import Path
import pops
import pytest
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      ConflictPolicy, EqualityPolicy, Hysteresis, Tag, Buffer)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant
from pops.math import Const, ValueExpr, ddt, div
from pops.mesh import (AxisQuadrature, CartesianGrid, LayoutPlanBuilder, LayoutMappingOperation,
                       LayoutRepresentation, LayoutSynchronization, PhysicalSupportMap,
                       PeriodicAxes, native_physical_mapping)
from pops.model import PhysicalDimension, PhysicalSupport
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.projection import ConservativeCellAverage
from pops.params import RuntimeParam
from pops.time import FixedDt

UNIT = PhysicalDimension()
PHASE = PhysicalSupport((("velocity", "v-domain"), ("position", "x-domain")))
SPACE = PhysicalSupport((("position", "x-domain"),))
ROOT = Path(__file__).resolve().parents[4]


def resolve_amr_maps(directory, *, initial=2.):
    case = pops.Case("distinct adaptive moment and extension")
    program = pops.Program("one accepted AMR mapping cadence")
    phase = Rectangle("v,x", (-2, 0), (2, 1)).frame(Cartesian2D())
    space = Rectangle("x,hidden", (0, 0), (1, 1)).frame(Cartesian2D())
    threshold = case.param(RuntimeParam("refine", default=-100.))
    states, grids = {}, {}
    for name, frame, support, shape, ratio in (
            ("distribution", phase, PHASE, (4, 8), (2, 2)),
            ("moment", space, SPACE, (8, 1), (2, 1)),
            ("extension", phase, PHASE, (4, 8), (2, 2))):
        model = pops.Model(name, frame=frame)
        state = model.state("U", components=("a",), support=support,
                            units=(UNIT,), sampling="cell_average")
        flux = model.flux("zero", frame=frame, state=state,
                          components={axis: (0 * state[0],) for axis in frame.axes},
                          waves={axis: (Const(0),) for axis in frame.axes})
        rate = model.rate("retain", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        states[name] = block[state]
        case.numerics(numerics, block=block)
        temporal = program.state(block[state])
        factor = 3 if name == "moment" else 1
        value = program.value(name + " update", factor * temporal.n, at=temporal.next.point)
        program.commit(temporal.next, value)
        case.initials.add(InitialCondition(state=block[state], value=Constant((initial,)),
                                          projection=ConservativeCellAverage()))
        transfer = AMRTransfer()
        transfer.state(block[state], StateTransfer())
        grids[name] = AMR(
            grid=CartesianGrid(frame=frame, cells=shape, periodic=PeriodicAxes(frame.axes)),
            hierarchy=AMRHierarchy(max_levels=2, ratios=(ratio,)),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=0)),
                hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid.frozen(), transfer=transfer, execution=AMRExecution.synchronous())
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    case = pops.validate(case)
    subjects = case.layout_subjects()
    builder = LayoutPlanBuilder(case.owner_path.canonical())
    layouts = {name: builder.layout(name, layout.resolve_for_case(case.resolve))
               for name, layout in grids.items()}
    for block in subjects.blocks:
        builder.assign_block(block, layouts[block.local_id])
    resolved_states = {state.block_ref.local_id: state for state in subjects.states}
    for name, state in resolved_states.items():
        builder.assign_state(state, layouts[name])
    requirements = []
    for source, target, physical, sync in (
            ("distribution", "moment", PhysicalSupportMap(PHASE, SPACE,
                reductions=(AxisQuadrature(0, -2, 2, 4, UNIT, weights=(-3, -1, 1, 5)),)),
             LayoutSynchronization.BEFORE_STEP_V1),
            ("moment", "extension", PhysicalSupportMap(SPACE, PHASE),
             LayoutSynchronization.AFTER_SOURCE_STEP_V1)):
        requirement, = builder.require_mapping(layouts[source], layouts[target],
            source=resolved_states[source], target=resolved_states[target],
            source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            target_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            operation=LayoutMappingOperation(physical.operation_abi),
            synchronization=sync, physical_map=physical)
        requirements.append(requirement)
    providers = tuple(native_physical_mapping(row, directory) for row in requirements)
    layout = builder.resolve(**subjects.to_dict(), providers=providers)
    return pops.resolve(case, layout=layout,
        layout_providers={layouts[name]: grid.resolve_for_case(case.resolve)
                          for name, grid in grids.items()},
        components=tuple(provider.component for provider in providers),
        compile_options={"include": str(ROOT / "include")})


def test_amr_maps_keep_exact_distinct_hierarchy_authorities(tmp_path):
    plan = resolve_amr_maps(tmp_path)
    assert len(plan.layout_amr_authorities) == 3
    assert plan.resolved_hierarchy is None and plan.amr_transfer is None
    assert len(plan.initial_condition_plan.bindings) == 3
    for layout_id, row in plan.layout_amr_authorities.items():
        assert row.layout_id == layout_id
        assert len(row.layout_plan.layouts) == 1
        assert len(row.authorities.initial_conditions.bindings) == 1
        assert row.authorities.hierarchy.plan.level_count == 2
        assert row.authorities.transfer.layout_plan_id == row.layout_plan.qualified_id
        with pytest.raises(TypeError):
            row.authorities.providers["tagger"]["layout_identity"] = "other"
    first, second, third = plan.layout_amr_authorities.values()
    swapped = dict(plan.layout_amr_authorities)
    swapped[first.layout_id] = second
    with pytest.raises(ValueError, match="parent projection"):
        replace(plan, layout_amr_authorities=swapped)
    from pops.codegen._compiled_artifact import CompiledPlanRecord
    detached = CompiledPlanRecord.from_resolved(plan)
    detached.verify()
    assert detached.layout_amr_authorities == plan.layout_amr_authorities
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_slicing import slice_program
    for row in plan.layout_amr_authorities.values():
        names = tuple(assignment.subject.local_id for assignment in row.layout_plan.assignments
                      if assignment.subject_kind == "block")
        program = slice_program(plan.time, names)
        graph = ProgramModelGraph.from_resolved_blocks(tuple(block for block in plan.blocks if block.name in names))
        source = emit_cpp_program(program, model_graph=graph, target="amr_system")
        assert "ctx.advance_hierarchy(dt, _advance_level)" in source
        assert "ctx.commit_many(" in source



def test_layout_projection_routes_selected_program_parameters_only():
    from types import SimpleNamespace
    from pops.model.bind_schema import BindSchema
    from pops.runtime._install_param_routing import route_program_params
    from pops.runtime._layout_install_projection import LayoutCompiledArtifactProjection
    from tests.python.integration.runtime.test_program_runtime_params import _authoring
    model, parameter, case, block, program = _authoring(
        RuntimeParam("k", default=2.), name="amr-projected-parameter")
    schema = BindSchema.from_problem(case)
    values = schema.resolve_bind({block[parameter]: 7.}, compile_values=schema.resolve_compile())
    routes = ((0, "k", 0, 2.),)
    compiled_program = SimpleNamespace(program=program, program_param_routes=routes,
                                       program_block_routes=((0, "gas"),))
    selected = SimpleNamespace(program=compiled_program)
    aggregate = SimpleNamespace(program=None, program_param_routes=None, program_block_routes=())
    projection = LayoutCompiledArtifactProjection(aggregate, selected, None, ())
    assert projection.program_param_routes is routes
    assert projection.program_block_routes is compiled_program.program_block_routes
    assert route_program_params(projection, schema, values) == {0: [7.]}
    compiled_program.program_param_routes = ()
    assert route_program_params(projection, schema, {}) == {}


def _hierarchy_image(instance):
    import numpy as np
    return {(name, level): np.asarray(instance.block_level_state_global(name, level)).copy()
            for name in ("distribution", "moment", "extension")
            for level in range(instance._executor.executor_for_block(name).n_levels())}


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_amr_moment_extension_and_restart(tmp_path):
    import numpy as np
    from tests.python.support.native_execution_context import artifact_execution_context
    artifact = pops.compile(resolve_amr_maps(tmp_path))
    instance = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    for name in ("distribution", "moment", "extension"):
        assert instance._executor.executor_for_block(name).n_levels() == 2
    pops.run(instance, t_end=0.01, max_steps=1)
    for (name, level), value in _hierarchy_image(instance).items():
        expected = 2. if name == "distribution" else 12.
        np.testing.assert_array_equal(value, np.full_like(value, expected))
    assert set(instance._executor.mapping_report().values()) == {1}
    checkpoint = instance.checkpoint(tmp_path / "amr-map-checkpoint")
    pops.run(instance, t_end=0.02, max_steps=1)
    accepted = _hierarchy_image(instance)
    instance.restart(checkpoint)
    pops.run(instance, t_end=0.02, max_steps=1)
    for key, value in _hierarchy_image(instance).items():
        np.testing.assert_array_equal(value, accepted[key])
    assert set(instance._executor.mapping_report().values()) == {2}


@pytest.mark.compiler
@pytest.mark.native_loader
def test_amr_integral_failure_restores_all_hierarchies_and_retries(tmp_path):
    import numpy as np
    from tests.python.support.native_execution_context import artifact_execution_context
    artifact = pops.compile(resolve_amr_maps(tmp_path, initial=1e308))
    instance = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    before = _hierarchy_image(instance)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="[Ii]ntegral|[Tt]ransfer"):
            pops.run(instance, t_end=0.01, max_steps=1)
        assert instance.time() == 0. and instance.macro_step() == 0
        for key, value in _hierarchy_image(instance).items():
            np.testing.assert_array_equal(value, before[key])
        assert set(instance._executor.mapping_report().values()) == {0}


def test_amr_internal_maps_resolve_to_native_hierarchy_continuations(tmp_path):
    from tests.python.integration.runtime.test_interstage_physical_maps import resolve_interstage_maps
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_slicing import slice_program
    plan = resolve_interstage_maps(tmp_path, adaptive=True)
    assert len(plan.layout_amr_authorities) == 3
    for row in plan.layout_plan.layouts:
        names = tuple(assignment.subject.local_id for assignment in plan.layout_plan.assignments
                      if assignment.subject_kind == "block" and assignment.layout == row.handle)
        child = slice_program(plan.time, names)
        graph = ProgramModelGraph.from_resolved_blocks(tuple(block for block in plan.blocks if block.name in names))
        source = emit_cpp_program(child, model_graph=graph, target="amr_system")
        assert "ctx.advance_mapping_hierarchy(dt" in source
        assert "ctx.suspend_map(" in source
        assert "ctx.advance_hierarchy(dt, _advance_level)" not in source


def _internal_map_image(instance):
    import numpy as np
    return {(name, level): np.asarray(instance.block_level_state_global(name, level)).copy()
            for name in ("population", "integral", "extended")
            for level in range(instance._executor.executor_for_block(name).n_levels())}


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_amr_internal_maps_preserve_stages_and_restart(tmp_path):
    import numpy as np
    from tests.python.integration.runtime.test_interstage_physical_maps import resolve_interstage_maps
    from tests.python.support.native_execution_context import artifact_execution_context
    artifact = pops.compile(resolve_interstage_maps(tmp_path, adaptive=True))
    instance = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    assert len(_internal_map_image(instance)) == 6
    pops.run(instance, t_end=0.01, max_steps=1)
    factors = {"population": 10., "integral": 20., "extended": 12.}
    for (name, _), value in _internal_map_image(instance).items():
        expected = np.array((2., 3.))[:, None, None] * factors[name]
        np.testing.assert_array_equal(value, np.broadcast_to(expected, value.shape))
    assert sorted(instance._executor.mapping_report().values()) == [1, 2]
    checkpoint = instance.checkpoint(tmp_path / "amr-internal-stage-checkpoint")
    pops.run(instance, t_end=0.02, max_steps=1)
    accepted = _internal_map_image(instance)
    instance.restart(checkpoint)
    pops.run(instance, t_end=0.02, max_steps=1)
    for key, value in _internal_map_image(instance).items():
        np.testing.assert_array_equal(value, accepted[key])
    assert sorted(instance._executor.mapping_report().values()) == [2, 4]


@pytest.mark.compiler
@pytest.mark.native_loader
def test_amr_internal_map_failure_then_shorter_step_reuses_same_instance(tmp_path, record_property):
    import numpy as np
    from tests.python.integration.runtime.test_interstage_physical_maps import resolve_interstage_maps
    from tests.python.support.native_execution_context import artifact_execution_context
    artifact = pops.compile(resolve_interstage_maps(tmp_path, adaptive=True, retry_by_dt=True))
    instance = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    before = _internal_map_image(instance)
    # Reject the actual non-finite native transfer; serial source packing preserves
    # invalid_argument as ValueError, while collective transfer failures use RuntimeError.
    with pytest.raises((ValueError, RuntimeError), match=(
            "AMR active physical source is non-finite|"
            "physical transfer produced a non-finite value|"
            "AMR physical candidate is non-finite|"
            "AMR transfer (?:source packing|intersection integrals) failed on a lane rank")) as failed:
        pops.run(instance, t_end=0.01, max_steps=1)
    record_property("native_map_failure", str(failed.value))
    assert instance.time() == 0. and instance.macro_step() == 0
    for key, value in _internal_map_image(instance).items():
        np.testing.assert_array_equal(value, before[key])
    assert set(instance._executor.mapping_report().values()) == {0}
    # FixedDt clips the final step at this public run boundary. The smaller dt makes
    # the same authored intermediate states finite, exercising real failure→success.
    step = 1e-5
    pops.run(instance, t_end=step, max_steps=1)
    assert instance.time() == step and instance.macro_step() == 1
    scale = (200 * step) * 1e307
    factors = {"population": 5., "integral": 10., "extended": 6.}
    for (name, _), value in _internal_map_image(instance).items():
        expected = np.array((2., 3.))[:, None, None] * scale * factors[name]
        assert np.all(np.isfinite(value))
        np.testing.assert_allclose(value, np.broadcast_to(expected, value.shape), rtol=2e-14, atol=0.)
    assert sorted(instance._executor.mapping_report().values()) == [1, 2]


def test_amr_internal_maps_refuse_recursive_timing_before_compile(tmp_path):
    from pops.amr import AMRExecution
    from tests.python.integration.runtime.test_interstage_physical_maps import resolve_interstage_maps
    with pytest.raises(ValueError, match="qualified stage map.*synchronous.*subcycled"):
        resolve_interstage_maps(tmp_path, adaptive=True,
                               adaptive_execution=AMRExecution.subcycled())
