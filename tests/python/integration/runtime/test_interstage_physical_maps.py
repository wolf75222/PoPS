"""Public multi-map Program, explicit weighted axes and native provider execution."""
from fractions import Fraction
from pathlib import Path
import numpy as np
import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.model import PhysicalDimension, PhysicalSupport
from pops.math import Const, ddt, div
from pops.layouts import Uniform
from pops.mesh import (AxisQuadrature, CartesianGrid, LayoutPlanBuilder, LayoutMappingOperation,
    LayoutRepresentation, LayoutSynchronization, PeriodicAxes, PhysicalSupportMap, native_physical_mapping)
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context

UNIT = PhysicalDimension()
PHASE = PhysicalSupport((("velocity", "velocity-interval"), ("position", "periodic-position")))
PHYSICAL = PhysicalSupport((("position", "periodic-position"),))
ROOT = Path(__file__).resolve().parents[4]


def resolve_interstage_maps(directory, *, reverse=False, adaptive=False, retry_by_dt=False,
                            adaptive_execution=None, with_fields=False):
    phase_frame = Rectangle("phase axes v then x", (-2, 0), (2, 1)).frame(Cartesian2D())
    physical_frame = Rectangle("physical x then hidden", (0, 0), (1, 1)).frame(Cartesian2D())
    case = pops.Case("independent weighted moments and explicit extension")
    program = pops.Program("intermediate moment twice and pullback")
    states = {}
    block_states = {}
    physical_field_inputs = None
    if adaptive:
        from pops.params import RuntimeParam
        threshold = case.param(RuntimeParam("refine", default=-100.))
    for name, frame, support in (("population", phase_frame, PHASE),
                                  ("integral", physical_frame, PHYSICAL),
                                  ("extended", phase_frame, PHASE)):
        model = pops.Model(name + " model", frame=frame)
        components = ("first", "second")
        state = model.state("U", components=components, support=support,
                            units=(UNIT,) * len(components), sampling="cell_average")
        flux = model.flux("zero_flux", frame=frame, state=state,
                          components={axis: tuple(0 * state[index] for index in range(len(components))) for axis in frame.axes},
                          waves={axis: (Const(0),) * len(components) for axis in frame.axes})
        if name == "integral" and with_fields:
            force = tuple(model.aux("force_" + component) for component in components)
            source = model.source("field_force", on=state, value=force)
            rate = model.rate("retain", equation=ddt(state) == -div(flux) + source)
        else:
            rate = model.rate("retain", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerics, block=block)
        states[name] = program.state(block[state])
        block_states[name] = block[state]
        if name == "integral" and with_fields:
            physical_field_inputs = (model, state, block, source)
        if adaptive:
            from pops.initial import InitialCondition
            from pops.lib.initial import Constant
            from pops.projection import ConservativeCellAverage
            case.initials.add(InitialCondition(state=block[state], value=Constant((2., 3.)),
                                               projection=ConservativeCellAverage()))
    reduction = PhysicalSupportMap(PHASE, PHYSICAL,
        reductions=(AxisQuadrature(0, -2, 2, 4, UNIT, weights=(-3, -1, 1, 5 if adaptive else 3)),))
    extension = PhysicalSupportMap(PHYSICAL, PHASE)
    if with_fields:
        from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
        from pops.fields.methods import CellCenteredSecondOrder
        from pops.math import Reaction, laplacian
        from pops.model import Handle, OwnerPath
        from pops.solvers import CompositeFieldGMRES
        model, state, block, source = physical_field_inputs
        unknowns = tuple(Handle("potential_" + name, kind="field", owner=OwnerPath.model("mapped fields"))
                         for name in ("first", "second"))
        problem = FieldProblem("two mapped Helmholtz fields", unknowns=unknowns,
            equations=tuple(-laplacian(unknown) + Reaction(unknown, index + 1) == state[index]
                            for index, unknown in enumerate(unknowns)),
            boundaries=tuple(FieldBoundary(unknown, bcs.BoundaryCondition(
                bcs.AllPhysicalBoundaries(), bcs.Periodic())) for unknown in unknowns))
        field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
            solver=CompositeFieldGMRES(max_iter=200, restart=20, rel_tol=1e-12, abs_tol=1e-13)))

    def consume_fields(moment, name):
        if not with_fields:
            return moment
        from pops.numerics.terms import SourceTerm
        from pops.time import FailRun
        model, state, block, source = physical_field_inputs
        observed = field.observe(program.solve(field, values={block[state]: moment}, at=moment.point)
                                 .consume(action=FailRun()))
        module = model.module
        carrier = block[module.field_handle(module.field_spaces()["fields"])]
        context = observed.publish({(carrier, "force_" + component): observed[field[unknown]]
                                    for component, unknown in zip(("first", "second"), unknowns, strict=True)})
        rhs = program.rhs(state=moment, fields=context,
                          terms=[SourceTerm(block[module.operator_handle("field_force")])])
        driven = program.value(name, moment + program.dt * rhs, at=moment.point)
        program.store_history(name + " history", driven, depth=1)
        return driven
    first = states["population"].stage("half", point=program.stage("half", c=Fraction(1, 2)))
    scale = (200 * program.dt) * 1e307 if retry_by_dt and not with_fields else 2
    program.value(first, scale * states["population"].n)
    moment = states["integral"].stage("half moment", point=program.stage("half moment", c=Fraction(1, 2)))
    imported_moment = program.map(reduction, source=first, target=moment)
    driven_moment = consume_fields(imported_moment, "half driven moment")
    transfer_scale = (200 * program.dt) * 1e307 if retry_by_dt and with_fields else 1
    transformed = program.value("three times intermediate moment", (3 * transfer_scale) * driven_moment,
                                at=moment.point)
    back = states["extended"].stage("half extension", point=program.stage("half extension", c=Fraction(1, 2)))
    program.map(extension, source=transformed, target=back)
    late = states["population"].stage("late", point=program.stage("late", c=Fraction(3, 4)))
    program.value(late, 5 * first)
    final_moment = states["integral"].stage("late moment", point=program.stage("late moment", c=Fraction(3, 4)))
    imported_final_moment = program.map(reduction, source=late, target=final_moment)
    driven_final_moment = consume_fields(imported_final_moment, "late driven moment")
    for name, value in (("population", late), ("integral", driven_final_moment), ("extended", back)):
        program.commit(states[name].next, program.value(name + " accepted", value, at=states[name].next.point))
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    validated = pops.validate(case)
    subjects = validated.layout_subjects()
    builder = LayoutPlanBuilder(validated.owner_path.canonical())
    position_cells = 8 if adaptive else 7
    phase_mesh = CartesianGrid(frame=phase_frame, cells=(4, position_cells), periodic=PeriodicAxes(phase_frame.axes))
    physical_mesh = CartesianGrid(frame=physical_frame, cells=(position_cells, 1), periodic=PeriodicAxes(physical_frame.axes))
    phase_grid, physical_grid = Uniform(phase_mesh), Uniform(physical_mesh)
    descriptors = {"population": phase_grid, "integral": physical_grid, "extended": phase_grid}
    if adaptive:
        from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                              ConflictPolicy, EqualityPolicy, Hysteresis, Tag, Buffer)
        from pops.layouts import AMR
        from pops.lib.amr import StateTransfer
        from pops.math import ValueExpr
        for name, _grid in tuple(descriptors.items()):
            transfer = AMRTransfer()
            transfer.state(block_states[name], StateTransfer())
            descriptors[name] = AMR(grid=physical_mesh if name == "integral" else phase_mesh,
                hierarchy=AMRHierarchy(max_levels=2, ratios=((2, 1) if name == "integral" else (2, 2),)),
                tagging=AMRTagging(rules=(Tag(ValueExpr(block_states[name])["first"] > case.value(threshold)), Buffer(cells=0)),
                    hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
                regrid=AMRRegrid.frozen(), transfer=transfer,
                execution=adaptive_execution or AMRExecution.synchronous()).resolve_for_case(validated.resolve)
    layouts = {name: builder.layout(name, descriptor) for name, descriptor in descriptors.items()}
    resolved_states = {row.block_ref.local_id: row for row in subjects.states}
    for block in subjects.blocks:
        builder.assign_block(block, layouts[block.local_id])
        builder.assign_state(resolved_states[block.local_id], layouts[block.local_id])
    for field_subject in subjects.fields:
        builder.assign_field(field_subject, layouts["integral"])
    requirements = []
    recipes = [("population", "integral", reduction, LayoutSynchronization.PROGRAM_POINT_V1),
               ("integral", "extended", extension, LayoutSynchronization.PROGRAM_POINT_V1)]
    for source, target, physical_map, synchronization in reversed(recipes) if reverse else recipes:
        requirement, = builder.require_mapping(layouts[source], layouts[target],
            source=resolved_states[source], target=resolved_states[target],
            source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            target_representation=LayoutRepresentation.CELL_AVERAGE_V1,
            operation=LayoutMappingOperation(physical_map.operation_abi), synchronization=synchronization,
            physical_map=physical_map)
        requirements.append(requirement)
    providers = tuple(native_physical_mapping(row, directory) for row in requirements)
    layout = builder.resolve(**subjects.to_dict(), providers=providers)
    resolved = pops.resolve(validated, layout=layout,
        layout_providers={layouts[name]: descriptor for name, descriptor in descriptors.items()},
        components=tuple(provider.component for provider in providers),
        compile_options={"include": str(ROOT / "include")})
    return resolved


def test_interstage_maps_resolve_and_emit_native_continuations(tmp_path):
    resolved = resolve_interstage_maps(tmp_path)
    from pops.codegen.program_mapping_regions import resolve_program_map_invocations
    from pops.time.references import canonical_handle
    calls = resolve_program_map_invocations(resolved.time, resolved.layout_plan,
                                            resolve=canonical_handle)
    assert len(calls) == 3
    assert len({row.requirement.qualified_id for row in calls}) == 2
    from pops.codegen.program_slicing import slice_program
    compiled_partitions = []
    for layout in resolved.layout_plan.layouts:
        names = tuple(row.subject.local_id for row in resolved.layout_plan.assignments
                      if row.subject_kind == "block" and row.layout == layout.handle)
        child = slice_program(resolved.time, names)
        assert any(node.op.startswith("layout_map_") for node in child._values)
        from pops.codegen.program_models import ProgramModelGraph
        from pops.codegen.program_codegen import emit_cpp_program
        from pops.time._program.detach import detach_compiled_program
        graph = ProgramModelGraph.from_resolved_blocks(tuple(row for row in resolved.blocks if row.name in names))
        detached = detach_compiled_program(child)
        compiled_partitions.append(detached)
        for authored in (child, detached):
            source = emit_cpp_program(authored, model_graph=graph, target="system")
            assert source.count("ctx.suspend_map(") == sum(node.op.startswith("layout_map_") for node in child._values)
            assert source.rindex("ctx.commit_many(") > source.rindex("ctx.suspend_map(")
            assert source.count("ctx.begin_step(dt)") == 1
            amr_source = emit_cpp_program(authored, model_graph=graph, target="amr_system")
            assert "ctx.advance_mapping_hierarchy(dt" in amr_source
            assert amr_source.count("ctx.suspend_map(") == sum(
                node.op.startswith("layout_map_") for node in child._values)
    from types import SimpleNamespace
    from pops.codegen.program_mapping_regions import compiled_program_map_invocations
    artifact = SimpleNamespace(layout_plan=resolved.layout_plan, layout_programs=tuple(
        SimpleNamespace(program=SimpleNamespace(program=program)) for program in compiled_partitions))
    installed_calls = compiled_program_map_invocations(artifact)
    assert {row.identity for row in installed_calls} == {row.identity for row in calls}



@pytest.mark.native_loader
def test_native_intermediate_maps_restart_and_repeated_calls(tmp_path):
    artifact = pops.compile(resolve_interstage_maps(tmp_path))
    x = np.arange(7, dtype=float)[:, None]
    v = np.array((-3, -1, 1, 3), dtype=float)[None, :]
    population = np.stack((2 + x + v, -1 + 2 * x - 3 * v))
    instance = pops.bind(artifact, initial_state={"population": population,
        "integral": np.full((2, 1, 7), -13.0), "extended": np.full((2, 7, 4), -17.0)},
        resources={"execution_context": artifact_execution_context(artifact)})
    moment = np.broadcast_to(np.array((20.0, -60.0))[:, None, None], (2, 1, 7))
    extension = np.broadcast_to(np.array((20.0, -60.0))[:, None, None], (2, 7, 4))
    pops.run(instance, t_end=0.01, max_steps=1)
    np.testing.assert_array_equal(instance.get_state("population"), 10 * population)
    np.testing.assert_array_equal(instance.get_state("integral"), 10 * moment)
    np.testing.assert_array_equal(instance.get_state("extended"), 6 * extension)
    checkpoint = instance.checkpoint(tmp_path / "stage-map-checkpoint")
    pops.run(instance, t_end=0.02, max_steps=1)
    expected = {name: instance.get_state(name).copy() for name in ("population", "integral", "extended")}
    instance.restart(checkpoint)
    pops.run(instance, t_end=0.02, max_steps=1)
    for name, values in expected.items():
        np.testing.assert_array_equal(instance.get_state(name), values)
    assert sorted(instance._executor.mapping_report().values()) == [2, 4]


@pytest.mark.native_loader
def test_failed_intermediate_map_rolls_back_all_layouts_and_clears_continuations(tmp_path):
    artifact = pops.compile(resolve_interstage_maps(tmp_path))
    initial = {"population": np.full((2, 7, 4), 1e308),
               "integral": np.full((2, 1, 7), -13.0),
               "extended": np.full((2, 7, 4), -17.0)}
    instance = pops.bind(artifact, initial_state=initial,
        resources={"execution_context": artifact_execution_context(artifact)})
    for _ in range(2):
        with pytest.raises(RuntimeError, match="finite|Transfer"):
            pops.run(instance, t_end=0.01, max_steps=1)
        for name, values in initial.items():
            np.testing.assert_array_equal(instance.get_state(name), values)
        assert instance.time() == 0.0
        assert set(instance._executor.mapping_report().values()) == {0}
