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
from pops.numerics.terms import Flux
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context

UNIT = PhysicalDimension()
PHASE = PhysicalSupport((("velocity", "velocity-interval"), ("position", "periodic-position")))
PHYSICAL = PhysicalSupport((("position", "periodic-position"),))
ROOT = Path(__file__).resolve().parents[4]


def resolve_generic_maps(directory, *, reverse=False):
    phase_frame = Rectangle("phase axes v then x", (-2, 0), (2, 1)).frame(Cartesian2D())
    physical_frame = Rectangle("physical x then hidden", (0, 0), (1, 1)).frame(Cartesian2D())
    case = pops.Case("independent weighted moments and explicit extension")
    program = pops.Program("two authored internal stages")
    states = {}
    for name, frame, support in (("population", phase_frame, PHASE),
                                  ("integral", physical_frame, PHYSICAL),
                                  ("weighted", physical_frame, PHYSICAL),
                                  ("extended", phase_frame, PHASE)):
        model = pops.Model(name + " model", frame=frame)
        state = model.state("U", components=("first", "second"), support=support,
                            units=(UNIT, UNIT), sampling="cell_average")
        flux = model.flux("zero_flux", frame=frame, state=state,
                          components={axis: tuple(0 * state[index] for index in range(2)) for axis in frame.axes},
                          waves={axis: (Const(0), Const(0)) for axis in frame.axes})
        rate = model.rate("retain", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerics, block=block)
        current = program.state(block[state])
        stage1 = program.stage(name + " first stage", c=Fraction(1, 3))
        first = program.value(name + " first predictor", current.n + Fraction(1, 3) * program.dt *
                              program.rhs(state=current.n, terms=[Flux()]), at=stage1)
        stage2 = program.stage(name + " second stage", c=Fraction(2, 3))
        second = program.value(name + " second predictor", current.n + Fraction(2, 3) * program.dt *
                               program.rhs(state=first, terms=[Flux()]), at=stage2)
        program.commit(current.next, program.value(name + " accepted", current.n + program.dt *
                       program.rhs(state=second, terms=[Flux()]), at=current.next.point))
        states[name] = block[state]
    program.step_strategy(FixedDt(0.01))
    case.program(program)
    validated = pops.validate(case)
    subjects = validated.layout_subjects()
    builder = LayoutPlanBuilder(validated.owner_path.canonical())
    phase_grid = Uniform(CartesianGrid(frame=phase_frame, cells=(4, 7), periodic=PeriodicAxes(phase_frame.axes)))
    physical_grid = Uniform(CartesianGrid(frame=physical_frame, cells=(7, 1), periodic=PeriodicAxes(physical_frame.axes)))
    layouts = {"population": builder.layout("source", phase_grid),
               "integral": builder.layout("moments", physical_grid),
               "extended": builder.layout("extension", phase_grid)}
    layouts["weighted"] = layouts["integral"]
    resolved_states = {row.block_ref.local_id: row for row in subjects.states}
    for block in subjects.blocks:
        builder.assign_block(block, layouts[block.local_id])
        builder.assign_state(resolved_states[block.local_id], layouts[block.local_id])
    requirements = []
    recipes = [("population", "integral", PhysicalSupportMap(PHASE, PHYSICAL,
                    reductions=(AxisQuadrature(0, -2, 2, 4, UNIT),)), LayoutSynchronization.BEFORE_STEP_V1),
               ("population", "weighted", PhysicalSupportMap(PHASE, PHYSICAL,
                    reductions=(AxisQuadrature(0, -2, 2, 4, UNIT, weights=(-3, -1, 1, 3)),)),
                    LayoutSynchronization.BEFORE_STEP_V1),
               ("weighted", "extended", PhysicalSupportMap(PHYSICAL, PHASE),
                    LayoutSynchronization.AFTER_SOURCE_STEP_V1)]
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
        layout_providers={layouts["population"]: phase_grid, layouts["integral"]: physical_grid,
                          layouts["extended"]: phase_grid},
        components=tuple(provider.component for provider in providers),
        compile_options={"include": str(ROOT / "include")})
    return resolved


def test_multiple_public_maps_resolve_without_field_solve_and_preserve_authored_stages(tmp_path):
    plan = resolve_generic_maps(tmp_path)
    assert len(plan.layout_plan.mappings) == 3
    assert len(plan.layout_plan.layouts) == 3
    assert plan.program_field_plans == {}
    for layout in plan.layout_plan.layouts:
        from pops.codegen.program_slicing import slice_program
        names = tuple(row.subject.local_id for row in plan.layout_plan.assignments
                      if row.subject_kind == "block" and row.layout == layout.handle)
        child = slice_program(plan.time, names)
        predictors = [value for value in child._values
                      if value.op == "linear_combine" and value.name.endswith("predictor")]
        assert len(predictors) == 2 * len(names)
        offsets = [value.point.to_data()["partitions"]["main"]["offset"] for value in predictors]
        assert offsets.count({"kind": "rational", "numerator": "1", "denominator": "3"}) == len(names)
        assert offsets.count({"kind": "rational", "numerator": "2", "denominator": "3"}) == len(names)


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("reverse", [False, True])
def test_native_weighted_axis_permutation_multiple_maps_and_stages(tmp_path, reverse):
    resolved = resolve_generic_maps(tmp_path, reverse=reverse)
    artifact = pops.compile(resolved)
    x = np.arange(7, dtype=float)[:, None]
    v = np.array((-3, -1, 1, 3), dtype=float)[None, :]
    population = np.stack((2 + x + v, -1 + 2 * x - 3 * v))
    initial = {"population": population, "integral": np.full((2, 1, 7), -11.0),
               "weighted": np.full((2, 1, 7), -13.0), "extended": np.full((2, 7, 4), -17.0)}
    instance = pops.bind(artifact, initial_state=initial,
                         resources={"execution_context": artifact_execution_context(artifact)})
    expected_integral = np.stack((8 + 4 * np.arange(7), -4 + 8 * np.arange(7)))[:, None, :]
    expected_weighted = np.broadcast_to(np.array((20.0, -60.0))[:, None, None], (2, 1, 7))
    expected_extension = np.broadcast_to(np.array((20.0, -60.0))[:, None, None], (2, 7, 4))
    for step in range(1, 4):
        pops.run(instance, t_end=step * 0.01, max_steps=1)
        np.testing.assert_allclose(instance.get_state("population"), population, rtol=0, atol=0)
        np.testing.assert_allclose(instance.get_state("integral"), expected_integral, rtol=0, atol=2e-13)
        np.testing.assert_allclose(instance.get_state("weighted"), expected_weighted, rtol=0, atol=2e-13)
        np.testing.assert_allclose(instance.get_state("extended"), expected_extension, rtol=0, atol=2e-13)
    assert set(instance._executor.mapping_report().values()) == {3}
