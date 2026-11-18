"""A vector state's selected component remains qualified through a refinement region.

These cases ran on the tag-graph VM until N12 lot 3c-2b removed it; they now write the same value
and gradient indicators as regions of ``cell`` (``cell.value`` and ``gradient_norm``).
"""
from __future__ import annotations

import pops
import pytest

from pops.amr import (
    AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging,
    AMRTransfer, Buffer, Coarsen, ConflictPolicy, EqualityPolicy, Hysteresis, Tag,
)
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer, gradient_norm
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every


def _case(*, component="mx", gradient=False, layout_frame=None,
          components=("rho", "mx", "my"), units=None, support=None, sampling="unspecified"):
    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("U", components=components, units=units,
                        support=support, sampling=sampling)
    flux = model.flux("advection", frame=frame, state=state,
                      components={axis: tuple(q for q in state) for axis in frame.axes},
                      waves={axis: (1.,) * len(components) for axis in frame.axes})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    case = pops.Case("vector_tags")
    block = case.block("fluid", model)
    subject = block[state]
    numerics = DiscretizationPlan()
    method = FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    numerics.rates.add(rate, method)
    case.numerics(numerics, block=block)
    program = pops.Program("advance")
    q = program.state(subject)
    updated = program.value("updated", q.n + program.dt * rate(q.n), at=q.next.point)
    program.commit(q.next, updated)
    program.step_strategy(FixedDt(.001))
    case.program(program)
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                      projection=ConservativeCellAverage()))
    high = case.param(RuntimeParam("high", default=1.2))
    low = case.param(RuntimeParam("low", default=.8))
    value = ValueExpr(subject)
    if component is not None:
        value = value[component]
    stencil = method.amr_indicator_stencil(dimension=2)

    def indicator(cell):
        if gradient:
            return gradient_norm(cell, value, axes=frame.axes, stencil=stencil)
        return cell.value(value)

    transfer = AMRTransfer()
    transfer.state(subject, StateTransfer())
    grid_frame = frame if layout_frame is None else layout_frame
    layout = AMR(
        grid=CartesianGrid(frame=grid_frame, cells=(16, 16),
                           periodic=PeriodicAxes(grid_frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(lambda cell: indicator(cell) > case.value(high)),
                   Coarsen(lambda cell: indicator(cell) < case.value(low)),
                   Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)))
    return case, subject, high, low, layout


@pytest.mark.parametrize("gradient", [False, True])
@pytest.mark.parametrize("component,expected", [("mx", (1, "mx")), (2, (2, "my"))])
def test_selected_component_reaches_the_resolved_criterion_and_its_native_tables(
        gradient, component, expected):
    from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

    case, subject, high, low, layout = _case(component=component, gradient=gradient)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    criterion = resolved.bootstrap_plan.tagging
    params = {case.resolve(high): 1.2, case.resolve(low): .8}
    data = criterion.runtime_criterion_data(params)
    index, name = expected
    # One field: the selected component, read at the cell or over the gradient stencil.
    (row,) = data["fields"]
    assert row["state"] == case.resolve(subject).qualified_id
    assert (row["block"], row["variables"], row["components"]) == ("fluid", [name], [index])
    assert row["reach"] == ([1, 1] if gradient else [0, 0])
    assert data["parameters"] == [1.2, .8]

    class NativeProbe:
        def _set_generated_tagging(self, *args):
            self.args = args

    native = NativeProbe()
    flow_bootstrap_tagging(native, resolved.bootstrap_plan, params, clock_identity="case::clock")
    assert native.args[0] == criterion.qualified_id
    assert native.args[1:7] == ([case.resolve(subject).qualified_id], ["fluid"], [[name]],
                                [[index]], [[1, 1] if gradient else [0, 0]], [1.2, .8])


@pytest.mark.parametrize("gradient", [False, True])
def test_two_component_state_with_full_physical_type_resolves_its_selected_indicator(gradient):
    from pops._ir.quantity import PhysicalDimension, PhysicalSupport

    case, subject, high, low, layout = _case(
        component="second", gradient=gradient, components=("first", "second"),
        units=(PhysicalDimension(()),) * 2,
        support=PhysicalSupport((("x", "position"), ("y", "position"))),
        sampling="cell_average")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    data = resolved.bootstrap_plan.tagging.runtime_criterion_data(
        {case.resolve(high): 1.2, case.resolve(low): .8})
    (row,) = data["fields"]
    assert (row["variables"], row["components"]) == (["second"], [1])
    assert row["state"] == case.resolve(subject).qualified_id


def test_component_selection_changes_the_authenticated_criterion_identity():
    criteria = []
    for component in ("rho", "my"):
        case, _, _, _, layout = _case(component=component)
        criteria.append(pops.resolve(pops.validate(case), layout=layout).bootstrap_plan.tagging)
    assert criteria[0].qualified_id != criteria[1].qualified_id


@pytest.mark.parametrize("gradient", [False, True])
def test_vector_indicator_requires_explicit_selection(gradient):
    # The region is traced where the rule is written, so the refusal comes from there.
    with pytest.raises(TypeError, match="one typed component of a state"):
        _case(component=None, gradient=gradient)


@pytest.mark.parametrize("component,error", [
    ("missing", ValueError), (-1, IndexError), (3, IndexError),
    (True, TypeError), (1.0, TypeError), (slice(None), TypeError),
])
def test_component_selector_rejects_invalid_or_implicit_coercions(component, error):
    with pytest.raises(error, match="component"):
        _case(component=component)


@pytest.mark.parametrize("gradient", [False, True])
def test_selected_component_keeps_physical_frame_qualification(gradient):
    foreign = Rectangle("foreign", lower=(0., 0.), upper=(2., 1.)).frame(Cartesian2D())
    case, _, _, _, layout = _case(layout_frame=foreign, gradient=gradient)
    with pytest.raises(ValueError, match="frame"):
        pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_tagging_reads_the_selected_last_component(
    isolated_native_cache, native_cxx, kokkos_root,
):
    """Moving the same profile to an unselected component must remove every fine patch."""
    import numpy as np
    from tests.python.support.native_execution_context import artifact_execution_context

    del isolated_native_cache, native_cxx, kokkos_root
    case, _, _, _, layout = _case(component="my")
    resolved = pops.resolve(pops.validate(case), layout=layout)
    artifact = pops.compile(resolved)
    (binding,) = resolved.initial_condition_plan.bindings
    coordinates = (np.arange(16) + .5) / 16
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    profile = 2. * np.exp(-((x - .4) ** 2 + (y - .6) ** 2) / .012)
    assert np.max(profile) > 1.2
    fine_cells = []
    for populated_component in (2, 0):
        initial = np.zeros((3, 16, 16))
        initial[populated_component] = profile
        runtime = pops.bind(
            artifact, initial_values={binding.subject: initial},
            resources={"execution_context": artifact_execution_context(artifact)})
        fine_cells.append(sum(
            int(np.prod(np.subtract(upper, lower) + 1))
            for level, lower, upper in runtime.patch_boxes() if int(level) == 1))
    assert 0 < fine_cells[0] < 32 * 32
    assert fine_cells[1] == 0
