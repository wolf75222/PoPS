"""CF15: two real generated rates require affine oriented Cartesian ghost reads."""

from fractions import Fraction

import numpy as np
import pytest

import pops
import pops.surface as surface
from examples.api_v02.common import state_array
from examples.api_v02.models_M01_M14 import _euler, _initial, _traced_ssprk2, rusanov
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.numerics import upwind
from pops.lib.time.methods import ssprk2
from pops.mesh import CartesianGrid, NativeSpatialLayout, PeriodicAxes
from pops.mesh.boundaries import (
    BoundaryHandle,
    BoundaryOrientation,
    BoundarySide,
    BoundaryTopology,
    PeriodicIdentification,
    PeriodicOrientation,
)
from pops.model import OwnerPath
from pops.numerics import DiscretizationPlan
from pops.physics import Density, Energy, Momentum
from pops.surface.math import ddt, div, vector
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.g09_oriented_halo_oracles import (
    CELLS,
    EULER_DT,
    GAMMA,
    P_INF,
    TWIST_DT,
    euler_initial,
    euler_step,
    self_check,
    twisted_initial,
    twisted_step,
)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def build_g09_twisted_transport(*, identified=True, shift=Fraction(1, 4)):
    if identified:
        box = surface.Box(x=(2, 3), y=surface.Periodic(-2, -1))
        domain = box.identify(box.boundary.x_min, box.boundary.x_max,
                              by={box.y: box.y + shift})
    else:
        domain = surface.Box(x=surface.Periodic(2, 3), y=surface.Periodic(-2, -1))
    model = surface.Model("offset_scalar_transport")
    tracer = model.quantity("u", surface.Scalar(domain))
    data = model.data("impulse", surface.Scalar(domain))
    flux = model.flux(of=tracer, value=vector(-1.0, 0.0) * tracer)
    model.balance(ddt(tracer) == -div(flux))
    numerics = surface.Numerics(represent={tracer: surface.CellAverage()}, schemes={flux: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=TWIST_DT),
                              initial={tracer: data})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=(CELLS, CELLS))})
    return case, tracer, data


def _quarter_topology():
    owner = OwnerPath.case("rotated_seam_Euler")
    west, east, south, north = (
        BoundaryHandle(name, owner=owner, orientation=BoundaryOrientation(axis, side))
        for name, axis, side in (
            ("x_min", 0, BoundarySide.LOWER), ("x_max", 0, BoundarySide.UPPER),
            ("y_min", 1, BoundarySide.LOWER), ("y_max", 1, BoundarySide.UPPER),
        )
    )
    return BoundaryTopology(owner, (west, east, south, north), (
        PeriodicIdentification(south, east, PeriodicOrientation((1, 0), (-1, 1))),
        PeriodicIdentification(west, north, PeriodicOrientation((1, 0), (1, -1))),
    ), ())


def build_g09_quarter_euler(*, identified=True):
    """The M03 constitutive law and method with both moment roles stated explicitly."""
    frame = Rectangle("offset_euler", lower=(2.0, -2.0), upper=(3.0, -1.0)).frame(Cartesian2D())
    model = pops.Model("offset_stiffened_Euler", frame=frame)
    x, y = frame.axes
    conserved = model.state("conserved", components=("rho", "mx", "my", "energy"), roles={
        "rho": Density(), "mx": Momentum(x), "my": Momentum(y), "energy": Energy(),
    })
    state, flux, rate, _ = _euler(
        model, frame, gamma=GAMMA, pressure_offset=P_INF, state=conserved)
    case = pops.Case("rotated_seam_Euler")
    block = case.block("fluid", model)
    discretization = DiscretizationPlan()
    discretization.rates.add(rate, rusanov(flux, state))
    case.numerics(discretization, block=block)
    _traced_ssprk2(case, block, state, rate, EULER_DT, name="Euler_SSPRK2")
    values = _initial(case, block[state], euler_initial())
    if identified:
        grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), identifications=_quarter_topology())
    else:
        grid = CartesianGrid(frame=frame, cells=(CELLS, CELLS), periodic=PeriodicAxes(frame.axes))
    return case, Uniform(grid), values


def test_shifted_scalar_consumes_the_wrapped_tangential_trace(tmp_path):
    case, tracer, datum = build_g09_twisted_transport()
    native, _ = native_profile(dimension=2, scope="G09 shifted scalar transport")
    config = native.SystemConfig()
    (row,) = case.elaborated.layout.mesh.native_spatial_data()["identifications"]
    config.identifications = (tuple(row),)
    assert config.identifications == (tuple(row),)
    with pytest.raises(TypeError, match="entries must be exact integers"):
        config.identifications = ((True, *row[1:]),)
    assert config.identifications == (tuple(row),)
    artifact = surface.compile(case, target=surface.OpenMP())
    simulation = surface.bind(artifact, data={datum: twisted_initial()}, directory=tmp_path)
    result = simulation.run(until=TWIST_DT)
    actual = np.asarray(result.state[tracer])
    expected = twisted_step(twisted_initial())
    np.testing.assert_array_equal(actual, expected)
    assert actual[-1, 1] == 1.125
    assert float(actual.sum()) == float(twisted_initial().sum())
    assert result.time == TWIST_DT and result.steps == 1 and result.rejections == 0


def test_mapped_layout_identity_includes_the_exact_resolved_offsets():
    def layout(**options):
        authored, _, _ = build_g09_twisted_transport(**options)
        elaborated = authored.elaborated
        plan = pops.resolve(pops.validate(elaborated.case), layout=elaborated.layout)
        (normalized,) = plan.layout_plan.layouts
        assert normalized.native_spatial_layout is not None
        return normalized.native_spatial_layout

    ordinary, first, second = layout(identified=False), layout(), layout(shift=Fraction(1, 8))
    assert first.shape == second.shape == ordinary.shape
    assert first.periodicity == second.periodicity == (False, True)
    assert "identifications" not in ordinary.to_data()
    assert first.identifications == ((0, 1, 0, 1, 1, 1, 8, 2),)
    assert second.identifications == ((0, 1, 0, 1, 1, 1, 8, 1),)
    assert len({ordinary.identity, first.identity, second.identity}) == 3
    assert NativeSpatialLayout.from_data(first.to_data()).identity == first.identity
    with pytest.raises(ValueError, match="tangential identification offset"):
        layout(shift=Fraction(1, 5))


def test_quarter_euler_consumes_density_and_both_oriented_moments():
    case, mesh, data = build_g09_quarter_euler()
    _native, world = native_profile(dimension=2, scope="G09 rotated M03 Euler")
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=mesh))
    artifact = compile_profile_plan_once(world, resolved, route="g09-euler-quarter",
                                         compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values=data,
        resources={"execution_context": artifact_execution_context(artifact)}))
    result = collective_call(world, lambda: pops.run(runtime, t_end=EULER_DT, max_steps=1, console=False))
    actual = collective_call(world, lambda: state_array(runtime, "fluid", (4, CELLS, CELLS)))
    with collective_check(world):
        assert result.accepted_steps == 1 and result.rejected_steps == 0
        assert runtime.time() == EULER_DT and runtime.macro_step() == 1
        if actual is not None:
            self_check()
            initial = euler_initial()
            np.testing.assert_allclose(actual, euler_step(initial), rtol=0, atol=16 * np.spacing(4.0))
            np.testing.assert_allclose(actual[[0, 3]].sum(axis=(1, 2)),
                                       initial[[0, 3]].sum(axis=(1, 2)),
                                       rtol=0, atol=4 * np.spacing(initial[3].sum()))
