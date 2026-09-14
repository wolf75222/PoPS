"""Generated full-disk Schur source with real analytic metric producers and sparse AMR."""
from __future__ import annotations

import math
import numpy as np
import pops
import pytest

from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.analytic import coordinate, cos, sin
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import NoFlux
from pops.domain import Rectangle
from pops.fields import AnalyticAux, AuxiliaryBoundary
from pops.fields.bcs import Dirichlet, Neumann, Periodic
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.linalg import LinearProblem
from pops.math import ValueExpr, Var, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers import CompositeTensorFAC, Hierarchy
from pops.time import FailRun, FixedDt, every
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
ALPHA, OMEGA, DT = 39.4784176e12, -6.28318531e12, 0.001


def _case(n, levels):
    frame = Rectangle("mapped_disk_mms", (0., 0.), (1., 2 * math.pi)).frame(Cartesian2D())
    radial, angular = frame.axes
    r, theta = coordinate(frame, radial), coordinate(frame, angular)
    radius, cosine, sine = Var("radius", "aux"), Var("cosine", "aux"), Var("sine", "aux")
    model = pops.Model("mapped_schur_mms", frame=frame)
    state = model.state("U", components=("rho", "mx", "my", "marker"))
    zeros = tuple(0. * component for component in state)
    flux = model.flux("unused_transport", state=state, frame=frame,
                     components={radial: zeros, angular: zeros},
                     waves={radial: (0.,) * 4, angular: (0.,) * 4})
    rate = model.rate("unused_transport", equation=ddt(state) == -div(flux))
    magnetic = model.operator("magnetic", returns=model.local_linear_operator(
        "magnetic", on=state, matrix=((0., 0., 0., 0.), (0., 0., OMEGA, 0.),
                                       (0., -OMEGA, 0., 0.), (0., 0., 0., 0.))))
    gradient = model.operator("gradient", returns=model.local_linear_operator(
        "gradient", on=state, matrix=((0., 0., 0., 0.), (0., cosine, -sine/radius, 0.),
                                       (0., sine, cosine/radius, 0.), (0., 0., 0., 0.))))
    metric = model.operator("metric", returns=model.local_linear_operator(
        "metric", on=state, matrix=((0., 0., 0., 0.), (0., radius, 0., 0.),
                                     (0., 0., 1./radius, 0.), (0., 0., 0., 0.))))
    module = model.module
    for name, expression in (("radius", r), ("cosine", cos(theta)), ("sine", sin(theta))):
        handle = module.aux_handle(module.aux_field(name, frame=frame.canonical_id))
        module.aux_provider(AnalyticAux(handle, expression, frame=frame,
                                       boundary=AuxiliaryBoundary(width=2, kind="foextrap")))
    module.operator_registry().get(gradient.name).requirements["aux"] = ("radius", "cosine", "sine")
    module.operator_registry().get(metric.name).requirements["aux"] = ("radius",)
    case = pops.Case("mapped_schur_native_mms")
    block = case.block("disk", model=model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
                                        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    numerics.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: NoFlux(state=block[state]),
        frame.boundaries.x_max: NoFlux(state=block[state])}, periodic=PeriodicAxes((angular,))))
    case.numerics(numerics, block=block)
    program = pops.Program("mapped_schur_source_only")
    old = program.state(block[state])
    s = 0.5 * program.dt
    coefficients = program.condensed_coeffs(state=old.n, linear_operator=magnetic,
        subset=(1, 2), th_dt=s, c=ALPHA*s*s, c_rho=0, gradient_map=gradient, base_tensor=metric)
    rhs = program.condensed_rhs(program.scalar_field("charge_rhs"), state=old.n,
        linear_operator=magnetic, subset=(1, 2), th_dt=s, g=s,
        gradient_map=gradient, base_tensor=metric, charge_component=0)
    previous = program.history("disk.potential", lag=1, ncomp=1, block=block)
    operator = program.matrix_free_operator("mapped_schur", scope=Hierarchy())
    program.set_apply(operator, lambda builder, _out, value:
        -builder.apply_laplacian_coeff(builder.scalar_field("elliptic_image"), value, coefficients))
    potential = program.solve(LinearProblem(operator, rhs, initial_guess=previous, scope=Hierarchy(), nullspace=None),
        solver=CompositeTensorFAC(max_iter=300, rel_tol=1e-10, abs_tol=1e-12,
            fine_sweeps=64, coarse_cycles=512, correction_damping=.5,
            boundary_conditions=(Neumann(0), Dirichlet(0), Periodic(), Periodic()),
            diagonal_average="arithmetic")).consume(action=FailRun())
    program.store_history("disk.potential", potential)
    independent = program.value("independent_means", 1*old.n, at=old.n.point)
    midpoint = program.condensed_reconstruct(state=independent, phi=potential,
        linear_operator=magnetic, subset=(1, 2), th_dt=s, c_rho=0,
        gradient_map=gradient, gradient_scale=ALPHA)
    endpoint = program.value("source_endpoint", 2*midpoint-old.n, at=old.next.point)
    program.commit(old.next, endpoint)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state],
        value=Analytic(frame=frame, components=(4*r, 0*r, 0*r, r*(1+.5*cos(theta)))),
        projection=ConservativeCellAverage()))
    threshold = case.param(RuntimeParam("refine_marker", default=.75))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(n, 4*n), periodic=PeriodicAxes((angular,))),
        hierarchy=AMRHierarchy(max_levels=levels, ratios=(2,)*(levels-1)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])["marker"] > case.value(threshold)), Buffer(cells=1)),
                          hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    return case, layout


def _measurement(n, levels):
    case, layout = _case(n, levels)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    artifact = pops.compile(resolved)
    runtime = pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})
    assert runtime.n_levels() == levels
    boxes = list(runtime.patch_boxes())
    if levels == 2:
        fine_cells = sum(np.prod(np.array(upper)-lower+1) for level, lower, upper in boxes if level == 1)
        assert 0 < fine_cells < (2*n)*(8*n)
    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    s, w = DT/2, DT*OMEGA/2
    scale = 1/(1+4*s*s*ALPHA/(1+w*w))
    field_error = mean_error = 0.
    for level in range(levels):
        nr, nt = n*2**level, 4*n*2**level
        values = np.asarray(runtime.block_level_state_global("disk", level)).reshape(4, nt, nr)
        potential = np.asarray(runtime.history_global("disk.potential", level, 0)).reshape(nt, nr)
        radial = (np.arange(nr)+.5)/nr
        theta = (np.arange(nt)+.5)*(2*math.pi/nt)
        r, theta = np.meshgrid(radial, theta, indexing="xy")
        exact = scale*(1-r*r)
        # The central gradient gives the continuum gradient exactly on a single level;
        # the potential's O(h^2) Dirichlet offset is deliberately not subtracted.
        velocity_x = 4*s*ALPHA*scale*r*(np.cos(theta)+w*np.sin(theta))/(1+w*w)
        velocity_y = 4*s*ALPHA*scale*r*(-w*np.cos(theta)+np.sin(theta))/(1+w*w)
        valid = np.full((nt, nr), level == 0, dtype=bool)
        for patch_level, lower, upper in boxes:
            if patch_level == level:
                valid[lower[1]:upper[1]+1, lower[0]:upper[0]+1] = True
        assert np.all(np.isfinite(potential[valid])) and np.all(np.isfinite(values[:, valid]))
        np.testing.assert_allclose(values[0][valid], 4*r[valid], rtol=2e-14, atol=2e-14)
        field_error = max(field_error, np.max(np.abs(potential[valid]-exact[valid])))
        mean_error = max(mean_error, np.max(np.abs(values[1][valid]/values[0][valid]-velocity_x[valid])),
                         np.max(np.abs(values[2][valid]/values[0][valid]-velocity_y[valid])))
    print("mapped_source_mms", dict(n=n, levels=levels, field_error=field_error, mean_error=mean_error))
    return field_error, mean_error


@pytest.mark.parametrize("levels", (1, 2))
def test_generated_mapped_disk_schur_uses_actual_auxiliary_geometry_and_converges(levels):
    coarse = _measurement(8, levels)
    fine = _measurement(16, levels)
    assert fine[0] < coarse[0]/2.5
    assert fine[1] < max(coarse[1]/1.5, 2e-8)
