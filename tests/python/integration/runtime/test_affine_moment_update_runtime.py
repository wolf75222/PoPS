"""Source-first common rotation preserves all FV moment integrals through AMR reflux."""
from __future__ import annotations

import json

import numpy as np
import pops
import pytest

from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import BindArray
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import moment_names
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers import DenseLU
from pops.time import FailRun, FixedDt, LocalLinear, every
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.requirements import repo_include

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
N, DT, OMEGA = 16, 0.0025, -6.28318531e12
INDICES = [(p, q) for q in range(5) for p in range(5 - q)]


def _case_and_layout(transport_order=1, *, subcycled=False):
    frame = Rectangle("affine_moments_square", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("affine_moment_transport", frame=frame)
    state = model.state("U", components=tuple(moment_names(4)))
    flux = model.flux(
        "transport", state=state, frame=frame,
        components={axis: tuple(speed * value for value in state)
                    for axis, speed in zip(frame.axes, (1., .25), strict=True)},
        waves={axis: (speed,) * 15 for axis, speed in zip(frame.axes, (1., .25), strict=True)})
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    matrix = [[0.] * 15 for _ in range(15)]
    matrix[1][5], matrix[5][1] = OMEGA, -OMEGA
    rotation = model.operator("rotation", returns=model.local_linear_operator(
        "rotation", on=state, matrix=tuple(tuple(row) for row in matrix)))
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("affine_moments_amr_case")
    block = case.block("moments", model=model)
    case.numerics(plan, block=block)
    program = pops.Program("affine_source_then_transport")
    temporal = program.state(block[state])
    # Solve only the actual two first-moment equations in this homogeneous B
    # witness. Higher moments below use the common map, not this degree-block CN.
    midpoint = program.solve(LocalLinear(
        operator=program.I - (program.dt / 2) * program.linear_source(rotation),
        rhs=temporal.n), solver=DenseLU()).consume(action=FailRun())
    mean = program.value("mean_endpoint", 2 * midpoint - temporal.n, at=temporal.n.point)
    source = program.affine_moment_update(
        temporal.n, mean, linear_operator=rotation, theta_dt=program.dt / 2)
    k0 = rate(source)
    if transport_order == 2:
        predictor = program.value("predictor", source + program.dt * k0,
                                  at=program.stage("predictor", c=1))
        k1 = rate(predictor)
        candidate = program.value("transported", source + (program.dt / 2) * (k0 + k1),
                                  at=temporal.next.point)
    else:
        candidate = program.value("transported", source + program.dt * k0,
                                  at=temporal.next.point)
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(InitialCondition(
        state=block[state], value=BindArray(), projection=ConservativeCellAverage()))
    threshold = case.param(RuntimeParam("refine_density", default=1.2))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    layout = AMR(
        grid=CartesianGrid(frame=frame, cells=(N, N), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block[state])["M00"] > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),))
        if subcycled else AMRExecution.synchronous())
    return case, layout


@pytest.mark.parametrize("transport_order,subcycled", ((1, False), (2, False), (1, True)),
                         ids=("FE", "SSPRK2", "unequal_windows_refused"))
def test_affine_rotation_uses_solved_means_and_conserves_through_partial_amr_reflux(
    isolated_native_cache, native_cxx, kokkos_root, transport_order, subcycled,
):
    del isolated_native_cache, kokkos_root
    case, layout = _case_and_layout(transport_order, subcycled=subcycled)
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production(),
                            compile_options={"include": repo_include(), "cxx": native_cxx})
    initial_subject = resolved.initial_condition_plan.bindings[0].subject
    from pops._native_selector import select_native_dimension
    from pops.codegen._native_mpi import native_mpi_communicator

    _pops = select_native_dimension(2)
    if native_mpi_communicator(_pops) == "MPI_COMM_WORLD":
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once

        artifact = compile_resolved_plan_once(
            _pops.mpi_world(), resolved, route="affine moment AMR",
            compile_artifact=pops.compile)
    else:
        artifact = pops.compile(resolved)
    nodes = np.array((-1., 0., 1.))
    vx, vy = np.meshgrid(.375 + .3 * nodes, -.625 + .2 * nodes, indexing="ij")
    weights = np.outer((.2, .5, .3), (.25, .5, .25))
    moments = np.array([np.sum(weights * vx**p * vy**q) for p, q in INDICES])
    coordinate = (np.arange(N) + .5) / N
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    density = 1 + .5 * np.exp(-((x - .5)**2 + (y - .5)**2) / .02)
    initial = np.ascontiguousarray(moments[:, None, None] * density)
    runtime = pops.bind(artifact, initial_values={initial_subject: initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    assert runtime.n_levels() == 2
    fine_cells = sum(np.prod(np.array(upper) - lower + 1)
                     for level, lower, upper in runtime.patch_boxes() if level == 1)
    assert 0 < fine_cells < 4 * N * N
    initial_integrals = np.array([runtime.integral("moments", component=i, levels=(0,))
                                  for i in range(15)])
    starting = np.asarray(runtime.block_level_state_global("moments", 0)).reshape((15, N, N))
    np.testing.assert_allclose(starting / starting[0],
                               np.broadcast_to(moments[:, None, None], starting.shape),
                               rtol=2e-14, atol=2e-15)
    if subcycled:
        snapshots = [np.array(runtime.block_level_state_global("moments", level), copy=True)
                     for level in range(runtime.n_levels())]
        # A second attempt proves the rejected trace does not become accepted
        # algorithm state or change the next failure into a stale-token failure.
        for _ in range(2):
            with pytest.raises((ValueError, RuntimeError), match="synchronous equal clock windows"):
                pops.run(runtime, t_end=DT, max_steps=1)
            assert runtime.time() == 0 and runtime.macro_step() == 0
            for level, snapshot in enumerate(snapshots):
                np.testing.assert_array_equal(
                    runtime.block_level_state_global("moments", level), snapshot)
        return
    angle = 2 * np.arctan(OMEGA * DT / 2)
    basis = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))
    position = {index: component for component, index in enumerate(INDICES)}
    for step in range(1, 4):
        c, s = np.cos(step * angle), np.sin(step * angle)
        # Independent particle trajectories supply every expected accepted moment.
        rotated_x, rotated_y = c * vx + s * vy, -s * vx + c * vy
        expected = initial_integrals[0] * np.array([
            np.sum(weights * rotated_x**p * rotated_y**q) for p, q in INDICES])
        report = pops.run(runtime, t_end=step * DT, max_steps=1)
        assert report.accepted_steps == 1
        actual = np.array([runtime.integral("moments", component=i, levels=(0,)) for i in range(15)])
        np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=2e-13)
        state = np.asarray(runtime.block_level_state_global("moments", 0)).reshape((15, N, N))
        gram = np.stack([np.stack([state[position[p+i, q+j]] for i, j in basis], axis=-1)
                         for p, q in basis], axis=-2)
        eigenvalues = np.linalg.eigvalsh(gram)
        worst = np.unravel_index(np.argmin(eigenvalues[:, :, 0]), (N, N))
        assert eigenvalues.min() > 0, (
            "step", step, "minimum eigenvalue", eigenvalues[worst][0], "cell", worst,
            "moments", state[:, worst[0], worst[1]], "patches", runtime.patch_boxes())
        expected_ratios = expected / initial_integrals[0]
        for level in range(runtime.n_levels()):
            cells = N * 2**level
            resolved_state = np.asarray(runtime.block_level_state_global("moments", level)).reshape(
                (15, cells, cells))
            valid = np.zeros((cells, cells), dtype=bool)
            if level == 0:
                valid[:] = True  # patch_boxes() lists only sparse refined levels.
            for patch_level, lower, upper in runtime.patch_boxes():
                if patch_level == level:
                    valid[lower[1]:upper[1]+1, lower[0]:upper[0]+1] = True
            level_gram = np.stack([
                np.stack([resolved_state[position[p+i, q+j], valid] for i, j in basis], axis=-1)
                for p, q in basis], axis=-2)
            minimum = np.linalg.eigvalsh(level_gram).min()
            assert minimum > 0, ("level", level, "step", step, "minimum Gram eigenvalue", minimum)
            ratios = resolved_state[:, valid] / resolved_state[0, valid]
            np.testing.assert_allclose(ratios, np.broadcast_to(expected_ratios[:, None], ratios.shape),
                                       rtol=1e-11, atol=2e-13)
            print("AFFINE_TRACE " + json.dumps({
                "transport_order": transport_order, "step": step, "level": level,
                "minimum_gram_eigenvalue": float(minimum),
                "maximum_moment_ratio_error": float(np.max(np.abs(ratios - expected_ratios[:, None]))),
                "maximum_integral_error": float(np.max(np.abs(actual - expected))),
            }, sort_keys=True))
    assert np.max(np.abs(state[0] - initial[0])) > 1e-5
