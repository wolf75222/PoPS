"""Public native reception of original, co-located nonlinear spatial products.

The manufactured forcing closes the declared discrete equations; it is never a
replacement for native solved fields. No M27 nonlinear qualification is claimed.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pops
import pytest

from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops._ir.handle_expr import ValueExpr
from pops.domain import CartesianDomain
from pops.fields import (CellCenteredNonlinearCoupled, FieldBoundary,
                         FieldDiscretization, FieldProblem, bcs)
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model import Handle, OwnerPath
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import FailRun, FixedDt
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.collective_checks import (
    collective_attempt, collective_call, collective_check, state_snapshots,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
CELLS, LENGTHS, DT = (5, 4), (1.0, 1.5), .01
DIFFUSION = np.array(((.04, .006, 0.), (-.003, .05, 0.), (0., 0., .03)))
STATE_ATOL, RESIDUAL_ATOL = 2e-8, 2e-8


def target_means(width):
    nx, ny = CELLS
    x = (np.arange(nx) + .5)[None, :] / nx
    y = (np.arange(ny) + .5)[:, None] / ny
    rows = (.15 + .02 * np.sinc(1 / nx) * np.cos(2 * np.pi * x) + 0 * y,
            .25 + .015 * np.sinc(1 / ny) * np.sin(2 * np.pi * y) + 0 * x,
            .18 + .01 * np.sinc(1 / nx) * np.sinc(1 / ny) * np.cos(2 * np.pi * (x + y)))
    return np.stack(rows[:width])


def parameter_means():
    nx, ny = CELLS
    x = (np.arange(nx) + .5)[None, :] / nx
    y = (np.arange(ny) + .5)[:, None] / ny
    return (1. + .04 * np.sinc(1 / nx) * np.sinc(1 / ny)
            * np.cos(2 * np.pi * (x + y)))[None]


def original_lhs(values, parameter):
    """Independent NumPy evaluation of all declared equations, without elimination."""
    laplacian = sum((np.roll(values, 1, axis=2 - axis) - 2 * values
                     + np.roll(values, -1, axis=2 - axis)) / (length / cells)**2
                    for axis, (length, cells) in enumerate(zip(LENGTHS, CELLS, strict=True)))
    q, v = values[:2]
    local = [parameter[0] * q + q**3 + .03 * q * v + .02 * v**2,
             1.3 * v + .4 * v**3 + .02 * v * q + .01 * q**2]
    if len(values) == 3:
        local.append(1.4 * values[2] + values[2]**3)
    return np.stack(local) - np.einsum("ij,jyx->iyx", DIFFUSION[:len(values), :len(values)], laplacian)


def prepared_case(order, *, failure=None):
    width = len(order)
    frame = CartesianDomain("original-product-box", lower=(0., 0.), upper=LENGTHS).frame(Cartesian2D())
    model = pops.Model("captured-equation-data", frame=frame)
    forcing = model.state("forcing", components=tuple("f%d" % i for i in range(width)))
    parameter_model = pops.Model("captured-reaction-parameter", frame=frame)
    parameter = parameter_model.state("parameter", components=("a",))
    case = pops.Case("original-spatial-product")
    block = case.block("forcing", model, states=(forcing,))
    parameter_block = case.block("parameter", parameter_model, states=(parameter,))
    unknowns = tuple(Handle(label, kind="field", owner=OwnerPath.model("independent-product"))
                     for label in ("q", "v", "z")[:width])
    q, v = unknowns[:2]
    reactions = [Reaction(q, parameter[0] + ValueExpr(q)**2 + .03 * ValueExpr(v))
                 + Reaction(v, .02 * ValueExpr(v)),
                 Reaction(v, 1.3 + .4 * ValueExpr(v)**2 + .02 * ValueExpr(q))
                 + Reaction(q, .01 * ValueExpr(q))]
    if width == 3:
        reactions.append(Reaction(unknowns[2], 1.4 + ValueExpr(unknowns[2])**2))
    if failure == "nonfinite":
        reactions[0] = Reaction(q, parameter[0] + sqrt(ValueExpr(q) - 1))
    equations = []
    for row in range(width):
        lhs = reactions[row]
        for column in range(width):
            if DIFFUSION[row, column] != 0:
                lhs = lhs - DivCoeffGrad(unknowns[column], float(DIFFUSION[row, column]))
        equations.append(lhs == forcing[row])
    problem = FieldProblem("uncondensed-equations", unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order), boundaries=tuple(FieldBoundary(
            unknowns[i], bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())) for i in order))
    solver = Newton(tolerance=1e-11, max_iterations=1 if failure == "iterations" else 20,
                    linear_tolerance=1e-9, linear_max_iterations=150, restart=60)
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=1e-7), boundaries=(), solver=solver))
    program = pops.Program("original-product-newton")
    load, coefficient = program.state(block[forcing]), program.state(parameter_block[parameter])
    request = field.bind_program_inputs(program=program,
        values={block[forcing]: load.n, parameter_block[parameter]: coefficient.n}, at=load.next.point, solver=solver)
    first = program.solve(request, solver=solver).consume(action=FailRun())
    # A second explicitly distinct seed changes initialization, never equation captures.
    nonzero_seed = program.value("scaled-first-solution-seed", .8 * first[0], at=load.next.point)
    second = program.solve(replace(request, seeds={"field_tuple": nonzero_seed}), solver=solver).consume(action=FailRun())
    for run, solution in enumerate((first, second)):
        observed = field.observe(solution)
        for component, unknown in enumerate(unknowns):
            program.store_history("solve%d_component%d" % (run, component), observed[field[unknown]], depth=1)
    program.commit(load.next, program.value("accepted-forcing", 1 * load.n, at=load.next.point))
    program.commit(coefficient.next, program.value("accepted-parameter", 1 * coefficient.n, at=coefficient.next.point))
    program.step_strategy(FixedDt(DT))
    case.program(program)
    for state in (block[forcing], parameter_block[parameter]):
        case.initials.add(InitialCondition(state=state, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=CELLS, periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout), block[forcing], parameter_block[parameter]


def bind_native(world, native, order, failure=None):
    resolved, forcing, parameter = collective_call(world, lambda: prepared_case(order, failure=failure))
    artifact = compile_resolved_plan_once(world, resolved, route="original nonlinear spatial field",
                                        compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    target, coefficients = target_means(len(order)), parameter_means()
    loads = original_lhs(target, coefficients)
    runtime = collective_call(world, lambda: pops.bind(artifact,
        initial_values={forcing: np.ascontiguousarray(loads), parameter: np.ascontiguousarray(coefficients)},
        resources={"execution_context": context}))
    return runtime, artifact, target, loads, coefficients


@pytest.mark.parametrize("order", ((0, 1), (1, 0), (2, 0, 1)))
def test_original_nonlinear_product_from_saved_native_observations(
        isolated_native_cache, tmp_path, record_property, order):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    package = Path(pops.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    native = select_native_dimension(2)
    world = native.mpi_world()
    runtime, artifact, target, loads, coefficients = bind_native(world, native, order)
    report = collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    results = []
    for run in range(2):
        rows = [collective_call(world, lambda run=run, component=component: runtime.history_global(
            "solve%d_component%d" % (run, component), 1)) for component in range(len(order))]
        results.append(np.asarray(rows))
    state = state_snapshots(runtime, world, ("forcing", "parameter"))
    native_time = collective_call(world, runtime.time)
    native_step = collective_call(world, runtime.macro_step)
    directory = tmp_path / ("nonlinear-field-order-" + "".join(map(str, order)))
    with collective_check(world):
        assert native_time == DT and native_step == 1
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        if world.rank == 0:
            solved = np.asarray(results).reshape(2, len(order), *reversed(CELLS))
            # These are actual runtime outputs; the target is checked independently.
            errors = [float(np.max(np.abs(row - target))) for row in solved]
            residuals = [float(np.max(np.abs(original_lhs(row, coefficients) - loads))) for row in solved]
            assert max(errors) < STATE_ATOL and max(residuals) < RESIDUAL_ATOL
            np.testing.assert_array_equal(state[0].reshape(loads.shape), loads)
            np.testing.assert_array_equal(state[1].reshape(coefficients.shape), coefficients)
            directory.mkdir()
            saved = directory / "native.npz"
            np.savez_compressed(saved, solved=solved, forcing=state[0].reshape(loads.shape),
                                parameter=state[1].reshape(coefficients.shape))
            receipt = {"contract": "pops.spatial-field-residual@1", "native_dimension": 2,
                "mpi_size": world.size, "native_time": native_time, "native_macro_step": native_step,
                "native_path": str(Path(native.__file__).resolve()), "package_path": str(package),
                "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "native_abi": str(native.abi_key()), "artifact_identity": artifact.artifact_identity.token,
                "platform_manifest": artifact.platform_manifest.to_data(),
                "fixture_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "saved_state_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(),
                "unknown_order": list(order), "cells": list(CELLS), "lengths": list(LENGTHS),
                "state_errors": errors, "original_residuals": residuals,
                "state_atol": STATE_ATOL, "original_residual_atol": RESIDUAL_ATOL}
            (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    record_property("spatial_field_residual_receipts", str(directory))
    record_property("artifact_identity", artifact.artifact_identity.token)
    record_property("native_dimension", 2)
    record_property("mpi_rank", world.rank)
    record_property("mpi_size", world.size)


@pytest.mark.parametrize("failure", ("iterations", "nonfinite"))
def test_original_nonlinear_failure_has_no_accepted_publication(isolated_native_cache, failure):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    runtime, _, _, _, _ = bind_native(world, native, (0, 1), failure=failure)
    before = state_snapshots(runtime, world, ("forcing", "parameter"))
    prior = collective_call(world, runtime.program_report)
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    after = state_snapshots(runtime, world, ("forcing", "parameter"))
    final = collective_call(world, runtime.program_report)
    native_time = collective_call(world, runtime.time)
    native_step = collective_call(world, runtime.macro_step)
    with collective_check(world):
        assert all(failures), failures
        reason = "field_newton_iteration_limit" if failure == "iterations" else "nonfinite_original_field_residual"
        assert all(reason in row[1] for row in failures), failures
        assert native_time == 0 and native_step == 0
        assert final.histories == prior.histories and final.diagnostics == prior.diagnostics
        for lhs, rhs in zip(before, after, strict=True):
            np.testing.assert_array_equal(lhs, rhs)
