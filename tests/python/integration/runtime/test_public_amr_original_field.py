"""Closed original nonlinear field mechanism on genuine synchronous partial AMR.

The public witness has constant solved fields and spatially varying captured data.
It qualifies coupling/transactions when run; the native nonconstant manufactured
fixture separately exercises coarse/fine flux action. Neither is an M27 campaign.
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
from pops.analytic import cos, x
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import BergerRigoutsos, StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import SourceTerm
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import FailRun, FixedDt, every
from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context

TARGET = np.array((.15, .25, .18))
DIFFUSION = np.array(((.04, .006, 0.), (-.003, .05, 0.), (0., 0., .03)))
DT, TOL = .01, 3e-8


def build(cells, order=(0, 1, 2), *, guarded=False, seed=False, right_preconditioner=None):
    width = len(order)
    frame = Rectangle("closed-original-field-box", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    fluid = pops.Model("actual-field-consumer", frame=frame)
    state = fluid.state("response", components=tuple("u%d" % i for i in range(width)))
    unknowns = tuple(fluid.field("potential%d" % i) for i in range(width))
    auxiliary = tuple(fluid.aux("observed%d" % i) for i in range(width))
    response = fluid.source("field_response", on=state, value=auxiliary)
    guard = fluid.local_transform("response_limit", tuple(state), valid_if=state[0] < .005) if guarded else None
    forcing_model = pops.Model("captured-original-load", frame=frame)
    forcing = forcing_model.state("forcing", components=tuple("f%d" % i for i in range(width)))
    parameter_model = pops.Model("captured-original-coefficient", frame=frame)
    parameter = parameter_model.state("coefficient", components=("a",))

    # The physical equations and consumer are declared before temporal Program wiring.
    reactions = [Reaction(unknowns[0], parameter[0] + ValueExpr(unknowns[0])**2)]
    if width > 1:
        reactions[0] += Reaction(unknowns[0], .03 * ValueExpr(unknowns[1])) + Reaction(unknowns[1], .02 * ValueExpr(unknowns[1]))
        reactions.append(Reaction(unknowns[1], 1.3 + .4 * ValueExpr(unknowns[1])**2 + .02 * ValueExpr(unknowns[0]))
                         + Reaction(unknowns[0], .01 * ValueExpr(unknowns[0])))
    if width == 3:
        reactions.append(Reaction(unknowns[2], 1.4 + ValueExpr(unknowns[2])**2))
    equations = []
    for row in range(width):
        lhs = reactions[row]
        for column in range(width):
            if DIFFUSION[row, column] != 0:
                lhs -= DivCoeffGrad(unknowns[column], float(DIFFUSION[row, column]))
        equations.append(lhs == forcing[row])
    problem = FieldProblem("closed-original-equations", unknowns=tuple(unknowns[i] for i in order),
        equations=tuple(equations[i] for i in order), boundaries=tuple(FieldBoundary(unknowns[i],
            bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())) for i in order))
    case = pops.Case("original-AMR-mechanism")
    blocks = []
    for name, model, model_state in (("fluid", fluid, state), ("forcing", forcing_model, forcing),
                                    ("parameter", parameter_model, parameter)):
        flux = model.flux("stationary_flux", frame=frame, state=model_state,
            components={axis: tuple(0 * q for q in model_state) for axis in frame.axes},
            waves={axis: tuple(0 * q for q in model_state) for axis in frame.axes})
        rate = model.rate("balance", equation=ddt(model_state) == (-div(flux) + response if model is fluid else -div(flux)))
        numerical = DiscretizationPlan()
        numerical.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(model_state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerical, block=block)
        blocks.append(block)
    realization = {} if right_preconditioner is None else {"right_preconditioner": right_preconditioner}
    solver = Newton(tolerance=1e-10, max_iterations=20, linear_tolerance=1e-8,
                     linear_max_iterations=240, restart=60, **realization)
    field = case.field(problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=1e-6), boundaries=(), solver=solver))
    program = pops.Program("one-original-composite-product")
    current, load, coefficient = (program.state(block[handle]) for block, handle in
        zip(blocks, (state, forcing, parameter), strict=True))
    program.store_history("accepted-response", current.n, depth=1)
    request = field.bind_program_inputs(program=program,
        values={blocks[1][forcing]: load.n, blocks[2][parameter]: coefficient.n},
        at=program.stage("original-field", c=0), solver=solver)
    if seed:
        declared_seed = program.scalar_field("declared-zero-seed", ncomp=width)
        request = replace(request, seeds={"field_tuple": declared_seed})
    observed = field.observe(program.solve(request, solver=solver).consume(action=FailRun()))
    outputs = tuple(observed[field[unknown]] for unknown in unknowns)
    module = fluid.module
    carrier = blocks[0][module.field_handle(module.field_spaces()["fields"])]
    publication = observed.publish({(carrier, "observed%d" % i): output for i, output in enumerate(outputs)},
                                   states={blocks[0][state]: current.n})
    rhs = program.rhs(state=current.n, fields=publication,
        terms=[SourceTerm(blocks[0][module.operator_handle("field_response")])])
    program.commit(current.next, program.value("response-update", current.n + program.dt * rhs, at=current.next.point))
    if guarded:
        def require_response(body):
            published = body.value("published-response", 1 * current.n, at=current.next.point)
            body.commit(current.next, body.transform(published, transform=guard))
        program.after_synchronization(require_response)
    for time in (load, coefficient):
        program.commit(time.next, program.value("fixed-" + time.n.name, 1 * time.n, at=time.next.point))
    # The guarded variant proposes a large interval; public run clips it to the
    # requested endpoint for the safe attempt without replacing its controller.
    program.step_strategy(FixedDt(10*DT if guarded else DT))
    case.program(program)

    a = 1 + .04 * cos(2 * np.pi * x(frame))
    q, v, z = map(float, TARGET)
    source = [a * q + q**3]
    if width > 1:
        source[0] += .03*q*v + .02*v*v
        source.append(1.3*v + .4*v**3 + .02*v*q + .01*q*q)
    if width == 3:
        source.append(1.4*z + z**3)
    transfer = AMRTransfer()
    for block, handle, initial in zip(blocks, (state, forcing, parameter),
        (tuple(0 * a for _ in range(width)), tuple(0 * a + value for value in source), (a,)), strict=True):
        case.initials.add(InitialCondition(state=block[handle], value=Analytic(frame=frame, components=initial),
                                          projection=ConservativeCellAverage()))
        transfer.state(block[handle], StateTransfer())
    # A central strip avoids wrapping disconnected tags into a full-domain fine
    # bounding box. This changes only the observation fixture's mesh selection.
    threshold = case.param(RuntimeParam("mesh-refinement-threshold", default=.97))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(blocks[2][parameter]) < case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous(), clustering=BergerRigoutsos(maximum_box_size=8))
    return case, layout


def bind_case(world, cells, order, *, guarded=False, seed=False, right_preconditioner=None):
    case, layout = collective_call(world, lambda: build(cells, order, guarded=guarded, seed=seed,
                                                      right_preconditioner=right_preconditioner))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_resolved_plan_once(world, resolved, route="original nonlinear composite AMR field", compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context": context}))
    return runtime, artifact


def capture(world, runtime):
    levels = collective_call(world, runtime.n_levels)
    rows = []
    for level in range(levels):
        fields = [collective_call(world, lambda block=block, level=level: runtime.block_level_state_global(block, level))
                  for block in ("fluid", "forcing", "parameter")]
        active = collective_call(world, lambda level=level: composite_active_mask(runtime, level, refinement_ratio=2))
        with collective_check(world):
            rows.append((*(np.asarray(field).copy() for field in fields), np.asarray(active).copy()))
    carriers = collective_call(world, lambda: tuple(tuple(row) for row in runtime._executor.checkpoint_rank_local_carrier_manifest()))
    # Each retrieval is a separate convergence boundary before the next gather.
    native = runtime._executor
    history = []
    for name in collective_call(world, native.history_names):
        history_levels = collective_call(world, lambda name=name: tuple(native.history_levels(name)))
        depth = collective_call(world, lambda name=name: native.history_depth(name))
        width = collective_call(world, lambda name=name: native.history_ncomp(name))
        with collective_check(world):
            assert history_levels == tuple(range(levels))
        images = []
        for level in history_levels:
            initialized = collective_call(world, lambda name=name, level=level: native.history_initialized(name, level))
            fill = collective_call(world, lambda name=name, level=level: native.history_fill_count(name, level))
            slots = []
            for slot in range(depth):
                duration = collective_call(world, lambda name=name, level=level, slot=slot: native.history_slot_dt(name, level, slot))
                value = collective_call(world, lambda name=name, level=level, slot=slot: native.history_global(name, level, slot))
                with collective_check(world):
                    slots.append((np.asarray(duration, dtype=np.float64).tobytes(), np.asarray(value).tobytes()))
            images.append((level, initialized, fill, tuple(slots)))
        history.append((name, width, depth, tuple(images)))
    diagnostics = collective_call(world, lambda: tuple(sorted(native.program_diagnostics().items())))
    history = (tuple(history), diagnostics)
    lifecycle = collective_call(world, lambda: (runtime.time(), runtime.macro_step(), tuple(runtime.patch_boxes())))
    return rows, carriers, history, lifecycle


def check_original_saved(rows, cells, width, time):
    assert len(rows) == 2
    assert np.any(rows[0][3]) and not np.all(rows[0][3]), "require genuine partial fine coverage"
    errors, residuals = [], []
    for level, (response, forcing, parameter, active) in enumerate(rows):
        shape = (cells * 2**level,) * 2
        response = response.reshape(width, *shape) / time
        forcing = forcing.reshape(width, *shape)
        a = parameter.reshape(*shape)
        expected = TARGET[:width, None, None] * np.ones(shape)
        np.testing.assert_allclose(response[:, active], expected[:, active], atol=TOL, rtol=0)
        q = response[0]
        lhs = [a*q + q**3]
        if width > 1:
            v = response[1]
            lhs[0] += .03*q*v + .02*v*v
            lhs.append(1.3*v + .4*v**3 + .02*v*q + .01*q*q)
        if width == 3:
            z = response[2]
            lhs.append(1.4*z + z**3)
        # The target is constant: the exact original diffusion is zero on every
        # valid patch and interface. This does not measure nonconstant AMR accuracy.
        local = np.asarray(lhs) - forcing
        errors.append(float(np.max(np.abs(response[:, active] - expected[:, active]))))
        residuals.append(float(np.max(np.abs(local[:, active]))))
        assert residuals[-1] < TOL
    return errors, residuals


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
@pytest.mark.parametrize("cells,order,seed", [(16, (0,), False), (16, (0, 1, 2), False), (32, (2, 0, 1), True)])
def test_public_original_amr_saved_reaction_and_exact_restart(isolated_native_cache, tmp_path, record_property, cells, order, seed):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    runtime, artifact = bind_case(world, cells, order, seed=seed, right_preconditioner="SpatialBasisJacobi@1")
    directory = collective_directory(world, tmp_path / "original-amr-field")
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = capture(world, runtime)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(directory / "accepted"))
    with collective_check(world):
        assert accepted[3][:2] == (DT, 1)
        if world.rank == 0:
            payload = {"level%d_%s" % (level, name): value for level, row in enumerate(accepted[0])
                       for name, value in zip(("response", "forcing", "parameter", "active"), row, strict=True)}
            saved = directory / "accepted.npz"
            np.savez_compressed(saved, **payload)
            with np.load(saved) as image:
                saved_rows = [tuple(image["level%d_%s" % (level, name)] for name in ("response", "forcing", "parameter", "active"))
                              for level in range(2)]
                errors, residuals = check_original_saved(saved_rows, cells, len(order), DT)
            (directory / "receipt.json").write_text(json.dumps({"artifact": artifact.artifact_identity.token,
                "native_dimension": 2, "mpi_size": world.size, "cells": cells, "order": order,
                "native_path": str(native.__file__), "native_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "saved_sha256": hashlib.sha256(saved.read_bytes()).hexdigest(), "errors": errors,
                "original_constant_solution_reaction_residuals": residuals, "tolerance": TOL,
                "fixture_contract": "sol61.public-original-amr-fixture@2",
                "right_preconditioner": "SpatialBasisJacobi@1"}, indent=2) + "\n")
    collective_call(world, lambda: pops.run(runtime, t_end=2*DT, max_steps=1, console=False))
    continuous = capture(world, runtime)
    restored = collective_call(world, lambda: pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)}))
    collective_call(world, lambda: restored.restart(checkpoint))
    reloaded = capture(world, restored)
    with collective_check(world):
        assert reloaded[1:] == accepted[1:]
        for left, right in zip(reloaded[0], accepted[0], strict=True):
            for a, b in zip(left, right, strict=True):
                np.testing.assert_array_equal(a, b)
    collective_call(world, lambda: pops.run(restored, t_end=2*DT, max_steps=1, console=False))
    replay = capture(world, restored)
    with collective_check(world):
        assert replay[1:] == continuous[1:]
        for left, right in zip(replay[0], continuous[0], strict=True):
            for a, b in zip(left, right, strict=True):
                np.testing.assert_array_equal(a, b)
    for name, value in (("artifact_identity", artifact.artifact_identity.token), ("native_dimension", 2),
                        ("mpi_rank", world.rank), ("mpi_size", world.size), ("evidence_path", str(directory))):
        record_property(name, value)


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_public_original_amr_published_fields_parent_rollback_and_retry(isolated_native_cache, record_property):
    del isolated_native_cache
    from pops._native_selector import select_native_dimension
    native = select_native_dimension(2)
    world = native.mpi_world()
    with collective_check(world):
        assert Path(pops.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    runtime, artifact = bind_case(world, 16, (2, 0, 1), guarded=True,
                                  right_preconditioner="SpatialBasisJacobi@1")
    before = capture(world, runtime)
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=runtime.time()+10*DT, max_steps=1, console=False))
    after = capture(world, runtime)
    with collective_check(world):
        assert all(errors), errors
        assert all("response_limit" in error[1] and "out-of-domain state" in error[1] for error in errors), errors
        assert after[1:] == before[1:]
        for left, right in zip(after[0], before[0], strict=True):
            for a, b in zip(left, right, strict=True):
                np.testing.assert_array_equal(a, b)
    collective_call(world, lambda: pops.run(runtime, t_end=runtime.time()+DT, max_steps=1, console=False))
    clean, _ = bind_case(world, 16, (2, 0, 1), guarded=True, right_preconditioner="SpatialBasisJacobi@1")
    collective_call(world, lambda: pops.run(clean, t_end=clean.time()+DT, max_steps=1, console=False))
    retried, fresh = capture(world, runtime), capture(world, clean)
    with collective_check(world):
        assert retried[1:] == fresh[1:]
        if world.rank == 0:
            check_original_saved(retried[0], 16, 3, DT)
        for left, right in zip(retried[0], fresh[0], strict=True):
            for a, b in zip(left, right, strict=True):
                np.testing.assert_array_equal(a, b)

    # Repeat the rejection after an accepted field image already exists. The
    # parent transaction must preserve its accepted fields and history bytes.
    prior = capture(world, runtime)
    _, errors = collective_attempt(world, lambda: pops.run(runtime, t_end=runtime.time()+10*DT, max_steps=1, console=False))
    restored = capture(world, runtime)
    with collective_check(world):
        assert all(errors), errors
        assert all("response_limit" in error[1] and "out-of-domain state" in error[1] for error in errors), errors
        assert restored[1:] == prior[1:]
        for left, right in zip(restored[0], prior[0], strict=True):
            for a, b in zip(left, right, strict=True):
                np.testing.assert_array_equal(a, b)
    collective_call(world, lambda: pops.run(runtime, t_end=runtime.time()+DT, max_steps=1, console=False))
    collective_call(world, lambda: pops.run(clean, t_end=clean.time()+DT, max_steps=1, console=False))
    final, reference = capture(world, runtime), capture(world, clean)
    with collective_check(world):
        assert final[1:] == reference[1:]
        for left, right in zip(final[0], reference[0], strict=True):
            for a, b in zip(left, right, strict=True):
                np.testing.assert_array_equal(a, b)
    for name, value in (("artifact_identity", artifact.artifact_identity.token), ("native_dimension", 2),
                        ("mpi_rank", world.rank), ("mpi_size", world.size), ("native_path", str(native.__file__))):
        record_property(name, value)
