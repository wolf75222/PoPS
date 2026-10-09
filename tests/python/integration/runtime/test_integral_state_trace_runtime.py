"""Native public receipt: one accepted FV exterior current advances a persistent scalar.

The C++ ProgramContextContract test separately exercises child accept/parent reject,
double consumption, retry, and POPSEX02 restore on the same native ledger.
"""

import numpy as np
import pops
import pytest
import re

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from pops._native_collectives import allgather_value
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory, save_public_snapshot
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT = 8, .01


def _world():
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator

    return _pops.mpi_world() if native_mpi_communicator(_pops) == "MPI_COMM_WORLD" else None


def _case(kind):
    frame = Rectangle("integral_square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("scalar_advection", frame=frame)
    state = model.state("U", components=("density",))
    x_axis, y_axis = frame.axes
    flux = model.flux("F", frame=frame, state=state,
                      components={x_axis: (state[0],), y_axis: (0 * state[0],)},
                      waves={x_axis: (1.,), y_axis: (0.,)})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("integral_native_" + kind)
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    plan.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Outflow(state=block[state]),
        frame.boundaries.x_max: Outflow(state=block[state]),
    }, periodic=PeriodicAxes((y_axis,))))
    case.numerics(plan, block=block)
    program = pops.Program("integral_forward_euler")
    temporal = program.state(block[state])
    selected = rate(temporal.n)
    accepted = program.value("accepted", temporal.n + program.dt * selected,
                             at=temporal.next.point)
    program.commit(temporal.next, accepted)
    program.step_strategy(FixedDt(DT))
    quantity = program.integral_state("q", initial=.7)
    # The ledger is oriented into the cell: right-face outgoing F=1 has amount -dt.
    # A capacitor receiving that current therefore authors the opposite sign, C=1.
    program.accept_external_trace(quantity, rate=selected, axis=0, side=1,
                                  component=0, scale=-1.)
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    n = 32 if kind == "amr2" else N
    grid = CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes((y_axis,)))
    if kind == "uniform":
        layout = Uniform(grid)
    else:
        from pops.amr import (AMRClockRelation, AMRExecution, AMRHierarchy, AMRRegrid,
                              AMRTagging, AMRTransfer, Buffer, ConflictPolicy,
                              EqualityPolicy, Hysteresis, Tag)
        from pops.layouts import AMR
        from pops.lib.amr import StateTransfer
        from pops.math import ValueExpr
        from pops.params import RuntimeParam
        from pops.time import every

        transfer = AMRTransfer()
        transfer.state(block[state], StateTransfer())
        threshold = case.param(RuntimeParam(
            "refinement_threshold", default=1.1 if kind == "amr2" else 1000.))
        layout = AMR(grid=grid,
            hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)) if kind == "amr2"
                      else AMRHierarchy(max_levels=1, ratios=()),
            tagging=AMRTagging(rules=(Tag(ValueExpr(block[state])[state.components[0]]
                                          > case.value(threshold)), Buffer(cells=1)),
                                     hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
                                     conflict_policy=ConflictPolicy.REFINE_WINS),
            regrid=AMRRegrid(schedule=every(100, clock=program.clock)), transfer=transfer,
            execution=(AMRExecution.subcycled((AMRClockRelation(0, 1, 2),))
                       if kind == "amr2" else AMRExecution.synchronous()))
    return case, layout, quantity


@pytest.mark.parametrize("kind", ("uniform", "amr1", "amr2"))
def test_native_accepted_face_amount_and_initial_read(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property, kind):
    world = _world()
    case, layout, quantity = collective_call(world, lambda: _case(kind))
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    if world is None:
        artifact = pops.compile(resolved)
    else:
        from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
        artifact = compile_resolved_plan_once(world, resolved, route="integral-" + kind,
                                              compile_artifact=pops.compile)
    with collective_check(world):
        assert artifact.resolved_dimension == 2
    n = 32 if kind == "amr2" else N
    right_value = 1.2 if kind == "amr2" else 1.
    with collective_check(world):
        subject = artifact.plan.initial_condition_plan.bindings[0].subject
        initial = np.ones((1, n, n), dtype=np.float64)
        if kind == "amr2":
            initial[:, :, n // 2:] = right_value
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values={subject: initial},
        resources={"execution_context": context}))
    with collective_check(world):
        assert runtime.integral_state(quantity) == .7
        assert runtime.time() == 0.
    if kind == "amr2":
        from tests.python.support.amr_snapshots import composite_active_mask
        coarse = collective_call(world, lambda: composite_active_mask(
            runtime, 0, refinement_ratio=2))
        with collective_check(world):
            assert runtime.n_levels() == 2
            assert coarse.any() and (~coarse).any()
            assert not coarse[:, -1].any(), "the physical right face must be fine-owned"
    directory = collective_directory(world, tmp_path)
    save_public_snapshot(world, runtime, artifact, quantity, directory, "initial", amr=kind != "uniform")
    if kind == "amr2":
        with collective_check(world):
            if world is None or int(world.rank) == 0:
                np.savez(directory / "composite-ownership.npz", coarse_active=coarse)
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    _, accepted = save_public_snapshot(
        world, runtime, artifact, quantity, directory, "accepted", amr=kind != "uniform")
    with collective_check(world):
        if world is None or int(world.rank) == 0:
            records = [row for ledger in accepted[2] for row in ledger["records"]
                       if row["exterior"] and (row["axis"], row["side"], row["component"]) == (0, 1, 0)]
            keys = [(row["operation"], row["occurrence"], row["context"], row["quadrature"])
                    for row in records]
            assert len(keys) == len(set(keys)), "MPI ranks duplicate a physical trace contribution"
            assert len(keys) == (4 * n if kind == "amr2" else n)
    if kind == "amr2":
        def physical_right_faces():
            selected = []
            for row in runtime._executor._program_exchange_records():
                match = re.match(r"pops\.exchange\.frame\.v1/\d+:[^/]+/\d+/(\d+)/(\d+)/",
                                 row["evaluation_context"])
                if match is None:
                    raise AssertionError("accepted AMR exchange lacks its exact level/substep")
                level, substep = map(int, match.groups())
                tokens = row["quadrature_identity"].split("/")
                cell_x = int(tokens[0].split(":")[1])
                axis, side = (int(token.split(":")[1]) for token in tokens[1:3])
                if axis == 0 and side == 1 and cell_x == n * 2**level - 1:
                    selected.append((level, substep, row))
            return selected

        local = collective_call(world, physical_right_faces)
        rows = (local if world is None else
                [row for batch in allgather_value(world, local) for row in batch])
        with collective_check(world):
            assert len({row["operation_identity"] for _, _, row in rows}) == 1
            groups = {}
            for level, substep, row in rows:
                groups.setdefault((level, substep), []).append(row)
            assert set(groups) == {(1, 0), (1, 1)}, {
                key: (len(group), sum(row["integrated_amount"] for row in group))
                for key, group in groups.items()}
            for group in groups.values():
                assert len(group) == n * 2
                assert {row["temporal_weight"] for row in group} == {DT / 2}
                assert {row["face_measure"] for row in group} == {1 / (n * 2)}
                assert {row["numerical_flux"] for row in group} == {right_value}
                assert abs(sum(row["integrated_amount"] for row in group) +
                           right_value * DT / 2) < 2e-14
    with collective_check(world):
        assert abs(runtime.integral_state(quantity) - (.7 + right_value * DT)) < 2e-13
        assert runtime.macro_step() == 1
        assert abs(runtime.time() - DT) < 2e-15
    record_property("integral_receipts", str(directory))
    record_property("artifact_identity", artifact.artifact_identity.token)
