"""Public three-field AMR qualification: nonzero MMS, consumption, and failed-attempt retry.

All 16/32/64 cases retain a genuine coarse/fine interface. Native execution is required;
source admission alone does not qualify the numerical or transaction assertions below.
"""
from __future__ import annotations

import numpy as np
import pytest
import pops
from pops._ir.elliptic import DivCoeffGrad, Reaction
from pops.analytic import cos, x, y
from pops.amr import (AMRExecution, AMRHierarchy, AMRRegrid, AMRTagging, AMRTransfer,
                      Buffer, ConflictPolicy, EqualityPolicy, Hysteresis, Tag)
from pops.domain import Rectangle
from pops.fields import ConstantModeGauge, FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Analytic
from pops.math import ValueExpr, ddt, div, grad
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.numerics.terms import SourceTerm
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.solvers import CompositeFieldGMRES
from pops.time import FailRun, FixedDt, every
from tests.python.support.amr_snapshots import composite_active_mask
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_amr_implicit_diffusion import accepted_histories

RESOLUTIONS = (16, 32, 64)
DT = 0.005
A = np.array(((2., -.35, .2), (-.35, 1.6, -.25), (.2, -.25, 1.3)))
V = np.array((1., 2., -1.))
R = np.outer(V, V)
MODES = ((2, -1, 0), (1, 0, 1))
MEANS = np.array((1.5, 1., 3.5))
# V @ MEANS = 0 and MODES @ MEANS = (2, 5): the constant reaction load is zero.
# The third amplitude 4 preserves those mean constraints while crossing zero. The same amplitudes enter both the manufactured forcing and reference.
AMPLITUDES = np.array((1., -.3, 4.))
WAVENUMBERS = ((1, 1), (2, 1), (1, 2))


def build(n, *, guarded=False):
    frame = Rectangle("joint field domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    fluid = pops.Model("field consumer", frame=frame)
    state = fluid.state("U", components=("p0", "p1", "p2", "gx", "gy"))
    unknowns = tuple(fluid.field("potential%d" % i) for i in range(3))
    gradient = fluid.vector("observed gradient", frame=frame,
        components={frame.x: grad(unknowns[2]).x, frame.y: grad(unknowns[2]).y})
    observed_potential = tuple(fluid.aux("observed_phi%d" % i) for i in range(3))
    source = fluid.source("field response", on=state,
        value=(*observed_potential, gradient.x, gradient.y))
    guard = fluid.local_transform("admissible_response", tuple(state), valid_if=state[2] < .1) if guarded else None
    driver = pops.Model("independent load", frame=frame)
    loads = driver.state("U", components=("b0", "b1", "b2"))
    marker = pops.Model("refinement marker", frame=frame)
    marker_state = marker.state("U", components=("indicator",))
    case = pops.Case("public joint AMR field")
    blocks = []
    for name, model, model_state in (("fluid", fluid, state), ("load", driver, loads), ("marker", marker, marker_state)):
        flux = model.flux("stationary flux", frame=frame, state=model_state,
            components={axis: tuple(0 * q for q in model_state) for axis in frame.axes},
            waves={axis: tuple(0 * q for q in model_state) for axis in frame.axes})
        equation = -div(flux) + source if model is fluid else -div(flux)
        rate = model.rate("balance", equation=ddt(model_state) == equation)
        numerical = DiscretizationPlan()
        numerical.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(model_state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerical, block=block)
        blocks.append(block)
    equations = []
    for i in range(3):
        terms = tuple(-DivCoeffGrad(unknowns[j], float(A[i,j])) +
                      Reaction(unknowns[j], float(R[i,j])) for j in range(3))
        equations.append(sum(terms[1:], terms[0]) == loads[i])
    problem = FieldProblem("joint electric relations", unknowns=unknowns, equations=tuple(equations),
        boundaries=tuple(FieldBoundary(q, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())) for q in unknowns),
        gauge=ConstantModeGauge(unknowns, modes=MODES, values=(2, 5)))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CompositeFieldGMRES(max_iter=4000, restart=80, rel_tol=1e-11, abs_tol=1e-11)))
    program = pops.Program("joint field step")
    current, load, tagging = (program.state(block[model_state]) for block, model_state in
                              zip(blocks, (state, loads, marker_state), strict=True))
    program.store_history("prior_fluid", current.n, depth=1)
    observations = field.observe(program.solve(field, values={blocks[1][loads]: load.n},
        at=program.stage("field solve", c=0)).consume(action=FailRun()))
    values = tuple(observations[field[q]] for q in unknowns)
    vector = observations.gradient(field[unknowns[2]], dimension=2)
    module = fluid.module
    carrier = blocks[0][module.field_handle(module.field_spaces()["fields"])]
    publication = observations.publish({
        **{(carrier, "observed_phi%d" % i): values[i] for i in range(3)},
        (carrier, "potential2_grad_x"): (vector, 0),
        (carrier, "potential2_grad_y"): (vector, 1),
    }, states={blocks[0][state]: current.n})
    for kind in ("sum", "min", "max"):
        program.record_scalar("phi2_" + kind, getattr(program, kind)(values[2]))
    program.record_scalar("phi2_abs_sum", program.abs_sum_component(values[2], 0))
    rhs = program.rhs(state=current.n, fields=publication,
        terms=[SourceTerm(blocks[0][module.operator_handle("field_response")])])
    candidate = program.value("response", current.n + program.dt * rhs, at=current.next.point)
    program.commit(current.next, candidate)
    if guarded:
        def require_admissible(body):
            synchronized = body.value("published_response", 1 * current.n, at=current.next.point)
            body.commit(current.next, body.transform(synchronized, transform=guard))
        program.after_synchronization(require_admissible)
    for time in (load, tagging):
        program.commit(time.next, program.value("fixed_" + time.n.name, 1 * time.n, at=time.next.point))
    program.step_strategy(FixedDt(10 * DT if guarded else DT))
    case.program(program)
    waves = tuple(cos(2 * np.pi * kx * x(frame)) * cos(2 * np.pi * ky * y(frame))
                  for kx, ky in WAVENUMBERS)
    forcing = tuple(sum(float((4*np.pi**2*(kx*kx+ky*ky)*A[i,j] + R[i,j]) * AMPLITUDES[j]) * waves[j]
                        for j, (kx, ky) in enumerate(WAVENUMBERS)) for i in range(3))
    initial = ((0 * waves[0],) * 5, forcing,
               (1 + .3 * cos(2 * np.pi * (x(frame) - .5)),))
    transfer = AMRTransfer()
    for block, model_state, expressions in zip(blocks, (state, loads, marker_state), initial, strict=True):
        case.initials.add(InitialCondition(state=block[model_state],
            value=Analytic(frame=frame, components=expressions), projection=ConservativeCellAverage()))
        transfer.state(block[model_state], StateTransfer())
    threshold = case.param(RuntimeParam("refine threshold", default=1 + .3 / np.sqrt(2)))
    layout = AMR(grid=CartesianGrid(frame=frame, cells=(n,n), periodic=PeriodicAxes(frame.axes)),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(rules=(Tag(ValueExpr(blocks[2][marker_state]) > case.value(threshold)),
                                  Buffer(cells=3*n//16-1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD), conflict_policy=ConflictPolicy.REFINE_WINS),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)), transfer=transfer,
        execution=AMRExecution.synchronous())
    return case, layout


def bind(n, *, guarded=False):
    case, layout = build(n, guarded=guarded)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    return pops.bind(artifact, resources={"execution_context": artifact_execution_context(artifact)})


def envelope(runtime):
    native = runtime._executor
    dirty = tuple(native.dirty_auxiliary_provider_identities())
    # The rank-local manifest includes grown payloads and accepted metadata even
    # before the initial dirty providers permit a durable auxiliary checkpoint.
    carriers = tuple(tuple(row) for row in native.checkpoint_rank_local_carrier_manifest())
    for kind in ("auxiliary-registry", "auxiliary-accepted-metadata"):
        assert sorted(int(row[3]) for row in carriers if row[1] == kind) == list(range(runtime.n_levels()))
    # A metadata-only manifest would miss a rollback that restores generations but
    # leaves published potential/gradient bytes behind. Each locally owned fluid
    # patch must carry at least all three observed potentials and both gradients.
    # Use patch identity rather than requiring local ownership on every MPI rank.
    local_fluid = {(row[3], row[5], row[7:11]) for row in carriers
                   if row[1] == "state" and row[2] == "fluid"}
    auxiliary_components = {}
    for row in carriers:
        if row[1] != "auxiliary":
            continue
        assert len(row) == 16  # Dim2: valid and grown boxes, component count, payload hash
        assert row[-1].startswith("pops.amr.rank-local-carrier.v1:sha256:")
        key = row[3], row[5], row[7:11]
        auxiliary_components[key] = auxiliary_components.get(key, 0) + int(row[6])
    assert all(auxiliary_components.get(key, 0) >= 5 for key in local_fluid)
    accepted = None if dirty else tuple(native.capture_auxiliary_checkpoint_accepted_state())
    if accepted is not None:
        assert len(accepted) == runtime.n_levels()
    return {
        "clock": (runtime.time(), runtime.macro_step()),
        "topology": tuple(runtime.patch_boxes()),
        "state": tuple(np.asarray(runtime.block_level_state_global(block, level)).tobytes()
                       for block in ("fluid", "load", "marker") for level in range(runtime.n_levels())),
        "histories": accepted_histories(runtime),
        "exchanges": native._program_exchange_records(),
        "flux_ledger": tuple(tuple(map(str, row)) for row in native.program_flux_ledger_manifest()),
        "diagnostics": tuple(sorted(native.program_diagnostics().items())),
        "dirty_providers": dirty,
        "carriers": carriers,
        "published_fields": tuple(row for row in carriers if row[1].startswith("auxiliary")),
        "accepted_auxiliary": accepted,
    }


def stationary_payload(runtime, n):
    # Covered coarse copies may be refreshed by conservative restriction; only leaf
    # values are independently evolved and must remain bitwise stationary here.
    return (tuple(runtime.patch_boxes()), tuple(
        np.asarray(runtime.block_level_state_global(block, level)).reshape(-1, n*2**level, n*2**level)
        [:, composite_active_mask(runtime, level, refinement_ratio=2)].tobytes()
        for block in ("load", "marker") for level in range(runtime.n_levels())))


def reference(n):
    z = (np.arange(n) + .5) / n
    waves = []
    for kx, ky in WAVENUMBERS:
        cx = np.cos(2*np.pi*kx*z) * np.sinc(kx/n)
        cy = np.cos(2*np.pi*ky*z) * np.sinc(ky/n)
        waves.append(cy[:, None] * cx[None, :])
    potential = MEANS[:, None, None] + AMPLITUDES[:, None, None] * np.stack(waves)
    kx, ky = WAVENUMBERS[2]
    cx, sx = np.cos(2*np.pi*kx*z)*np.sinc(kx/n), np.sin(2*np.pi*kx*z)*np.sinc(kx/n)
    cy, sy = np.cos(2*np.pi*ky*z)*np.sinc(ky/n), np.sin(2*np.pi*ky*z)*np.sinc(ky/n)
    gradient = -2*np.pi*AMPLITUDES[2] * np.stack((kx*cy[:, None]*sx[None, :], ky*sy[:, None]*cx[None, :]))
    return np.concatenate((potential, gradient))


def inspect_solution(runtime, n):
    assert runtime.n_levels() == 2
    coarse = composite_active_mask(runtime, 0, refinement_ratio=2)
    assert coarse.any() and (~coarse).any(), "qualification requires a partial refinement interface"
    squared, mean, raw = np.zeros(5), np.zeros(3), []
    for level in range(2):
        width = n * 2**level
        active = composite_active_mask(runtime, level, refinement_ratio=2)
        response = np.asarray(runtime.block_level_state_global("fluid", level)).reshape(5, width, width) / DT
        exact = reference(width)
        squared += np.sum((response[:, active] - exact[:, active])**2, axis=1) / width**2
        mean += np.sum(response[:3, active], axis=1) / width**2
        raw.append(response[2, active])
    np.testing.assert_allclose(mean, MEANS, rtol=0, atol=2e-8)
    np.testing.assert_allclose(np.asarray(MODES) @ mean, (2,5), rtol=0, atol=2e-8)
    assert abs(V @ mean) < 2e-8  # integrated reaction balance, independent of the authored gauge
    raw = np.concatenate(raw)
    assert raw.min() < 0 < raw.max(), "abs_sum must be tested on a sign-changing observation"
    # Separate the two oracles by more than the diagnostic comparison tolerance;
    # substituting signed sum for abs_sum must fail at every declared resolution.
    absolute_sum = np.abs(raw).sum()
    assert absolute_sum - raw.sum() > 2 * (2e-12 * absolute_sum + 2e-10)
    diagnostics = runtime._executor.program_diagnostics()
    for kind, expected in (("sum", raw.sum()), ("abs_sum", np.abs(raw).sum()), ("min", raw.min()), ("max", raw.max())):
        assert diagnostics["phi2_" + kind] == pytest.approx(float(expected), rel=2e-12, abs=2e-10)
    return np.sqrt(squared)


@pytest.mark.parametrize("n", RESOLUTIONS)
def test_public_joint_field_amr_source_gate(n):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    for guarded in (False, True):
        case, layout = build(n, guarded=guarded)
        resolved = pops.resolve(pops.validate(case), layout=layout)
        code = emit_cpp_program(case._time, model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
        for required in ("ctx.solve_hierarchy_field(", "ctx.stage_field_components(",
                         "ctx.publish_staged_field_components();", "ctx.observe_hierarchy_field_gradient(",
                         "ctx.reduce_hierarchy_field_component("):
            assert required in code
        assert code.index(".observe(hierarchy_dt)") < code.index("ctx.publish_staged_field_components();") < code.index(".publish(hierarchy_dt)")


@pytest.mark.compiler
@pytest.mark.native_loader
def test_public_joint_field_amr_full_resolution_matrix(isolated_native_cache, native_cxx, kokkos_root, record_property):
    errors = []
    for n in RESOLUTIONS:
        runtime = bind(n)
        fixed = stationary_payload(runtime, n)
        report = pops.run(runtime, t_end=DT, max_steps=1, console=False)
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        errors.append(inspect_solution(runtime, n))
        assert fixed == stationary_payload(runtime, n)
    errors = np.asarray(errors)
    orders = np.log2(errors[:-1] / errors[1:])
    # Composite potential accuracy and cell-centred gradient accuracy are assessed separately.
    assert np.all(orders[:, :3] > 1.5), (errors, orders)
    assert np.all(orders[:, 3:] > 1.0), (errors, orders)
    assert np.max(errors[-1, :3]) < .015 and np.max(errors[-1, 3:]) < .2, errors
    record_property("public_composite_field_matrix", {"resolutions": RESOLUTIONS, "l2": errors.tolist(), "orders": orders.tolist()})


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.parametrize("n", RESOLUTIONS)
def test_public_joint_field_publication_rollback_and_same_instance_retry(n, isolated_native_cache, native_cxx, kokkos_root):
    retried = bind(n, guarded=True)
    before = envelope(retried)
    with pytest.raises(RuntimeError, match="(?i)(invalid|evaluation|admissible|transform)"):
        pops.run(retried, t_end=10*DT, max_steps=1, console=False)
    assert envelope(retried) == before
    fresh = bind(n, guarded=True)
    for runtime in (retried, fresh):
        report = pops.run(runtime, t_end=DT, max_steps=1, console=False)
        assert report.accepted_steps == 1
        inspect_solution(runtime, n)
    assert envelope(retried) == envelope(fresh)

    # Repeat rejection after a real accepted publication: initial dirty-provider
    # rollback alone cannot prove restoration of an existing accepted value/point.
    accepted = envelope(retried)
    assert accepted["published_fields"] != before["published_fields"]
    with pytest.raises(RuntimeError, match="(?i)(invalid|evaluation|admissible|transform)"):
        pops.run(retried, t_end=11*DT, max_steps=1, console=False)
    assert envelope(retried) == accepted
    for runtime in (retried, fresh):
        report = pops.run(runtime, t_end=2*DT, max_steps=1, console=False)
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        assert runtime.time() == 2*DT and runtime.macro_step() == 2
    assert envelope(retried) == envelope(fresh)
