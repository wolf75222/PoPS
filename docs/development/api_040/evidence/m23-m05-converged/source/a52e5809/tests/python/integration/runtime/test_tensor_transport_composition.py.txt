"""One public composed update retains transport, cross-tensor diffusion and their ledgers."""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops import math
from pops._native_collectives import allgather_value
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (
    DiscretizationPlan, FiniteVolume, TensorDiffusion, reconstruction, riemann, variables,
)
from pops.projection import ConservativeCellAverage
from pops.solvers import Newton
from pops.time import DerivativeStrategy, FailRun, FixedDt, ImplicitDiffusionStage, SolveUnknown
from tests.python.integration.runtime.test_multicomponent_diffusion_exchange_ledger import (
    _global_exchange_records,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
N, VALID_DT = 16, 2e-4
SPEEDS = (.7, -.4)
TENSOR = np.array(((.004, .0006), (.0006, .002)))
GRADIENT_JACOBIAN = np.array(((2., .25), (.25, 1.5)))


def composition_case(*, n=N, dt=VALID_DT, dimension=2, implicit=False, competing_transport=False):
    frame = CartesianDomain("composition-domain", lower=(0.,)*dimension,
                            upper=(1.,)*dimension).frame(
        (Cartesian1D, Cartesian2D, Cartesian3D)[dimension-1]())
    model = pops.Model("composed_inventory", frame=frame)
    state = model.state("inventory", components=("first", "second"))
    u, v = state
    speeds = (.7, -.4, .2)[:dimension]
    transport = model.flux("transport", frame=frame, state=state,
        components={axis: tuple(speed*q for q in state)
                    for axis, speed in zip(frame.axes, speeds, strict=True)},
        waves={axis: (speed, speed) for axis, speed in zip(frame.axes, speeds, strict=True)})
    tensor = tuple(tuple(.004 if row == column == 0 else .002 if row == column else .0006
                         for column in range(dimension)) for row in range(dimension))
    diffusion = model.diffusive_flux("conduction", state=state, value=(
        math.CoeffGradient(2*u+.25*v, tensor), math.CoeffGradient(.25*u+1.5*v, tensor)))
    rhs = -math.div(transport)+math.div(diffusion)
    if competing_transport:
        other = model.flux("other_transport", frame=frame, state=state,
            components={axis: tuple(.2*q for q in state) for axis in frame.axes})
        rhs = rhs-math.div(other)
    balance = model.rate("balance", equation=math.ddt(state) == rhs)
    advective, diffusive = balance.select(transport), balance.select(diffusion)
    if competing_transport:
        other_rate = balance.select(other)
    case = pops.Case("tensor_transport_composition")
    block = case.block("mixture", model, states=(state,))
    methods = DiscretizationPlan()
    methods.rates.add(advective, FiniteVolume(flux=transport,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov()))
    methods.rates.add(diffusive, TensorDiffusion(flux=diffusion))
    if competing_transport:
        methods.rates.add(other_rate, FiniteVolume(flux=other,
            variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.HLL()))
    case.numerics(methods, block=block)
    program = pops.Program("partition_composition")
    q = program.state(block[state])
    if implicit:
        predictor = program.value("transport_predictor", q.n+program.dt*advective(q.n),
                                  at=q.next.point)
        coordinate = program.value("coordinate", q.n, at=q.next.point)
        request = ImplicitDiffusionStage(diffusive, predictor, program.dt).request(
            unknown=SolveUnknown("coordinate", coordinate), seed=q.n,
            derivative=DerivativeStrategy("finite_difference"))
        solved = program.solve(request, solver=Newton(
            tolerance=1e-12, max_iterations=20, linear_tolerance=1e-8,
            linear_max_iterations=100, restart=30)).consume(action=FailRun())[0]
        candidate = program.value("accepted", solved, at=q.next.point)
    else:
        rate = advective(q.n)+diffusive(q.n)
        if competing_transport:
            rate = rate+other_rate(q.n)
        candidate = program.value("accepted", q.n+program.dt*rate, at=q.next.point)
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(n,)*dimension,
                                  periodic=PeriodicAxes(frame.axes)))
    return case, layout, model


def fourier_oracle(n=N, *, dt=VALID_DT, implicit=False):
    """Exact symbols of upwind transport and the declared negative-adjoint G/H operator."""
    y, x = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n, indexing="ij")
    initial = np.broadcast_to(np.array((2., 3.))[:, None, None], (2, n, n)).copy()
    expected = initial.copy()
    transport = np.zeros_like(initial)
    diffusion = np.zeros_like(initial)
    for wave, amplitudes in (((1, 1), (.3, .1)), ((2, -1), (-.07j, -.2j))):
        theta = 2*np.pi*np.asarray(wave)/n
        phase = np.exp(2j*np.pi*(wave[0]*x+wave[1]*y))*np.prod(np.sinc(np.asarray(wave)/n))
        amplitudes = np.asarray(amplitudes)
        upwind = -sum(abs(speed)*n*(1-np.exp(-1j*np.sign(speed)*angle))
                      for speed, angle in zip(SPEEDS, theta, strict=True))
        gradient = n*np.sin(theta)
        stabilization = 2*n*np.sin(theta/2)**2
        adjoint = -(gradient@TENSOR@gradient+np.diag(TENSOR)@(stabilization**2))
        predictor = (1+dt*upwind)*amplitudes
        if implicit:
            updated = np.linalg.solve(np.eye(2)-dt*adjoint*GRADIENT_JACOBIAN, predictor)
        else:
            updated = predictor+dt*adjoint*(GRADIENT_JACOBIAN@amplitudes)
        initial += np.real(amplitudes[:, None, None]*phase)
        expected += np.real(updated[:, None, None]*phase)
        transport += np.real(upwind*amplitudes[:, None, None]*phase)
        diffusive_state = updated if implicit else amplitudes
        diffusion += np.real(adjoint*(GRADIENT_JACOBIAN@diffusive_state)[:, None, None]*phase)
    return initial, expected, transport, diffusion


def _bind(dt, *, implicit=False):
    case, layout, _ = composition_case(dt=dt, implicit=implicit)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    context = artifact_execution_context(artifact)
    initial = np.ascontiguousarray(fourier_oracle()[0])
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    runtime = pops.bind(artifact, initial_values={subject: initial},
                        resources={"execution_context": context})
    return runtime, context, initial


@pytest.mark.parametrize("implicit", (False, True), ids=("explicit", "implicit"))
def test_separate_transport_tensor_update_and_signed_ledgers(
    isolated_native_cache, native_cxx, kokkos_root, record_property, implicit,
):
    del isolated_native_cache, native_cxx, kokkos_root
    runtime, context, initial = _bind(VALID_DT, implicit=implicit)
    _, expected, transport, diffusion = fourier_oracle(implicit=implicit)
    assert np.linalg.norm(transport) > .1 and np.linalg.norm(diffusion) > .1
    report = pops.run(runtime, t_end=VALID_DT, max_steps=1, console=False)
    assert report.accepted_steps == 1 and report.rejected_steps == 0
    actual = np.asarray(runtime.state_global("mixture")).reshape(initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-13)
    if implicit:
        diagnostics = runtime.program_report().diagnostics
        residuals = [name for name in diagnostics if name.endswith(".residual_norm")]
        assert residuals, "the accepted implicit tensor solve must report its actual residual"
        for name in residuals:
            reference = diagnostics[name.removesuffix(".residual_norm")+".reference_residual_norm"]
            assert diagnostics[name] <= 1e-12*max(1., reference)
    records = _global_exchange_records(runtime, context)
    assert len(records) == 2*2*4*N*N  # components, physical fluxes, cell-face incidences
    increments = {kind: np.zeros_like(initial) for kind in ("transport", "diffusion")}
    occurrences = {kind: set() for kind in increments}
    contexts = {kind: set() for kind in increments}
    identities = set()
    for row in records:
        cell_token, axis_token, side_token, component_token = row["quadrature_identity"].split("/")
        i, j = map(int, cell_token.split(":")[1:])
        axis, side, component = (int(token.split(":")[1])
                                 for token in (axis_token, side_token, component_token))
        assert axis in (0, 1) and side in (0, 1) and component in (0, 1)
        identity = (row["operation_identity"], row["occurrence_identity"],
                    row["evaluation_context"], row["quadrature_identity"])
        assert identity not in identities
        identities.add(identity)
        kind = "transport" if row["orientation"] == (1 if side == 0 else -1) else "diffusion"
        assert row["orientation"] in (-1, 1)
        occurrences[kind].add(row["occurrence_identity"])
        contexts[kind].add(row["evaluation_context"])
        assert row["multiplicity"] == 1 and row["face_measure"] == 1/N
        assert row["temporal_weight"] == VALID_DT
        assert row["integrated_amount"] == (
            row["orientation"]*row["face_measure"]*row["numerical_flux"]*row["temporal_weight"])
        increments[kind][component, j, i] += row["integrated_amount"]
    assert len(occurrences["transport"]) == len(occurrences["diffusion"]) == 1
    assert occurrences["transport"].isdisjoint(occurrences["diffusion"])
    assert all(len(values) == 1 for values in contexts.values())
    for kind, rate in (("transport", transport), ("diffusion", diffusion)):
        np.testing.assert_allclose(increments[kind], VALID_DT*rate/N**2, rtol=0, atol=2e-13)
        np.testing.assert_allclose(increments[kind].sum(axis=(1, 2)), 0, rtol=0, atol=2e-13)
    np.testing.assert_allclose((actual-initial)/N**2, sum(increments.values()), rtol=0, atol=2e-13)
    ranks = (1 if context.communicator.identity == "serial"
             else len(allgather_value(context.communicator.handle, None)))
    record_property("mpi_ranks", ranks)
    record_property("cells_per_axis", N)
    record_property("tensor_partition", "implicit" if implicit else "explicit")
    record_property("maximum_update_defect", float(np.max(abs(actual-expected))))


def test_independent_bounds_do_not_admit_a_sum_exceeding_the_combined_bound(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    transport_frequency = max(map(abs, SPEEDS))*2*N
    tensor_frequency = (2.5*np.linalg.norm(TENSOR, ord=np.inf)
                        *np.linalg.norm(GRADIENT_JACOBIAN, ord=np.inf)*2*N**2)
    dt = .9/max(transport_frequency, tensor_frequency)
    assert dt*transport_frequency < 1 and dt*tensor_frequency < 1
    assert dt*(transport_frequency+tensor_frequency) > 1
    runtime, _, initial = _bind(dt)
    before = (runtime.time(), runtime.macro_step(), runtime._executor._program_exchange_records(),
              runtime.program_report().diagnostics.copy())
    with pytest.raises(RuntimeError, match="combined_transport_diffusion_stability|rejected"):
        pops.run(runtime, t_end=dt, max_steps=1, console=False)
    assert (runtime.time(), runtime.macro_step(), runtime._executor._program_exchange_records(),
            runtime.program_report().diagnostics) == before
    np.testing.assert_array_equal(np.asarray(runtime.state_global("mixture")).reshape(initial.shape), initial)
