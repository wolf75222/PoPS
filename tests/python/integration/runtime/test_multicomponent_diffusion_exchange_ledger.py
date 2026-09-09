"""Signed accepted component quadrature for public coupled transport/diffusion."""
from fractions import Fraction

import numpy as np
import pops
import pytest
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.integration.runtime.test_combined_diffusion_exchange_ledger import _rates

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
REFINEMENTS = (16, 32, 64)
SPEEDS, DIFFUSIVITIES = (.7, -.4), (.1, .17)


def component_case(n, dt, scheme):
    frame = Rectangle("component_ledger", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("two_inventories", frame=frame)
    state = model.state("inventory", components=("energy", "species"))
    flux = model.flux("transport", frame=frame, state=state,
        components={axis: tuple(speed*q for q in state)
                    for axis, speed in zip(frame.axes, SPEEDS, strict=True)},
        waves={axis: (speed, speed) for axis, speed in zip(frame.axes, SPEEDS, strict=True)})
    diffusion = model.diffusive_flux("conduction", state=state,
        value=tuple(nu*math.grad(q) for nu, q in zip(DIFFUSIVITIES, state, strict=True)))
    rate = model.rate("balance", equation=math.ddt(state)==-math.div(flux)+math.div(diffusion))
    case = pops.Case("component_quadrature")
    block = case.block("mixture", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=diffusion, transport=FiniteVolume(
        flux=flux, variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov() if scheme=="rusanov" else riemann.HLL())))
    case.numerics(plan, block=block)
    program = pops.Program("two_stages")
    q = program.state(block[state])
    first = rate(q.n)
    predictor = program.value("predictor", q.n+program.dt*first, at=program.stage("predictor", c=1))
    last = rate(predictor)
    candidate = program.value("accepted", q.n+Fraction(1,2)*program.dt*(first+last), at=q.next.point)
    program.commit(q.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, Uniform(CartesianGrid(frame=frame, cells=(n,n), periodic=PeriodicAxes(frame.axes)))


def _bind(n, dt, scheme, initial):
    case, layout = component_case(n, dt, scheme)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    return pops.bind(artifact, initial_state={"mixture": np.ascontiguousarray(initial)},
        resources={"execution_context": artifact_execution_context(artifact)})


def _initial(n):
    y,x = np.meshgrid((np.arange(n)+.5)/n, (np.arange(n)+.5)/n, indexing="ij")
    return np.stack((2+.3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y),
                     3+.2*np.cos(4*np.pi*x)+.1*np.sin(2*np.pi*y)))


@pytest.mark.parametrize("scheme", ("rusanov", "hll"))
def test_component_signed_values_exact_multiplicity_and_accepted_stages(
        isolated_native_cache, native_cxx, kokkos_root, scheme):
    del isolated_native_cache, native_cxx, kokkos_root
    for n in REFINEMENTS:
        initial = _initial(n)
        dt = .15/(4*max(DIFFUSIVITIES)*n*n+sum(map(abs,SPEEDS))*n)
        first_rates = tuple(_rates(q, SPEEDS, nu) for q,nu in zip(initial,DIFFUSIVITIES,strict=True))
        predictor = initial+dt*np.stack([sum(row) for row in first_rates])
        last_rates = tuple(_rates(q, SPEEDS, nu) for q,nu in zip(predictor,DIFFUSIVITIES,strict=True))
        expected = .5*(initial+predictor+dt*np.stack([sum(row) for row in last_rates]))
        runtime = _bind(n, dt, scheme, initial)
        report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
        assert report.accepted_steps == 1
        actual = np.asarray(runtime.state_global("mixture")).reshape(initial.shape)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        records = runtime._executor._program_exchange_records()
        assert len(records) == 2*2*8*n*n  # components, stages, two fluxes with four incidences
        contexts = tuple(dict.fromkeys(row["evaluation_context"] for row in records))
        assert len(contexts)==2
        stage_states = dict(zip(contexts, (initial,predictor), strict=True))
        increments = {kind:np.zeros_like(initial) for kind in ("transport","diffusion")}
        identities = set()
        for row in records:
            cell_token, axis_token, side_token, component_token = row["quadrature_identity"].split("/")
            i,j = map(int,cell_token.split(":")[1:])
            axis,side,component = (int(token.split(":")[1]) for token in
                                   (axis_token,side_token,component_token))
            identity = (row["occurrence_identity"],row["evaluation_context"],row["quadrature_identity"])
            assert identity not in identities
            identities.add(identity)
            kind = "transport" if row["orientation"] == (1 if side==0 else -1) else "diffusion"
            assert row["multiplicity"]==1 and row["face_measure"]==1/n
            assert row["temporal_weight"]==dt/2
            state = stage_states[row["evaluation_context"]][component]
            cell = (j,i)
            neighbor = list(cell)
            neighbor[1-axis] = (neighbor[1-axis]+(1 if side else -1)) % n
            left,right = (state[cell],state[tuple(neighbor)]) if side else (state[tuple(neighbor)],state[cell])
            oracle = (SPEEDS[axis]*(left if SPEEDS[axis]>=0 else right) if kind=="transport"
                      else DIFFUSIVITIES[component]*n*(right-left))
            assert abs(row["numerical_flux"]-oracle) < 2e-12
            increments[kind][component,j,i] += row["integrated_amount"]
        for index,kind in enumerate(("transport","diffusion")):
            expected_increment = dt/(2*n*n)*np.stack([a[index]+b[index]
                for a,b in zip(first_rates,last_rates,strict=True)])
            np.testing.assert_allclose(increments[kind], expected_increment, rtol=0, atol=2e-13)
            np.testing.assert_allclose(np.sum(increments[kind],axis=(1,2)),0,rtol=0,atol=2e-13)
        np.testing.assert_allclose((actual-initial)/n**2, sum(increments.values()),rtol=0,atol=2e-13)


@pytest.mark.parametrize("scheme", ("rusanov", "hll"))
def test_component_rejection_publishes_no_quadrature(isolated_native_cache, native_cxx, kokkos_root, scheme):
    del isolated_native_cache, native_cxx, kokkos_root
    n = 16
    dt = 2/(4*max(DIFFUSIVITIES)*n*n+sum(map(abs,SPEEDS))*n)
    initial = _initial(n)
    runtime = _bind(n,dt,scheme,initial)
    before = (runtime.time(),runtime.macro_step(),runtime._executor._program_exchange_records(),
              runtime.program_report().diagnostics)
    with pytest.raises(RuntimeError,match="combined_transport_diffusion_stability|rejected"):
        pops.run(runtime,t_end=dt,max_steps=1,console=False)
    assert (runtime.time(),runtime.macro_step(),runtime._executor._program_exchange_records(),
            runtime.program_report().diagnostics)==before
    np.testing.assert_array_equal(np.asarray(runtime.state_global("mixture")).reshape(initial.shape),initial)
