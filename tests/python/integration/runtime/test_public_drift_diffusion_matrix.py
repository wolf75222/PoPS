"""Full declared Dim1 fitted-flux matrix; every row uses the public native lifecycle."""
from __future__ import annotations
import json
import math as pmath
import os
from pathlib import Path

import numpy as np
import pops
import pytest
from pops import math
from pops.codegen import Production
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.provider_pack import ProviderPack
from pops.numerics import Diffusion, DiscretizationPlan, ScharfetterGummel, StateStorage
from pops.physics.diffusion import DiffusiveBoundary
from pops.time import FixedDt, Program
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
REFINEMENTS = (16, 32, 64)


def _record(name, payload):
    destination = os.environ.get("POPS_DIFFUSION_QUALIFICATION_EVIDENCE_DIR")
    if destination:
        root = Path(destination)
        root.mkdir(parents=True, exist_ok=True)
        (root / (name+".json")).write_text(json.dumps(payload, indent=2)+"\n")


def _case(n, dt, *, mobility=.1, manufactured=False, physical=False, transport=None):
    frame = CartesianDomain("line", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("drift_diffusion", frame=frame)
    state = model.state("U", components=("n",))
    density = state[0]
    case = pops.Case("fitted_case")
    density_bc = potential_bc = None
    if physical:
        density_bc = (DiffusiveBoundary(0, "lower", "value", 1.),
                      DiffusiveBoundary(0, "upper", "value", pmath.exp(-1)))
        potential_bc = (DiffusiveBoundary(0, "lower", "value", 0.),
                        DiffusiveBoundary(0, "upper", "value", 1.))
    diffusion = model.diffusive_flux("diffusion", state=state,
                                    value=.1*math.grad(state), boundaries=density_bc)
    if transport is None:
        potential = model.aux("potential")
        factor = model.aux("source_factor") if manufactured else None
        drift = model.drift_flux("drift", state=state, mobility=mobility,
                                 potential=potential, boundaries=potential_bc)
        rhs = -math.div(drift)+math.div(diffusion)
        if manufactured:
            rhs += model.source("manufactured", on=state, value=(factor*density,))
        method = ScharfetterGummel(drift=drift, flux=diffusion)
    else:
        from pops.numerics import FiniteVolume, variables, reconstruction, riemann
        advective = model.flux("transport", frame=frame, state=state,
            components={frame.axes[0]: (transport*density,)},
            waves={frame.axes[0]: (transport,)})
        rhs = -math.div(advective)+math.div(diffusion)
        method = Diffusion(flux=diffusion, transport=FiniteVolume(flux=advective,
            variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov()))
    rate = model.rate("physical_rate", equation=math.ddt(state) == rhs)
    block = case.block("density", model, states=(state,))
    plan = DiscretizationPlan()
    plan.rates.add(rate, method)
    case.numerics(plan, block=block)
    if transport is None:
        program = Program("fitted_euler")
        values = program.state(block[state])
        fields = program.input_fields(values.n, for_rate=rate)
        end = program.value("end", values.n+program.dt*rate(values.n, fields), at=values.next.point)
        program.commit(values.next, end)
    else:
        program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, Uniform(CartesianGrid(frame=frame, cells=(n,),
        periodic=None if physical else PeriodicAxes(frame.axes)))


def _bind(case, layout, initial, inputs):
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout, backend=Production()))
    pack = ProviderPack.from_data(artifact.plan.blocks[0].resolved_operations.to_data()
                                  ["provider_evidence"]["auxiliary"])
    keys = {key.component: key for key in pack if pack.declared_entry(key).producer == "runtime_input"}
    assert set(keys) == set(inputs)
    runtime = pops.bind(artifact, initial_state={"density": np.ascontiguousarray(initial[None])},
        aux={keys[name]: np.ascontiguousarray(value) for name, value in inputs.items()},
        resources={"execution_context": artifact_execution_context(artifact)})
    return runtime, artifact


def _run(runtime, final, steps):
    report = pops.run(runtime, t_end=final, max_steps=steps+1, console=False)
    assert steps <= report.accepted_steps <= steps+1
    if report.accepted_steps != steps:
        assert 0 < runtime._executor.program_last_dt() <= np.spacing(final)*max(4, steps)
    assert abs(runtime.time()-final) <= 2e-13
    return np.asarray(runtime.state_global("density")).reshape(-1)


def _solved_potential_case(n, dt):
    from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, SharedMeanGauge, bcs
    from pops.fields.methods import CellCenteredSecondOrder
    from pops.model import Handle, OwnerPath
    from pops.solvers import CG
    from pops.time import FailRun
    frame = CartesianDomain("solved-line", (0.,), (1.,)).frame(Cartesian1D())
    case = pops.Case("solved-potential-sg")
    driver_model = pops.Model("frozen-Poisson-driver", frame=frame)
    driver = driver_model.state("U", components=("charge",))
    frozen = driver_model.source("frozen", on=driver, value=(0*driver[0],))
    driver_rate = driver_model.rate("frozen-law", equation=math.ddt(driver) == frozen)
    driver_block = case.block("driver", driver_model)
    driver_numerics = DiscretizationPlan()
    driver_numerics.rates.add(driver_rate, StateStorage())
    case.numerics(driver_numerics, block=driver_block)
    model = pops.Model("solved-drift-diffusion", frame=frame)
    state = model.state("U", components=("n",))
    potential = model.aux("potential")
    drift = model.drift_flux("drift", state=state, mobility=.1, potential=potential)
    diffusion = model.diffusive_flux("diffusion", state=state, value=.1*math.grad(state))
    rate = model.rate("physical-rate", equation=math.ddt(state) == -math.div(drift)+math.div(diffusion))
    block = case.block("density", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, ScharfetterGummel(drift=drift, flux=diffusion))
    case.numerics(numerics, block=block)
    unknown = Handle("phi", kind="field", owner=OwnerPath.model("physical-Poisson"))
    problem = FieldProblem("Poisson", unknowns=(unknown,), equations=(-math.laplacian(unknown) == driver[0],),
        boundaries=(FieldBoundary(unknown, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic())),),
        gauge=SharedMeanGauge((unknown,)))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
        solver=CG(max_iter=4000, rel_tol=1e-12, abs_tol=1e-13)))
    program = Program("solved-potential-step")
    driver_time, density_time = program.state(driver_block[driver]), program.state(block[state])
    point = program.stage("potential", c=0)
    solution = field.observe(program.solve(field, values={driver_block[driver]:driver_time.n}, at=point)
                              .consume(action=FailRun()))
    module = model.module
    carrier = block[module.field_handle(module.field_spaces()["fields"])]
    solved = solution[field[unknown]]
    context = solution.publish({(carrier,"potential"):solved}, states={block[state]:density_time.n})
    rhs = rate(density_time.n, context)
    end = program.value("end", density_time.n+program.dt*rhs, at=density_time.next.point)
    frozen_end = program.value("driver-end", 1*driver_time.n, at=driver_time.next.point)
    program.commit_many({density_time.next:end,driver_time.next:frozen_end})
    program.store_history("solved-potential", solved, depth=1)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    return case, Uniform(CartesianGrid(frame=frame,cells=(n,),periodic=PeriodicAxes(frame.axes)))


def test_full_sg_manufactured_refinement(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence, errors = [], []
    for n in REFINEMENTS:
        x = (np.arange(n)+.5)/n
        k, d, mu = 2*np.pi, .1, .1
        f = 2+np.sin(k*x)
        phi = .4*np.cos(k*x)
        divergence = -d*k*k*np.sin(k*x)+mu*(k*np.cos(k*x)*(-.4*k*np.sin(k*x))
                                                   + f*(-.4*k*k*np.cos(k*x)))
        inputs = {"potential": phi, "source_factor": -1-divergence/f}
        initial = 2+np.sinc(1/n)*np.sin(k*x)
        final = .05
        steps = pmath.ceil(final/(.025/(d*n*n)))
        case, layout = _case(n, final/steps, manufactured=True)
        runtime, artifact = _bind(case, layout, initial, inputs)
        actual = _run(runtime, final, steps)
        error = actual-np.exp(-final)*initial
        norms = [float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2))),
                 float(np.max(np.abs(error)))]
        errors.append(norms)
        assert actual.min() > 0
        evidence.append({"n": n, "dt": final/steps, "planned_steps": steps,
            "accepted_steps": runtime.macro_step(), "norms_L1_L2_Linf": norms,
            "artifact": artifact.artifact_identity.token})
        _record("diffusion-sg-mms-progress", evidence)
    assert np.all(np.asarray(errors)[1:] < np.asarray(errors)[:-1])
    orders = np.log2(np.asarray(errors)[-2]/np.asarray(errors)[-1])
    assert min(orders) >= 1.75
    _record("diffusion-sg-mms", {"rows": evidence, "final_orders": orders.tolist()})


def test_full_sg_boundary_equilibrium_and_oriented_exchanges(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence = []
    for n in REFINEMENTS:
        x = (np.arange(n)+.5)/n
        initial = np.exp(-x)
        final, bound = .05, .025/(.1*n*n)
        steps = pmath.ceil(final/bound)
        case, layout = _case(n, final/steps, physical=True)
        runtime, artifact = _bind(case, layout, initial, {"potential": x})
        actual = _run(runtime, final, steps)
        records = runtime._executor._program_exchange_records()
        assert len(records) == 2*n
        assert all("joint-occurrences:0,1" in record["occurrence_identity"] for record in records)
        flux = max(abs(record["numerical_flux"]) for record in records)
        drift = float(np.max(np.abs(actual-initial)))
        assert flux <= 2e-12
        assert drift <= 2e-11
        evidence.append({"n": n, "accepted_steps": runtime.macro_step(), "face_flux": flux,
                         "state_drift": drift, "records": len(records),
                         "artifact": artifact.artifact_identity.token})
    _record("diffusion-sg-boundary-equilibrium", evidence)


def test_full_sg_zero_jump_is_centered_diffusion(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence = []
    for n in REFINEMENTS:
        x = (np.arange(n)+.5)/n
        initial = 2+.4*np.sin(2*np.pi*x)
        dt = .025/(.1*n*n)
        case, layout = _case(n, dt)
        runtime, artifact = _bind(case, layout, initial, {"potential": np.zeros_like(x)})
        actual = _run(runtime, dt, 1)
        expected = initial+dt*.1*n*n*(np.roll(initial, 1)-2*initial+np.roll(initial, -1))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-13)
        records = runtime._executor._program_exchange_records()
        delta = np.zeros(n)
        for record in records:
            cell = int(record["quadrature_identity"].split("/")[0].split(":")[1])
            delta[cell] += record["integrated_amount"]
        np.testing.assert_allclose((actual-initial)/n, delta, rtol=0, atol=2e-11)
        evidence.append({"n": n, "max_error": float(np.max(np.abs(actual-expected))),
                         "artifact": artifact.artifact_identity.token})
    _record("diffusion-sg-zero-jump", evidence)


def test_full_sg_consumes_actual_solved_potential_at_faces(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence = []
    for n in REFINEMENTS:
        x = (np.arange(n)+.5)/n
        charge = .4*(2*np.pi)**2*np.cos(2*np.pi*x)
        # Independent eigenvalue of the selected centered periodic Laplacian.
        expected_potential = charge/(4*n*n*np.sin(np.pi/n)**2)
        initial = np.exp(-expected_potential)
        final = .05
        steps = pmath.ceil(final/(.025/(.1*n*n)))
        case, layout = _solved_potential_case(n,final/steps)
        resolved = pops.resolve(pops.validate(case),layout=layout)
        claims = next(block.resolved_operations.provider_evidence["program_field_publications"]
                      for block in resolved.blocks if block.name == "density")
        assert len(claims) == 1 and claims[0]["key"]["component"] == "potential"
        artifact = pops.compile(resolved)
        runtime = pops.bind(artifact,initial_state={"density":initial[None],"driver":charge[None]},
            resources={"execution_context":artifact_execution_context(artifact)})
        actual = _run(runtime,final,steps)
        observed = np.asarray(runtime.history_global("solved-potential",0)).reshape(-1)
        np.testing.assert_allclose(observed,expected_potential,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(np.asarray(runtime.state_global("driver")).reshape(-1),charge)
        records = runtime._executor._program_exchange_records()
        flux = max(abs(record["numerical_flux"]) for record in records)
        drift = float(np.max(np.abs(actual-initial)))
        assert flux <= 2e-12 and drift <= 2e-11
        evidence.append({"n":n,"accepted_steps":runtime.macro_step(),"face_flux":flux,
            "state_drift":drift,"potential_error":float(np.max(np.abs(observed-expected_potential))),
            "artifact":artifact.artifact_identity.token})
        _record("diffusion-sg-solved-potential-progress",evidence)
    _record("diffusion-sg-solved-potential",evidence)


@pytest.mark.parametrize("c,r,accepted", ((.6,.3,False), (.2,.2,True)))
def test_dim1_actual_combined_transport_diffusion_bound(isolated_native_cache, native_cxx, kokkos_root, c, r, accepted):
    del isolated_native_cache, native_cxx, kokkos_root
    n = 32
    dt = r/(.1*n*n)
    speed = c/(dt*n)
    assert c <= 1 and 2*r <= 1
    initial = np.zeros(n)
    initial[n//2] = 1
    case, layout = _case(n, dt, transport=speed)
    runtime, artifact = _bind(case, layout, initial, {})
    if accepted:
        actual = _run(runtime, dt, 1)
        expected = (r+c)*np.roll(initial, 1)+(1-c-2*r)*initial+r*np.roll(initial, -1)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        assert actual.min() >= 0
    else:
        with pytest.raises(RuntimeError, match="combined_transport_diffusion_stability|pointwise|rejected"):
            pops.run(runtime, t_end=dt, max_steps=1, console=False)
        np.testing.assert_array_equal(np.asarray(runtime.state_global("density")).reshape(-1), initial)
        assert runtime.time() == 0 and runtime.macro_step() == 0
        assert not runtime._executor._program_exchange_records()
    _record("diffusion-dim1-combined-"+str(accepted), {"c":c,"r":r,"accepted":accepted,
                                                    "artifact":artifact.artifact_identity.token})
