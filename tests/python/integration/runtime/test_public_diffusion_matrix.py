"""Predeclared full public explicit diffusion matrix, with independent FV oracles."""
from __future__ import annotations

import json
import math as pmath
import os
from pathlib import Path

import numpy as np
import pops
import pytest
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler, SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.provider_pack import ProviderPack
from pops.numerics import Diffusion, DiscretizationPlan
from pops.physics.diffusion import DiffusiveBoundary
from pops.time import FixedDt
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
REFINEMENTS = (16, 32, 64)


def _record(name, payload):
    destination = os.environ.get("POPS_DIFFUSION_QUALIFICATION_EVIDENCE_DIR")
    if destination:
        path = Path(destination)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(json.dumps(payload, indent=2) + "\n")


def _centers(n):
    return np.meshgrid((np.arange(n) + .5)/n, (np.arange(n) + .5)/n, indexing="xy")


def _average(function, n):
    """Five-point tensor Gauss quadrature, exact for the declared polynomial MMS."""
    x, y = _centers(n)
    nodes, weights = np.polynomial.legendre.leggauss(5)
    result = np.zeros_like(x)
    for a, wa in zip(nodes, weights, strict=True):
        for b, wb in zip(nodes, weights, strict=True):
            result += .25*wa*wb*function(x+a/(2*n), y+b/(2*n))
    return result


def _smooth(x, y):
    return 1+x+y+16*x*x*(1-x)**2*y*y*(1-y)**2


def _smooth_source(x, y):
    def p(z):
        return z*z*(1-z)**2
    def dp(z):
        return 2*z-6*z*z+4*z**3
    def ddp(z):
        return 2-12*z+12*z*z
    return -.75-16*(((1+.25*x)*ddp(x)+.25*dp(x))*p(y)
                   + ((2+.5*y)*ddp(y)+.5*dp(y))*p(x))


def _build(kind, n, dt, *, method="euler", transport=None, physical=None, cells=None):
    frame = Rectangle("diffusion-square", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("qualified_diffusion", frame=frame)
    state = model.state("U", components=("u",))
    u = state[0]
    fields = {}
    source = None
    coefficient = .1
    variable = u**3 if kind == "nonlinear" else u
    if kind == "variable":
        coefficient = model.aux("diffusivity")
        factor = model.aux("source_factor")
        source = model.source("manufactured", on=state, value=(factor*u,))
        fields = {"diffusivity": coefficient, "source_factor": factor}
    elif kind.startswith("diagonal"):
        ax, ay, forcing = (model.aux(name) for name in ("a_x", "a_y", "forcing"))
        coefficient = ((ax, 0), (0, ay))
        source = model.source("manufactured", on=state, value=(forcing,))
        fields = {"a_x": ax, "a_y": ay, "forcing": forcing}
        physical = tuple(DiffusiveBoundary(axis, side, "value", 1., (1., 1.))
                         if axis == 0 else DiffusiveBoundary(axis, side, "conormal",
                                                            -2. if side == "lower" else 2.5)
                         for axis in range(2) for side in ("lower", "upper"))
    flux = model.diffusive_flux("conduction", state=state,
        value=math.CoeffGradient(variable, coefficient), boundaries=physical)
    rhs = math.div(flux)
    transport_method = None
    if transport is not None:
        from pops.numerics import FiniteVolume, variables, reconstruction, riemann
        adv = model.flux("advection", frame=frame, state=state,
            components={axis: (speed*u,) for axis, speed in zip(frame.axes, transport, strict=True)},
            waves={axis: (speed,) for axis, speed in zip(frame.axes, transport, strict=True)})
        rhs = -math.div(adv) + rhs
        transport_method = FiniteVolume(flux=adv, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    if source is not None:
        rhs = rhs + source
    rate = model.rate("physical_balance", equation=math.ddt(state) == rhs)
    case = pops.Case("explicit_diffusion_"+kind)
    block = case.block("heat", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux, transport=transport_method))
    case.numerics(numerics, block=block)
    if fields:
        # Static inputs are explicitly observed at the rate's actual state. This
        # token authenticates InputAux and does not perform a field solve.
        program = pops.Program("explicit_external_data_"+method)
        temporal = program.state(block[state])
        first_rate = rate(temporal.n, program.input_fields(temporal.n, for_rate=rate))
        candidate = program.value("predictor", temporal.n+program.dt*first_rate,
                                  at=temporal.next.point)
        if method == "ssprk2":
            last_rate = rate(candidate, program.input_fields(candidate, for_rate=rate))
            candidate = program.value("corrector", .5*temporal.n+.5*candidate+.5*program.dt*last_rate,
                                      at=temporal.next.point)
        program.commit(temporal.next, candidate)
    else:
        program = (ForwardEuler if method == "euler" else SSPRK2)(block[state], rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    grid = CartesianGrid(frame=frame, cells=(n, n) if cells is None else cells,
                         periodic=None if kind.startswith("diagonal") else PeriodicAxes(frame.axes))
    return case, Uniform(grid), fields


def _bind(case, layout, initial, auxiliary):
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    pack = ProviderPack.from_data(artifact.plan.blocks[0].resolved_operations.to_data()
                                 ["provider_evidence"]["auxiliary"])
    keys = {key.component: key for key in pack if pack.declared_entry(key).producer == "runtime_input"}
    assert set(keys) == set(auxiliary)
    runtime = pops.bind(artifact, initial_state={"heat": np.ascontiguousarray(initial[None])},
        aux={keys[name]: np.ascontiguousarray(value) for name, value in auxiliary.items()},
        resources={"execution_context": artifact_execution_context(artifact)})
    return runtime, artifact


def _run(case, layout, initial, auxiliary, *, final, steps):
    runtime, artifact = _bind(case, layout, initial, auxiliary)
    report = pops.run(runtime, t_end=final, max_steps=steps+1, console=False)
    # Floating accumulation can leave one rounding-scale horizon remainder. It is
    # real accepted work and is reported, rather than silently rounded away here.
    assert steps <= report.accepted_steps <= steps+1
    if report.accepted_steps != steps:
        assert 0 < runtime._executor.program_last_dt() <= np.spacing(final)*max(4, steps)
    assert abs(runtime.time()-final) < 2e-13
    return np.asarray(runtime.state_global("heat")).reshape(initial.shape), runtime, artifact


def _norms(error):
    return tuple(float(value) for value in (np.mean(np.abs(error)), np.sqrt(np.mean(error**2)),
                                           np.max(np.abs(error))))


@pytest.mark.parametrize("kind", ("constant", "variable", "diagonal_linear", "diagonal_smooth"))
def test_full_explicit_spatial_matrix(isolated_native_cache, native_cxx, kokkos_root, kind):
    del isolated_native_cache, native_cxx, kokkos_root
    evidence = {"schema": "pops.m5.explicit-spatial.v1", "results": []}
    errors = []
    for n in REFINEMENTS:
        x, y = _centers(n)
        auxiliary = {}
        final = .05
        if kind == "constant":
            average_mode = np.sinc(1/n)**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
            initial = 2+average_mode
            exact = 2+np.exp(-8*np.pi**2*.1*final)*average_mode
            bound = .1/(.1*n*n)
        elif kind == "variable":
            initial = _average(lambda x,y: 2+np.sin(2*np.pi*x)*np.sin(2*np.pi*y), n)
            exact = np.exp(-final)*initial
            a = 1+.25*np.sin(2*np.pi*x)
            s = np.sin(2*np.pi*x)*np.sin(2*np.pi*y)
            # S/U is an analytic physical reaction coefficient; no hidden clock update.
            divergence = -8*np.pi**2*a*s+np.pi**2*np.cos(2*np.pi*x)**2*np.sin(2*np.pi*y)
            auxiliary = {"diffusivity": a, "source_factor": -1-divergence/(2+s)}
            bound = .05/(1.25*n*n)
        else:
            smooth = kind == "diagonal_smooth"
            initial = _average(_smooth if smooth else lambda x,y: 1+x+y, n)
            exact = initial.copy()
            auxiliary = {"a_x": 1+.25*x, "a_y": 2+.5*y,
                         "forcing": _average(_smooth_source, n) if smooth else np.full_like(x,-.75)}
            bound = .05/(2.5*n*n)
        steps = pmath.ceil(final/bound)
        case, layout, _ = _build(kind, n, final/steps)
        actual, runtime, artifact = _run(case, layout, initial, auxiliary, final=final, steps=steps)
        norms = _norms(actual-exact)
        mass_defect = float(np.mean(actual-initial))
        boundary_source_rate_defect = None
        if kind == "diagonal_linear":
            assert norms[-1] <= 2e-11
            exchanges = runtime._executor._program_exchange_records()
            final_dt = runtime._executor.program_last_dt()
            assert final_dt > 0
            # A rounding-scale final horizon clamp must not make the physical
            # balance oracle vacuous: compare rates, independently of its dt.
            boundary_source_rate_defect = abs(
                sum(row["integrated_amount"] for row in exchanges)/final_dt
                + float(np.mean(auxiliary["forcing"])))
            assert boundary_source_rate_defect <= 2e-11
        if kind == "constant":
            assert abs(mass_defect) <= 2e-11*abs(float(initial.mean()))
        errors.append(norms)
        evidence["results"].append({"case": kind, "n": n, "dt": final/steps,
            "planned_steps": steps, "accepted_steps": runtime.macro_step(),
            "last_dt": runtime._executor.program_last_dt(), "L1_L2_Linf": norms, "mass_change": mass_defect,
            "physical_boundary_source_rate_defect": boundary_source_rate_defect,
            "artifact_identity": artifact.artifact_identity.token,
            "last_accepted_exchange_count": len(runtime._executor._program_exchange_records())})
        _record("diffusion-explicit-spatial-"+kind+"-progress", evidence)
    if kind != "diagonal_linear":
        assert np.all(np.asarray(errors)[1:] < np.asarray(errors)[:-1])
        orders = np.log2(np.asarray(errors)[-2]/np.asarray(errors)[-1])
        assert min(orders) >= 1.75
        evidence.setdefault("orders", {})[kind] = orders.tolist()
    _record("diffusion-explicit-spatial-"+kind, evidence)


def _periodic_diffusion(u, nu=.1, *, nonlinear=False):
    w = u**3 if nonlinear else u
    return nu*u.shape[-1]**2*sum(np.roll(w,1,axis)-2*w+np.roll(w,-1,axis) for axis in (0,1))


@pytest.mark.parametrize("method", ("euler", "ssprk2"))
def test_full_explicit_exchange_quadrature(isolated_native_cache, native_cxx, kokkos_root, method):
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        x, y = _centers(n)
        initial = 2+.4*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        dt = .05/(.1*n*n)
        case, layout, _ = _build("constant", n, dt, method=method)
        actual, runtime, _artifact = _run(case, layout, initial, {}, final=dt, steps=1)
        first = initial+dt*_periodic_diffusion(initial)
        expected = first if method == "euler" else .5*initial+.5*(first+dt*_periodic_diffusion(first))
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        records = runtime._executor._program_exchange_records()
        stages = 1 if method == "euler" else 2
        assert len(records) == stages*4*n*n
        assert len({row["evaluation_context"] for row in records}) == stages
        expected_weight = dt/stages
        assert all(abs(row["temporal_weight"]-expected_weight) < 2e-16 for row in records)
        cell_delta = np.zeros((n,n))
        for row in records:
            cell = row["quadrature_identity"].split("/")[0].split(":")
            i, j = map(int, cell[1:])
            cell_delta[j,i] += row["integrated_amount"]
        np.testing.assert_allclose((actual-initial)/n**2, cell_delta, rtol=0, atol=2e-11)
        rows.append({"n":n,"stages":stages,"records":len(records),
                     "mass_exchange_defect":float(np.sum(actual-initial)/n**2-sum(row["integrated_amount"] for row in records))})
    _record("diffusion-explicit-exchanges-"+method, rows)


def test_full_nonlinear_gradient_variable(isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        x,y = _centers(n)
        initial = 1+.3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        dt = .01/(.1*n*n)
        case, layout, _ = _build("nonlinear", n, dt)
        actual, _runtime, _artifact = _run(case, layout, initial, {}, final=dt, steps=1)
        expected = initial+dt*_periodic_diffusion(initial, nonlinear=True)
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
        midpoint = np.zeros_like(initial)
        for axis in (0,1):
            high,low = np.roll(initial,-1,axis),np.roll(initial,1,axis)
            midpoint += .1*n*n*(3*((high+initial)/2)**2*(high-initial)
                                  -3*((low+initial)/2)**2*(initial-low))
        distinction = float(np.max(np.abs(actual-(initial+dt*midpoint))))
        assert distinction > 1e-10
        rows.append({"n":n,"midpoint_chain_rule_difference":distinction})
    _record("diffusion-nonlinear-gradient-variable", rows)


def test_combined_stability_and_exact_two_dimensional_weights(
    isolated_native_cache, native_cxx, kokkos_root,
):
    """Full Dim2 grid; both independent bounds hold but their sum must also hold."""
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        r = .15
        dt = r/(.1*n*n)
        x, y = _centers(n)
        initial = 1+.4*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        for c in (.2, .6):
            speed = c/(dt*n)
            case, layout, _ = _build("constant", n, dt, transport=(speed,0.))
            runtime, _artifact = _bind(case, layout, initial, {})
            if c == .6:
                assert c < 1 and 4*r < 1 and c+4*r > 1
                before = (runtime.time(), runtime.macro_step(), runtime._executor._program_exchange_records())
                with pytest.raises(RuntimeError, match="combined_transport_diffusion_stability"):
                    pops.run(runtime, t_end=dt, max_steps=1, console=False)
                assert (runtime.time(), runtime.macro_step(), runtime._executor._program_exchange_records()) == before
                np.testing.assert_array_equal(np.asarray(runtime.state_global("heat")).reshape(initial.shape), initial)
            else:
                report = pops.run(runtime, t_end=dt, max_steps=1, console=False)
                assert report.accepted_steps == 1
                actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
                expected = ((1-c-4*r)*initial+(c+r)*np.roll(initial,1,1)
                            +r*np.roll(initial,-1,1)+r*np.roll(initial,1,0)+r*np.roll(initial,-1,0))
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
                assert np.min(actual) > 0
            rows.append({"n":n,"cells":[n,n],"c":c,"r_x":r,"r_y":r,"accepted":c==.2})
    _record("diffusion-combined-stability-dim2", rows)


def test_physical_diffusion_boundary_authority_is_not_optional():
    with pytest.raises(ValueError, match="covered exactly once"):
        _build("constant", 16, .0001, physical=())
    duplicated = tuple(DiffusiveBoundary(axis, side, "value", 0.) for axis in range(2)
                       for side in ("lower","upper")) + (DiffusiveBoundary(0,"lower","conormal",0.),)
    with pytest.raises(ValueError, match="covered exactly once"):
        _build("constant", 16, .0001, physical=duplicated)
