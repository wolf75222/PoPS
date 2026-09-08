"""An actual transport-only derived provider refreshes at each accepted interval."""
import numpy as np
import pops
import pytest
from pops import math
from pops._ir import ValueExpr
from pops._ir.expr import Var
from pops.domain import Rectangle
from pops.fields import AuxiliaryBoundary, DerivedAux
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.model.provider_pack import ProviderPack
from pops.numerics import (
    Diffusion, DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables,
)
from pops.time import FixedDt

from tests.python.integration.runtime.test_combined_diffusion_exchange_ledger import _rates
from tests.python.integration.runtime.test_public_diffusion_matrix import (
    REFINEMENTS, _bind, _centers, _record,
)

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _build_derived_transport(n, dt, method):
    frame = Rectangle("derived-transport", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("derived-transport", frame=frame)
    state = model.state("U", components=("u",))
    imposed = model.aux("imposed")
    # AuxSpace consumers use the established emitter expression coordinate;
    # the Module below resolves it to its exact qualified provider component.
    speed = Var("velocity", "aux")
    flux = model.flux("transport", frame=frame, state=state,
        components={frame.axes[0]: (speed*state[0],), frame.axes[1]: (0*state[0],)},
        waves={frame.axes[0]: (speed,), frame.axes[1]: (0*state[0],)})
    diffusion = model.diffusive_flux("diffusion", state=state,
                                    value=math.CoeffGradient(state[0], .1))
    forcing = model.source("forcing", on=state, value=(imposed,))
    rate = model.rate("combined", equation=math.ddt(state) ==
                      -math.div(flux)+math.div(diffusion)+forcing)
    module = model.module
    velocity = module.aux_handle(module.aux_field("velocity"))
    module.aux_provider(DerivedAux(velocity,
        2*ValueExpr(module.field_handle(module.field_spaces()["fields"])),
        boundary=AuxiliaryBoundary(width=1, kind="foextrap")))
    module.operator_registry().get(flux.reg_name).requirements["aux"] = ("velocity",)
    module.eigenvalues(x=(speed,), y=(0*state[0],))
    case = pops.Case("derived-combined")
    block = case.block("heat", module)
    plan = DiscretizationPlan()
    plan.rates.add(rate, Diffusion(flux=diffusion, transport=FiniteVolume(flux=flux,
        variables=variables.Conservative(state), reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.Rusanov())))
    case.numerics(plan, block=block)
    program = pops.Program("derived-transport-"+method)
    current = program.state(block[state])
    rhs = rate(current.n, program.input_fields(current.n, for_rate=rate))
    candidate = program.value("predictor", current.n+program.dt*rhs, at=current.next.point)
    if method == "ssprk2":
        last = rate(candidate, program.input_fields(candidate, for_rate=rate))
        candidate = program.value("corrector", .5*current.n+.5*candidate+.5*program.dt*last,
                                  at=current.next.point)
    program.commit(current.next, candidate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(n, n), periodic=PeriodicAxes(frame.axes)))
    return case, layout


def _advance(value, imposed, dt, method):
    first = _rates(value, (2*imposed, 0.), .1)
    predictor = value+dt*(sum(first)+imposed)
    if method == "euler":
        return predictor, (value,), (first,)
    last = _rates(predictor, (2*imposed, 0.), .1)
    return .5*value+.5*(predictor+dt*(sum(last)+imposed)), (value, predictor), (first, last)


@pytest.mark.parametrize("method", ("euler", "ssprk2"))
def test_combined_transport_consumes_changed_derived_auxiliary(
        isolated_native_cache, native_cxx, kokkos_root, method):
    del isolated_native_cache, native_cxx, kokkos_root
    rows = []
    for n in REFINEMENTS:
        x, y = _centers(n)
        initial = 2+.3*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)+.1*np.cos(4*np.pi*x)
        dt = .15/(.4*n*n+.7*n)
        case, layout = _build_derived_transport(n, dt, method)
        runtime, artifact = _bind(case, layout, initial, {"imposed": np.full_like(initial, .2)})
        pack = ProviderPack.from_data(artifact.plan.blocks[0].resolved_operations.to_data()
                                     ["provider_evidence"]["auxiliary"])
        key = next(key for key in pack if key.component == "imposed")
        previous = initial
        for step, imposed in enumerate((.2, .35), 1):
            if step == 2:
                runtime._executor.stage_auxiliary_input(key, np.full_like(initial, imposed))
            expected, stage_states, stage_rates = _advance(previous, imposed, dt, method)
            report = pops.run(runtime, t_end=step*dt, max_steps=1, console=False)
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            actual = np.asarray(runtime.state_global("heat")).reshape(initial.shape)
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)
            records = runtime._executor._program_exchange_records()
            stages = len(stage_states)
            assert len(records) == 8*n*n*stages
            contexts = tuple(dict.fromkeys(row["evaluation_context"] for row in records))
            assert len(contexts) == stages
            context_stage = dict(zip(contexts, range(stages), strict=True))
            increments = {"transport": np.zeros_like(initial), "diffusion": np.zeros_like(initial)}
            flux_defect = 0.
            for row in records:
                cell_token, axis_token, side_token = row["quadrature_identity"].split("/")
                i, j = map(int, cell_token.split(":")[1:])
                axis, side = int(axis_token.split(":")[1]), int(side_token.split(":")[1])
                kind = "transport" if row["orientation"] == (1 if side == 0 else -1) else "diffusion"
                assert row["face_measure"] == 1/n and row["multiplicity"] == 1
                assert abs(row["temporal_weight"]-dt/stages) <= 2e-16
                state = stage_states[context_stage[row["evaluation_context"]]]
                cell = (j, i)
                other = list(cell)
                other[1-axis] = (other[1-axis]+(1 if side else -1)) % n
                neighbor = state[tuple(other)]
                left, right = (state[cell], neighbor) if side else (neighbor, state[cell])
                if kind == "transport":
                    speed = 2*imposed if axis == 0 else 0.
                    oracle = .5*speed*(left+right)-.5*abs(speed)*(right-left)
                else:
                    oracle = .1*n*(right-left)
                flux_defect = max(flux_defect, abs(row["numerical_flux"]-oracle))
                increments[kind][cell] += row["integrated_amount"]
            assert flux_defect < 2e-12
            for index, kind in enumerate(("transport", "diffusion")):
                oracle = dt/stages*sum(rate[index] for rate in stage_rates)/n**2
                np.testing.assert_allclose(increments[kind], oracle, rtol=0, atol=2e-13)
            delta = increments["transport"]+increments["diffusion"]+dt*imposed/n**2
            np.testing.assert_allclose((actual-previous)/n**2, delta, rtol=0, atol=2e-13)
            rows.append({"n": n, "method": method, "step": step, "imposed": imposed,
                         "face_flux_defect": flux_defect,
                         "cell_change_defect": float(np.max(np.abs((actual-previous)/n**2-delta))),
                         "artifact": artifact.artifact_identity.token})
            previous = actual.copy()
    _record("combined-derived-transport-"+method, rows)
