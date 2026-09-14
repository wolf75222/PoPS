"""Scalar conservation is insufficient to certify a collective species constraint."""
import numpy as np
import pops
import pytest
from pops import math
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.numerics import Diffusion, DiscretizationPlan
from interaction_test_layout import interaction_grid
from tests.python.support.native_execution_context import artifact_execution_context


def _case(n, coefficient):
    grid = interaction_grid(n=n)
    case = pops.Case("collective_constraint")
    model = pops.Model("scalar_species", frame=grid.frame)
    state = model.state("U", components=("y",))
    flux = model.diffusive_flux("flux", state=state, value=coefficient*math.grad(state))
    rate = model.rate("rate", equation=math.ddt(state) == math.div(flux))
    block = case.block("species", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(pops.time.FixedDt(1e-6))
    case.program(program)
    return pops.resolve(pops.validate(case), layout=Uniform(grid))


@pytest.mark.compiler
@pytest.mark.native_loader
@pytest.mark.integration
@pytest.mark.parametrize("n", (16,32,64))
@pytest.mark.parametrize("diffusivities", ((1.,1.),(1.,2.)), ids=("compatible","unequal-counterexample"))
def test_native_scalar_conservation_does_not_certify_collective_constraint(n, diffusivities, record_property):
    from pops._native_selector import select_native_dimension
    select_native_dimension(2)
    coordinate = (np.arange(n)+.5)/n
    mode = np.sin(2*np.pi*coordinate[None,:])*np.ones((n,1))
    initial = ((.5+.1*mode)[None], (.5-.1*mode)[None])
    end = 0.
    for _ in range(100):
        end += 1e-6
    actual = []
    eigenvalue = 4*n*n*np.sin(np.pi/n)**2
    for index, coefficient in enumerate(diffusivities):
        artifact = pops.compile(_case(n, coefficient))
        simulation = pops.bind(artifact, initial_state={"species": initial[index]},
            resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(simulation, t_end=end, max_steps=100)
        assert report.accepted_steps == 100 and report.rejected_steps == 0
        actual.append(simulation.state_global("species")[0])
        z = 1e-6*coefficient*eigenvalue
        amplitude = (.1 if index==0 else -.1)*(1-z+.5*z*z)**100
        np.testing.assert_allclose(actual[index], .5+amplitude*mode, rtol=0, atol=1e-11)
        assert abs(simulation.integral("species",0)-.5) < 1e-11
    collective_error = float(np.max(np.abs(actual[0]+actual[1]-1)))
    if diffusivities[0] == diffusivities[1]:
        assert collective_error < 1e-11
    else:
        assert collective_error > 3e-4
    record_property("n",n)
    record_property("diffusivities",diffusivities)
    record_property("accepted_steps",report.accepted_steps)
    record_property("scalar_inventory_tolerance",1e-11)
    record_property("pointwise_collective_constraint_error",collective_error)
    record_property("claim", "each scalar conserves; only compatible joint law certifies y1+y2=1")
