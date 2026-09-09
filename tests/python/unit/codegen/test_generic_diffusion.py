"""Public multicomponent constitutive laws reach generic native face kernels."""
import pytest
import pops
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.numerics import Diffusion, DiscretizationPlan
from pops.lib.time import ForwardEuler
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.codegen import Production
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage


def generic_case(dimension, name="mixture", state_name="inventory"):
    frame = CartesianDomain("box", lower=(0.,)*dimension, upper=(1.,)*dimension).frame(
        (Cartesian1D, Cartesian2D, Cartesian3D)[dimension-1]())
    model = pops.Model(name, frame=frame)
    state = model.state(state_name, components=("first", "second"))
    u, v = state
    from pops.params import RuntimeParam
    nu = model.value(model.param(RuntimeParam("second_diffusivity", default=.2)))
    flux = model.diffusive_flux("constitutive", state=state,
        value=((.1+.01*v*v)*math.grad(u**3), (nu+.02*u*u)*math.grad(v)))
    rate = model.rate("balance", equation=math.ddt(state)==math.div(flux))
    case = pops.Case("coupled_diffusion")
    block = case.block(name, model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1e-5))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    resolved = pops.resolve(pops.validate(case), layout=Uniform(CartesianGrid(
        frame=frame, cells=(8,)*dimension, periodic=PeriodicAxes(frame.axes))), backend=Production())
    return model, resolved


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_coupled_coefficient_laws_have_all_components_axes_and_runtime_parameters(dimension):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_emit_params import program_param_entries
    model, resolved = generic_case(dimension)
    emitter = lower_and_validate(model)[0]
    code = emit_cpp_program(resolved.time, model=emitter)
    assert "PreparedDiffusion<pops::kNativeDimension, 2>" in code
    assert "DiffusiveLawResult<pops::kNativeDimension, 2>" in code
    assert program_param_entries(resolved.time, emitter)==[(0,"second_diffusivity",0,.2)]
    assert "std::pow(" in code or "pops::pow(" in code or "cons[0]" in code
    assert all(row.refusal is None for plan in resolved.resolved_operations.values() for row in plan.operations)


def test_cross_gradient_is_physical_before_monotone_realization_is_selected():
    model = pops.Model("cross_gradient", frame=Cartesian2D())
    state = model.state("U", components=("u", "v"))
    u, v = state
    flux = model.diffusive_flux("cross", state=state, value=(math.grad(2*u+v), math.grad(u+2*v)))
    assert len(flux.law.flux_expressions()) == 4
    with pytest.raises(ValueError, match="cross-component gradient"):
        Diffusion(flux=flux)
