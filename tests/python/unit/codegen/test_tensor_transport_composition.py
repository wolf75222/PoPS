"""Resolved methods retain independent transport and tensor ownership on one block."""
import pops
import pytest
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.integration.runtime.test_tensor_transport_composition import composition_case


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_separate_transport_and_tensor_emit_both_operators_and_required_halos(dimension):
    case, layout, model = composition_case(dimension=dimension)
    authored_flux = model._dsl._m._flux
    authored_eigenvalues = model._dsl._m._eig
    resolved = pops.resolve(pops.validate(case), layout=layout)
    operations = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=operations)[0]
    code = emit_cpp_program(resolved.time, model=emitter)
    body = _emit_bricks(emitter._m)[1]
    assert "PreparedDiffusion<pops::kNativeDimension, 2, true>" in code
    assert "ctx.neg_div_flux_default_into(" in code
    assert "program_state_ghost_depth = 2;" in body
    assert "program_only_storage = true" not in body
    assert max(operations.halo_requirements().values()) == 2
    selected = [row.guarantees.get("numerical_method") for row in operations.operations]
    assert any(method and method.get("method") == "tensor_diffusion" for method in selected)
    assert any(method and method.get("method") == "finite_volume" for method in selected)
    assert model._dsl._m._flux is authored_flux and model._dsl._m._eig is authored_eigenvalues
    assert not hasattr(model._dsl._m, "_program_state_ghost_depth")


def test_implicit_tensor_partition_is_not_charged_to_explicit_transport_bound():
    case, layout, model = composition_case(implicit=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    operations = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=operations)[0]
    code = emit_cpp_program(resolved.time, model=emitter)
    assert "PreparedDiffusion<pops::kNativeDimension, 2, true>" in code
    assert "ctx.neg_div_flux_default_into(" in code
    assert "pops::runtime::program::PreparedSpatialResidual<" in code
    assert ".explicit_frequency()" not in code
    assert "program_state_ghost_depth = 2;" in _emit_bricks(emitter._m)[1]


def test_competing_native_transport_configurations_are_still_refused():
    case, layout, _ = composition_case(competing_transport=True)
    with pytest.raises(ValueError, match="distinct runtime configurations"):
        pops.resolve(pops.validate(case), layout=layout)


@pytest.mark.parametrize("flux", ("rusanov", "hll"))
def test_shared_endpoint_frequency_contract_authenticates_supported_fluxes(flux):
    from types import SimpleNamespace
    from pops.numerics import reconstruction, riemann
    from pops.numerics.transport_frequency import transport_frequency_contract

    selected = SimpleNamespace(reconstruction=reconstruction.FirstOrder(),
                               riemann=riemann.Rusanov() if flux == "rusanov" else riemann.HLL())
    assert transport_frequency_contract(selected)["provider"] == "native_endpoint_model_wave_envelope"


@pytest.mark.parametrize("reconstruction_name,flux_name,waves", (
    ("WENO5", "Rusanov", None), ("FirstOrder", "Roe", None),
    ("FirstOrder", "HLLC", None), ("FirstOrder", "HLL", "einfeldt"),
    ("FirstOrder", "HLL", "davis"),
))
def test_shared_endpoint_frequency_contract_refuses_unproved_providers(
        reconstruction_name, flux_name, waves):
    from types import SimpleNamespace
    from pops.numerics import reconstruction, riemann
    from pops.numerics.transport_frequency import transport_frequency_contract

    wave_provider = None if waves is None else getattr(riemann.waves, waves.title())()
    flux = getattr(riemann, flux_name)(**({} if wave_provider is None else {"waves": wave_provider}))
    selected = SimpleNamespace(reconstruction=getattr(reconstruction, reconstruction_name)(), riemann=flux)
    with pytest.raises(ValueError, match="combined diffusion"):
        transport_frequency_contract(selected)


def test_separate_transport_and_joint_diffusion_keep_both_transport_multiplicities():
    from pops import math
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.numerics import Diffusion, DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.time import FixedDt

    frame = Rectangle("domain", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("multiple_roots", frame=frame)
    state = model.state("U", components=("u",))
    flux = model.flux("transport", frame=frame, state=state,
                      components={axis: (.2*state[0],) for axis in frame.axes},
                      waves={axis: (.2,) for axis in frame.axes})
    diffusion = model.diffusive_flux("conduction", state=state, value=.1*math.grad(state))
    transport = model.rate("transport_root", equation=math.ddt(state) == -math.div(flux))
    joint = model.rate("joint_root", equation=math.ddt(state) == -math.div(flux)+math.div(diffusion))
    fv = FiniteVolume(flux=flux, variables=variables.Conservative(state),
                      reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    case = pops.Case("multiple_roots")
    block = case.block("inventory", model, states=(state,))
    methods = DiscretizationPlan()
    methods.rates.add(transport, fv)
    methods.rates.add(joint, Diffusion(flux=diffusion, transport=fv))
    case.numerics(methods, block=block)
    program = pops.Program("sum")
    q = program.state(block[state])
    update = program.value("accepted", q.n+program.dt*(transport(q.n)+joint(q.n)), at=q.next.point)
    program.commit(q.next, update)
    program.step_strategy(FixedDt(.001))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout)
    operations = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=operations)[0]
    code = emit_cpp_program(resolved.time, model=emitter)
    assert code.count("ctx.max_wave_speed(") == 2
    assert ".explicit_frequency() + ctx.max_wave_speed" in code
    assert any("transport_frequency_" in line and "diffusion_frequency_" in line
               for line in code.splitlines())
    assert code.count("combined_transport_diffusion_stability") == 1
