"""A retained reversible component law has its own selected native route."""
import pytest
import pops
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D
from pops.numerics import CoupledGradient, Diffusion, TensorDiffusion, DiscretizationPlan
from pops.lib.time import ForwardEuler
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.codegen import Production
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage


def hall_case(*, hall=.3, dimension=1, components=("transverse_a", "transverse_b")):
    frame = CartesianDomain(
        "periodic", lower=(0.,) * dimension, upper=(6.283185307179586,) * dimension
    ).frame((Cartesian1D, Cartesian2D)[dimension - 1]())
    model = pops.Model("hall_fourier", frame=frame)
    state = model.state("w", components=components)
    flux = model.coupled_gradient_flux(
        "signed_gradient", state=state,
        dissipative=((0., 0.), (0., 0.)),
        reversible=((0., -hall), (hall, 0.)),
    )
    rate = model.rate("balance", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("hall_fourier")
    block = case.block("hall", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, CoupledGradient(flux=flux))
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(1e-5))
    case.program(program)
    case.initials.add(InitialCondition(
        state=block[state], value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(
        frame=frame, cells=(32,) * dimension, periodic=PeriodicAxes(frame.axes)))
    return model, case, layout, flux


def test_hall_source_has_separate_physical_parts_and_native_provider():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.module_codegen import _emit_bricks
    from pops.codegen.program_codegen import emit_cpp_program
    model, case, layout, flux = hall_case()
    assert flux.kind == "coupled_gradient_flux"
    assert flux.law.to_data()["reversible_components"] == ((0., -.3), (.3, 0.))
    with pytest.raises(TypeError, match="exact constitutive"):
        Diffusion(flux=flux)
    with pytest.raises(TypeError, match="exact constitutive"):
        TensorDiffusion(flux=flux)
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
    plan = next(iter(resolved.resolved_operations.values()))
    emitter = lower_and_validate(model, resolved_operations=plan)[0]
    code = emit_cpp_program(resolved.time, model=emitter)
    assert "PreparedCoupledGradient<pops::kNativeDimension, 2>" in code
    assert "DiffusiveLawResult<pops::kNativeDimension, 2, true>" in code
    assert ".stage_accepted_exchanges(" in code
    assert "program_state_ghost_depth = 2;" in _emit_bricks(emitter._m)[1]


def test_hall_zero_and_component_names_do_not_change_scientific_sign():
    a = hall_case(hall=0, components=("left", "right"))[3].law
    b = hall_case(hall=.3, components=("north", "east"))[3].law
    assert a.reversible_components == ((0., 0.), (0., 0.))
    assert b.reversible_components[0][1] == -.3
    assert b.reversible_components[1][0] == .3


def test_physical_law_general_but_v1_realization_rejects_second_axis():
    with pytest.raises(ValueError, match="one periodic axis"):
        hall_case(dimension=2)


@pytest.mark.parametrize("matrix", [
    ((0., 1.), (0., 0.)),
    ((0., -.3), (-.3, 0.)),
])
def test_non_skew_reversible_matrix_is_rejected(matrix):
    frame = Cartesian1D()
    model = pops.Model("bad_hall", frame=frame)
    state = model.state("w", components=("a", "b"))
    with pytest.raises(ValueError, match="skew R"):
        model.coupled_gradient_flux(
            "bad", state=state, dissipative=((0., 0.), (0., 0.)),
            reversible=matrix)


def test_three_component_coupling_is_generic_and_keeps_two_physical_parts():
    model = pops.Model("three_gradient", frame=Cartesian1D())
    state = model.state("w", components=("u", "v", "z"))
    flux = model.coupled_gradient_flux(
        "constitutive", state=state,
        dissipative=((.1, 0., 0.), (0., .2, 0.), (0., 0., .3)),
        reversible=((0., -.4, .2), (.4, 0., -.1), (-.2, .1, 0.)))
    assert len(flux.law.flux_expressions()) == 3
    assert CoupledGradient(flux=flux).validate()
    assert flux.law.dissipative_components[2][2] == .3
    assert flux.law.reversible_components[0][1] == -.4


def test_negative_symmetric_part_is_not_relabelled_reversible():
    model = pops.Model("bad_dissipation", frame=Cartesian1D())
    state = model.state("w", components=("a", "b"))
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.coupled_gradient_flux(
            "bad", state=state, dissipative=((-1., 0.), (0., 0.)),
            reversible=((0., -.3), (.3, 0.)))


def test_hall_oracle_distinguishes_continuous_phase_spatial_symbol_and_ssprk2_norm():
    import sys
    from pathlib import Path
    examples = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    sys.path.insert(0, str(examples))
    try:
        from api040_m23_hall_fourier import discrete_oracle, initial_means, DT, K, STEPS
        import numpy as np
        n = 32
        a = initial_means(n, (0, 1))
        expected = np.array([1., .2])[:, None]*np.sinc(K/n)*np.cos(
            K*2*np.pi*(np.arange(n)+.5)/n)[None, :]
        assert np.max(np.abs(a-expected)) == 0
        amp, symbol = discrete_oracle(n, .3)
        z = .3*symbol*DT
        assert symbol < K*K
        assert abs(amp) > 1  # SSPRK2 is not an imaginary-axis norm certificate.
        assert abs(amp-(1-.5*z*z-1j*z)**STEPS) < 1e-15
        assert discrete_oracle(n, 0.)[0] == 1
    finally:
        sys.path.remove(str(examples))


@pytest.mark.parametrize("eta_h,order", [
    (.3, (0, 1)), (.3, (1, 0)), (0., (0, 1)),
])
def test_scientific_variants_resolve_without_spd_substitution(eta_h, order):
    import sys
    from pathlib import Path
    examples = Path(__file__).resolve().parents[4] / "examples/migration/scientific"
    sys.path.insert(0, str(examples))
    try:
        from api040_m23_hall_fourier import build_case
        from pops.codegen.module_lowering import lower_and_validate
        from pops.codegen.program_codegen import emit_cpp_program
        case, layout = build_case(32, eta_h=eta_h, order=order)
        resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
        model = case._block_registry.spec("transverse")["model"]
        plan = next(iter(resolved.resolved_operations.values()))
        emitted = emit_cpp_program(
            resolved.time, model=lower_and_validate(model, resolved_operations=plan)[0])
        assert "PreparedCoupledGradient<pops::kNativeDimension, 2>" in emitted
        assert "PreparedDiffusion<pops::kNativeDimension, 2, true>" not in emitted
        assert emitted.count(".stage_accepted_exchanges(") == 2
    finally:
        sys.path.remove(str(examples))
