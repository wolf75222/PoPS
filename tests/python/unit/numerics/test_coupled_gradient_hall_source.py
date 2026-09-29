"""A retained reversible component law has its own selected native route."""
import pytest
import pops
from pops import math
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D, Cartesian2D, Cartesian3D
from pops.numerics import CoupledGradient, Diffusion, TensorDiffusion, DiscretizationPlan
from pops.lib.time import ForwardEuler
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.codegen import Production
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage


def hall_case(*, hall=.3, dimension=1, components=("transverse_a", "transverse_b"),
              weights=(1,)):
    frame = CartesianDomain(
        "periodic", lower=(0.,) * dimension, upper=(6.283185307179586,) * dimension
    ).frame((Cartesian1D, Cartesian2D, Cartesian3D)[dimension - 1]())
    model = pops.Model("hall_fourier", frame=frame)
    state = model.state("w", components=components)
    flux = model.coupled_gradient_flux(
        "signed_gradient", state=state,
        dissipative=((0., 0.), (0., 0.)),
        reversible=((0., -hall), (hall, 0.)),
    )
    rhs = weights[0] * math.div(flux)
    for weight in weights[1:]:
        rhs += weight * math.div(flux)
    rate = model.rate("balance", equation=math.ddt(state) == rhs)
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
    return model, case, layout, flux, rate


def test_hall_source_has_separate_physical_parts_and_native_provider():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.module_codegen import _emit_bricks
    from pops.codegen.program_codegen import emit_cpp_program
    model, case, layout, flux, _ = hall_case()
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


def test_repeated_weighted_exact_flux_keeps_rhs_sum_and_each_accepted_occurrence():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    model, case, layout, flux, rate = hall_case(weights=(1, 2))
    assert tuple(row.coefficient for row in rate.occurrences) == (1, 2)
    assert all(row.payload == flux for row in rate.occurrences)
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
    plan = next(iter(resolved.resolved_operations.values()))
    code = emit_cpp_program(
        resolved.time, model=lower_and_validate(model, resolved_operations=plan)[0])
    scale = next(line for line in code.splitlines()
                 if "pops::scale(diffusive_rhs_" in line)
    assert "pops::Real(3)" in scale
    exchanges = [line for line in code.splitlines()
                 if ".stage_accepted_exchanges(" in line]
    assert len(exchanges) == 2
    assert "/occurrence:0" in exchanges[0] and exchanges[0].endswith("*pops::Real(1));")
    assert "/occurrence:1" in exchanges[1] and exchanges[1].endswith("*pops::Real(2));")


def test_single_positive_nonunit_weight_is_admitted_but_nonpositive_weight_is_not():
    _, case, layout, _, _ = hall_case(weights=(.5,))
    assert pops.resolve(pops.validate(case), layout=layout, backend=Production())
    _, bad, _, _, _ = hall_case(weights=(-.5,))
    with pytest.raises(Exception, match="positive uses of one exact physical flux"):
        pops.validate(bad)


def test_hall_zero_and_component_names_do_not_change_scientific_sign():
    a = hall_case(hall=0, components=("left", "right"))[3].law
    b = hall_case(hall=.3, components=("north", "east"))[3].law
    assert a.reversible_components == ((0., 0.), (0., 0.))
    assert b.reversible_components[0][1] == -.3
    assert b.reversible_components[1][0] == .3


@pytest.mark.parametrize("dimension", (2, 3))
def test_periodic_multiaxis_law_resolves_to_the_same_native_component_provider(dimension):
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    _, case, layout, flux, _ = hall_case(dimension=dimension)
    assert CoupledGradient(flux=flux).validate()
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
    graph = ProgramModelGraph.from_resolved_blocks(resolved.blocks)
    source = emit_cpp_program(resolved.time, model_graph=graph)
    assert "PreparedCoupledGradient<pops::kNativeDimension, 2>" in source
    assert source.count(".stage_accepted_exchanges(") == 1
    assert "pops::kNativeDimension" in source


def test_multiaxis_extension_does_not_admit_unprepared_physical_traces():
    from pops.physics.diffusion import DiffusiveBoundary
    model = pops.Model("boundary_obligation", frame=Cartesian2D())
    state = model.state("w", components=("a", "b"))
    faces = (DiffusiveBoundary(0, "lower", "periodic"),
             DiffusiveBoundary(0, "upper", "periodic"),
             DiffusiveBoundary(1, "lower", "value", 0.),
             DiffusiveBoundary(1, "upper", "value", 0.))
    flux = model.coupled_gradient_flux(
        "with_traces", state=state,
        dissipative=((1., 0.), (0., 1.)),
        reversible=((0., -.2), (.2, 0.)),
        boundaries={component: faces for component in state.components})
    with pytest.raises(ValueError, match="requires periodic Cartesian axes"):
        CoupledGradient(flux=flux)


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


def test_exact_rank_one_dissipation_is_admitted_without_changing_coefficients():
    model = pops.Model("rank_one_dissipation", frame=Cartesian1D())
    state = model.state("w", components=("a", "b", "c"))
    ones = ((1., 1., 1.),) * 3
    zeros = ((0., 0., 0.),) * 3
    law = model.coupled_gradient_flux(
        "rank_one", state=state, dissipative=ones, reversible=zeros).law
    assert law.dissipative_components == ones


@pytest.mark.parametrize("matrix", [
    ((1., 0., 0.), (0., 1., 0.), (0., 0., -2**-50)),
    ((0., 1., 0.), (1., 0., 0.), (0., 0., 0.)),
])
def test_exact_negative_dissipation_modes_are_rejected(matrix):
    model = pops.Model("indefinite_dissipation", frame=Cartesian1D())
    state = model.state("w", components=("a", "b", "c"))
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.coupled_gradient_flux(
            "indefinite", state=state, dissipative=matrix,
            reversible=((0., 0., 0.),) * 3)


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
        case, layout, _ = build_case(32, eta_h=eta_h, order=order)
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
