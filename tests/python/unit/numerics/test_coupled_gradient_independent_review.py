"""Independent public Hall route checks: signed occurrence and accepted faces."""
import pops
import pytest
from pops import math
from pops.codegen import Production
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.domain import CartesianDomain
from pops.frames import Cartesian1D
from pops.layouts import Uniform
from pops.lib.time import SSPRK2
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import CoupledGradient, DiscretizationPlan
from pops.time import FixedDt


def emitted_hall(*, loader=False):
    frame = CartesianDomain("line", lower=(0.,), upper=(6.283185307179586,)).frame(Cartesian1D())
    model = pops.Model("coupled_review", frame=frame)
    state = model.state("w", components=("first", "second"))
    flux = model.coupled_gradient_flux(
        "hall", state=state, dissipative=((0., 0.), (0., 0.)),
        reversible=((0., -.3), (.3, 0.)))
    rate = model.rate("balance", equation=math.ddt(state) == math.div(flux))
    case = pops.Case("coupled_review")
    block = case.block("fields", model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, CoupledGradient(flux=flux))
    case.numerics(numerics, block=block)
    program = SSPRK2(block[state], rate=rate)
    program.step_strategy(FixedDt(.002))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(32,), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production())
    plan = next(iter(resolved.resolved_operations.values()))
    native_model = lower_and_validate(model, resolved_operations=plan)[0]
    if loader:
        return native_model._m.emit_cpp_native_loader(name="ReviewedHall", target="system")
    return emit_cpp_program(resolved.time, model=native_model)


def test_accepted_coupled_gradient_faces_are_published_for_both_ssprk2_stages():
    source = emitted_hall()
    assert source.count("PreparedCoupledGradient<pops::kNativeDimension, 2>") >= 2
    assert source.count(".stage_accepted_exchanges(") == 2


def test_hall_storage_loader_needs_no_fabricated_hyperbolic_flux():
    source = emitted_hall(loader=True)
    assert "program_only_storage = true" in source
    assert "void pops_install_native(" in source
    assert "State flux(" not in source and "max_wave_speed(" not in source


def test_foreign_same_named_component_state_is_not_authenticated_as_local():
    first = pops.Model("first_owner", frame=Cartesian1D())
    second = pops.Model("second_owner", frame=Cartesian1D())
    state = first.state("w", components=("first", "second"))
    second.state("w", components=("first", "second"))
    with pytest.raises(ValueError, match="exact state"):
        second.coupled_gradient_flux(
            "foreign", state=state, dissipative=((0., 0.), (0., 0.)),
            reversible=((0., -.3), (.3, 0.)))


def test_component_tensor_shape_and_skew_parts_are_retained_without_spd_projection():
    model = pops.Model("three_components", frame=Cartesian1D())
    state = model.state("w", components=("z", "a", "q"))
    dissipative = ((.3, -.1, 0.), (-.1, .3, 0.), (0., 0., 0.))
    reversible = ((0., -.4, .7), (.4, 0., -.2), (-.7, .2, 0.))
    flux = model.coupled_gradient_flux(
        "joint", state=state, dissipative=dissipative, reversible=reversible)
    _ = model.module  # freeze the public declaration registry before canonical data
    data = flux.law.to_data()
    assert data["dissipative_components"] == dissipative
    assert data["reversible_components"] == reversible
    assert CoupledGradient(flux=flux).validate()


def test_constant_coefficient_realization_refuses_a_runtime_capture_explicitly():
    from pops.params import RuntimeParam

    model = pops.Model("constant_contract", frame=Cartesian1D())
    state = model.state("w", components=("first", "second"))
    parameter = model.param(RuntimeParam("eta", default=.3))
    eta = model.value(parameter)
    with pytest.raises(ValueError, match="finite real constants"):
        model.coupled_gradient_flux(
            "not_frozen", state=state, dissipative=((0., 0.), (0., 0.)),
            reversible=((0., -eta), (eta, 0.)))


def test_exact_rank_one_dissipation_is_not_rejected_by_eigensolver_roundoff():
    model = pops.Model("rank_one", frame=Cartesian1D())
    state = model.state("w", components=("a", "b", "c"))
    # v.T D v = (v[0]+v[1]+v[2])**2: PSD including two exact null modes.
    dissipative = ((1., 1., 1.),) * 3
    flux = model.coupled_gradient_flux(
        "rank_one", state=state, dissipative=dissipative,
        reversible=((0., -.3, 0.), (.3, 0., 0.), (0., 0., 0.)))
    assert flux.law.dissipative_components == dissipative


def test_small_true_negative_mode_is_not_projected_into_the_psd_domain():
    model = pops.Model("negative_mode", frame=Cartesian1D())
    state = model.state("w", components=("a", "b", "c"))
    with pytest.raises(ValueError, match="positive semidefinite"):
        model.coupled_gradient_flux(
            "negative_mode", state=state,
            dissipative=((1., 0., 0.), (0., 1., 0.), (0., 0., -2.**-50)),
            reversible=((0., -.3, 0.), (.3, 0., 0.), (0., 0., 0.)))
