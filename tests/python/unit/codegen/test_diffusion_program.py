"""Real public Case resolution and native Program generation for diffusion."""
import pytest
import pops
from pops import math
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import Diffusion, DiscretizationPlan
from pops.lib.time import ForwardEuler, SSPRK2
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.codegen import Production
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage


def resolved_heat(*, n=16, coefficient=0.1, method="euler", variable=None, transport=None,
                  runtime_coefficient=False, transport_riemann="rusanov"):
    frame=Rectangle("diffusion_domain",lower=(0.,0.),upper=(1.,1.)).frame(Cartesian2D())
    model=pops.Model("heat",frame=frame)
    state=model.state("U",components=("u",))
    expression=state if variable is None else variable(state[0])
    selected_coefficient=coefficient
    if runtime_coefficient:
        from pops.params import RuntimeParam
        selected_coefficient=model.value(model.param(RuntimeParam("nu",default=coefficient)))
    flux=model.diffusive_flux("conduction",state=state,value=selected_coefficient*math.grad(expression))
    rhs=math.div(flux)
    transport_method=None
    if transport is not None:
        from pops.numerics import FiniteVolume, variables, reconstruction, riemann
        adv=model.flux("transport",frame=frame,state=state,
            components={axis:(speed*state[0],) for axis,speed in zip(frame.axes,transport,strict=True)},
            waves={axis:(speed,) for axis,speed in zip(frame.axes,transport,strict=True)})
        rhs=-math.div(adv)+rhs
        transport_method=FiniteVolume(flux=adv,variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov() if transport_riemann=="rusanov" else riemann.HLL())
    rate=model.rate("heat_rate",equation=math.ddt(state)==rhs)
    case=pops.Case("heat_case")
    block=case.block("heat",model,states=(state,))
    numerics=DiscretizationPlan()
    numerics.rates.add(rate,Diffusion(flux=flux,transport=transport_method))
    case.numerics(numerics,block=block)
    program=(ForwardEuler if method=="euler" else SSPRK2)(block[state],rate=rate)
    program.step_strategy(FixedDt(0.1/(coefficient*n*n)))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state],value=BindArray(),
                                      projection=ConservativeCellAverage()))
    resolved=pops.resolve(pops.validate(case),layout=Uniform(CartesianGrid(
        frame=frame,cells=(n,n),periodic=PeriodicAxes(frame.axes))),backend=Production())
    return resolved,block[state],model


@pytest.mark.parametrize("method",["euler","ssprk2"])
def test_scalar_diffusion_public_case_resolves_and_emits_native_faces(method):
    from pops.codegen.program_codegen import emit_cpp_program
    resolved,_,model=resolved_heat(method=method)
    from pops.codegen.module_lowering import lower_and_validate
    emitter,_=lower_and_validate(model)
    code=emit_cpp_program(resolved.time,model=emitter)
    assert "PreparedDiffusion<pops::kNativeDimension>" in code
    assert code.index('ctx.require_cartesian_generated_operator(0, "diffusive_face_evaluation")') < code.index("PreparedDiffusion<pops::kNativeDimension>")
    assert ".apply(" in code
    assert "ctx.rhs_into(" not in code
    assert not model._dsl._m._flux
    assert all(row.refusal is None for plan in resolved.resolved_operations.values() for row in plan.operations)


@pytest.mark.parametrize("method,weights",[("euler",[1]),("ssprk2",[.5,.5])])
def test_diffusion_accepted_quadrature_does_not_count_predictor_ancestry(method,weights):
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
    resolved,_,_=resolved_heat(method=method)
    rows=accepted_diffusive_quadrature(resolved.time)
    assert [float(weight[1]) for _,weight in rows]==weights
    assert len({value.id for value,_ in rows})==len(weights)


@pytest.mark.parametrize("transport_riemann", ["rusanov", "hll"])
def test_combined_transport_retains_native_riemann_and_adds_stability_frequencies(transport_riemann):
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_codegen import emit_cpp_program
    resolved,_,model=resolved_heat(transport=(.2,.2), transport_riemann=transport_riemann)
    code=emit_cpp_program(resolved.time,model=lower_and_validate(model)[0])
    assert code.count("ctx.neg_div_flux_default_with_faces_into(") == 1
    assert "ctx.prepare_provider_values(" not in code  # field-free native flux stays field-free
    assert "_transport_faces" in code
    assert "diffusive_transport_frequency_" in code
    assert "ctx.max_wave_speed(" not in code
    assert "combined_transport_diffusion_stability" in code
    assert ".explicit_frequency() + diffusive_transport_frequency_" in code
    assert code.index(".stage_accepted_exchanges(")<code.index("ctx.commit_many(")
    assert model._dsl._m._flux


def test_runtime_diffusion_coefficient_is_retained_in_program_parameter_metadata():
    from pops.codegen.module_lowering import lower_and_validate
    from pops.codegen.program_emit_params import program_param_entries
    resolved,_,model=resolved_heat(runtime_coefficient=True)
    assert program_param_entries(resolved.time,lower_and_validate(model)[0])==[(0,"nu",0,.1)]


def test_unproved_accepted_transform_cannot_silently_drop_diffusive_exchanges():
    from types import SimpleNamespace
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature
    resolved,_,_=resolved_heat()
    accepted=next(iter(resolved.time._commits.values()))
    transformed=SimpleNamespace(op="local_transform",inputs=(accepted,))
    with pytest.raises(ValueError,match="hides diffusive exchanges"):
        accepted_diffusive_quadrature(SimpleNamespace(_commits={"state":transformed}))


@pytest.mark.parametrize("frontier", [None, "rhs", "source", "coupled_rate"])
def test_deep_diffusion_search_retains_fresh_residual_frontiers(frontier):
    from types import SimpleNamespace
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature

    value = SimpleNamespace(op="diffusive_rhs", inputs=())
    if frontier is not None:
        value = SimpleNamespace(op=frontier, inputs=(value,))
    for _ in range(1500):
        value = SimpleNamespace(op="product_get", inputs=(value,))
    program = SimpleNamespace(_commits={"state": value})
    if frontier is None:
        with pytest.raises(ValueError, match="hides diffusive exchanges"):
            accepted_diffusive_quadrature(program)
    else:
        assert accepted_diffusive_quadrature(program) == ()


def test_shared_deep_diffusion_free_structure_has_no_hidden_exchange():
    from types import SimpleNamespace
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature

    value = SimpleNamespace(op="state", inputs=())
    for _ in range(1500):
        value = SimpleNamespace(op="product_get", inputs=(value, value))
    assert accepted_diffusive_quadrature(SimpleNamespace(_commits={"state": value})) == ()


def test_deep_affine_quadrature_preserves_shared_path_multiplicity_order_and_exact_weights():
    from fractions import Fraction
    from types import SimpleNamespace
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature

    first = SimpleNamespace(op="diffusive_rhs", inputs=(), id=11)
    second = SimpleNamespace(op="diffusive_rhs", inputs=(), id=12)
    value = SimpleNamespace(op="linear_combine", inputs=(first, second, first), attrs={
        "coeffs": ({1: Fraction(1, 3)}, {1: Fraction(-1, 2)}, {1: Fraction(2, 3)})})
    for _ in range(1500):
        value = SimpleNamespace(op="linear_combine", inputs=(value,), attrs={"coeffs": ({0: 1},)})
    program = SimpleNamespace(_commits={"state": value})
    rows = accepted_diffusive_quadrature(program, authored_stability_rates=(11, 12))
    assert [(rate.id, weights) for rate, weights in rows] == [
        (11, {1: Fraction(1)}), (12, {1: Fraction(-1, 2)})]
    with pytest.raises(ValueError, match="SSP coefficient-one convex proof"):
        accepted_diffusive_quadrature(program)


def test_deep_local_euler_atoms_keep_exact_coefficients_and_duplicate_paths():
    from fractions import Fraction
    from types import SimpleNamespace
    from pops.codegen.program_diffusion_exchanges import _single_forward_euler_with_frozen_inputs

    class Reference:
        block_ref = "block"

    reference = Reference()
    initial = SimpleNamespace(op="state", block="block", point="n", state_ref=reference)
    rate = SimpleNamespace(op="diffusive_rhs", block="block", point="n", inputs=(initial,))

    def wrapped(inputs, coefficients):
        value = SimpleNamespace(op="linear_combine", inputs=inputs, attrs={"coeffs": coefficients})
        for _ in range(1500):
            value = SimpleNamespace(op="linear_combine", inputs=(value,), attrs={"coeffs": ({0: 1},)})
        return SimpleNamespace(_commits={reference: value})

    rows = ((rate, {1: Fraction(1)}),)
    assert _single_forward_euler_with_frozen_inputs(
        wrapped((initial, rate), ({0: 1}, {1: 1})), rows)
    assert not _single_forward_euler_with_frozen_inputs(
        wrapped((initial, rate), ({0: 1}, {1: Fraction(3, 2)})), rows)
    # The local recognizer intentionally accepts one actual Euler rate atom.
    # Sharing a node must not silently collapse two authored occurrences.
    assert not _single_forward_euler_with_frozen_inputs(
        wrapped((initial, rate, rate), ({0: 1}, {1: Fraction(1, 2)}, {1: Fraction(1, 2)})), rows)


def test_explicit_affine_cycle_is_rejected_without_unbounded_work():
    from fractions import Fraction
    from types import SimpleNamespace
    from pops.codegen.program_diffusion_exchanges import accepted_diffusive_quadrature

    value = SimpleNamespace(op="linear_combine", attrs={"coeffs": ({0: Fraction(1)},)})
    value.inputs = (value,)
    with pytest.raises(ValueError, match="quadrature contains a cycle"):
        accepted_diffusive_quadrature(SimpleNamespace(_commits={"state": value}))
