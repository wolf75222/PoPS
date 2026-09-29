"""Public C11 contracts, independently of the native execution witness."""
import pytest

import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.numerics import FiniteVolume
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.numerics.variables import Conservative


def _principal_model(size, *, permuted=False, parameter=False):
    frame = Rectangle("principal-contract", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("principal-law", frame=frame)
    scale = 1.
    if parameter:
        from pops.params import RuntimeParam
        model.param(RuntimeParam("a_unused", default=3.))
        scale = model.value(model.param(RuntimeParam("speed", default=1.)))
    names = tuple("s%d" % j for j in range(size))
    if permuted:
        names = names[1:] + names[:1]
    states = tuple(model.species(name, state=("q",)) for name in names)
    x, y = frame.axes
    bound = 1. + .25 * (size - 1)
    fluxes = tuple(model.flux("F%d" % i, frame=frame, state=state,
        components={x: (scale * sum((1. if i == j else .25) * other[0]
                            for j, other in enumerate(states)),), y: (0.,)},
        waves={x: (scale * bound,), y: (0.,)}) for i, state in enumerate(states))
    return model, states, fluxes


@pytest.mark.parametrize("size", (2, 3, 5))
@pytest.mark.parametrize("permuted", (False, True))
def test_every_physical_flux_retains_its_complete_principal_inputs(size, permuted):
    model, states, fluxes = _principal_model(size, permuted=permuted)
    expected = tuple(state.space for state in states)
    for flux, state in zip(fluxes, states, strict=True):
        signature = model.module.operator_handle(flux.reg_name).signature
        assert signature.inputs == expected
        assert signature.output.base_space == state.space


@pytest.mark.parametrize("size", (2, 3, 5))
def test_reconstruction_sampling_is_an_explicit_typed_method_choice(size):
    model, states, fluxes = _principal_model(size)
    before = tuple(model.module.operator_handle(flux.reg_name).signature for flux in fluxes)
    methods = tuple(FiniteVolume(flux=flux, variables=Conservative(state),
        reconstruction=FirstOrder(), riemann=Rusanov(),
        sampling=tuple(other for other in states if other != state))
        for flux, state in zip(fluxes, states, strict=True))
    for method, state in zip(methods, states, strict=True):
        assert method.sampling == tuple(other for other in states if other != state)
        assert method.options()["sampling"] == method.sampling
    assert tuple(model.module.operator_handle(flux.reg_name).signature for flux in fluxes) == before


def test_foreign_same_named_state_is_not_a_principal_dependency():
    model, states, _ = _principal_model(2)
    foreign = pops.Model("principal-law", frame=model.frame)
    other = foreign.state("s1", components=("q",))
    x, y = model.frame.axes
    with pytest.raises(ValueError, match="foreign|owner|declared"):
        model.flux("foreign", frame=model.frame, state=states[0],
                   components={x: (states[0][0] + other[0],), y: (0.,)})


def _principal_case(size, *, parameter=False, reconstruction=None):
    from pops.math import ddt, div
    from pops.numerics import DiscretizationPlan
    model, states, fluxes = _principal_model(size, parameter=parameter)
    rates = tuple(model.rate("R%d" % i, equation=ddt(state) == -div(flux))
                  for i, (state, flux) in enumerate(zip(states, fluxes, strict=True)))
    case = pops.Case("principal_case")
    blocks = tuple(case.block("b%d" % i, model, states=(state,)) for i, state in enumerate(states))
    for block, state, flux, rate in zip(blocks, states, fluxes, rates, strict=True):
        plan = DiscretizationPlan()
        plan.rates.add(rate, FiniteVolume(flux=flux, variables=Conservative(state),
            reconstruction=FirstOrder() if reconstruction is None else reconstruction, riemann=Rusanov(),
            sampling=tuple(other for other in states if other != state)))
        case.numerics(plan, block=block)
    program = pops.Program("principal_step")
    temporal = tuple(program.state(block[state]) for block, state in zip(blocks, states, strict=True))
    bindings = dict(zip(states, (q.n for q in temporal), strict=True))
    evaluations = tuple(rate(q.n, bindings=bindings) for rate, q in zip(rates, temporal, strict=True))
    for q, rhs in zip(temporal, evaluations, strict=True):
        endpoint = program.value("endpoint", q.n + program.dt * rhs, at=q.next.point)
        program.commit(q.next, endpoint)
    case.program(program)
    return case, model, states, rates, blocks, program, evaluations


@pytest.mark.parametrize("size", (2, 3, 5))
def test_resolved_principal_group_has_every_physical_row_and_exact_instance(size):
    case, _, states, rates, blocks, _, evaluations = _principal_case(size)
    for value in evaluations:
        assert value.op == "principal_rate"
        assert len(value.inputs) == size
    groups = tuple(case._resolved_numerics_for(block.local_id).principal_groups[0] for block in blocks)
    assert len({group.identity for group in groups}) == 1
    group = groups[0]
    assert group.component_counts == (1,) * size
    assert group.states == tuple(case.resolve(block[state]) for block, state in zip(blocks, states, strict=True))
    assert group.rates == tuple(case.resolve(rate, block=block) for block, rate in zip(blocks, rates, strict=True))


def test_resolved_binding_keys_authenticate_the_exact_authoring_block():
    case, _, states, rates, blocks, _, evaluations = _principal_case(2)
    inputs = evaluations[0].inputs
    bindings = {case.resolve(block[state]): value
                for block, state, value in zip(blocks, states, inputs, strict=True)}
    assert rates[0](inputs[0], bindings=bindings).op == "principal_rate"


def test_resolved_binding_key_from_another_instance_is_not_interchangeable():
    case, model, states, rates, blocks, _, evaluations = _principal_case(2)
    inputs = evaluations[0].inputs
    clone = case.block("clone", model, states=(states[0],))
    bindings = {case.resolve(block[state]): value
                for block, state, value in zip(blocks, states, inputs, strict=True)}
    del bindings[case.resolve(blocks[0][states[0]])]
    bindings[case.resolve(clone[states[0]])] = inputs[0]
    with pytest.raises(ValueError, match="another block instance"):
        rates[0](inputs[0], bindings=bindings)
