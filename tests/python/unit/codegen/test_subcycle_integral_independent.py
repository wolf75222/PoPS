"""Independent FV balance witnesses for accepted transport across child clocks.

These are Source tests. They do not qualify Native rollback or execute a PDE.
"""
import ast
from fractions import Fraction
import re
from types import SimpleNamespace

import pops
import pytest
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_transport_quadrature import accepted_transport_quadrature
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.time import Clock, FixedDt, SampleAndHold, StagePoint, TimePoint


def _case(*, ssprk2=False, nested=False, preface=False):
    frame = Rectangle('unit_domain', lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model('independent_advection', frame=frame)
    u = model.state('u', components=('density',))
    x, y = frame.axes
    flux = model.flux('F', frame=frame, state=u, components={x: tuple(u), y: (0*u[0],)},
                      waves={x: (1.,), y: (0.,)})
    rate = model.rate('balance', equation=ddt(u) == -div(flux))
    case = pops.Case('independent_child_balance')
    block = case.block('fluid', model)
    numerical = DiscretizationPlan()
    numerical.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(u),
                          reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    numerical.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Outflow(state=block[u]),
        frame.boundaries.x_max: Outflow(state=block[u]),
    }, periodic=PeriodicAxes((y,))))
    case.numerics(numerical, block=block)
    p = pops.Program('weighted_child_transport')
    state = p.state(block[u])
    quantity = p.integral_state('q', initial=.7)
    rates = []
    initial = state.n
    if preface:
        preceding_rate = rate(state.n)
        rates.append(preceding_rate)
        p.accept_external_trace(quantity, rate=preceding_rate, axis=0, side=1, component=0,
                                scale=-1.)
        initial = p.value('prelude', state.n+p.dt*preceding_rate, at=state.n.point)

    def child_window(previous, parent_clock, depth):
        clock = Clock('child_%d' % depth, owner=p.owner_path)
        child_state = p.state(block[u], clock=clock)
        seed = p.synchronize(previous, at=TimePoint(clock), relation=SampleAndHold())

        def tick(builder, current):
            if nested and depth == 0:
                inside = child_window(current, clock, 1)
                return builder.synchronize(inside, at=child_state.next.point,
                                           relation=SampleAndHold())
            first = rate(current)
            rates.append(first)
            builder.accept_external_trace(quantity, rate=first, axis=0, side=1, component=0,
                                          scale=-1.)
            predictor = builder.value('predictor_%d' % depth, current+builder.dt*first,
                                      at=StagePoint('predictor_%d' % depth,
                                                    {'main': TimePoint(clock, 1)}))
            if not ssprk2:
                return builder.value('euler_%d' % depth, predictor, at=child_state.next.point)
            second = rate(predictor)
            rates.append(second)
            builder.accept_external_trace(quantity, rate=second, axis=0, side=1, component=0,
                                          scale=-1.)
            return builder.value('ssprk2_%d' % depth,
                Fraction(1, 2)*current+Fraction(1, 2)*predictor+Fraction(1, 2)*builder.dt*second,
                at=child_state.next.point)

        return p.subcycle(seed, clock=clock, within=parent_clock, count=2, body_fn=tick)

    advanced = child_window(initial, p.clock, 0)
    returned = p.synchronize(advanced, at=state.next.point, relation=SampleAndHold())
    accepted = p.value('outer_half', Fraction(1, 2)*state.n+Fraction(1, 2)*returned,
                       at=state.next.point)
    p.commit(state.next, accepted)
    p.step_strategy(FixedDt(.1))
    case.program(p)
    layout = Uniform(CartesianGrid(frame=frame, cells=(4, 4), periodic=PeriodicAxes((y,))))
    return case, layout, p, accepted, rates


def _fv_reference(parent_dt, ticks, ssprk2):
    """Exact upwind cell balance, independent of compiler graph/weights/identities."""
    initial = tuple(map(Fraction, (1, 2, 3, 4)))
    current = initial
    q = Fraction(7, 10)
    h = parent_dt/ticks
    traces = []

    def rhs(values):
        return tuple(-4*(values[i]-values[max(0, i-1)]) for i in range(4))

    for _ in range(ticks):
        predictor = tuple(u+h*r for u, r in zip(current, rhs(current)))
        if ssprk2:
            traces.extend((current[-1], predictor[-1]))
            q += h*(current[-1]+predictor[-1])/4
            current = tuple((u+v+h*r)/2 for u, v, r in zip(current, predictor, rhs(predictor)))
        else:
            traces.append(current[-1])
            q += h*current[-1]/2
            current = predictor
    accepted = tuple((u+v)/2 for u, v in zip(initial, current))
    return initial, accepted, q, traces


def _coefficient(expression, dt):
    """Read only scalar C++ arithmetic from the physical ExchangeRecord argument."""
    expression = expression.replace('static_cast<pops::Real>', '').replace('pops::Real', '')

    def read(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.Name) and node.id == 'dt':
            return dt
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -read(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, ast.Add):
                return left+right
            if isinstance(node.op, ast.Mult):
                return left*right
            if isinstance(node.op, ast.Div):
                return left/right
        raise AssertionError('unrecognized physical temporal coefficient: '+expression)

    return read(ast.parse(expression, mode='eval').body)


@pytest.mark.parametrize('nested', (False, True))
@pytest.mark.parametrize('ssprk2', (False, True))
def test_child_scoped_transport_keeps_outer_weight_and_exact_tick_duration(nested, ssprk2):
    case, layout, program, _, rates = _case(ssprk2=ssprk2, nested=nested)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    ticks = 4 if nested else 2
    stages = 2 if ssprk2 else 1
    weights = re.findall(r'face_values\.axes\[axis\]\(face, component\)/measure, (.*), 1,', source)
    assert len(weights) == len(rates) == stages
    # Count the actual child-duration scopes. A neutral (0,1) scope may preserve the
    # caller's stage around face staging; it must not be mistaken for another child tick.
    child_scopes = re.findall(r'ctx\.logical_evaluation_scope\([^,\n]+,\s*2\)', source)
    assert len(child_scopes) == (2 if nested else 1)
    assert source.count('ctx.subcycle_scope(') == (2 if nested else 1)
    # The observable physical ExchangeRecord must carry the global outer half weight, evaluated
    # with the runtime child dt, not parent dt and not a sum that loses the stage identity.
    dt = Fraction(1, 10)
    physical = [_coefficient(weight, dt/ticks) for weight in weights]
    assert physical == [dt/(2*ticks*stages)]*stages
    contexts = re.findall(r'/evaluation:(\d+)', source)
    assert set(map(int, contexts)) == {value.id for value in rates}
    initial, accepted, q, traces = _fv_reference(dt, ticks, ssprk2)
    assert accepted != initial  # A future Native rollback witness cannot be a constant state.
    assert Fraction(7, 10)+sum(weight*flux for weight, flux in
        zip(physical*ticks, traces, strict=True)) == q
    assert sum(accepted)/4+q == sum(initial)/4+Fraction(7, 10)+dt/2


def test_nonconstant_reference_distinguishes_ssprk_stages_and_retry_from_rejection():
    initial, euler, q_euler, _ = _fv_reference(Fraction(1, 10), 2, False)
    _, rk, q_rk, _ = _fv_reference(Fraction(1, 10), 2, True)
    assert q_euler == Fraction(179, 200)
    assert q_rk == Fraction(89, 100)
    assert euler != rk and euler != initial
    _, _, retry_q, _ = _fv_reference(Fraction(2, 25), 2, True)
    assert retry_q == Fraction(1067, 1250)
    # No Native rejection was executed here. These values are a mathematical reference only;
    # the integration witness must restore q=.7 and all State/history/cursors/ledger bytes.


def test_loop_input_transport_is_counted_once_at_its_own_parent_duration():
    case, layout, _, _, rates = _case(preface=True)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    source = emit_cpp_program(resolved.time,
        model_graph=ProgramModelGraph.from_resolved_blocks(resolved.blocks))
    weights = re.findall(r'face_values\.axes\[axis\]\(face, component\)/measure, (.*), 1,', source)
    assert len(weights) == len(rates) == 2
    parent_dt = Fraction(1, 10)
    assert _coefficient(weights[0], parent_dt) == parent_dt/2
    assert _coefficient(weights[1], parent_dt/2) == parent_dt/4
    parent_evaluation = '/evaluation:%d' % rates[0].id
    child_evaluation = '/evaluation:%d' % rates[1].id
    for scope in re.finditer(r'auto (subcycle_scope_\d+) = ctx\.subcycle_scope\(', source):
        finish = source.index(scope.group(1)+'.finish();', scope.end())
        assert not scope.start() < source.index(parent_evaluation) < finish
        assert scope.start() < source.index(child_evaluation) < finish
    # Initial right flux4, then the Euler parent seed has right flux18/5; the next
    # child input has right flux17/5. Each new evaluation is an actual face contribution.
    expected_q = Fraction(7, 10)+parent_dt*4/2+parent_dt*(Fraction(18, 5)+Fraction(17, 5))/4
    assert expected_q == Fraction(43, 40)


@pytest.mark.parametrize('scheduled', (False, True))
def test_unauthenticated_transforms_or_schedules_still_refuse_transport(scheduled):
    _, _, program, accepted, rates = _case()
    attrs = {'schedule': object()} if scheduled else {}
    opaque = SimpleNamespace(id=99999, op='local_transform', inputs=(accepted,), attrs=attrs)
    boundary = SimpleNamespace(_commits={'state': opaque})
    with pytest.raises(ValueError, match='hides transport|scoped quadrature|authenticated'):
        accepted_transport_quadrature(boundary, {value.id: value for value in rates})
