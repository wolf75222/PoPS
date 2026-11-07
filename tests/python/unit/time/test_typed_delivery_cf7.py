"""CF7 regression case (spec 0.3 IR19, B4, D5): resolve concludes on owned deliveries without
enumerating them, inside a loop of 10**4 iterations and inside a map over 10**5 entries."""
import pytest

import pops
from pops import math as pm
from pops._ir.refusals import CoverageNotExact, DeliveryNotStructural
from pops.codegen import typed_delivery


def exchange(values):
    return (values[1]-values[0], values[0]-values[1])


def loop_case(count, *, accumulate=False):
    """The literal fold of realization_loop_checks.py ('overwrite2', 'accumulate2') at ``count``."""
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, ProgramRealization
    from pops.realization import Realize, Use
    from pops.time import FixedDt
    frame = Rectangle('cf7_loop_domain', (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model('cf7_loop_exchange', frame=frame)
    state = model.state('U', components=('a', 'b'), sampling='cell_average')
    source = model.source('exchange', on=state, value=exchange(state))
    balance = model.rate('balance', equation=pm.ddt(state) == source)
    use = Use(balance.occurrences[0], 'cell', 'accepted_step', 'rate')
    numerics = DiscretizationPlan()
    numerics.rates.add(balance, ProgramRealization(uses=(use,)))
    case = pops.Case('cf7_loop')
    block = case.block('material', model)
    case.numerics(numerics, block=block)
    p = pops.Program('cf7_loop')
    q = p.state(block[state])
    old = q.n
    def body(builder, carried, index):
        scaled = builder.value('input_at_iteration', (builder.float64(index)+1.)*old)
        emitted = Realize('owned_exchange', owns=(use,), bindings={'old': scaled},
                          body=lambda b, old: exchange(old), outputs={use: ()}).evaluate(builder)[use]
        return builder.value('accumulated', carried+emitted) if accumulate else emitted
    selected = p.fold(p.value('unowned_seed', (0*old[0], 0*old[1])), count, body)
    p.commit(q.next, p.value('candidate', old+p.dt*selected, at=q.next.point))
    p.step_strategy(FixedDt(.02))
    case.program(p)
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    return case, layout


def map_case(count, *, consumer='pointwise', literal=False):
    """A zero-block Continuation whose owned Use is realized element by element in a map.

    ``consumer`` updates each element with its own increment ('pointwise'), with the increment
    of element 0 ('element0_everywhere'), or with both ('twice_i_and_0'). With ``literal`` the
    maps range over the literal set of ``count`` entries, not over the set read back from the
    Continuation (whose capacity is ``count``).
    """
    from pops.continuations import ContinuationSchema
    from pops.frames import Cartesian1D
    from pops.graph_functions import graph
    from pops.realization import Realize, Use, require_continuation_coverage
    from pops.time import FixedDt
    from pops.time.variant_types import Bool, Product, type_of
    p = pops.Program('cf7_map')
    domain = p.range_set(count, capacity=count, ownership='replicated')
    initial = p.product(domain=domain, q=p.map(domain, lambda b, i: 1+.1*b.float64(b.index_value(i))),
                        time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    model = pops.Model('cf7_map_decay', frame=Cartesian1D())
    u = model.state('u', components=('u',))
    balance = model.rate('balance', equation=pm.ddt(u) == model.source('decay', on=u, value=(-u[0],)))
    use = Use(balance.occurrences[0], 'cells', 'interval_increment', 'point_value')
    h = p.requested_dt()
    def element(b, i):
        return Realize('decay', owns=(use,), bindings={'old': b.at(current['q'], i), 'h': h},
                       body=lambda r, old, h: -h*old, outputs={use: ()}).evaluate(b)[use]
    over = domain if literal else current['domain']
    increment = p.map(over, element)
    read = {'pointwise': lambda b, i: b.at(increment, i),
            'element0_everywhere': lambda b, i: b.at(increment, b.index(over, 0)),
            'twice_i_and_0': lambda b, i: b.at(increment, i)+b.at(increment, b.index(over, 0))}[consumer]
    updated = p.map(over, lambda b, i: b.at(current['q'], i)+read(b, i))
    continued = p.product(domain=over, q=updated, time=current['time']+h, step=current['step']+p.int64(1))
    envelope = p.product(continuation=continued, increment=p.product(), observations=p.product(),
                         checks=p.product())
    proposal = p.product(h=h)
    candidate = p.continuation_freeze(base, proposal, envelope)
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product()))
    def assess(b, a):
        return b.product(accepted=b.boolean(True), checks=b.product())
    publication = p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    require_continuation_coverage(publication, {('q',): (use,)})
    p.step_strategy(FixedDt(.02))
    case = pops.Case('cf7_map')
    case.program(p)
    return case, p


@pytest.fixture
def visits(monkeypatch):
    """Count the node visits of every delivery analysis that runs."""
    counted = []
    original = typed_delivery._Analysis._visit
    def counting(self, value, path, context):
        counted.append(value.id)
        return original(self, value, path, context)
    monkeypatch.setattr(typed_delivery._Analysis, '_visit', counting)
    return counted


def test_resolve_concludes_on_a_delivery_inside_a_loop_of_ten_thousand_iterations(visits):
    case, layout = loop_case(10**4)
    plan = pops.resolve(pops.validate(case), layout=layout)
    assert len(plan.blocks[0].numerics.rates[0].method.uses) == 1
    large = len(visits)
    visits.clear()
    case, layout = loop_case(10)
    pops.resolve(pops.validate(case), layout=layout)
    # The analysis visits the same nodes for 10 and for 10**4 iterations.
    assert large == len(visits) > 0


def test_resolve_refuses_an_accumulated_loop_delivery_naming_its_term():
    case, layout = loop_case(10**4, accumulate=True)
    with pytest.raises(CoverageNotExact, match=r'exactly one owned SSA delivery.*term balance\[0\] source '
                                                r'exchange \(cell, accepted_step, rate, whole\) is delivered '
                                                r'at least twice'):
        pops.resolve(pops.validate(case), layout=layout)


def emitted(count, **options):
    """The C++ of map_case; Program.validate, which the emitter runs, checks its coverage."""
    from examples.api_v02.models_M01_M14 import _domain
    from pops.codegen.program_codegen import emit_cpp_program
    _, layout = _domain('cf7_map_context', (4, 4))
    return emit_cpp_program(pops.resolve(pops.validate(map_case(count, **options)[0]), layout=layout).time)


@pytest.mark.parametrize('literal', [False, True])
def test_resolve_concludes_on_a_delivery_inside_a_map_over_a_hundred_thousand_entries(visits, literal):
    emitted(10**5, literal=literal)
    large = len(visits)
    visits.clear()
    emitted(10, literal=literal)
    # One visit of the map body stands for every element (IR19), on a literal set of 10**5
    # entries as on the set read back from the Continuation.
    assert large == len(visits) > 0


@pytest.mark.parametrize('consumer', ['element0_everywhere', 'twice_i_and_0'])
def test_resolve_refuses_a_map_element_read_at_another_element_naming_its_term(consumer):
    with pytest.raises(DeliveryNotStructural, match=r'\(a gather\).*; term balance\[0\] source decay '
                                                    r'\(cells, interval_increment, point_value, whole\)$'):
        emitted(10**5, consumer=consumer)


def test_a_map_element_that_delivers_twice_is_refused_naming_its_term():
    p = pops.Program('cf7_map_twice')
    domain = p.range_set(10**5, capacity=10**5, ownership='replicated')
    owned = []
    def body(g, i):
        owned.extend((g.float64(g.index_value(i)), 2*g.float64(g.index_value(i))))
        return owned[-2]+owned[-1]
    result = p.map(domain, body)
    with pytest.raises(CoverageNotExact, match='term element_term is delivered at least twice'):
        typed_delivery.require_single_delivery(result, lambda node: any(node is x for x in owned),
                                               term='element_term', missing='missing', duplicate='duplicate')
