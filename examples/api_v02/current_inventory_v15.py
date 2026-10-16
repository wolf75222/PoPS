"""A retained current equation with explicitly old history, in the real Program."""
import pops
from pops.continuations import Candidate, ContinuationSchema
from pops.current_dependencies import CurrentDependency
from pops.contributions import FiniteInterval
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of
from examples.api_v02.common import CaseSpec
from examples.api_v02.inventory_publication_v15 import DT, _layout, _linear_branch


@graph(inputs=Product(q=Float64, history=Float64), output=Float64)
def current_equation(p, a):
    return 2*a['q']+a['history']


def build_current_inventory_v15(*, steps=2):
    p = pops.Program('inventory_current_equation')
    initial = p.product(q=p.float64(0), z=p.float64(0), history=p.float64(0),
                        time=p.float64(0), step=p.int64(0))
    dependency = CurrentDependency(current_equation, target=('z',),
                                   inputs={'q': ('q',), 'history': ('history',)})
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
                                current_dependencies=(dependency,))
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    h = p.requested_dt(); proposal = p.product(h=h)
    window = FiniteInterval(old['time'], old['time']+h, 'physical')
    values = p.branch(old['step'] == 0,
        lambda g: _linear_branch(g, old, window, replaced=False),
        lambda g: _linear_branch(g, old, window, replaced=True))
    # History is intentionally the old q. It is the current candidate's retained
    # history path; no timestamp relabel or implicit recomputation is used.
    current = p.product(q=values['q'], history=old['q'])
    z = dependency.evaluate(p, current=current)
    after = p.product(q=current['q'], z=z, history=current['history'],
                      time=old['time']+h, step=old['step']+1)
    candidate = Candidate(p, base=base, proposal=proposal, continuation=after,
        increment=values['increment'], observations=p.product(), checks=p.product())
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal),
                         candidate=type_of(p.continuation_candidate_value(candidate))),
           output=Product(accepted=Bool, checks=Product(equation=Float64)))
    def assess(g, a):
        now = a['candidate']['continuation']
        error = now['z']-2*now['q']-now['history']
        accepted = g.branch(error == 0,
            lambda h: now['history'] == a['base']['q'], lambda h: h.boolean(False))
        return g.product(accepted=accepted, checks=g.product(equation=error))
    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    p.step_strategy(FixedDt(DT))
    case = pops.Case('inventory_current_equation'); case.program(p)
    return CaseSpec(case, _layout(), {}, {}, steps*DT, DT,
        lambda runtime: {'passed': runtime.macro_step() == steps and runtime.time() == steps*DT},
        'Retained z=2*q_current+history_old; explicit schema dependency, dynamic inventory '
        'replacement, zero-block Continuation and strict restart.')
