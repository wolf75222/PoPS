"""Finite inventory publication equations authored in the actual PoPS Program.

This constructor is not evidence of native execution; the associated native tests
record it independently. The context has no physical Field carrier. All q, inventories, records and histories are native
Continuation values. The bounded dimensionless law is Q(q)=q or q+q²; it is not
an additional physical model. Independent expected values live only in tests.
"""
import pops
import pops.math as pm
from pops.continuations import Candidate, ContinuationSchema
from pops.contributions import Contribution, Delivery, FiniteInterval, Increment, LinearMap
from pops.graph_functions import graph
from pops.inventory_decode import checked_inventory_decode
from pops.ledger_provenance import InventoryBinding
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Int64, Product, type_of
from examples.api_v02.common import CaseSpec

MODES = ('dynamic_replacement', 'nonlinear_exact', 'nonlinear_tolerated',
         'bad_inverse', 'negative_branch')
DT = .125


def _layout():
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    frame = Rectangle('execution_context', (0., 0.), (1., 1.)).frame(Cartesian2D())
    return Uniform(CartesianGrid(frame=frame, cells=(4, 4),
                                periodic=PeriodicAxes(tuple(frame.axes))))


def _identity_map(recipient):
    @graph(inputs=Product(value=type_of(recipient)), output=type_of(recipient))
    def identity(g, a):
        return a['value']
    return LinearMap.differential(identity, at=recipient.prog.product(value=recipient), wrt='value')


def _linear_branch(g, old, window, *, replaced):
    recipient = old['q']
    action = _identity_map(recipient)
    original = Contribution('J', g.float64(1), window)
    old_ledger = Increment((original,), (Delivery(original, recipient, action),))
    old_value = old_ledger.apply(g, recipient, name='original_inventory')
    replacement = Contribution('J', g.float64(2), window)
    new_ledger = old_ledger.replace(original, replacement)
    new_value = new_ledger.apply(g, recipient, name='recomputed_inventory')
    ledger, retained, contribution = ((new_ledger, new_value, replacement) if replaced
                                       else (old_ledger, old_value, original))
    return g.product(q=retained, encoded=retained, target=retained,
                     defect=g.float64(0), delta=contribution.value,
                     increment=ledger.pack(g))


def _inventory_value(q, *, nonlinear):
    """Original inventory equation, reused by encoding and assessment."""
    return q + q*q if nonlinear else q


def _nonlinear(g, old, window, mode):
    data = g.product(step=old['step'])
    @graph(inputs=Product(value=Float64, data=type_of(data)), output=Float64)
    def encode(z, a):
        return _inventory_value(a['value'], nonlinear=True)
    @graph(inputs=Product(inventory=Float64, data=type_of(data), budget=Int64), output=Float64)
    def decode(z, a):
        target = a['inventory']
        exact = 2*target/(1+pm.sqrt(1+4*target))
        if mode == 'nonlinear_tolerated':
            return z.branch(a['data']['step'] == 0, lambda h: h.float64(.9), lambda h: exact)
        if mode == 'bad_inverse':
            return z.branch(a['data']['step'] > 0, lambda h: h.float64(0), lambda h: exact)
        if mode == 'negative_branch':
            return z.branch(a['data']['step'] > 0,
                            lambda h: (-1-pm.sqrt(1+4*target))/2, lambda h: exact)
        return exact
    @graph(inputs=Product(inventory=Float64, data=type_of(data)), output=Bool)
    def domain(z, a):
        return a['inventory'] >= 0
    @graph(inputs=Product(value=Float64, data=type_of(data)), output=Bool)
    def admissible(z, a):
        # Both real roots are admissible on this bounded chart; selection below
        # chooses the nonnegative root, even when the negative root solves Q.
        return z.isfinite(a['value'])
    @graph(inputs=Product(value=Float64, data=type_of(data)), output=Bool)
    def selection(z, a):
        return a['value'] >= 0
    recipient = encode(g, value=old['q'], data=data)
    action = _identity_map(recipient)
    source = Contribution('J', g.float64(2), window)
    ledger = Increment((source,), (Delivery(source, recipient, action),))
    target = ledger.apply(g, recipient, name='nonlinear_target_inventory')
    result = g.require_success(checked_inventory_decode(g, recipient=recipient,
        application=target, base_value=old['q'], encode=encode, decode=decode,
        inventory_domain=domain, admissible=admissible, selection=selection,
        before_data=data, after_data=data, budget=g.int64(4),
        atol=g.float64(.3 if mode == 'nonlinear_tolerated' else 1.e-12), rtol=g.float64(0)))
    residual = Contribution('conversion_residual', result['defect'], window, kind='defect')
    final = Increment((source, residual),
                      (Delivery(source, recipient, action), Delivery(residual, recipient, action)))
    # The measured R is a declaration in L+R, not a second update of the decoded q.
    packed = final.pack(g, bindings=(InventoryBinding(recipient, ('q',), conversion=result['value']),))
    return g.product(q=result['value'], encoded=result['encoded'], target=target,
                     defect=result['defect'], delta=source.value, increment=packed)


def build_inventory_publication_v15(*, mode='dynamic_replacement', steps=2):
    if mode not in MODES:
        raise ValueError('unknown bounded inventory publication mode')
    if type(steps) is not int or steps < 1:
        raise ValueError('positive integer number of steps required')
    p = pops.Program('inventory_publication_'+mode)
    initial = p.product(q=p.float64(0), inventory=p.float64(0), time=p.float64(0), step=p.int64(0),
        history=p.product(q=p.float64(0), inventory=p.float64(0), defect=p.float64(0)),
        record=p.product(delta=p.float64(0), defect=p.float64(0), target=p.float64(0)),
        controller=p.product(accepted=p.int64(0)))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    h = p.requested_dt()
    proposal = p.product(h=h)
    window = FiniteInterval(old['time'], old['time']+h, 'physical')
    if mode == 'dynamic_replacement':
        values = p.branch(old['step'] == 0,
            lambda g: _linear_branch(g, old, window, replaced=False),
            lambda g: _linear_branch(g, old, window, replaced=True))
    else:
        values = _nonlinear(p, old, window, mode)
    after = p.product(q=values['q'], inventory=values['encoded'], time=old['time']+h,
        step=old['step']+1,
        history=p.product(q=old['q'], inventory=old['inventory'], defect=old['record']['defect']),
        record=p.product(delta=values['delta'], defect=values['defect'], target=values['target']),
        controller=p.product(accepted=old['controller']['accepted']+1))
    candidate = Candidate(p, base=base, proposal=proposal, continuation=after,
        increment=values['increment'], observations=p.product(), checks=p.product())
    envelope = type_of(p.continuation_candidate_value(candidate))
    check_type = Product(balance=Float64, original_equation=Float64, reached=Float64)
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=envelope),
           output=Product(accepted=Bool, checks=check_type))
    def assess(g, a):
        before = a['base']; now = a['candidate']['continuation']; q = now['q']
        Q = _inventory_value(q, nonlinear=mode != 'dynamic_replacement')
        balance = now['inventory']-before['inventory']-now['record']['delta']-now['record']['defect']
        original = now['inventory']-Q
        reached = now['time']-before['time']-a['proposal']['h']
        accepted = g.branch(balance*balance <= 1.e-24,
            lambda z: z.branch(original*original <= 1.e-24,
                lambda y: y.branch(reached == 0, lambda x: now['step'] == before['step']+1,
                                    lambda x: x.boolean(False)), lambda y: y.boolean(False)),
            lambda z: z.boolean(False))
        return g.product(accepted=accepted,
                         checks=g.product(balance=balance, original_equation=original, reached=reached))
    accepted = p.continuation_accept(candidate, p.continuation_attest(candidate, assess))
    p.continuation_publish(accepted)
    p.step_strategy(FixedDt(DT))
    case = pops.Case('inventory_publication_'+mode)
    case.program(p)
    return CaseSpec(case, _layout(), {}, {}, DT*steps, DT,
        lambda runtime: {'passed': runtime.macro_step() == steps and runtime.time() == DT*steps},
        'Finite dimensionless inventory publication; exact original equations, ledger version, '
        'branch selection, measured defect and full finite Continuation.')
