"""Differentiate the full checked inventory conversion and publish its primal ledger."""
import math
import pops
from pops.continuations import Candidate, ContinuationSchema
from pops.contributions import FiniteInterval
from pops.graph_derivatives import jvp
from pops.graph_functions import GraphFunction, graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of
from examples.api_v02.common import CaseSpec
from examples.api_v02.inventory_publication_v15 import DT, _layout, _nonlinear, _inventory_value


def build_derivative_inventory_v15(*, steps=2, initial_q=.5):
    if type(initial_q) not in (int, float) or not math.isfinite(initial_q) or initial_q < 0:
        raise ValueError('initial_q must be finite and nonnegative on the selected chart')
    contribution = Product(value=Float64, interval=Product(begin=Float64, end=Float64))
    ledger = Product(J=contribution, conversion_residual=contribution)
    output = Product(q=Float64, encoded=Float64, target=Float64, defect=Float64,
                     delta=Float64, increment=ledger)
    @graph(inputs=Product(q=Float64, time=Float64, h=Float64), output=output)
    def conversion(g, a):
        old = g.product(q=a['q'], step=g.int64(0))
        window = FiniteInterval(a['time'], a['time']+a['h'], 'physical')
        return _nonlinear(g, old, window, 'nonlinear_exact')
    # Round-trip the transformed closed graph. The executable primal guards and
    # the exact conversion/ledger references must survive both transformations.
    derivative = GraphFunction.from_data(jvp(conversion).to_data())
    p = pops.Program('inventory_conversion_derivative')
    # q=0 is a genuine boundary of the selected q>=0 chart. Its ambient
    # derivative must fail with 91001; the receiving profile starts inside it.
    initial = p.product(q=p.float64(initial_q), inventory=p.float64(initial_q+initial_q*initial_q), derivative=p.float64(0),
        time=p.float64(0), step=p.int64(0), history=p.product(q=p.float64(initial_q), derivative=p.float64(0)),
        record=p.product(defect=p.float64(0), target=p.float64(0)))
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',))
    base = p.continuation_open(initial, schema); old = p.continuation_read(base)
    h = p.requested_dt(); proposal = p.product(h=h)
    out = derivative(p, primal=p.product(q=old['q'], time=old['time'], h=h),
        tangent=p.product(q=p.float64(1), time=p.float64(0), h=p.float64(0)))
    primal = out['value']; tangent = out['tangent']['q']
    after = p.product(q=primal['q'], inventory=primal['encoded'], derivative=tangent,
        time=old['time']+h, step=old['step']+1,
        history=p.product(q=old['q'], derivative=old['derivative']),
        record=p.product(defect=primal['defect'], target=primal['target']))
    candidate = Candidate(p, base=base, proposal=proposal, continuation=after,
        increment=primal['increment'], observations=p.product(derivative=tangent), checks=p.product())
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal),
                         candidate=type_of(p.continuation_candidate_value(candidate))),
           output=Product(accepted=Bool, checks=Product(original=Float64, balance=Float64)))
    def assess(g, a):
        before=a['base']; now=a['candidate']['continuation']
        original=now['inventory']-_inventory_value(now['q'], nonlinear=True)
        balance=now['inventory']-before['inventory']-2-now['record']['defect']
        accepted=g.branch(original*original <= 1.e-24,
            lambda z: z.branch(balance*balance <= 1.e-24,
                lambda y: y.isfinite(now['derivative']), lambda y: y.boolean(False)),
            lambda z: z.boolean(False))
        return g.product(accepted=accepted, checks=g.product(original=original,balance=balance))
    p.continuation_publish(p.continuation_accept(candidate,p.continuation_attest(candidate,assess)))
    p.step_strategy(FixedDt(DT));case=pops.Case('inventory_conversion_derivative');case.program(p)
    return CaseSpec(case, _layout(), {}, {}, DT*steps, DT,
        lambda runtime: {'passed': runtime.macro_step()==steps and runtime.time()==DT*steps},
        f'JVP of the complete checked Q=q+q² conversion and ledger, initial q={initial_q}; '
        'ambient derivative only inside the nonnegative selection chart; primal proof publication, '
        'independent tangent oracle and strict checkpoint/restart.')
