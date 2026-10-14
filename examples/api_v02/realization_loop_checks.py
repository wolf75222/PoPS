"""Real Model/Case delivery tests; no native execution at import or construction."""
import numpy as np
import pops
from pops import math as pm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, ProgramRealization
from pops.projection import ConservativeCellAverage
from pops.realization import Realize, Use
from pops.time import FixedDt
from .common import CaseSpec, state_array, non_owner_metrics


def exchange(values):
    """The same equations build the model declaration and the method's native pointwise DAG."""
    return (values[1]-values[0], values[0]-values[1])


def assemble_delivery_case(kind='loop1', *, dt=.02, nested_failure=False):
    """Return CaseSpec plus the actual Program/Use/candidate/owned outputs for source audit.

    Literal folds overwrite one carried RHS with successive source evaluations.
    The last evaluation reads (index+1)*old, so count1 and count2 differ numerically.
    This is a finite delivery witness, not an order/consistency claim for this method.
    """
    allowed = {'loop0', 'loop1', 'overwrite2', 'dynamic_empty', 'dynamic_owned',
               'accumulate2', 'legacy_recovery', 'legacy_guarded', 'typed_guarded', 'where_predicate', 'expression_where_predicate'}
    if kind not in allowed:
        raise ValueError('unknown delivery case '+kind)
    frame = Rectangle('delivery_domain', (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model('finite_exchange', frame=frame)
    state = model.state('U', components=('a', 'b'), sampling='cell_average')
    source = model.source('exchange', on=state, value=exchange(state))
    balance = model.rate('balance', equation=pm.ddt(state) == source)
    use = Use(balance.occurrences[0], 'cell', 'accepted_step', 'rate')
    numerics = DiscretizationPlan()
    numerics.rates.add(balance, ProgramRealization(uses=(use,)))
    case = pops.Case('delivery_'+kind)
    block = case.block('material', model)
    case.numerics(numerics, block=block)
    p = pops.Program('delivery_'+kind)
    q = p.state(block[state])
    old = q.n
    owned = []
    def evaluate(builder, input_value):
        value = Realize('owned_exchange', owns=(use,), bindings={'old': input_value},
            body=lambda b, old: exchange(old), outputs={use: ()}).evaluate(builder)[use]
        owned.append(value)
        return value
    seed = p.value('unowned_seed', (0*old[0], 0*old[1]))
    if kind in {'loop0', 'loop1', 'overwrite2', 'dynamic_empty', 'dynamic_owned', 'accumulate2'}:
        initial = evaluate(p, old) if kind == 'dynamic_owned' else seed
        def body(builder, carried, index):
            scaled = builder.value('input_at_iteration', (builder.float64(index)+1.)*old)
            emitted = evaluate(builder, scaled)
            return builder.value('accumulated', carried+emitted) if kind == 'accumulate2' else emitted
        if kind.startswith('dynamic_'):
            selected = p.fold(initial, p.int64(p.requested_dt()*100.), body, max_steps=2)
            factor = 2.
        else:
            count = {'loop0': 0, 'loop1': 1, 'overwrite2': 2, 'accumulate2': 2}[kind]
            selected = p.fold(initial, count, body)
            factor = float(count)
    elif kind in {'legacy_recovery', 'legacy_guarded', 'typed_guarded'}:
        unowned = p.value('unowned_source', exchange(old))
        condition = p.sum(old) > (1e6 if kind == 'legacy_recovery' else 0.)
        from pops.time.variant_types import Result, type_of, Int64
        result_type = Result(type_of(old), Int64)
        def success(builder, value):
            return (builder.variant(result_type, 'ok', value) if kind == 'typed_guarded'
                    else builder.success(value))
        def failed(builder):
            error = (builder.variant(result_type, 'err', builder.int64(7))
                     if kind == 'typed_guarded' else builder.failure(unowned, 7))
            if nested_failure:
                # This success constructor is unreachable: its aliased payload
                # requires success from the deliberately failed Result first.
                alias = builder.value('unreachable_alias', builder.require_success(error))
                return success(builder, alias)
            return error
        result = p.branch(condition, lambda b: success(b, evaluate(b, old)), failed)
        if kind == 'legacy_recovery':
            selected = p.match(result, ok=lambda b, value: value,
                err=lambda b, failure: b.value('recovered_without_delivery', b.float64(failure['code'])*unowned))
            factor = 7.
        else:
            selected = p.require_success(result)
            factor = 1.
    else:
        emitted = evaluate(p, old)
        unowned = p.value('unowned_source', exchange(old))
        if kind == 'where_predicate':
            selected = p.where(p.cell_gt(emitted, 0.), unowned, unowned)
        else:
            selected = p.value('predicate_only_expression', tuple(
                pm.where(emitted[0] > 0., lambda i=i: unowned[i], lambda i=i: unowned[i])
                for i in range(2)))
        factor = 1.
    candidate = p.value('candidate', old+p.dt*selected, at=q.next.point)
    p.commit(q.next, candidate)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    subject = block[state]
    case.initials.add(InitialCondition(state=subject, value=BindArray(), projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(8, 4), periodic=PeriodicAxes(frame.axes)))
    initial_array = np.empty((2, 4, 8))
    x = (np.arange(8)+.5)/8
    initial_array[0] = .6+.1*np.cos(2*np.pi*x)[None, :]
    initial_array[1] = 1.1+.03*np.sin(2*np.pi*x)[None, :]
    rate = np.stack((initial_array[1]-initial_array[0], initial_array[0]-initial_array[1]))
    expected = initial_array+dt*factor*rate
    def invariants(runtime):
        actual = state_array(runtime, 'material', initial_array.shape)
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        inventory = float(np.max(np.abs(actual.sum(axis=0)-initial_array.sum(axis=0))))
        return {'passed': bool(np.isfinite(actual).all() and error < 3e-14 and inventory < 3e-14),
                'max_error': error, 'inventory_error': inventory, 'retained_factor': factor,
                'metrics_owner': True}
    spec = CaseSpec(case, layout, {subject: initial_array}, {'material': subject}, dt, dt,
                    invariants, 'finite native loop delivery '+kind+'; no time-order claim')
    return spec, p, use, candidate, tuple(owned)


def build_delivery_case(kind='loop1', *, dt=.02):
    return assemble_delivery_case(kind, dt=dt)[0]
