"""N03: authored periodic PDE, Newton and nested CG/GMRES in native Array SSA.

Coordinates, time and unknowns are explicitly nondimensional. This example checks
scientific typing on the complete finite solver graph, including its original
residual. It declares no model-specific native terminal or runtime callback.
"""
from __future__ import annotations

import pops
from pops._ir.quantity import PhysicalDimension, PhysicalSupport
from pops.model.spaces import Space
from pops.graph_functions import graph
from pops.relations import (Problem, Algorithm, Accuracy, Approximation, Evidence,
                            Check, RelationErrors, solve)
from pops.representations import ScientificType, ScientificMap, ScientificValue
from pops.time.variant_types import Bool, Float64, Int64, Product, Result, type_of


def _map(p, a, body):
    return p.map(p.domain(a), lambda b, i: body(b, i))


def _dot(p, a, b):
    return p.reduce(_map(p, a, lambda c, i: c.at(a, i)*c.at(b, i)), p.float64(0),
                    lambda c, total, term: total+term, order='ascending')


def _axpy(p, a, scale, b):
    return _map(p, a, lambda c, i: c.at(a, i)+scale*c.at(b, i))


def _laplacian(p, a):
    domain = p.domain(a); n = p.size(domain)
    def cell(b, i):
        j = b.index_value(i)
        return b.at(a, b.index(domain, (j+n-1)%n))-2*b.at(a, i)+b.at(a, b.index(domain, (j+1)%n))
    return p.map(domain, cell)


def _upwind(p, a):
    domain = p.domain(a); n = p.size(domain)
    return p.map(domain, lambda b, i: b.at(a, i)-b.at(a, b.index(domain, (b.index_value(i)+n-1)%n)))


def _residual(p, unknown, data):
    u = unknown['u']; lap = _laplacian(p, u); upwind = _upwind(p, u)
    return p.product(r=_map(p, u, lambda b, i: b.at(u, i)-b.at(data['old'], i)
        -data['mu']*b.at(lap, i)+data['beta']*b.at(upwind, i)+data['h']*b.at(u, i)**3))


def _jacobian_action(p, base, vector, h, mu, beta):
    lap = _laplacian(p, vector); upwind = _upwind(p, vector)
    return _map(p, vector, lambda b, i:
        (1+3*h*b.at(base, i)**2)*b.at(vector, i)-mu*b.at(lap, i)+beta*b.at(upwind, i))


def _ok(p, value):
    return p.variant(Result(type_of(value), RelationErrors), 'ok', value)


def spatial_newton_map(values, *, dishonest=False, krylov="cg"):
    """Return a checked finite map q_old,h,mu,beta -> backward-Euler state.

    Newton uses at most eight iterations. Each correction is a distinct original-
    equation solve J delta = F by CG (SPD) or GMRES (with upwind advection);
    the final solve independently rechecks F.
    The periodic Laplacian is non-diagonal for every cardinality >= 3.
    """
    if krylov not in ("cg", "gmres"):
        raise ValueError("choose the authored cg or gmres method")
    array = type_of(values); domain = type_of(values.prog.domain(values))
    U = Product(u=array); R = Product(r=array)
    data_type = Product(old=array, h=Float64, mu=Float64, beta=Float64)
    linear_data = Product(base=array, rhs=array, h=Float64, mu=Float64, beta=Float64)
    accuracy_type = Product(atol=Float64, rtol=Float64, scale=Float64, request=Product())
    evidence_type = Product(estimates=Product(), valid=Bool)
    approximation = Product(values=U, evidence=evidence_type, termination=Product(iterations=Int64))
    checked = Product(accepted=Bool, observations=Product())
    def problem(data, residual_body):
        args = Product(unknown=U, data=data)
        residual = graph(inputs=args, output=R)(lambda p, a: residual_body(p, a['unknown'], a['data']))
        def admissible(p, a):
            data = a['data']
            beta = data['beta'] == 0 if krylov == 'cg' else data['beta'] >= 0
            return p.branch(data['h'] >= 0,
                lambda b:b.branch(data['mu'] >= 0, lambda c:beta, lambda c:c.boolean(False)),
                lambda b:b.boolean(False))
        valid = graph(inputs=args, output=Bool)(admissible)
        request = graph(inputs=Product(unknown=U, data=data, residual=R, accuracy=accuracy_type,
                                      evidence=evidence_type), output=Result(checked, RelationErrors))(
            lambda p, a: _ok(p, Check(p, accepted=p.boolean(True), observations=p.product())))
        return Problem(residual, valid, valid, valid, request)
    nonlinear = problem(data_type, _residual)
    def linear_residual(p, unknown, data):
        applied = _jacobian_action(p, data['base'], unknown['u'], data['h'], data['mu'], data['beta'])
        return p.product(r=_axpy(p, applied, -1, data['rhs']))
    linear = problem(linear_data, linear_residual)
    norm = graph(inputs=Product(residual=R), output=Float64)(
        lambda p, a: _dot(p, a['residual']['r'], a['residual']['r']))
    def parameters(p, tolerance):
        return p.product(atol=p.float64(tolerance), rtol=p.float64(0), scale=p.float64(1), request=p.product())
    def approximation_value(p, answer, iterations):
        return _ok(p, Approximation(p, values=p.product(u=answer),
            evidence=Evidence(p, estimates=p.product(), valid=p.boolean(True)),
            termination=p.product(iterations=iterations)))
    @graph(inputs=Product(data=linear_data, guess=U, accuracy=accuracy_type, budget=Product()),
           output=Result(approximation, RelationErrors))
    def cg(p, a):
        data = a['data']; x = a['guess']['u']
        if krylov == 'gmres':
            from .scientific_krylov_v14 import gmres
            answer, iterations = gmres(p, data['rhs'], x,
                lambda b, vector:_jacobian_action(b, data['base'], vector, data['h'], data['mu'], data['beta']),
                atol=a['accuracy']['atol'], capacity=array.signature['capacity'])
            if dishonest: answer = _map(p, answer, lambda b, i:b.float64(0))
            return approximation_value(p, answer, iterations)
        original = linear_residual(p, a['guess'], data)['r']
        r = _map(p, original, lambda b, i: -b.at(original, i))
        initial = p.product(x=x, r=r, direction=r)
        def step(b, state):
            d, r = state['direction'], state['r']
            image = _jacobian_action(b, data['base'], d, data['h'], data['mu'], data['beta'])
            rr = _dot(b, r, r); alpha = rr/_dot(b, d, image)
            updated = _axpy(b, state['x'], alpha, d)
            residual = _axpy(b, r, -alpha, image)
            beta = _dot(b, residual, residual)/rr
            return b.product(x=updated, r=residual, direction=_axpy(b, residual, beta, d))
        outcome = p.require_success(p.iterate(initial, step,
            lambda b, state: _dot(b, state['r'], state['r']) <= a['accuracy']['atol'],
            max_steps=p.size(p.domain(x))+2))
        answer = outcome['state']['x']
        if dishonest:
            answer = _map(p, answer, lambda b, i: b.float64(0))
        return approximation_value(p, answer, outcome['iterations'])
    @graph(inputs=Product(data=data_type, guess=U, accuracy=accuracy_type, budget=Product()),
           output=Result(approximation, RelationErrors))
    def newton(p, a):
        data = a['data']
        def step(b, state):
            rhs = _residual(b, b.product(u=state), data)['r']
            inner_data = b.product(base=state, rhs=rhs, h=data['h'], mu=data['mu'], beta=data['beta'])
            zero = _map(b, state, lambda c, i: c.float64(0))
            correction = b.require_success(solve(b, linear, data=inner_data,
                algorithm=Algorithm(cg), guess=b.product(u=zero),
                accuracy=Accuracy(parameters(b, 1.e-26), norm), budget=b.product()))['values']['u']
            return _axpy(b, state, -1, correction)
        def stop(b, state):
            r = _residual(b, b.product(u=state), data)['r']
            return _dot(b, r, r) <= a['accuracy']['atol']
        outcome = p.require_success(p.iterate(a['guess']['u'], step, stop, max_steps=8))
        return approximation_value(p, outcome['state'], outcome['iterations'])
    @graph(inputs=data_type, output=array)
    def full_step(p, a):
        data = p.product(old=a['old'], h=a['h'], mu=a['mu'], beta=a['beta'])
        return p.require_success(solve(p, nonlinear, data=data, algorithm=Algorithm(newton),
            guess=p.product(u=a['old']), accuracy=Accuracy(parameters(p, 1.e-24), norm),
            budget=p.product()))['values']['u']
    def scientific(finite, dofs=None):
        space = Space('nondimensional_q', ('q',), support=PhysicalSupport((('x', 'periodic_line'),)),
                      units=(PhysicalDimension(),), value_shape=(), sampling='point', representation='coordinates')
        return ScientificType(finite, space, 'point_value', (), (PhysicalDimension(),),
                              {'name': 'uniform_counting'}, dofs=dofs)
    q_type = scientific(array, domain)
    scalar = scientific(Float64)
    return ScientificMap(full_step, inputs={'old':q_type, 'h':scalar, 'mu':scalar, 'beta':scalar},
                         output=q_type, strict=True)


def build_scientific_spatial_v14(*, count=5, steps=2, dishonest=False, krylov="cg"):
    """Zero-block Continuation carrying the complete nonlinear spatial solution.

    The original-equation relation and scientific map are checked. A finite Use
    occurrence binding is added at the generic integration seam, not inferred
    from this example's name or a dummy Model block.
    """
    from pops.continuations import ContinuationSchema
    from pops.time import FixedDt
    from .common import CaseSpec
    from .models_M01_M14 import _domain
    if type(count) is not int or count < 3 or count > 8:
        raise ValueError('this receiver samples 3..8 periodic cells')
    _, layout = _domain('scientific_array_context', (4, 4))
    p = pops.Program('nondimensional_periodic_diffusion_reaction')
    domain = p.range_set(count, capacity=8, ownership='replicated')
    q = p.map(domain, lambda b, i: 1+.1*b.float64(b.index_value(i)))
    initial = p.product(domain=domain, q=q, time=p.float64(0), step=p.int64(0))
    mapping = spatial_newton_map(q, dishonest=dishonest, krylov=krylov)
    schema = ContinuationSchema(initial, fields=(), time=('time',), step=('step',),
        scientific={('q',):mapping.output, ('time',):mapping.inputs['h']})
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    q = current['q']; h = p.requested_dt()
    mu = h*.1*p.float64(p.size(current['domain']))**2
    beta = h*(.3 if krylov == 'gmres' else 0)*p.float64(p.size(current['domain']))
    bindings = dict(old=ScientificValue(q, mapping.inputs['old']),
                    h=ScientificValue(h, mapping.inputs['h']),
                    mu=ScientificValue(mu, mapping.inputs['mu']),
                    beta=ScientificValue(beta, mapping.inputs['beta']))
    solved = mapping(p, **bindings)
    # These are the two actual terms of the declared PDE, evaluated at the
    # accepted implicit candidate. The mathematical Model has no runtime block.
    import pops.math as pm
    from pops.frames import Cartesian1D
    from pops.realization import Realize, Use
    model = pops.Model('periodic_diffusion_reaction_equations', frame=Cartesian1D())
    u = model.state('u', components=('u',))
    diffusion = model.diffusive_flux('diffusion', state=u, value=.1*pm.grad(u))
    reaction = model.source('reaction', on=u, value=(-u[0]**3,))
    rhs = pm.div(diffusion)+reaction
    if krylov == 'gmres':
        advection = model.flux('advection', state=u, frame=model.frame,
                               components={model.frame.axes[0]:(.3*u[0],)})
        rhs = rhs-pm.div(advection)
    balance = model.rate('physical_balance', equation=pm.ddt(u) == rhs)
    qtype = mapping.output
    diffusion_use = Use(balance.select(diffusion).occurrences[0], 'periodic_cells',
                        'interval_increment', 'point_value', scientific_type=qtype)
    reaction_use = Use(balance.select(reaction).occurrences[0], 'periodic_cells',
                       'interval_increment', 'point_value', scientific_type=qtype)
    array = qtype.finite_type
    @graph(inputs=Product(u=array, mu=Float64), output=array)
    def diffusion_increment(b, a):
        lap = _laplacian(b, a['u'])
        return _map(b, lap, lambda c, i: a['mu']*c.at(lap, i))
    @graph(inputs=Product(u=array, h=Float64), output=array)
    def reaction_increment(b, a):
        return _map(b, a['u'], lambda c, i: -a['h']*c.at(a['u'], i)**3)
    diff_map = ScientificMap(diffusion_increment, inputs={'u':qtype, 'mu':mapping.inputs['mu']},
                             output=qtype, strict=True)
    react_map = ScientificMap(reaction_increment, inputs={'u':qtype, 'h':mapping.inputs['h']},
                              output=qtype, strict=True)
    uses = [diffusion_use, reaction_use]
    outputs = {diffusion_use:('diffusion',), reaction_use:('reaction',)}
    if krylov == 'gmres':
        advection_use = Use(balance.select(advection).occurrences[0], 'periodic_cells',
            'interval_increment', 'point_value', scientific_type=qtype)
        @graph(inputs=Product(u=array, beta=Float64), output=array)
        def advection_increment(b, a):
            upwind = _upwind(b, a['u'])
            return _map(b, upwind, lambda c, i:-a['beta']*c.at(upwind, i))
        adv_map = ScientificMap(advection_increment, inputs={'u':qtype, 'beta':mapping.inputs['beta']},
                                output=qtype, strict=True)
        uses.append(advection_use); outputs[advection_use] = ('advection',)
    def realize(b, solution, h, mu, beta):
        out = {'diffusion':diff_map(b, u=solution, mu=mu),
               'reaction':react_map(b, u=solution, h=h)}
        if krylov == 'gmres': out['advection'] = adv_map(b, u=solution, beta=beta)
        return out
    realized = Realize('implicit_equation_terms', owns=tuple(uses),
        bindings={'solution':solved, 'h':bindings['h'], 'mu':bindings['mu'], 'beta':bindings['beta']},
        outputs=outputs, body=realize, strict=True).evaluate(p)
    result = q
    for use in uses: result = _axpy(p, result, 1, realized[use])
    proposal = p.product(h=h, mu=mu, beta=beta)
    continued = p.product(domain=current['domain'], q=result,
                         time=current['time']+h, step=current['step']+p.int64(1))
    envelope = p.product(continuation=continued, increment=p.product(), observations=p.product(), checks=p.product())
    candidate = p.continuation_freeze(base, proposal, envelope)
    assessment_type = Product(accepted=Bool, checks=Product(residual_squared=Float64))
    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=assessment_type)
    def assess(b, a):
        after = a['candidate']['continuation']
        residual = _residual(b, b.product(u=after['q']), b.product(old=a['base']['q'],
                             h=a['proposal']['h'], mu=a['proposal']['mu'], beta=a['proposal']['beta']))['r']
        norm = _dot(b, residual, residual)
        accepted = b.branch(b.same_domain(after['domain'], b.domain(after['q'])),
                           lambda c: norm < 1.e-23, lambda c: c.boolean(False))
        return b.product(accepted=accepted, checks=b.product(residual_squared=norm))
    publication = p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    from pops.realization import require_continuation_coverage
    require_continuation_coverage(publication, {('q',):tuple(uses)})
    p.step_strategy(FixedDt(.02))
    case = pops.Case('scientific_spatial_newton_'+krylov+'_v14'); case.program(p)
    return CaseSpec(case, layout, {}, {}, steps*.02, .02,
        lambda runtime: {'passed': runtime.macro_step()==steps, 'frontier_only':True},
        'Nondimensional periodic nonlinear diffusion/reaction'+(' and upwind advection' if krylov=='gmres' else '')+
        '; strict full Newton/'+krylov.upper()+' equations; zero-block complete Continuation')
