"""Magnetic Euler--Poisson with an authored midpoint source and PDE Schur solve.

Primary source: https://arxiv.org/html/2510.11808v3, equations (8), (9),
(12), (14), (16), (24).  This is an explicitly specified periodic finite-volume
adaptation, not the paper's finite-element discretization.  Every spatial cell,
conserved component, potential, Krylov iterate and time update below belongs to
the native Program algebra. NumPy is used only in the independent test oracle.

Forward differences G and their exact counting-measure adjoints G* define the
electric energy and K=G*G. Both components of velocity live at every site; the
electric gradient is first-order at that site. This choice has only constants
in the periodic nullspace, including even grids. The zero-mean gauge is explicit.
The hyperbolic and source substeps separately conserve their respective energy
balances. Strang splitting does not exactly maintain Gauss: its measured defect
is retained, never replaced by an unrelated Poisson solve after the source step.
"""
from __future__ import annotations

import math

import pops
from pops import math as pm
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .models_M01_M14 import _domain
from .scientific_krylov_v14 import gmres


COMPONENTS = ("rho", "mx", "my", "energy", "phi", "omega")
FLUID = COMPONENTS[:4]


def _map(p, values, body):
    return p.map(p.domain(values), body)


def _get(p, values, offset):
    return p.at(values, p.index(p.domain(values), offset))


def _sum(p, values):
    return p.reduce(values, p.float64(0), lambda b, total, x: total+x,
                    order="ascending")


def _dot(p, a, b):
    return _sum(p, _map(p, a, lambda c, i: c.at(a, i)*c.at(b, i)))


def _axpy(p, a, scale, b):
    return _map(p, a, lambda c, i: c.at(a, i)+scale*c.at(b, i))


def _component(p, state, key):
    return _map(p, state, lambda b, i: b.at(state, i)[key])


def _neighbor(j, axis, direction, cells):
    nx, ny = cells
    if axis == 0:
        return (j//nx)*nx+(j%nx+direction+nx)%nx
    return ((j//nx+direction+ny)%ny)*nx+j%nx


def _difference(p, values, axis, cells, lengths, *, adjoint=False):
    """G=(shift(+1)-I)/dx; G*=(shift(-1)-I)/dx, exactly adjoint."""
    direction = -1 if adjoint else 1
    scale = cells[axis]/lengths[axis]
    return _map(p, values, lambda b, i:
        scale*(_get(b, values, _neighbor(b.index_value(i), axis, direction, cells))
               -b.at(values, i)))


def _gradient(p, phi, cells, lengths):
    return tuple(_difference(p, phi, a, cells, lengths) for a in range(2))


def _adjoint(p, vector, cells, lengths):
    x, y = (_difference(p, vector[a], a, cells, lengths, adjoint=True)
            for a in range(2))
    return _axpy(p, x, 1, y)


def _stiffness(p, phi, cells, lengths):
    return _adjoint(p, _gradient(p, phi, cells, lengths), cells, lengths)


def _inverse_rotation(p, x, y, omega, s):
    """(I-s cross(Omega))^-1, with cross(v,Omega)=(vy,-vx) Omega."""
    def entry(b, i, axis):
        a = s*b.at(omega, i)
        vx, vy = b.at(x, i), b.at(y, i)
        return (vx+a*vy if axis == 0 else vy-a*vx)/(1+a*a)
    return tuple(_map(p, x, lambda b, i, a=a: entry(b, i, a)) for a in range(2))


def _multiply(p, a, b):
    return _map(p, a, lambda c, i: c.at(a, i)*c.at(b, i))


def _source_midpoint(p, state, h, *, cells, lengths, alpha, solver, tolerance):
    """Solve the two original midpoint equations, then extrapolate endpoints.

    B v* + s G phi* = v_n,
    K phi* - s alpha G*(rho v*) = K phi_n, with s=h/2.
    The same density is used on both sides of the electric-work pairing.
    """
    count = math.prod(cells)
    rho, old_phi, omega = (
        _component(p, state, key) for key in ("rho", "phi", "omega"))
    old_v = tuple(_map(p, state, lambda b, i, key=key:
        b.at(state, i)[key]/b.at(state, i)["rho"]) for key in ("mx", "my"))
    s = h/2
    old_k = _stiffness(p, old_phi, cells, lengths)
    zero = _map(p, old_phi, lambda b, i: b.float64(0))
    old_mean = _sum(p, old_phi)/count

    def add_gauge(b, value, phi):
        mean = _sum(b, phi)/count
        return _map(b, value, lambda c, i: c.at(value, i)+mean)

    if solver == "schur":
        rotated = _inverse_rotation(p, *old_v, omega, s)
        current = tuple(_multiply(p, rho, v) for v in rotated)
        rhs = _axpy(p, old_k, s*alpha, _adjoint(p, current, cells, lengths))
        rhs = _map(p, rhs, lambda b, i: b.at(rhs, i)+old_mean)

        def action(b, phi):
            rotated_grad = _inverse_rotation(b, *_gradient(b, phi, cells, lengths), omega, s)
            current = tuple(_multiply(b, rho, v) for v in rotated_grad)
            image = _axpy(b, _stiffness(b, phi, cells, lengths), s*s*alpha,
                          _adjoint(b, current, cells, lengths))
            return add_gauge(b, image, phi)

        mid_phi, iterations = gmres(p, rhs, zero, action,
            atol=p.float64(tolerance*tolerance), capacity=count)
        gradient = _gradient(p, mid_phi, cells, lengths)
        driving = tuple(_axpy(p, old_v[a], -s, gradient[a]) for a in range(2))
        mid_v = _inverse_rotation(p, *driving, omega, s)
    elif solver == "monolithic":
        domain = p.range_set(3*count, capacity=3*count, ownership="replicated")

        def pack(b, vx, vy, phi):
            def entry(c, i):
                j = c.index_value(i)
                return c.branch(j < count, lambda d: _get(d, vx, j),
                    lambda d: d.branch(j < 2*count, lambda e: _get(e, vy, j-count),
                        lambda e: _get(e, phi, j-2*count)))
            return b.map(domain, entry)

        def unpack(b, values):
            return tuple(_map(b, old_phi, lambda c, i, a=a:
                _get(c, values, c.index_value(i)+a*count)) for a in range(3))

        potential_rhs = _map(p, old_k, lambda b, i: b.at(old_k, i)+old_mean)
        rhs = pack(p, *old_v, potential_rhs)
        guess = p.map(domain, lambda b, i: b.float64(0))

        def action(b, values):
            vx, vy, phi = unpack(b, values)
            gx, gy = _gradient(b, phi, cells, lengths)
            first = _map(b, vx, lambda c, i: c.at(vx, i)
                -s*c.at(omega, i)*c.at(vy, i)+s*c.at(gx, i))
            second = _map(b, vy, lambda c, i: c.at(vy, i)
                +s*c.at(omega, i)*c.at(vx, i)+s*c.at(gy, i))
            current = (_multiply(b, rho, vx), _multiply(b, rho, vy))
            last = _axpy(b, _stiffness(b, phi, cells, lengths), -s*alpha,
                         _adjoint(b, current, cells, lengths))
            return pack(b, first, second, add_gauge(b, last, phi))

        answer, iterations = gmres(p, rhs, guess, action,
            atol=p.float64(tolerance*tolerance), capacity=3*count)
        vx, vy, mid_phi = unpack(p, answer)
        mid_v = (vx, vy)
    else:
        raise ValueError("solver must be 'schur' or 'monolithic'")

    # Independently reconstruct BOTH original equations, not just the Schur norm.
    gradient = _gradient(p, mid_phi, cells, lengths)
    defects = tuple(_map(p, mid_v[a], lambda b, i, a=a:
        b.at(mid_v[a], i)-b.at(old_v[a], i)+s*b.at(gradient[a], i)
        -(1 if a == 0 else -1)*s*b.at(omega, i)*b.at(mid_v[1-a], i))
        for a in range(2))
    current = tuple(_multiply(p, rho, v) for v in mid_v)
    potential_defect = _axpy(p,
        _axpy(p, _stiffness(p, mid_phi, cells, lengths), -1, old_k),
        -s*alpha, _adjoint(p, current, cells, lengths))
    squared = sum(_dot(p, r, r) for r in (*defects, potential_defect))
    p.require_success(p.branch(squared < 64*tolerance*tolerance,
        lambda b: b.success(squared), lambda b: b.failure(squared, 90901)))
    endpoint_v = tuple(_map(p, mid_v[a], lambda b, i, a=a:
        2*b.at(mid_v[a], i)-b.at(old_v[a], i)) for a in range(2))
    endpoint_phi = _map(p, old_phi, lambda b, i: 2*b.at(mid_phi, i)-b.at(old_phi, i))

    def endpoint(b, i):
        before = b.at(state, i)
        r = before["rho"]
        vx, vy = (b.at(v, i) for v in endpoint_v)
        kinetic_change = .5*r*(vx*vx+vy*vy)-.5*(before["mx"]**2+before["my"]**2)/r
        return b.product(rho=r, mx=r*vx, my=r*vy, energy=before["energy"]+kinetic_change,
                         phi=b.at(endpoint_phi, i), omega=before["omega"])
    return p.map(p.domain(state), endpoint), squared, iterations


def _flux(q, axis, gamma):
    rho, mx, my, energy = (q[key] for key in FLUID)
    normal = (mx, my)[axis]/rho
    pressure = (gamma-1)*(energy-.5*(mx*mx+my*my)/rho)
    values = (rho*normal, mx*normal+(pressure if axis == 0 else 0),
              my*normal+(pressure if axis == 1 else 0), (energy+pressure)*normal)
    return values, abs(normal)+pm.sqrt(gamma*pressure/rho)


def _hyperbolic_rhs(p, state, cells, lengths, gamma):
    """Two-dimensional conservative Rusanov flux, all four Euler components."""
    def face(b, left, right, axis):
        fl, al = _flux(left, axis, gamma)
        fr, ar = _flux(right, axis, gamma)
        speed = pm.maximum(al, ar)
        return b.product(**{key: .5*(fl[k]+fr[k]-speed*(right[key]-left[key]))
                            for k, key in enumerate(FLUID)})
    fluxes = tuple(p.map(p.domain(state), lambda b, i, axis=axis:
        face(b, b.at(state, i), _get(b, state,
             _neighbor(b.index_value(i), axis, 1, cells)), axis)) for axis in range(2))

    def divergence(b, i):
        j = b.index_value(i)
        faces = tuple((b.at(fluxes[a], i), _get(b, fluxes[a],
                      _neighbor(j, a, -1, cells))) for a in range(2))
        return b.product(**{key: -sum(cells[a]/lengths[a]*(faces[a][0][key]-faces[a][1][key])
                                     for a in range(2)) for key in FLUID})
    return p.map(p.domain(state), divergence)


def _hyperbolic_step(p, state, h, cells, lengths, gamma):
    def admit(b, values):
        admissible = b.reduce(values, b.boolean(True), lambda c, valid, q:
            c.branch(valid, lambda d: d.branch(q["rho"] > 0,
                lambda z: q["energy"]-.5*(q["mx"]**2+q["my"]**2)/q["rho"] > 0,
                lambda z: z.boolean(False)), lambda d: d.boolean(False)), order="ascending")
        b.require_success(b.branch(admissible, lambda c: c.success(h),
                                   lambda c: c.failure(h, 90902)))
        speed = tuple(b.reduce(values, b.float64(0), lambda c, maximum, q, a=a:
            pm.maximum(maximum, _flux(q, a, gamma)[1]), order="ascending") for a in range(2))
        cfl = h*sum(speed[a]*cells[a]/lengths[a] for a in range(2))
        b.require_success(b.branch(cfl <= .5, lambda c: c.success(cfl),
                                   lambda c: c.failure(cfl, 90903)))

    def advance(b, old, rate):
        return b.map(b.domain(old), lambda c, i: c.product(
            **{key: c.at(old, i)[key]+h*c.at(rate, i)[key] for key in FLUID},
            phi=c.at(old, i)["phi"], omega=c.at(old, i)["omega"]))
    admit(p, state)
    first = advance(p, state, _hyperbolic_rhs(p, state, cells, lengths, gamma))
    admit(p, first)
    second = advance(p, first, _hyperbolic_rhs(p, first, cells, lengths, gamma))
    return p.map(p.domain(state), lambda b, i: b.product(
        **{key: .5*(b.at(state, i)[key]+b.at(second, i)[key]) for key in FLUID},
        phi=b.at(state, i)["phi"], omega=b.at(state, i)["omega"]))


def _total_energy(p, state, cells, lengths, alpha):
    phi = _component(p, state, "phi")
    grad = _gradient(p, phi, cells, lengths)
    return (math.prod(lengths)/math.prod(cells))*(
        _sum(p, _component(p, state, "energy"))
        +sum(_dot(p, g, g) for g in grad)/(2*alpha))


def build_m09_midpoint(cells=(4, 3), *, lengths=(2*math.pi, 2*math.pi), dt=.003,
                       steps=2, alpha=1., omega=4., gamma=1.4, solver="schur",
                       source_only=False, tolerance=2.e-12):
    """Complete 2D periodic Euler/source composition with persistent potential.

    ``source_only`` selects the source subproblem for its independent energy and
    phase reception; the default advances the full Euler--Poisson PDE. The
    owned array is replicated in this profile, with explicit cardinality supplied
    by the caller. This does not claim distributed memory scaling of this profile.
    """
    if len(cells) != 2 or any(type(n) is not int or n < 3 for n in cells):
        raise ValueError("cells must contain two integers >= 3")
    if len(lengths) != 2 or any(not math.isfinite(x) or x <= 0 for x in lengths):
        raise ValueError("lengths must contain two finite positive values")
    if (not all(math.isfinite(x) for x in (dt, alpha, omega, gamma, tolerance))
            or dt <= 0 or alpha <= 0 or gamma <= 1 or tolerance <= 0
            or type(steps) is not int or steps < 1):
        raise ValueError("invalid duration, coefficients, tolerance or step count")
    if solver not in ("schur", "monolithic"):
        raise ValueError("solver must be 'schur' or 'monolithic'")
    count = math.prod(cells)
    _, layout = _domain("magnetic_midpoint_native_context", (4, 4))
    p = pops.Program("magnetic_midpoint_"+solver)
    domain = p.range_set(count, capacity=count, ownership="replicated")
    eigen = tuple(4*math.sin(math.pi/n)**2*(n/L)**2 for n, L in zip(cells, lengths, strict=True))

    def initialize(b, i):
        j = b.index_value(i)
        x = 2*math.pi*(b.float64(j % cells[0])+.5)/cells[0]
        y = 2*math.pi*(b.float64(j//cells[0])+.5)/cells[1]
        rho = 1+.08*pm.cos(x)+.04*pm.sin(y)
        vx, vy = .12+.03*pm.sin(x+y), .07-.02*pm.cos(x-y)
        return b.product(rho=rho, mx=rho*vx, my=rho*vy,
            energy=2.5+.5*rho*(vx*vx+vy*vy),
            phi=alpha*(.08*pm.cos(x)/eigen[0]+.04*pm.sin(y)/eigen[1]),
            omega=omega*(1+.2*pm.cos(y)))
    initial_state = p.map(domain, initialize)
    initial = p.product(domain=domain, state=initial_state, time=p.float64(0), step=p.int64(0),
                        source_residual=p.float64(0), source_iterations=p.int64(0),
                        energy_defect=p.float64(0))
    schema = ContinuationSchema(initial, fields=(), time=("time",), step=("step",))
    base = p.continuation_open(initial, schema)
    current = p.continuation_read(base)
    old = current["state"]
    h = p.requested_dt()
    before = old if source_only else _hyperbolic_step(p, old, h/2, cells, lengths, gamma)
    sourced, residual, iterations = _source_midpoint(p, before, h, cells=cells,
        lengths=lengths, alpha=alpha, solver=solver, tolerance=tolerance)
    after = sourced if source_only else _hyperbolic_step(p, sourced, h/2, cells, lengths, gamma)
    defect = _total_energy(p, after, cells, lengths, alpha)-_total_energy(p, old, cells, lengths, alpha)
    continued = p.product(domain=current["domain"], state=after,
        time=current["time"]+h, step=current["step"]+p.int64(1),
        source_residual=residual, source_iterations=iterations, energy_defect=defect)
    proposal = p.product(h=h)
    envelope = p.product(continuation=continued, increment=p.product(), observations=p.product(), checks=p.product())
    candidate = p.continuation_freeze(base, proposal, envelope)

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product(energy_defect=Float64)))
    def assess(b, data):
        result = data["candidate"]["continuation"]
        e = result["energy_defect"]
        valid = b.reduce(result["state"], b.boolean(True), lambda c, valid, q:
            c.branch(valid, lambda d: d.branch(q["rho"] > 0,
                lambda z: q["energy"]-.5*(q["mx"]**2+q["my"]**2)/q["rho"] > 0,
                lambda z: z.boolean(False)), lambda d: d.boolean(False)), order="ascending")
        accepted = b.branch(valid, lambda c: e*e < 1.e-18, lambda c: c.boolean(False))
        return b.product(accepted=accepted, checks=b.product(energy_defect=e))
    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    p.step_strategy(FixedDt(dt))
    case = pops.Case("M09_periodic_midpoint_"+solver)
    case.program(p)
    return CaseSpec(case, layout, {}, {}, steps*dt, dt,
        lambda runtime: dict(passed=runtime.macro_step() == steps, frontier_only=True),
        "M09: full periodic 2D Euler Rusanov/SSPRK2 halves and magnetic coupled midpoint "
        +solver+" source in native finite-array Continuation; first-order adjoint spatial "
        "coupling, original two-equation residual and total-energy acceptance; "
        "replicated ownership, Gauss splitting defect measured separately")
