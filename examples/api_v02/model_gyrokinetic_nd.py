"""Full-distribution electromagnetic gyrokinetics on distributed 3x2v arrays.

Equations: Mandell et al. arXiv:1908.05653v2 (5),(6),(8),(11),(13)--(18).
This finite-volume realization retains perpendicular transport, magnetic drift,
mirror force, magnetic flutter and the same-discrete-velocity Ampere derivative.
The default nonlinear polarization uses n_s[f], together with
H_2=-m_s |grad_perp(phi)|^2/(2 B^2), as in Hammett--Shi--Hakim, Oxford 2016,
slide 23 (https://w3.pppl.gov/~hammett/refs/2016/Hammett-Shi-Hakim-2016_Oxford_v4.pdf).
The linear-polarization variant remains available and omits H_2 consistently.
The optional collision operator uses the declared finite velocity box and an
explicit moment projection by default. This projects out quadrature defects in
particle number, parallel momentum and kinetic energy; the raw face flux is
available with collision_projection=False and has measurable energy defects.

All spatial operators are actual multidimensional stencils. There is no fixed
k_perp symbol, tabulated field inverse, callback, or scientific C++ terminal.
The periodic magnetic geometry is divergence free, has b.curl(b)=0, nonzero
curvature and mirror terms. This states a geometry/ordering; it does not claim
to represent a diverted tokamak or to inherit DG energy proofs.
"""

from __future__ import annotations

from itertools import product
from math import pi

import pops
from pops import math as pm
from pops.algebra import Distributed
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .phase_space_transport import DistributedTensor, TensorGrid, fibre_sum, lift


def _gradient(space, value, *, forward=False):
    if not forward:
        return tuple(space.gradient(value, a) for a in range(3))
    return tuple(
        space.zip(
            (value, space.shift(value, a, 1)),
            lambda b, gid, v, a=a: (v[1] - v[0]) / space.grid.widths[a],
        )
        for a in range(3)
    )


def _curl(space, vector):
    derivatives = tuple(_gradient(space, component) for component in vector)
    return tuple(
        space.axpy(derivatives[(a + 2) % 3][(a + 1) % 3], -1, derivatives[(a + 1) % 3][(a + 2) % 3])
        for a in range(3)
    )


def _perpendicular(space, vector, direction):
    parallel = space.zip(
        (*vector, *direction),
        lambda b, gid, values: sum(values[a] * values[a + 3] for a in range(3)),
    )
    return tuple(
        space.zip((vector[a], direction[a], parallel), lambda b, gid, v: v[0] - v[1] * v[2])
        for a in range(3)
    )


def _operator(space, value, coefficient, direction):
    """G^T coefficient*(I-b b^T)*G, with explicit periodic incidence maps."""
    gradient = _perpendicular(space, _gradient(space, value, forward=True), direction)
    flux = tuple(
        space.zip((component, coefficient), lambda b, gid, v: v[0] * v[1]) for component in gradient
    )
    result = space.constant(0)
    for a, component in enumerate(flux):
        previous = space.shift(component, a, -1)
        divergence = space.zip(
            (component, previous), lambda b, gid, v, a=a: (v[1] - v[0]) / space.grid.widths[a]
        )
        result = space.axpy(result, 1, divergence)
    return result


def _cg(space, rhs, action, *, gauge, tolerance=1e-11):
    p = space.program
    mean = space.sum(rhs) / space.grid.size if gauge else p.float64(0)
    residual = space.zip((rhs,), lambda b, gid, value: value[0] - mean)
    initial = p.product(x=space.constant(0), r=residual, d=residual)

    def step(b, state):
        local = space.rebind(b)
        image = action(local, state["d"])
        rr = local.dot(state["r"], state["r"])
        alpha = rr / local.dot(state["d"], image)
        r = local.axpy(state["r"], -alpha, image)
        beta = local.dot(r, r) / rr
        return b.product(
            x=local.axpy(state["x"], alpha, state["d"]), r=r, d=local.axpy(r, beta, state["d"])
        )

    solved = p.require_success(
        p.iterate(
            initial,
            step,
            lambda b, state: space.rebind(b).dot(state["r"], state["r"]) <= tolerance**2,
            max_steps=4 * space.grid.size,
        )
    )["state"]["x"]
    original = space.axpy(rhs, -1, action(space, solved))
    return solved, space.dot(original, original), mean


def _moment(phase, space, velocity, distribution, mass, power):
    weighted = phase.zip(
        (distribution,),
        lambda b, gid, values: (
            2 * pi / mass * velocity.coordinate(gid % velocity.size, 0) ** power * values[0]
        ),
    )
    return fibre_sum(phase, space, weighted)


def _geometry(space, amplitude):
    vector = (
        space.map(lambda b, gid, i: 1 + amplitude * pm.sin(2 * pi * space.grid.coordinate(gid, 1))),
        space.map(
            lambda b, gid, i: 0.37 + amplitude * pm.sin(2 * pi * space.grid.coordinate(gid, 0))
        ),
        space.constant(0),
    )
    magnitude = space.zip(vector, lambda b, gid, v: pm.sqrt(sum(x * x for x in v)))
    direction = tuple(
        space.zip((component, magnitude), lambda b, gid, v: v[0] / v[1]) for component in vector
    )
    return magnitude, direction, _curl(space, direction)


def _initial(phase, space, velocity, magnitude, species, mass):
    magnetic = lift(space, phase, magnitude)
    temperature = 1.0 + 0.1 * species
    raw = phase.zip(
        (magnetic,),
        lambda b, gid, values: pm.exp(
            -mass * velocity.coordinate(gid % velocity.size, 0) ** 2 / (2 * temperature)
            - velocity.coordinate(gid % velocity.size, 1) * values[0] / temperature
        ),
    )
    normalization = lift(space, phase, _moment(phase, space, velocity, raw, mass, 0))
    return phase.zip(
        (raw, normalization),
        lambda b, gid, values: (
            (
                1.0
                + (-1) ** species
                * 0.01
                * sum(pm.cos(2 * pi * phase.grid.coordinate(gid, a) + 0.2 * a) for a in range(3))
                / 3
            )
            * values[0]
            / values[1]
        ),
    )


def _flux_divergence(phase, distribution, speed, axis, *, centered=False):
    right, right_speed = phase.shift(distribution, axis, 1), phase.shift(speed, axis, 1)

    def face(b, gid, values):
        velocity = (values[2] + values[3]) / 2
        flux = velocity * (values[0] + values[1]) / 2
        if not centered:
            flux = flux - abs(velocity) * (values[1] - values[0]) / 2
        if phase.grid.boundaries[axis] == "zero_flux":
            return b.branch(
                phase.grid.ordinal(gid, axis) == phase.grid.shape[axis] - 1,
                lambda c: c.float64(0),
                lambda c: flux,
            )
        return flux

    faces = phase.zip((distribution, right, speed, right_speed), face)
    previous = phase.shift(faces, axis, -1)
    return phase.zip((faces, previous), lambda b, gid, v: (v[0] - v[1]) / phase.grid.widths[axis])


def _collision_projection(phase, space, velocity, rate, magnitude, mass):
    """Remove the finite-quadrature defects in (1,v,v²+2μB/m).

    This is an explicit chosen discretization, separate from the continuum
    operator. The symmetric v grid diagonalizes its odd moment. Geometry and
    moment defects remain runtime arrays; only velocity quadrature constants
    are elaborated in Python. No positivity theorem follows from projection.
    """
    points = tuple(
        (
            velocity.bounds[0][0] + (i + 0.5) * velocity.widths[0],
            velocity.bounds[1][0] + (j + 0.5) * velocity.widths[1],
        )
        for i, j in product(*(range(n) for n in velocity.shape))
    )
    w = velocity.volume
    one = w * len(points)
    vv = w * sum(v**2 for v, mu in points)
    vvvv = w * sum(v**4 for v, mu in points)
    mu1 = w * sum(mu for v, mu in points)
    mu2 = w * sum(mu**2 for v, mu in points)
    vv_mu = w * sum(v**2 * mu for v, mu in points)
    magnetic = lift(space, phase, magnitude)

    def invariant(b, gid, values, component):
        v = velocity.coordinate(gid % velocity.size, 0)
        mu = velocity.coordinate(gid % velocity.size, 1)
        basis = (1, v, v**2 + 2 * mu * values[1] / mass)[component]
        return basis * values[0]

    defects = tuple(
        fibre_sum(
            phase,
            space,
            phase.zip((rate, magnetic), lambda b, gid, values, a=a: invariant(b, gid, values, a)),
        )
        for a in range(3)
    )

    def coefficient(b, gid, values, component):
        n, momentum, energy, B = values
        first = vv + 2 * B / mass * mu1
        second = vvvv + 4 * B / mass * vv_mu + 4 * B**2 / mass**2 * mu2
        determinant = one * second - first**2
        if component == 0:
            return (second * n - first * energy) / determinant
        if component == 1:
            return momentum / vv
        return (one * energy - first * n) / determinant

    multipliers = tuple(
        lift(
            space,
            phase,
            space.zip((*defects, magnitude), lambda b, gid, v, a=a: coefficient(b, gid, v, a)),
        )
        for a in range(3)
    )
    return phase.zip(
        (rate, magnetic, *multipliers),
        lambda b, gid, values: (
            values[0]
            - values[2]
            - values[3] * velocity.coordinate(gid % velocity.size, 0)
            - values[4]
            * (
                velocity.coordinate(gid % velocity.size, 0) ** 2
                + 2 * velocity.coordinate(gid % velocity.size, 1) * values[1] / mass
            )
        ),
    )


def _collision(phase, space, velocity, distribution, magnitude, mass, frequency, project):
    """Declared finite-box realization of the like-species operator (17)."""
    number = _moment(phase, space, velocity, distribution, mass, 0)
    momentum = _moment(phase, space, velocity, distribution, mass, 1)
    magnetic = lift(space, phase, magnitude)
    energy_density = phase.zip(
        (distribution, magnetic),
        lambda b, gid, v: (
            2
            * pi
            / mass
            * (
                velocity.coordinate(gid % velocity.size, 0) ** 2
                + 2 * velocity.coordinate(gid % velocity.size, 1) * v[1] / mass
            )
            * v[0]
        ),
    )
    second = fibre_sum(phase, space, energy_density)
    mean = space.zip((number, momentum), lambda b, gid, v: v[1] / v[0])
    thermal = space.zip((number, second, mean), lambda b, gid, v: (v[1] / v[0] - v[2] ** 2) / 3)
    u, theta = lift(space, phase, mean), lift(space, phase, thermal)
    result = phase.constant(0)
    for a in (0, 1):
        right = phase.shift(distribution, 3 + a, 1)

        def face(b, gid, values, a=a):
            coordinate = velocity.coordinate(gid % velocity.size, a) + velocity.widths[a] / 2
            drift = coordinate - values[2] if a == 0 else 2 * coordinate
            diffusivity = values[3] if a == 0 else 2 * coordinate * mass / values[4] * values[3]
            flux = frequency * (
                drift * (values[0] + values[1]) / 2
                + diffusivity * (values[1] - values[0]) / velocity.widths[a]
            )
            return b.branch(
                velocity.ordinal(gid % velocity.size, a) == velocity.shape[a] - 1,
                lambda c: c.float64(0),
                lambda c: flux,
            )

        faces = phase.zip((distribution, right, u, theta, magnetic), face)
        previous = phase.shift(faces, 3 + a, -1)
        result = phase.axpy(
            result,
            1,
            phase.zip((faces, previous), lambda b, gid, v, a=a: (v[0] - v[1]) / velocity.widths[a]),
        )
    return (
        _collision_projection(phase, space, velocity, result, magnitude, mass)
        if project
        else result
    )


def _fields(
    phase, space, velocity, distributions, magnitude, direction, masses, charges, nonlinear, mu0
):
    numbers = tuple(
        _moment(phase, space, velocity, value, mass, 0)
        for value, mass in zip(distributions, masses, strict=True)
    )
    charge, current, coefficient = space.constant(0), space.constant(0), space.constant(0)
    for value, number, mass, q in zip(distributions, numbers, masses, charges, strict=True):
        charge = space.axpy(charge, q, number)
        current = space.axpy(current, q, _moment(phase, space, velocity, value, mass, 1))
        susceptibility = space.zip(
            (number, magnitude),
            lambda b, gid, v, mass=mass: mass * (v[0] if nonlinear else 1.0) / v[1] ** 2,
        )
        coefficient = space.axpy(coefficient, 1, susceptibility)
    one = space.constant(1)
    phi, phi_residual, neutral = _cg(
        space, charge, lambda local, x: _operator(local, x, coefficient, direction), gauge=True
    )
    apar, apar_residual, net_current = _cg(
        space,
        space.zip((current,), lambda b, gid, v: mu0 * v[0]),
        lambda local, x: _operator(local, x, one, direction),
        gauge=True,
    )
    return phi, apar, (phi_residual, apar_residual, neutral, net_current)


def _rhs(
    phase,
    space,
    velocity,
    distributions,
    magnitude,
    direction,
    curvature,
    phi,
    apar,
    masses,
    charges,
    nonlinear,
    mu0,
    collision_frequency,
    collision_projection,
):
    grad_phi, grad_B = _gradient(space, phi), _gradient(space, magnitude)
    # curl(A b), including A curl(b); it is perpendicular in this planar geometry.
    vector_potential = tuple(
        space.zip((apar, component), lambda b, gid, v: v[0] * v[1]) for component in direction
    )
    flutter = _curl(space, vector_potential)
    perpendicular = _perpendicular(space, _gradient(space, phi, forward=True), direction)
    flow_energy = space.zip(
        (*perpendicular, magnitude),
        lambda b, gid, v: sum(value * value for value in v[:3]) / v[3] ** 2,
    )
    grad_energy = _gradient(space, flow_energy)
    static = tuple(
        lift(space, phase, value)
        for value in (magnitude, *direction, *curvature, *flutter, *grad_B, *grad_phi, *grad_energy)
    )
    star, velocity_derivatives = [], []
    for distribution, mass, charge in zip(distributions, masses, charges, strict=True):

        def characteristics(b, gid, values, component, mass=mass, charge=charge):
            B = values[0]
            unit = values[1:4]
            curl_b = values[4:7]
            perturbed = values[7:10]
            gB = values[10:13]
            gphi = values[13:16]
            guE = values[16:19]
            v = velocity.coordinate(gid % velocity.size, 0)
            mu = velocity.coordinate(gid % velocity.size, 1)
            effective = tuple(
                B * unit[a] + mass * v / charge * curl_b[a] + perturbed[a] for a in range(3)
            )
            force = tuple(
                mu * gB[a] + charge * gphi[a] - (0.5 * mass * guE[a] if nonlinear else 0.0)
                for a in range(3)
            )
            if component == 3:
                return -sum(effective[a] * force[a] for a in range(3)) / (mass * B)
            cross = (
                unit[(component + 1) % 3] * force[(component + 2) % 3]
                - unit[(component + 2) % 3] * force[(component + 1) % 3]
            )
            return effective[component] * v / B + cross / (charge * B)

        rates = phase.constant(0)
        for axis in range(4):
            speed = phase.zip(
                static, lambda b, gid, values, axis=axis: characteristics(b, gid, values, axis)
            )
            rates = phase.axpy(rates, -1, _flux_divergence(phase, distribution, speed, axis))
        if collision_frequency:
            rates = phase.axpy(
                rates,
                1,
                _collision(
                    phase,
                    space,
                    velocity,
                    distribution,
                    magnitude,
                    mass,
                    collision_frequency,
                    collision_projection,
                ),
            )
        star.append(rates)
        velocity_derivatives.append(
            _flux_divergence(phase, distribution, phase.constant(1), 3, centered=True)
        )
    induction_rhs, mass_coefficient = space.constant(0), space.constant(0)
    for s, (mass, charge) in enumerate(zip(masses, charges, strict=True)):
        induction_rhs = space.axpy(
            induction_rhs, mu0 * charge, _moment(phase, space, velocity, star[s], mass, 1)
        )
        mass_coefficient = space.axpy(
            mass_coefficient,
            -mu0 * charge**2 / mass,
            _moment(phase, space, velocity, velocity_derivatives[s], mass, 1),
        )
    one = space.constant(1)

    def induction_operator(local, value):
        result = _operator(local, value, one, direction)
        return local.axpy(
            result, 1, local.zip((mass_coefficient, value), lambda b, gid, v: v[0] * v[1])
        )

    adot, residual, _ = _cg(space, induction_rhs, induction_operator, gauge=False)
    lifted = lift(space, phase, adot)
    rhs = tuple(
        phase.axpy(rate, charge / mass, phase.zip((dv, lifted), lambda b, gid, v: v[0] * v[1]))
        for rate, dv, charge, mass in zip(star, velocity_derivatives, charges, masses, strict=True)
    )
    # Check the original differentiated Ampere equation with the actual full RHS.
    original = _operator(space, adot, one, direction)
    for value, mass, charge in zip(rhs, masses, charges, strict=True):
        original = space.axpy(
            original, -mu0 * charge, _moment(phase, space, velocity, value, mass, 1)
        )
    return rhs, adot, space.dot(original, original), residual


def build_gyrokinetic_nd(
    *,
    configuration_shape=(3, 3, 3),
    velocity_shape=(4, 3),
    ranks=1,
    nonlinear_polarization=True,
    geometry_amplitude=0.1,
    mu0=0.2,
    collision_frequency=0.0,
    collision_projection=True,
    dt=1e-5,
    steps=2,
):
    if len(configuration_shape) != 3 or len(velocity_shape) != 2:
        raise ValueError("this gyrokinetic problem uses R=(x,y,z) and (v_parallel,mu)")
    if not 0 <= geometry_amplitude < 0.25 or mu0 <= 0 or dt <= 0 or collision_frequency < 0:
        raise ValueError(
            "positive metric, permeability, dt and nonnegative collision rate required"
        )
    masses, charges = (1.0, 2.0), (-1.0, 1.0)
    xgrid = TensorGrid(tuple(configuration_shape), ((0.0, 1.0),) * 3, ("periodic",) * 3)
    velocity = TensorGrid(tuple(velocity_shape), ((-4.0, 4.0), (0.0, 2.0)), ("zero_flux",) * 2)
    from .models_M01_M14 import _domain

    _, layout = _domain("gyrokinetic_5d_execution_context", (4, 4))
    p = pops.Program("full_f_electromagnetic_equations")
    space = DistributedTensor.create(p, xgrid, ranks)
    phase = DistributedTensor.create(p, xgrid.product(velocity), ranks)
    magnitude, direction, curvature = _geometry(space, geometry_amplitude)
    fields = {
        "g" + str(s): _initial(phase, space, velocity, magnitude, s, mass)
        for s, mass in enumerate(masses)
    }
    initial = p.product(
        configuration=space.domain,
        phase=phase.domain,
        fields=p.product(**fields),
        time=p.float64(0),
        step=p.int64(0),
    )
    schema = ContinuationSchema(
        initial, fields=(), time=("time",), step=("step",), byte_capacity=64 << 20
    )
    base = p.continuation_open(initial, schema)
    before = p.continuation_read(base)
    space = space.rebind(p, before["configuration"])
    phase = phase.rebind(p, before["phase"])
    magnitude, direction, curvature = _geometry(space, geometry_amplitude)
    distributions = tuple(before["fields"]["g" + str(s)] for s in range(2))
    phi, apar, field_checks = _fields(
        phase,
        space,
        velocity,
        distributions,
        magnitude,
        direction,
        masses,
        charges,
        nonlinear_polarization,
        mu0,
    )
    rhs, adot, derivative_residual, solver_residual = _rhs(
        phase,
        space,
        velocity,
        distributions,
        magnitude,
        direction,
        curvature,
        phi,
        apar,
        masses,
        charges,
        nonlinear_polarization,
        mu0,
        collision_frequency,
        collision_projection,
    )
    h = p.requested_dt()
    updated = tuple(phase.axpy(old, h, rate) for old, rate in zip(distributions, rhs, strict=True))
    after = p.product(
        configuration=before["configuration"],
        phase=before["phase"],
        fields=p.product(**{"g" + str(s): value for s, value in enumerate(updated)}),
        time=before["time"] + h,
        step=before["step"] + 1,
    )
    checks = p.product(
        field_residual=field_checks[0] + field_checks[1],
        compatibility=field_checks[2] ** 2 + field_checks[3] ** 2,
        derivative_residual=derivative_residual + solver_residual,
    )
    envelope = p.product(
        continuation=after, increment=p.product(), observations=checks, checks=checks
    )
    proposal = p.product(h=h)
    candidate = p.continuation_freeze(base, proposal, envelope)

    @graph(
        inputs=Product(
            base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)
        ),
        output=Product(accepted=Bool, checks=Product(number_defect=Float64, minimum=Float64)),
    )
    def assess(b, args):
        old, new = args["base"], args["candidate"]["continuation"]
        local = phase.rebind(b, new["phase"])
        defect, minimum = b.float64(0), b.float64(1e300)
        for s, mass in enumerate(masses):
            value = new["fields"]["g" + str(s)]
            change = local.axpy(value, -1, old["fields"]["g" + str(s)])
            delta = local.sum(change) * local.grid.volume * 2 * pi / mass
            defect = pm.maximum(defect, abs(delta))
            minimum = pm.minimum(
                minimum,
                Distributed(b).reduce(
                    value,
                    b.float64(1e300),
                    lambda c, total, item: pm.minimum(total, item),
                    order="global_gid_ascending",
                    byte_capacity=64 << 20,
                ),
            )
        valid = b.boolean(True)
        for condition in (
            defect < 1e-10,
            minimum > 0,
            args["candidate"]["checks"]["field_residual"] < 1e-19,
            args["candidate"]["checks"]["compatibility"] < 1e-20,
            args["candidate"]["checks"]["derivative_residual"] < 1e-19,
        ):
            valid = b.branch(condition, lambda c, valid=valid: valid, lambda c: c.boolean(False))
        return b.product(accepted=valid, checks=b.product(number_defect=defect, minimum=minimum))

    p.continuation_publish(
        p.continuation_accept(candidate, p.continuation_attest(candidate, assess))
    )
    p.step_strategy(FixedDt(dt))
    case = pops.Case("M21_full_f_distributed_5d")
    case.program(p)
    return CaseSpec(
        case,
        layout,
        {},
        {},
        dt * steps,
        dt,
        lambda runtime: {"passed": runtime.macro_step() == steps, "frontier_only": True},
        "M21 complete selected periodic 3x2v EM gyrokinetic equations, variable metric, curvature/mirror/ExB/flutter, "
        "nonlinear polarization with Hamiltonian correction and discrete compatible dA/dt; FV, not DG",
    )
