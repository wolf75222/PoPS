"""Dimension-independent Vlasov--Poisson--BGK and multispecies Fokker--Planck.

The selected finite-volume realization has periodic configuration space and a
finite zero-flux velocity box. Full distributions live on distributed x*v index
sets. Species number, every momentum direction and total kinetic energy are
explicit velocity integrals. All field/transport/collision equations and the CG
algorithm are Python algebra compiled by the ordinary native Program route.

M20 uses the symmetric mixture subclass of Habbershaw et al., arXiv:2404.11775,
eqs. (1), (6)--(9): the pair velocity and temperature enforce continuous pair
momentum/energy conservation. A declared polynomial correction removes discrete
truncation defects. Its finite-grid positivity is checked, not inferred from the
continuous entropy theorem. M19 uses the same correction for its BGK equilibrium.
"""

from __future__ import annotations

from math import pi

import numpy as np
import pops
from pops import math as pm
from pops.algebra import Distributed
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .phase_space_transport import DistributedTensor, TensorGrid, fibre_sum, lift, poisson_cg


def grids(configuration_shape, velocity_shape, velocity_extent=4.0):
    configuration_shape, velocity_shape = tuple(configuration_shape), tuple(velocity_shape)
    if len(configuration_shape) > len(velocity_shape):
        raise ValueError("each configuration direction needs a velocity direction")
    if any(n < 3 for n in velocity_shape):
        raise ValueError("energy and momentum require at least three velocity cells per axis")
    x = TensorGrid(
        configuration_shape,
        ((0.0, 1.0),) * len(configuration_shape),
        ("periodic",) * len(configuration_shape),
    )
    v = TensorGrid(
        velocity_shape,
        ((-velocity_extent, velocity_extent),) * len(velocity_shape),
        ("zero_flux",) * len(velocity_shape),
    )
    return x, v


def _coordinates(grid):
    return np.array(
        [
            [grid.coordinate(gid, axis) for axis in range(len(grid.shape))]
            for gid in range(grid.size)
        ]
    )


def _basis(velocity):
    nodes = _coordinates(velocity)
    return np.vstack((np.ones(velocity.size), nodes.T, np.sum(nodes**2, axis=1)))


def _monomial(velocity, gid, component):
    if component == 0:
        return 1.0
    if component <= len(velocity.shape):
        return velocity.coordinate(gid, component - 1)
    return sum(velocity.coordinate(gid, a) ** 2 for a in range(len(velocity.shape)))


def moments(phase, configuration, velocity, distribution):
    return tuple(
        fibre_sum(
            phase,
            configuration,
            phase.zip(
                (distribution,),
                lambda b, gid, values, k=k: _monomial(velocity, gid % velocity.size, k) * values[0],
            ),
        )
        for k in range(len(velocity.shape) + 2)
    )


def _initial(phase, velocity, species, mass):
    # Normalize the sampled equilibrium with its declared finite quadrature.
    nodes = _coordinates(velocity)
    mean = np.array([0.12 * (species + 1) * (-1) ** axis for axis in range(len(velocity.shape))])
    temperature = 1.0 + 0.1 * species
    normalizer = float(
        velocity.volume
        * np.sum(np.exp(-mass * np.sum((nodes - mean) ** 2, axis=1) / (2 * temperature)))
    )
    xdim = len(phase.grid.shape) - len(velocity.shape)

    def cell(b, gid, i):
        vgid = gid % velocity.size
        exponent = sum(
            (velocity.coordinate(vgid, axis) - float(mean[axis])) ** 2
            for axis in range(len(velocity.shape))
        )
        modulation = 1.0 + sum(
            0.025 / xdim * pm.cos(2 * pi * phase.grid.coordinate(gid, a) + 0.3 * species + 0.2 * a)
            for a in range(xdim)
        )
        return modulation * pm.exp(-mass * exponent / (2 * temperature)) / normalizer

    return phase.map(cell)


def _transport(phase, velocity, distribution, electric=(), charge_mass=0.0):
    xdim = len(phase.grid.shape) - len(velocity.shape)
    total = phase.constant(0)
    for axis in range(xdim):

        def flux(b, gid, left, right, axis=axis):
            speed = velocity.coordinate(gid % velocity.size, axis)
            return (speed * (left + right) - abs(speed) * (right - left)) / 2

        total = phase.axpy(total, -1, phase.divergence(distribution, axis, flux))
    for axis, force in enumerate(electric):
        # A field is explicitly lifted from x to every velocity degree of freedom.
        right = phase.shift(distribution, xdim + axis, 1)

        def face(b, gid, values, axis=axis):
            speed = charge_mass * values[2]
            expression = (
                speed * (values[0] + values[1]) - abs(speed) * (values[1] - values[0])
            ) / 2
            return b.branch(
                velocity.ordinal(gid % velocity.size, axis) == velocity.shape[axis] - 1,
                lambda c: c.float64(0),
                lambda c: expression,
            )

        faces = phase.zip((distribution, right, force), face)
        previous = phase.shift(faces, xdim + axis, -1)
        divergence = phase.zip(
            (faces, previous),
            lambda b, gid, values, axis=axis: (values[0] - values[1]) / velocity.widths[axis],
        )
        total = phase.axpy(total, -1, divergence)
    return total


def _project_to_moments(phase, configuration, velocity, sampled, target):
    basis = _basis(velocity)
    # This is a static quadrature Gram matrix, not a tabulated physical solve.
    inverse = np.linalg.inv(velocity.volume * basis @ basis.T)
    actual = moments(phase, configuration, velocity, sampled)
    defects = tuple(
        configuration.axpy(wanted, -1, measured)
        for wanted, measured in zip(target, actual, strict=True)
    )
    lifted = tuple(lift(configuration, phase, defect) for defect in defects)
    return phase.zip(
        (sampled, *lifted),
        lambda b, gid, values: (
            values[0]
            + sum(
                _monomial(velocity, gid % velocity.size, k)
                * sum(float(inverse[k, l]) * values[l + 1] for l in range(len(defects)))
                for k in range(len(defects))
            )
        ),
    )


def _bgk(phase, configuration, velocity, distribution, moment_values, mass, frequency):
    lifted = tuple(lift(configuration, phase, value) for value in moment_values)
    dimension = len(velocity.shape)

    def sample(b, gid, values):
        number = values[0]
        mean = tuple(values[a + 1] / number for a in range(dimension))
        theta = (values[-1] / number - sum(u * u for u in mean)) / dimension
        speed2 = sum(
            (velocity.coordinate(gid % velocity.size, a) - mean[a]) ** 2 for a in range(dimension)
        )
        return number * pm.exp(-speed2 / (2 * theta)) / ((2 * pi * theta) ** (dimension / 2))

    sampled = phase.zip(lifted, sample)
    equilibrium = _project_to_moments(phase, configuration, velocity, sampled, moment_values)
    return phase.zip(
        (equilibrium, distribution), lambda b, gid, value: frequency * (value[0] - value[1])
    )


def _fp_collision(
    phase, configuration, velocity, distributions, moments_by_species, masses, frequency
):
    """All self and cross pairs, with a declared global invariant projection."""
    dimension, species_count = len(velocity.shape), len(masses)
    lifted = tuple(
        tuple(lift(configuration, phase, value) for value in values)
        for values in moments_by_species
    )
    p = phase.program
    species_domain = p.range_set(species_count, capacity=species_count, ownership="replicated")
    rows_by_species = tuple(
        p.product(mass=p.float64(mass), **{"m" + str(k): value for k, value in enumerate(values)})
        for mass, values in zip(masses, lifted, strict=True)
    )

    def select_species(b, index, k=0):
        if k == species_count - 1:
            return rows_by_species[k]
        return b.branch(
            index == k, lambda c: rows_by_species[k], lambda c: select_species(c, index, k + 1)
        )

    species_values = p.map(species_domain, lambda b, i: select_species(b, b.index_value(i)))
    result = []
    for s, distribution in enumerate(distributions):
        # The count is replicated and identical on every rank. Each pair has a
        # native lexical scope, so its temporary arrays are released before the
        # next pair. Unrolling these scopes kept all immutable SSA arrays alive.
        def pair_step(b, total, t, s=s, distribution=distribution):
            local = phase.rebind(b)
            other = b.at(species_values, b.index(species_domain, t))
            pair = lifted[s] + tuple(other["m" + str(k)] for k in range(dimension + 2))
            other_mass = other["mass"]

            def parameters(c, gid, values):
                first, second = values[: dimension + 2], values[dimension + 2 :]
                ns, nt = first[0], second[0]
                us = tuple(first[a + 1] / ns for a in range(dimension))
                ut = tuple(second[a + 1] / nt for a in range(dimension))
                ts = masses[s] * (first[-1] / ns - sum(u * u for u in us)) / dimension
                tt = other_mass * (second[-1] / nt - sum(u * u for u in ut)) / dimension
                rs, rt = masses[s] * ns, other_mass * nt
                mean = tuple((rs * us[a] + rt * ut[a]) / (rs + rt) for a in range(dimension))
                temperature = (
                    ns * ts
                    + nt * tt
                    + rs
                    * rt
                    / (rs + rt)
                    * sum((us[a] - ut[a]) ** 2 for a in range(dimension))
                    / dimension
                ) / (ns + nt)
                return mean, temperature

            for axis in range(dimension):
                right = local.shift(distribution, len(configuration.grid.shape) + axis, 1)

                def face(b, gid, values, axis=axis, s=s, parameters=parameters):
                    u, temperature = parameters(b, gid, values[2:])
                    location = (
                        velocity.coordinate(gid % velocity.size, axis) + velocity.widths[axis] / 2
                    )
                    expression = frequency * (
                        (location - u[axis]) * (values[0] + values[1]) / 2
                        + temperature / masses[s] * (values[1] - values[0]) / velocity.widths[axis]
                    )
                    return b.branch(
                        velocity.ordinal(gid % velocity.size, axis) == velocity.shape[axis] - 1,
                        lambda c: c.float64(0),
                        lambda c: expression,
                    )

                faces = local.zip((distribution, right, *pair), face)
                previous = local.shift(faces, len(configuration.grid.shape) + axis, -1)
                divergence = local.zip(
                    (faces, previous),
                    lambda b, gid, value, axis=axis: (value[0] - value[1]) / velocity.widths[axis],
                )
                total = local.axpy(total, 1, divergence)
            return total

        result.append(p.fold(phase.constant(0), species_count, pair_step))
    # Projection acts jointly over species, retaining each species number and
    # only the physically shared total momentum and energy constraints.
    local_basis = _basis(velocity)
    rows = species_count + dimension + 1
    basis = np.zeros((rows, species_count * velocity.size))
    for s, mass in enumerate(masses):
        segment = slice(s * velocity.size, (s + 1) * velocity.size)
        basis[s, segment] = 1
        basis[species_count : species_count + dimension, segment] = mass * local_basis[1:-1]
        basis[-1, segment] = 0.5 * mass * local_basis[-1]
    inverse = np.linalg.inv(velocity.volume * basis @ basis.T)
    raw_moments = tuple(moments(phase, configuration, velocity, value) for value in result)
    defects = [row[0] for row in raw_moments]
    for k in range(1, dimension + 2):
        combined = configuration.constant(0)
        for s, mass in enumerate(masses):
            combined = configuration.axpy(
                combined, mass * (0.5 if k == dimension + 1 else 1), raw_moments[s][k]
            )
        defects.append(combined)
    lifted_defects = tuple(lift(configuration, phase, value) for value in defects)
    corrected = []
    for s, mass in enumerate(masses):

        def correction(b, gid, values, s=s, mass=mass):
            coordinates = tuple(
                velocity.coordinate(gid % velocity.size, a) for a in range(dimension)
            )
            vector = (
                tuple(float(k == s) for k in range(species_count))
                + tuple(mass * v for v in coordinates)
                + (0.5 * mass * sum(v * v for v in coordinates),)
            )
            return values[0] - sum(
                vector[k] * sum(float(inverse[k, l]) * values[l + 1] for l in range(rows))
                for k in range(rows)
            )

        corrected.append(phase.zip((result[s], *lifted_defects), correction))
    return tuple(corrected)


def build_kinetic_nd(
    *,
    model="M19",
    configuration_shape=(4, 4, 4),
    velocity_shape=(4, 4, 4),
    ranks=1,
    masses=(1.0,),
    charges=(-1.0,),
    frequency=0.2,
    dt=1e-5,
    steps=2,
    byte_capacity=64 << 20,
    storage_bytes=64 << 20,
    storage_versions=65536,
):
    if model not in ("M19", "M20"):
        raise ValueError("this module defines the two explicitly stated kinetic models")
    masses, charges = tuple(masses), tuple(charges)
    if not masses or any(m <= 0 for m in masses) or len(charges) != len(masses):
        raise ValueError("each species needs a positive mass and a charge")
    if frequency < 0 or dt <= 0 or type(steps) is not int or steps < 1:
        raise ValueError("nonnegative frequency, positive dt and steps are required")
    xgrid, velocity = grids(configuration_shape, velocity_shape)
    grid = xgrid.product(velocity)
    from .models_M01_M14 import _domain

    _, layout = _domain("kinetic_nd_execution_context", (4, 4))
    p = pops.Program("distributed_phase_space_equations")
    Distributed(p, storage_bytes=storage_bytes, storage_versions=storage_versions)
    xspace = DistributedTensor.create(p, xgrid, ranks, byte_capacity=byte_capacity)
    phase = DistributedTensor.create(p, grid, ranks, byte_capacity=byte_capacity)
    fields = {"f" + str(s): _initial(phase, velocity, s, mass) for s, mass in enumerate(masses)}
    initial = p.product(
        configuration=xspace.domain,
        phase=phase.domain,
        fields=p.product(**fields),
        time=p.float64(0),
        step=p.int64(0),
    )
    schema = ContinuationSchema(
        initial, fields=(), time=("time",), step=("step",), byte_capacity=byte_capacity
    )
    base = p.continuation_open(initial, schema)
    before = p.continuation_read(base)
    xspace = xspace.rebind(p, before["configuration"])
    phase = phase.rebind(p, before["phase"])
    distributions = tuple(before["fields"]["f" + str(s)] for s in range(len(masses)))
    all_moments = tuple(moments(phase, xspace, velocity, value) for value in distributions)
    electric, poisson_residual, neutrality = (), p.float64(0), p.float64(0)
    if model == "M19":
        rho = xspace.constant(-sum(charges))  # prescribed homogeneous neutralizing background
        for charge, value in zip(charges, all_moments, strict=True):
            rho = xspace.axpy(rho, charge, value[0])
        potential, poisson_residual, neutrality = poisson_cg(xspace, rho)
        electric = tuple(
            lift(
                xspace,
                phase,
                xspace.zip((xspace.gradient(potential, axis),), lambda b, gid, value: -value[0]),
            )
            for axis in range(len(xgrid.shape))
        )
        collision = tuple(
            _bgk(phase, xspace, velocity, distribution, moment_values, mass, frequency)
            for distribution, moment_values, mass in zip(
                distributions, all_moments, masses, strict=True
            )
        )
    else:
        # One typed evaluation scope returns only the collision fields. Pair
        # temporaries already die inside their folds; the joint projection's
        # requests and moments must also die before transport is evaluated.
        def collision_step(b, fields, _):
            values = tuple(fields["f" + str(s)] for s in range(len(masses)))
            sources = _fp_collision(
                phase.rebind(b), xspace.rebind(b), velocity, values,
                all_moments, masses, frequency,
            )
            return b.product(**{"f" + str(s): value for s, value in enumerate(sources)})

        collision_fields = p.fold(before["fields"], 1, collision_step)
        collision = tuple(collision_fields["f" + str(s)] for s in range(len(masses)))
    updated = []
    h = p.requested_dt()
    for distribution, source, charge, mass in zip(
        distributions, collision, charges, masses, strict=True
    ):
        rhs = phase.axpy(
            _transport(phase, velocity, distribution, electric, charge / mass), 1, source
        )
        updated.append(phase.axpy(distribution, h, rhs))
    after = p.product(
        configuration=before["configuration"],
        phase=before["phase"],
        fields=p.product(**{"f" + str(s): value for s, value in enumerate(updated)}),
        time=before["time"] + h,
        step=before["step"] + 1,
    )
    checks = p.product(poisson_residual_squared=poisson_residual, charge_mean=neutrality)
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
    def assess(b, arguments):
        old, new = arguments["base"], arguments["candidate"]["continuation"]
        local = phase.rebind(b, new["phase"])
        defect, minimum = b.float64(0), b.float64(1e300)
        for s in range(len(masses)):
            f = new["fields"]["f" + str(s)]
            change = local.axpy(f, -1, old["fields"]["f" + str(s)])
            delta = local.sum(change) * grid.volume
            defect = pm.maximum(defect, abs(delta))
            smallest = Distributed(b).reduce(
                f,
                b.float64(1e300),
                lambda c, a, x: pm.minimum(a, x),
                order="global_gid_ascending",
                byte_capacity=byte_capacity,
            )
            minimum = pm.minimum(minimum, smallest)
        valid = b.boolean(True)
        for condition in (
            defect < 1e-10,
            minimum > 0,
            arguments["candidate"]["checks"]["poisson_residual_squared"] < 1e-20,
            abs(arguments["candidate"]["checks"]["charge_mean"]) < 1e-11,
        ):
            valid = b.branch(condition, lambda c, valid=valid: valid, lambda c: c.boolean(False))
        return b.product(accepted=valid, checks=b.product(number_defect=defect, minimum=minimum))

    p.continuation_publish(
        p.continuation_accept(candidate, p.continuation_attest(candidate, assess))
    )
    p.step_strategy(FixedDt(dt))
    case = pops.Case(model + "_distributed_tensor_phase_space")
    case.program(p)
    return CaseSpec(
        case,
        layout,
        {},
        {},
        dt * steps,
        dt,
        lambda runtime: {"passed": runtime.macro_step() == steps, "frontier_only": True},
        model
        + ": explicit FV on "
        + str(len(xgrid.shape))
        + "x"
        + str(len(velocity.shape))
        + "v distributed index sets; all species/directions, native CG and velocity moments; finite zero-flux velocity box",
    )
