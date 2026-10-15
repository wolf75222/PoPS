"""Reynolds update on distributed moving tensor control volumes in N dimensions.

Geometry, endpoint measures, exact swept face volumes and relative fluxes are
all Python equations lowered into the ordinary PoPS Continuation. No geometry
callback or model-specific native routine is used. The chosen geometry is a
separable Cartesian deformation, linear in time within each accepted interval.
"""
from __future__ import annotations

from itertools import combinations
from math import pi

import pops
from pops import math as pm
from pops.algebra import Distributed
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .models_M01_M14 import _domain
from .phase_space_transport import DistributedTensor, TensorGrid


def _geometry(space, time, amplitudes):
    grid = space.grid
    dimension = len(grid.shape)

    def cell(b, gid, index):
        coordinates = {}
        volume = b.float64(1)
        for axis in range(dimension):
            endpoints = []
            for side in (0, 1):
                node = grid.ordinal(gid, axis)+side
                reference = b.float64(node)/grid.shape[axis]
                # Periodic endpoints have exactly equal displacement.
                displacement = b.branch(node % grid.shape[axis] == 0,
                    lambda c: c.float64(0), lambda c, axis=axis, reference=reference:
                    c.float64(amplitudes[axis]*pm.sin(2*pi*reference)*pm.sin(2*pi*time)))
                endpoints.append(reference+displacement)
                coordinates[("lower" if side == 0 else "upper")+str(axis)] = endpoints[-1]
            volume = volume*(endpoints[1]-endpoints[0])
        return b.product(volume=volume, **coordinates)
    return space.map(cell)


def _integrated_area(old, new, axis, dimension):
    """Integral_0^1 product_j!=axis (width_old+s*delta_width) ds."""
    transverse = tuple(j for j in range(dimension) if j != axis)
    result = 0.
    for count in range(len(transverse)+1):
        for selected in combinations(transverse, count):
            term = 1.
            for j in transverse:
                width = old["upper"+str(j)]-old["lower"+str(j)]
                delta = new["upper"+str(j)]-new["lower"+str(j)]-width
                term = term*(delta if j in selected else width)
            result = result+term/(count+1)
    return result


def build_ale_nd(*, shape=(8, 6), amplitudes=(.025, .018), velocities=(.7, -.3),
                 ranks=1, dt=.001, steps=4, nonuniform=True):
    shape, amplitudes, velocities = tuple(shape), tuple(amplitudes), tuple(velocities)
    dimension = len(shape)
    if len(amplitudes) != dimension or len(velocities) != dimension:
        raise ValueError("one deformation and velocity per physical axis is required")
    if any(abs(a)*2*pi >= .5 for a in amplitudes) or dt <= 0 or type(steps) is not int or steps < 1:
        raise ValueError("positive steps and a nonsingular declared geometry are required")
    grid = TensorGrid(shape, ((0., 1.),)*dimension, ("periodic",)*dimension)
    p = pops.Program("authored_multidimensional_Reynolds_balance")
    space = DistributedTensor.create(p, grid, ranks)
    geometry = _geometry(space, p.float64(0), amplitudes)
    density = space.map(lambda b, gid, i: b.float64(1.3 + (sum(
        .04/dimension*pm.cos(2*pi*grid.coordinate(gid, a)) for a in range(dimension))
        if nonuniform else 0.)))
    uniform = space.constant(.8)
    initial = p.product(domain=space.domain, geometry=geometry, density=density, uniform=uniform,
                        time=p.float64(0), step=p.int64(0))
    schema = ContinuationSchema(initial, fields=(), time=("time",), step=("step",),
                                byte_capacity=32 << 20)
    base = p.continuation_open(initial, schema)
    old = p.continuation_read(base)
    space = space.rebind(p, old["domain"])
    h = p.requested_dt()
    next_geometry = _geometry(space, old["time"]+h, amplitudes)
    swept_divergence = space.constant(0)
    amounts = {name: space.zip((old[name], old["geometry"]),
        lambda b, gid, v: v[0]*v[1]["volume"]) for name in ("density", "uniform")}
    outgoing = space.constant(0)
    for axis in range(dimension):
        measures = space.zip((old["geometry"], next_geometry),
            lambda b, gid, v, axis=axis: b.product(
                area=_integrated_area(v[0], v[1], axis, dimension),
                swept=(v[1]["upper"+str(axis)]-v[0]["upper"+str(axis)])
                    *_integrated_area(v[0], v[1], axis, dimension)))
        previous = space.shift(measures, axis, -1)
        swept_divergence = space.axpy(swept_divergence, 1,
            space.zip((measures, previous), lambda b, gid, v: v[0]["swept"]-v[1]["swept"]))
        coefficients = space.zip((measures,), lambda b, gid, v, axis=axis:
            h*velocities[axis]*v[0]["area"]-v[0]["swept"])
        previous_coefficients = space.shift(coefficients, axis, -1)
        outgoing = space.axpy(outgoing, 1, space.zip((coefficients, previous_coefficients),
            lambda b, gid, v: pm.maximum(v[0], 0.)+pm.maximum(-v[1], 0.)))
        for name in amounts:
            right = space.shift(old[name], axis, 1)
            faces = space.zip((old[name], right, coefficients),
                lambda b, gid, v: (v[2]*(v[0]+v[1])-abs(v[2])*(v[1]-v[0]))/2)
            left = space.shift(faces, axis, -1)
            amounts[name] = space.axpy(amounts[name], 1,
                space.zip((left, faces), lambda b, gid, v: v[0]-v[1]))
    values = {name: space.zip((amount, next_geometry), lambda b, gid, v: v[0]/v[1]["volume"])
              for name, amount in amounts.items()}
    gcl = space.zip((next_geometry, old["geometry"], swept_divergence),
        lambda b, gid, v: v[0]["volume"]-v[1]["volume"]-v[2])
    cfl_margin = space.zip((old["geometry"], outgoing), lambda b, gid, v: v[0]["volume"]-v[1])
    minimum_margin = Distributed(p).reduce(cfl_margin, p.float64(1e300),
        lambda b, a, v: pm.minimum(a, v), order="global_gid_ascending", byte_capacity=32 << 20)
    checks = p.product(gcl=space.dot(gcl, gcl), cfl_margin=minimum_margin)
    after = p.product(domain=old["domain"], geometry=next_geometry, **values,
                      time=old["time"]+h, step=old["step"]+1)
    envelope = p.product(continuation=after, increment=p.product(), observations=checks, checks=checks)
    proposal = p.product(h=h)
    candidate = p.continuation_freeze(base, proposal, envelope)

    @graph(inputs=Product(base=schema.read_type, proposal=type_of(proposal), candidate=type_of(envelope)),
           output=Product(accepted=Bool, checks=Product(mass=Float64, uniform=Float64, gcl=Float64)))
    def assess(b, data):
        before, after = data["base"], data["candidate"]["continuation"]
        current = space.rebind(b, after["domain"])
        delta = current.zip((before["density"], before["geometry"], after["density"], after["geometry"]),
            lambda c, gid, v: v[2]*v[3]["volume"]-v[0]*v[1]["volume"])
        mass = abs(current.sum(delta))
        uniform_error = current.zip((after["uniform"],), lambda c, gid, v: (v[0]-.8)**2)
        uniform_error = current.sum(uniform_error)
        gcl_error = data["candidate"]["checks"]["gcl"]
        valid = b.boolean(True)
        for condition in (mass < 1e-11, uniform_error < 1e-20, gcl_error < 1e-22,
                          data["candidate"]["checks"]["cfl_margin"] > 0.):
            valid = b.branch(condition, lambda c, valid=valid: valid,
                             lambda c: c.boolean(False))
        return b.product(accepted=valid, checks=b.product(mass=mass, uniform=uniform_error, gcl=gcl_error))

    p.continuation_publish(p.continuation_accept(candidate, p.continuation_attest(candidate, assess)))
    p.step_strategy(FixedDt(dt))
    case = pops.Case("M28_authored_moving_tensor_volumes")
    case.program(p)
    _, layout = _domain("ALE_execution_context", (4, 4))
    return CaseSpec(case, layout, {}, {}, steps*dt, dt,
        lambda runtime: dict(passed=runtime.macro_step() == steps, frontier_only=True),
        "M28: distributed "+str(dimension)+"D moving control volumes, source-authored geometry, "
        "exact swept measures, shared relative fluxes, volume update, GCL, uniform-state and mass assessment")
