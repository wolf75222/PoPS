"""Periodic 2.5D adiabatic Hall/Ohmic MHD with explicit Python CT algebra.

In units mu0=q=1, E=-u×B+h(J×B)/rho+eta*J and J=curl(B).
Mass, Maxwell stress and total-energy/Poynting flux are conservative.
The Hall energy flux follows Kimura--Morrison (2014), III.A,
https://arxiv.org/abs/1406.2745; no Hall heating is added. This is a declared
Cartesian CT/RK4 variant, not the moving-mesh scheme of arXiv:2309.15907.

Every periodic entity set has nx*ny labels. At label (i,j), rho/m/E/Bz are
cell-centred, Bx/Ey are at (i,j+1/2), By/Ex at (i+1/2,j), and Ez/Jz at (i,j).
Incidences, interpolation and their transposes are equations in this file.
The engine only executes distributed maps, imports, reductions and products.
"""

from __future__ import annotations

from math import pi

import pops
from pops import math as pm
from pops.algebra import Distributed
from pops.continuations import ContinuationSchema
from pops.graph_functions import graph
from pops.time import FixedDt
from pops.time.variant_types import Bool, Float64, Product, type_of

from .common import CaseSpec
from .phase_space_transport import DistributedTensor, TensorGrid

NAMES = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz")


def _average(space, value, axis, direction):
    return space.zip(
        (value, space.shift(value, axis, direction)), lambda b, gid, v: (v[0] + v[1]) / 2
    )


def _difference(space, value, axis, direction):
    return space.zip(
        (value, space.shift(value, axis, direction)),
        lambda b, gid, v: direction * (v[1] - v[0]) / space.grid.widths[axis],
    )


def _cross(left, right):
    return tuple(
        left[(a + 1) % 3] * right[(a + 2) % 3] - left[(a + 2) % 3] * right[(a + 1) % 3]
        for a in range(3)
    )


def _edge_to_cell(space, values):
    return (
        _average(space, values[0], 1, 1),
        _average(space, values[1], 0, 1),
        _average(space, _average(space, values[2], 0, 1), 1, 1),
    )


def _cell_to_edge(space, values):
    """The exact transpose of _edge_to_cell for periodic uniform measures."""
    return (
        _average(space, values[0], 1, -1),
        _average(space, values[1], 0, -1),
        _average(space, _average(space, values[2], 0, -1), 1, -1),
    )


def _magnetic(space, state):
    bx, by, bz = (state[name] for name in ("bx", "by", "bz"))
    cell = (_average(space, bx, 0, 1), _average(space, by, 1, 1), bz)
    square = tuple(space.zip((value,), lambda b, gid, v: v[0] ** 2) for value in (bx, by, bz))
    energy = space.zip(
        (_average(space, square[0], 0, 1), _average(space, square[1], 1, 1), square[2]),
        lambda b, gid, v: sum(v) / 2,
    )
    current = (
        _difference(space, bz, 1, -1),
        space.zip((_difference(space, bz, 0, -1),), lambda b, gid, v: -v[0]),
        space.axpy(_difference(space, by, 0, -1), -1, _difference(space, bx, 1, -1)),
    )
    return cell, energy, current


def _divergence(space, state):
    return space.axpy(
        _difference(space, state["bx"], 0, 1), 1, _difference(space, state["by"], 1, 1)
    )


def _pressure(space, state, magnetic_energy, gamma):
    return space.zip(
        tuple(state[name] for name in NAMES[:5]) + (magnetic_energy,),
        lambda b, gid, v: (gamma - 1) * (v[4] - sum(q * q for q in v[1:4]) / (2 * v[0]) - v[5]),
    )


def _rhs(space, state, hall, ohmic, gamma):
    cell_B, magnetic_energy, current = _magnetic(space, state)
    cell_J = _edge_to_cell(space, current)
    pressure = _pressure(space, state, magnetic_energy, gamma)
    velocity = tuple(
        space.zip((state[name], state["rho"]), lambda b, gid, v: v[0] / v[1])
        for name in ("mx", "my", "mz")
    )
    ideal_cell = tuple(
        space.zip((*velocity, *cell_B), lambda b, gid, v, a=a: -_cross(v[:3], v[3:])[a])
        for a in range(3)
    )
    hall_cell = tuple(
        space.zip(
            (*cell_J, *cell_B, state["rho"]),
            lambda b, gid, v, a=a: hall * _cross(v[:3], v[3:6])[a] / v[6],
        )
        for a in range(3)
    )
    ideal, electric_hall = _cell_to_edge(space, ideal_cell), _cell_to_edge(space, hall_cell)
    electric = tuple(
        space.zip((i, h, j), lambda b, gid, v: v[0] + v[1] + ohmic * v[2])
        for i, h, j in zip(ideal, electric_hall, current, strict=True)
    )
    electric_cell = _edge_to_cell(space, electric)
    result = {name: space.constant(0) for name in NAMES}
    for axis in range(2):
        for component, name in enumerate(NAMES[:5]):

            def flux(b, gid, v, axis=axis, component=component):
                rho, momentum, energy = v[0], v[1:4], v[4]
                B, E, p, magnetic = v[5:8], v[8:11], v[11], v[12]
                u = tuple(x / rho for x in momentum)
                if component == 0:
                    return momentum[axis]
                if component < 4:
                    a = component - 1
                    return (
                        momentum[a] * u[axis]
                        + ((p + sum(x * x for x in B) / 2) if a == axis else 0)
                        - B[a] * B[axis]
                    )
                return (energy - magnetic + p) * u[axis] + _cross(E, B)[axis]

            values = space.zip(
                tuple(state[name] for name in NAMES[:5])
                + (*cell_B, *electric_cell, pressure, magnetic_energy),
                flux,
            )
            faces = _average(space, values, axis, 1)
            result[name] = space.axpy(result[name], -1, _difference(space, faces, axis, -1))
    result["bx"] = space.zip((_difference(space, electric[2], 1, 1),), lambda b, gid, v: -v[0])
    result["by"] = _difference(space, electric[2], 0, 1)
    result["bz"] = space.axpy(
        _difference(space, electric[0], 1, 1), -1, _difference(space, electric[1], 0, 1)
    )
    hall_work = (
        sum(space.dot(j, e) for j, e in zip(current, electric_hall, strict=True))
        * space.grid.volume
    )
    ohmic_heat = ohmic * sum(space.dot(j, j) for j in current) * space.grid.volume
    field_work = (
        sum(space.dot(j, e) for j, e in zip(current, electric, strict=True)) * space.grid.volume
    )
    magnetic_rate = (
        sum(space.dot(state[name], result[name]) for name in NAMES[5:]) * space.grid.volume
    )
    checks = space.program.product(
        hall_work=hall_work, ohmic_heat=ohmic_heat, poynting_identity=magnetic_rate + field_work
    )
    return result, checks


def _add(space, left, right, scale):
    return {name: space.axpy(left[name], scale, right[name]) for name in NAMES}


def _initial(space, gamma):
    potential = space.map(
        lambda b, gid, i: (
            0.03
            * pm.sin(2 * pi * (space.grid.coordinate(gid, 0) - space.grid.widths[0] / 2))
            * pm.sin(2 * pi * (space.grid.coordinate(gid, 1) - space.grid.widths[1] / 2))
            + 0.01
            * pm.cos(
                4 * pi * (space.grid.coordinate(gid, 0) - space.grid.widths[0] / 2)
                + 2 * pi * (space.grid.coordinate(gid, 1) - space.grid.widths[1] / 2)
            )
        )
    )
    bx = space.zip((_difference(space, potential, 1, 1),), lambda b, gid, v: 1 + v[0])
    by = space.zip((_difference(space, potential, 0, 1),), lambda b, gid, v: 0.3 - v[0])
    bz = space.map(
        lambda b, gid, i: (
            0.4
            + 0.08
            * pm.cos(2 * pi * (space.grid.coordinate(gid, 0) + space.grid.coordinate(gid, 1)))
        )
    )
    rho = space.map(
        lambda b, gid, i: (
            1
            + 0.05
            * pm.sin(2 * pi * space.grid.coordinate(gid, 0))
            * pm.cos(2 * pi * space.grid.coordinate(gid, 1))
        )
    )
    velocity = (
        space.map(lambda b, gid, i: 0.1 * pm.sin(2 * pi * space.grid.coordinate(gid, 1))),
        space.map(lambda b, gid, i: 0.1 * pm.cos(2 * pi * space.grid.coordinate(gid, 0))),
        space.map(
            lambda b, gid, i: (
                0.05
                * pm.sin(2 * pi * (space.grid.coordinate(gid, 0) + space.grid.coordinate(gid, 1)))
            )
        ),
    )
    pressure = space.map(
        lambda b, gid, i: (
            1
            + 0.03
            * pm.cos(2 * pi * (space.grid.coordinate(gid, 0) - space.grid.coordinate(gid, 1)))
        )
    )
    state = dict(rho=rho, bx=bx, by=by, bz=bz)
    for name, component in zip(("mx", "my", "mz"), velocity, strict=True):
        state[name] = space.zip((rho, component), lambda b, gid, v: v[0] * v[1])
    _, energy, _ = _magnetic(space, state)
    state["energy"] = space.zip(
        (pressure, rho, *velocity, energy),
        lambda b, gid, v: v[0] / (gamma - 1) + v[1] * sum(x * x for x in v[2:5]) / 2 + v[5],
    )
    return {name: state[name] for name in NAMES}


def build_hall_mhd_ct(*, cells=(6, 8), ranks=1, hall=0.15, ohmic=0.01, gamma=1.4, dt=1e-4, steps=3):
    if (
        len(cells) != 2
        or min(cells) < 3
        or ohmic < 0
        or gamma <= 1
        or dt <= 0
        or type(steps) is not int
        or steps < 1
    ):
        raise ValueError(
            "periodic 2D MHD requires >=3 cells per axis, eta>=0, gamma>1 and positive step"
        )
    from .models_M01_M14 import _domain

    _, layout = _domain("Hall_CT_execution_context", (4, 4))
    p = pops.Program("Hall_CT_equations_RK4")
    space = DistributedTensor.create(
        p, TensorGrid(tuple(cells), ((0.0, 1.0),) * 2, ("periodic",) * 2), ranks
    )
    initial = p.product(
        domain=space.domain,
        fields=p.product(**_initial(space, gamma)),
        time=p.float64(0),
        step=p.int64(0),
    )
    schema = ContinuationSchema(
        initial, fields=(), time=("time",), step=("step",), byte_capacity=64 << 20
    )
    base = p.continuation_open(initial, schema)
    before = p.continuation_read(base)
    space = space.rebind(p, before["domain"])
    state = {name: before["fields"][name] for name in NAMES}
    h = p.requested_dt()
    k1, c1 = _rhs(space, state, hall, ohmic, gamma)
    k2, c2 = _rhs(space, _add(space, state, k1, h / 2), hall, ohmic, gamma)
    k3, c3 = _rhs(space, _add(space, state, k2, h / 2), hall, ohmic, gamma)
    k4, c4 = _rhs(space, _add(space, state, k3, h), hall, ohmic, gamma)
    updated = {
        name: space.zip(
            (state[name], k1[name], k2[name], k3[name], k4[name]),
            lambda b, gid, v: v[0] + h * (v[1] + 2 * v[2] + 2 * v[3] + v[4]) / 6,
        )
        for name in NAMES
    }
    after = p.product(
        domain=before["domain"],
        fields=p.product(**updated),
        time=before["time"] + h,
        step=before["step"] + 1,
    )
    checks = p.product(
        hall_work=sum(abs(c["hall_work"]) for c in (c1, c2, c3, c4)),
        ohmic_heat=sum(c["ohmic_heat"] for c in (c1, c2, c3, c4)),
        poynting_identity=sum(abs(c["poynting_identity"]) for c in (c1, c2, c3, c4)),
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
        output=Product(
            accepted=Bool, checks=Product(inventory_defect=Float64, divergence_defect=Float64)
        ),
    )
    def assess(b, args):
        old, new = args["base"], args["candidate"]["continuation"]
        local = space.rebind(b, new["domain"])
        old_fields = {name: old["fields"][name] for name in NAMES}
        new_fields = {name: new["fields"][name] for name in NAMES}
        inventory = b.float64(0)
        for name in NAMES:
            defect = (
                local.sum(local.axpy(new_fields[name], -1, old_fields[name])) * local.grid.volume
            )
            inventory = pm.maximum(inventory, abs(defect))
        div_old, div_new = _divergence(local, old_fields), _divergence(local, new_fields)
        delta = local.axpy(div_new, -1, div_old)
        divergence = local.dot(delta, delta) + local.dot(div_new, div_new)
        _, magnetic, _ = _magnetic(local, new_fields)
        pressure = _pressure(local, new_fields, magnetic, gamma)
        minimum = b.float64(1e300)
        for value in (new_fields["rho"], pressure):
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
        c = args["candidate"]["checks"]
        valid = b.boolean(True)
        for condition in (
            inventory < 1e-11,
            divergence < 1e-20,
            minimum > 0,
            c["hall_work"] < 1e-12,
            c["ohmic_heat"] >= 0,
            c["poynting_identity"] < 1e-12,
        ):
            valid = b.branch(condition, lambda t, valid=valid: valid, lambda t: t.boolean(False))
        return b.product(
            accepted=valid,
            checks=b.product(inventory_defect=inventory, divergence_defect=divergence),
        )

    p.continuation_publish(
        p.continuation_accept(candidate, p.continuation_attest(candidate, assess))
    )
    p.step_strategy(FixedDt(dt))
    case = pops.Case("M23_Hall_Ohm_CT_2d")
    case.program(p)
    return CaseSpec(
        case,
        layout,
        {},
        {},
        dt * steps,
        dt,
        lambda runtime: {"passed": runtime.macro_step() == steps, "frontier_only": True},
        "M23 periodic nonlinear 2.5D Hall/Ohm MHD, face CT, conservative Maxwell stress/Poynting flux, RK4; explicit coefficients and finite-step qualification",
    )
