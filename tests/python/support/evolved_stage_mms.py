"""Independent FV data/oracles for installed EvolvedOriginalFieldStage witnesses.

This module manufactures no output. Initial fields, loads and acceptance bounds
are fixed before native execution; actual accepted fields must come from ROOT's
installed runtime. No PoPS spatial operator is called by the numerical oracle.
"""
from __future__ import annotations

import numpy as np

from tests.python.support.captured_diffusion_mms import CONTROLS, FD_STEP

ACCEPTANCE = 3e-8
DIFFUSION = np.array([[.012, .002], [-.001, .014]])
CANDIDATE_BETA = .4


def accumulation(temperature):
    t = np.asarray(temperature)
    if t.shape[0] == 1:
        return t + t*t
    if t.shape[0] != 2:
        raise ValueError("the declared witness has one or two evolved fields")
    a, b = t
    return np.stack((a+a*a+.1*b*b, b+b*b+.2*a*b))


def diffusion_action(temperature, *, candidate_diffusion=False):
    """Periodic div(D grad T) from oriented arithmetic face fluxes.

    Each face flux is integrated with its true face length. Cell differences
    divide by the declared cell volume. Signed cross entries are retained.
    """
    t = np.asarray(temperature)
    width, ny, nx = t.shape
    base = DIFFUSION[:width, :width]
    cell_d = np.broadcast_to(base[:, :, None, None], (width, width, ny, nx)).copy()
    if candidate_diffusion:
        cell_d *= 1+CANDIDATE_BETA*t[None]**2
    volume = 1/(nx*ny)
    result = np.zeros_like(t)
    face_fluxes = []
    for axis, cells, face_length in ((1, ny, 1/nx), (2, nx, 1/ny)):
        face_d = .5*cell_d + .5*np.roll(cell_d, -1, axis+1)
        gradient = cells*(np.roll(t, -1, axis)-t)
        flux = np.einsum("ijyx,jyx->iyx", face_d, gradient)*face_length
        result += (flux-np.roll(flux, 1, axis))/volume
        face_fluxes.append(flux)
    return result, tuple(face_fluxes), np.full((ny, nx), volume)


def manufactured_data(cells, width, dt, *, candidate_diffusion=False):
    if cells not in (8, 16) or width not in (1, 2) or dt not in (.01, .02):
        raise ValueError("explicit witness inventory is N8/N16, width1/2, dt.01/.02")
    coordinate = (np.arange(cells)+.5)/cells
    y, x = np.meshgrid(coordinate, coordinate, indexing="ij")
    initial_t = np.stack(tuple(.2+.03*np.sin(2*np.pi*x)+.02*np.cos(2*np.pi*y)
                              +.07*i for i in range(width)))
    target = np.stack(tuple(.21+.025*np.cos(2*np.pi*(i+1)*x)+.018*np.sin(2*np.pi*y)
                           +.07*i for i in range(width)))
    q0 = accumulation(initial_t)
    spatial, faces, volumes = diffusion_action(target, candidate_diffusion=candidate_diffusion)
    forcing = (accumulation(target)-q0)/dt-spatial
    initial = {"Q%d" % i: np.ascontiguousarray(q0[i:i+1]) for i in range(width)}
    initial["forcing"] = np.ascontiguousarray(forcing)
    return initial, initial_t, target, faces, volumes


def check_original(temperature, conserved, previous, forcing, dt, *,
                   candidate_diffusion=False, target=None):
    t = np.asarray(temperature)
    q = np.asarray(conserved)
    spatial, faces, volumes = diffusion_action(t, candidate_diffusion=candidate_diffusion)
    original = q-dt*(spatial+forcing)-previous
    scale = max(float(np.linalg.norm(previous)), float(np.linalg.norm(dt*forcing)), 1e-300)
    residual = float(np.linalg.norm(original)/scale)
    projection = float(np.max(np.abs(q-accumulation(t))))
    # Telescoping periodic flux sums independently prove the global amount law.
    balance = np.sum(volumes[None]*(q-previous-dt*forcing), axis=(1, 2))
    assert residual <= ACCEPTANCE
    assert projection <= ACCEPTANCE
    assert np.max(np.abs(balance)) <= ACCEPTANCE
    assert np.max(np.abs(spatial)) > .005
    assert all(np.max(np.abs(face)) > 0 for face in faces)
    result = {"original_relative_l2": residual, "Q_projection_linf": projection,
              "global_balance_linf": float(np.max(np.abs(balance))),
              "spatial_linf": float(np.max(np.abs(spatial))),
              "volume_sum": float(np.sum(volumes))}
    if target is not None:
        error = float(np.max(np.abs(t-target)))
        assert error <= ACCEPTANCE
        result["physical_field_linf"] = error
    return result


def build(cells, width, dt, *, candidate_diffusion=False, spatial=True, reverse=False, omit_forcing=False):
    """Physical declarations, then explicit stage/method, then execution layout."""
    import pops
    from pops._ir.elliptic import DivCoeffGrad, Reaction
    from pops._ir.quantity import PhysicalSupport
    from pops.domain import Rectangle
    from pops.fields import CellCenteredNonlinearCoupled, FieldBoundary, FieldDiscretization, bcs
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.math import ValueExpr, ddt, div
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.solvers import Newton
    from pops.time import EvolvedOriginalFieldStage, EvolvedOriginalFieldRate, FailRun, FixedDt

    frame = Rectangle("stage-material", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    model = pops.Model("accumulation-material", frame=frame)
    support = PhysicalSupport((("x", "stage-material"), ("y", "stage-material")))
    materials = (model, *tuple(pops.Model("accumulation-material-%d" % i, frame=frame)
                               for i in range(1, width)))
    conserved = tuple(material.state("Q%d" % i, components=("amount",), sampling="cell_average",
                                     support=support) for i, material in enumerate(materials))
    temperature = tuple(model.field("T%d" % i) for i in range(width))
    auxiliary = model.field("z") if width == 2 else None
    load_model = pops.Model("prescribed-load", frame=frame)
    forcing = load_model.state("forcing", components=tuple("f%d" % i for i in range(width)))
    # Q and the spatial rate are the physical declarations. No computed field
    # or time controller is hidden in the material model.
    q = (Reaction(temperature[0], 1+ValueExpr(temperature[0])),)
    if width == 2:
        a, b = temperature
        q = (Reaction(a, 1+ValueExpr(a))+Reaction(b, .1*ValueExpr(b)),
             Reaction(b, 1+ValueExpr(b))+Reaction(a, .2*ValueExpr(b)))
    rates = []
    for row in range(width):
        terms = []
        for column in range(width):
            coefficient = float(DIFFUSION[row, column])
            if candidate_diffusion:
                coefficient *= 1+CANDIDATE_BETA*ValueExpr(temperature[column])**2
            terms.append(DivCoeffGrad(temperature[column], coefficient))
        operator = sum(terms[1:], terms[0]) if spatial else None
        rates.append(EvolvedOriginalFieldRate(spatial=operator, additive=forcing[row]))
    constraints = ({auxiliary: Reaction(auxiliary, 1)+Reaction(temperature[0], -.25)
                    +Reaction(temperature[1], -.5) == 0} if auxiliary is not None else {})

    # The immutable load has an explicit physical zero flux; Program preserves
    # its accepted State instead of supplying a mutable parameter to the solver.
    flux = load_model.flux("stationary", frame=frame, state=forcing,
        components={axis: tuple(0*c for c in forcing) for axis in frame.axes},
        waves={axis: tuple(0*c for c in forcing) for axis in frame.axes})
    rate = load_model.rate("prescribed", equation=ddt(forcing) == -div(flux))
    case = pops.Case("evolved-original-stage-MMS")
    material_blocks = tuple(case.block("Q%d" % i, material) for i, material in enumerate(materials))
    load_block = case.block("forcing", load_model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(forcing),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=load_block)
    program = pops.Program("evolved-original-stage-step")
    previous = tuple(program.state(block[state]) for block, state in zip(material_blocks, conserved, strict=True))
    load = program.state(load_block[forcing])
    point = previous[0].next.point
    unknowns = (*temperature, *((auxiliary,) if auxiliary is not None else ()))
    order = tuple(reversed(range(width))) if reverse else tuple(range(width))
    stage = EvolvedOriginalFieldStage("original-evolution", unknowns=unknowns,
        evolved_unknowns=tuple(temperature[i] for i in order),
        accumulation=tuple(q[i] for i in order), spatial_rhs=tuple(rates[i] for i in order),
        previous=tuple(conserved[i][0] for i in order),
        tau=program.temporal_tau(program.dt, at=point), constraints=constraints,
        boundaries=tuple(FieldBoundary(t, bcs.BoundaryCondition(
            bcs.AllPhysicalBoundaries(), bcs.Periodic())) for t in unknowns))
    solver = Newton(**CONTROLS)
    field = case.field(stage.problem, FieldDiscretization(
        method=CellCenteredNonlinearCoupled(finite_difference_step=FD_STEP,
            face_policy="Arithmetic@1", coefficient_evaluation="PerCandidate@1" if candidate_diffusion else None),
        boundaries=(), solver=solver))
    captures = {block[state]: time.n for block, state, time in zip(material_blocks, conserved, previous, strict=True)}
    if not omit_forcing:
        captures[load_block[forcing]] = load.n
    request = field.bind_program_inputs(program=program, values=captures, at=point, solver=solver)
    observed = field.observe(program.solve(request, solver=solver).consume(action=FailRun()))
    for time in previous:
        program.commit(time.next, observed.evolved_state(target=time.next))
    for unknown in unknowns:
        program.store_history(unknown.name, observed[field[unknown]], depth=1)
    program.commit(load.next, program.value("preserved-load", 1*load.n, at=load.next.point))
    program.step_strategy(FixedDt(dt))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells, cells), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program
