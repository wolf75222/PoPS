"""Declared periodic compressible NS benchmark; no output is manufactured here."""
import numpy as np

from tests.python.support.captured_diffusion_mms import CONTROLS, FD_STEP

GAMMA, GAS_R, CV, MU, KAPPA = 1.4, 1., 2.5, .03, .02
DT, ACCEPTANCE = .0002, 3e-8


def primitives(q):
    q = np.asarray(q)
    rho, momentum, energy = q
    velocity = momentum/rho
    temperature = (energy/rho-.5*velocity**2)/CV
    return np.stack((rho, velocity, temperature))


def amounts(psi):
    rho, velocity, temperature = np.asarray(psi)
    return np.stack((rho, rho*velocity, rho*(CV*temperature+.5*velocity**2)))


def initial_data(cells):
    """Analytic cell means of Q, not center samples relabeled as means.

    A fixed 32-point Gauss rule resolves these smooth trigonometric products to
    double precision. The distinct inverse-EOS guess is evaluated in Program.
    """
    nodes, weights = np.polynomial.legendre.leggauss(32)
    x = (np.arange(cells)[:, None]+.5+.5*nodes[None])/cells
    rho = 1+.08*np.cos(2*np.pi*x)
    u = .2+.04*np.sin(2*np.pi*x)
    t = 1+.06*np.cos(2*np.pi*x+.3)
    q = amounts(np.stack((rho, u, t)))
    return np.ascontiguousarray(np.sum(q*weights, axis=2)/2)


def convective_action(q):
    """Independent oriented FirstOrder/Rusanov Euler faces."""
    rho, u, t = primitives(q)
    assert np.all(rho > 0) and np.all(t > 0)
    pressure = rho*GAS_R*t
    physical = np.stack((q[1], q[1]*u+pressure, u*(q[2]+pressure)))
    wave = np.abs(u)+np.sqrt(GAMMA*GAS_R*t)
    speed = np.maximum(wave, np.roll(wave, -1))
    face = .5*(physical+np.roll(physical, -1, axis=1))-.5*speed*(np.roll(q, -1, axis=1)-q)
    return -q.shape[1]*(face-np.roll(face, 1, axis=1)), face


def viscous_action(psi):
    """Arithmetic candidate matrix and physical stress/heat oriented faces."""
    _, u, t = np.asarray(psi)
    n = len(u)
    stress = (4*MU/3)*n*(np.roll(u, -1)-u)
    energy = .5*(u+np.roll(u, -1))*stress+KAPPA*n*(np.roll(t, -1)-t)
    face = np.stack((np.zeros(n), stress, energy))
    return n*(face-np.roll(face, 1, axis=1)), face


def check_step(q, psi, previous, dt=DT):
    convective, euler_faces = convective_action(previous)
    viscous, viscous_faces = viscous_action(psi)
    issued = previous+dt*convective
    original = amounts(psi)-dt*viscous-issued
    volume = np.full(q.shape[1], 1/q.shape[1])
    relative = np.linalg.norm(original)/np.linalg.norm(issued)
    assert np.all(np.isfinite(q)) and np.min(psi[0]) > 0 and np.min(psi[2]) > 0
    assert relative <= ACCEPTANCE
    assert np.max(np.abs(q-amounts(psi))) <= ACCEPTANCE
    assert np.max(np.abs(np.sum((q-previous)*volume, axis=1))) <= ACCEPTANCE
    assert np.max(np.abs(euler_faces)) > 0 and np.max(np.abs(viscous_faces[1:])) > 0
    gradients = q.shape[1]*(np.roll(psi[1:], -1, axis=1)-psi[1:])
    # Physical constitutive entropy production is positive; this is distinct
    # from Rusanov dissipation and does not subtract energy a second time.
    face_t = .5*(psi[2]+np.roll(psi[2], -1))
    dissipation = np.sum(volume*((4*MU/3)*gradients[0]**2/face_t
                                  +KAPPA*gradients[1]**2/face_t**2))
    assert dissipation > 0
    return {"original_relative_l2": float(relative), "physical_entropy_production": float(dissipation)}, {
        "issued_Q": issued, "Euler_face_flux": euler_faces, "viscous_thermal_face_flux": viscous_faces,
        "original_F": original, "cell_volumes": volume}


def build(cells=8, *, previous_scope="issued", mutation=None):
    import pops
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.fields import FieldBoundary, FieldDiscretization, bcs
    from pops.fields.methods import CellCenteredNonlinearCoupled
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.layouts import Uniform
    from pops.math import DivCoeffGrad, Reaction, ValueExpr, ddt, div, sqrt
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops._ir.quantity import PhysicalSupport
    from pops._ir.quantity import PhysicalDimension
    from pops.model.spaces import FieldSpace
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.projection import ConservativeCellAverage
    from pops.solvers import Newton
    from pops.time import EvolvedOriginalFieldStage, EvolvedOriginalFieldRate, FailRun, FixedDt

    # Physical declarations precede every temporal/numerical decision.
    frame = CartesianDomain("gas-domain", (0.,), (1.,)).frame(Cartesian1D())
    model = pops.Model("compressible-viscous-thermal-material", frame=frame)
    state = model.state("Q", components=("rho", "rhou", "rhoE"), sampling="cell_average",
                        support=PhysicalSupport((("x", "gas-domain"),)))
    rho, momentum, energy = state
    velocity = momentum/rho
    pressure = (GAMMA-1)*(energy-.5*momentum**2/rho)
    sound = sqrt(GAMMA*pressure/rho)
    flux = model.flux("Euler", frame=frame, state=state,
        components={frame.axes[0]: (momentum, momentum*velocity+pressure, velocity*(energy+pressure))},
        waves={frame.axes[0]: (velocity-sound, velocity, velocity+sound)})
    transport = model.rate("convective-balance", equation=ddt(state) == -div(flux))
    density, speed, temperature = (model.field(name) for name in ("density", "velocity", "temperature"))
    # This benchmark uses nondimensional variables. Psi is a distinct physical
    # product, not the conservative StateSpace of Q used for cell allocation.
    primitive_space = FieldSpace("Psi", components=("density", "velocity", "temperature"),
        representation="primitive", centering="cell", sampling="cell_value",
        units=(PhysicalDimension(),)*3, frame=state.space.frame, clock=state.space.clock,
        support=state.space.support)
    r, u, t = (ValueExpr(row) for row in (density, speed, temperature))
    h = (Reaction(density, 1), Reaction(density, u), Reaction(density, CV*t+.5*u*u))
    viscous = (EvolvedOriginalFieldRate(spatial=None, additive=0),
        DivCoeffGrad(speed, 4*MU/3),
        DivCoeffGrad(speed, (4*MU/3)*u)+DivCoeffGrad(temperature, KAPPA))

    case = pops.Case("periodic-compressible-NS-conduction@1")
    block = case.block("fluid", model)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(), projection=ConservativeCellAverage()))
    discretization = DiscretizationPlan()
    discretization.rates.add(transport, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(discretization, block=block)
    program = pops.Program("Euler-then-original-viscous-thermal-stage")
    temporal = program.state(block[state])
    rhs = transport(temporal.n)
    issued = program.value("issued-convective-Q", temporal.n+program.dt*rhs, at=temporal.next.point)
    if mutation is not None:
        issued = mutation(program, temporal, issued)
    stage = EvolvedOriginalFieldStage("original-NS-viscous-thermal", unknowns=(density, speed, temperature),
        evolved_unknowns=(density, speed, temperature), accumulation=h, spatial_rhs=viscous,
        previous=tuple(state), tau=program.temporal_tau(program.dt, at=temporal.next.point),
        boundaries=tuple(FieldBoundary(row, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
                         for row in (density, speed, temperature)),
        **({"previous_scope": previous_scope} if previous_scope is not None else {}))
    solver = Newton(**CONTROLS)
    field = case.field(stage.problem, FieldDiscretization(method=CellCenteredNonlinearCoupled(
        finite_difference_step=FD_STEP, face_policy="Arithmetic@1", coefficient_evaluation="PerCandidate@1"),
        boundaries=(), solver=solver))
    # The old endpoint and thermal domain are checked by native reductions
    # before any inverse-EOS seed or coefficient evaluation consumes them.
    density_domain = program.value("issued-density-domain", (issued[0],)*3, at=temporal.next.point)
    issued = program.guard("positive-issued-density", issued, program.min(density_domain) > 0, action=FailRun())
    rho_star, momentum_star, energy_star = tuple(issued[i] for i in range(3))
    velocity_star = momentum_star/rho_star
    temperature_star = (energy_star/rho_star-.5*velocity_star**2)/CV
    thermal = program.value("issued-temperature-domain", (temperature_star,)*3, at=temporal.next.point)
    issued = program.guard("positive-issued-temperature", issued, program.min(thermal) > 0, action=FailRun())
    request = field.bind_program_inputs(program=program, values={block[state]: issued},
                                       at=temporal.next.point, solver=solver)
    request = request.seed_product(program=program, name="inverse-EOS-primitive-guess", space=primitive_space,
        expressions=(issued[0], issued[1]/issued[0], (issued[2]/issued[0]-.5*(issued[1]/issued[0])**2)/CV))
    seed = request.seeds["field_tuple"]
    for name, value in (("issued-convective-Q", issued), ("primitive-seed", seed), ("Euler-RHS", rhs)):
        program.store_history(name, value, depth=1)
    observed = field.observe(program.solve(request, solver=solver).consume(action=FailRun()))
    candidate = observed.evolved_state(target=temporal.next)
    candidate = program.guard("positive-final-density", candidate,
                              program.min(observed[field[density]]) > 0, action=FailRun())
    candidate = program.guard("positive-final-temperature", candidate,
                              program.min(observed[field[temperature]]) > 0, action=FailRun())
    program.commit(temporal.next, candidate)
    for name, unknown in zip(("density", "velocity", "temperature"), (density, speed, temperature), strict=True):
        program.store_history(name, observed[field[unknown]], depth=1)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(cells,), periodic=PeriodicAxes(frame.axes)))
    return case, layout, program
