"""Closed API-v0.2 reception cases; generated Program kernels own every step.

The cases use normalized scalar quantities on a genuine periodic spatial grid.
The callbacks below only check gathered results after native execution. They are
not passed to the scientific Program or called by a native cell kernel.
"""
from __future__ import annotations

from math import pi

from examples.api_v02.common import CaseSpec, MissingCapabilityError, state_array, non_owner_metrics


def _periodic_scalar(name, cells, dimension):
    import pops
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D, Cartesian2D
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    if type(cells) is not int or cells < 8 or dimension not in (1, 2):
        raise ValueError("reception uses >=8 genuine cells and dimension 1 or 2")
    frame = CartesianDomain(name, lower=(0.,)*dimension, upper=(1.,)*dimension).frame(
        Cartesian1D() if dimension == 1 else Cartesian2D())
    model = pops.Model(name, frame=frame)
    declaration = model.state("q", components=("q",), sampling="cell_average")
    case = pops.Case(name)
    block = case.block("density", model, states=(declaration,))
    subject = block[declaration]
    case.initials.add(InitialCondition(state=subject, value=BindArray(),
                                      projection=ConservativeCellAverage()))
    shape = (cells,)+(1,)*(dimension-1)
    layout = Uniform(CartesianGrid(frame=frame, cells=shape, periodic=PeriodicAxes(frame.axes)))
    return case, layout, subject, shape


def _laplacian(program, state, midpoint, name):
    """The actual finite Cartesian Laplacian, including native ghost fill."""
    field = program.cell_field(state, quadrature=midpoint, name=name+"_sample")
    result = program.laplacian(program.scalar_field(name+"_buffer", like=field), field)
    return program.cell_mean(result, template=state, quadrature=midpoint, name=name)


def build_m26(*, cells=16, dimension=2, steps=4, dt=None, strength=-.4,
              perturbation=None):
    """Aggregation--diffusion: xi=rho+W*rho; rho_t=div(rho grad xi).

    W(x,y)=strength*cos(2*pi*(x0-y0)), periodic and symmetric. Nonlocal
    quadrature covers all actual source cells, including other MPI ranks.
    In the 1D profile the conservative centered realization is
        q_i=-rho_i*D0(xi)_i,  rho'_i=-D0(q)_i.
    In FV form the face flux is (q_i+q_{i+1})/2. No upwind/positivity or
    unconditional time-dissipation theorem is claimed for this chosen method.
    The 2D profile uses arithmetic face mobility with nearest-neighbour flux;
    its exact Laplacian composition is written below. There is one y cell and
    the solution is x-dependent, while all x cells remain real spatial cells.
    """
    import numpy as np
    import pops
    from pops import math
    from pops.fields import SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
    from pops.model.spaces import FieldSpace
    from pops.time import FixedDt
    if type(steps) is not int or steps < 2:
        raise ValueError("reception must exercise more than one native step")
    dt = 1e-4*(16/cells)**2 if dt is None else float(dt)
    if not 0 < dt <= 1e-4*(16/cells)**2:
        raise ValueError("requested step is outside this reception envelope")
    case, layout, subject, shape = _periodic_scalar("api_v02_m26", cells, dimension)
    p = pops.Program("nonlocal_conservative_euler")
    rho = p.state(subject)
    midpoint = CellMidpoint()
    space = FieldSpace("W_rho", components=rho.n.space.components,
        representation="field", frame=rho.n.space.frame, clock=rho.n.space.clock,
        support=rho.n.space.support, sampling="cell_center", units=rho.n.space.units)
    integral = p.spatial_interaction(rho.n,
        SpatialInteractionKernel(dimension, lambda x, y: strength*math.cos(2*pi*(x[0]-y[0]))),
        output_space=space, measure=CellVolumeMeasure(), quadrature=midpoint,
        realization=DirectSpatialInteraction(16 << 20), source_scope="issued")
    potential = p.cell_mean(integral, template=rho.n, quadrature=midpoint, name="W_rho_mean")
    xi = p.value("chemical_potential", (rho.n[0]+potential[0],))
    xi_field = p.cell_field(xi, quadrature=midpoint, name="xi_sample")
    if dimension == 1:
        grad = p.gradient(p.scalar_field("grad_xi", like=xi_field), xi_field)
        grad_mean = p.cell_mean(grad, template=rho.n, quadrature=midpoint, name="grad_xi_mean")
        flux = p.value("flux", (-rho.n[0]*grad_mean[0],))
        flux_field = p.cell_field(flux, quadrature=midpoint, name="flux_sample")
        divergence = p.divergence(p.scalar_field("div_flux", like=flux_field), flux_field)
        rate = p.cell_mean(divergence, template=rho.n, quadrature=midpoint, name="div_flux_mean")
        candidate = p.value("rho_next", rho.n-p.dt*rate, at=rho.next.point)
    else:
        # Identity div(rho grad xi)=.5*(Lap(rho*xi)+rho*Lap(xi)-xi*Lap(rho))
        # defines the arithmetic-face-mobility FV stencil exactly for the
        # shared nearest-neighbour Laplacian. It is valid in every native axis
        # and avoids padding a 1D gradient as a 2D vector.
        lap_xi = _laplacian(p, xi, midpoint, "lap_xi")
        lap_rho = _laplacian(p, rho.n, midpoint, "lap_rho")
        product = p.value("rho_xi", (rho.n[0]*xi[0],))
        lap_product = _laplacian(p, product, midpoint, "lap_rho_xi")
        rate = p.value("div_rho_grad_xi", ((lap_product[0]+rho.n[0]*lap_xi[0]-xi[0]*lap_rho[0])/2,))
        candidate = p.value("rho_next", rho.n+p.dt*rate, at=rho.next.point)
    p.commit(rho.next, candidate)
    p.step_strategy(FixedDt(dt))
    case.program(p)
    x = (np.arange(cells)+.5)/cells
    initial_line = 1+.1*np.cos(2*pi*x)+.05*np.sin(4*pi*x)
    if perturbation is not None:
        index, amplitude = perturbation
        if type(index) is not int or not 0 <= index < cells:
            raise ValueError("perturbation requires a valid spatial cell index")
        initial_line[index] += float(amplitude)
    if np.min(initial_line) <= 0:
        raise ValueError("initial density must be positive")
    initial = initial_line.reshape((1,)+tuple(reversed(shape)))
    W = strength*np.cos(2*pi*(x[:, None]-x[None, :]))/cells
    dx = 1/cells
    def lap(v):
        return (np.roll(v,-1)-2*v+np.roll(v,1))/dx**2
    def discrete_rate(v):
        xi0 = v+W@v
        if dimension == 1:
            grad0 = (np.roll(xi0,-1)-np.roll(xi0,1))/(2*dx)
            q = -v*grad0
            return -(np.roll(q,-1)-np.roll(q,1))/(2*dx)
        return .5*(lap(v*xi0)+v*lap(xi0)-xi0*lap(v))
    expected = initial_line.copy()
    for _ in range(steps):
        expected += dt*discrete_rate(expected)
    def energy(v):
        return dx*(np.dot(v,v)/2+np.dot(v,W@v)/2)
    def invariants(runtime):
        actual = state_array(runtime, "density", (cells,))
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        mass = float(abs(dx*np.sum(actual-initial_line)))
        energy_change = float(energy(actual)-energy(initial_line))
        min_density = float(np.min(actual))
        return dict(passed=error<3e-10 and mass<2e-12 and energy_change<=1e-12 and min_density>=0.,
                    native_discrete_error=error, mass_defect=mass, energy_change=energy_change,
                    min_density=min_density,
                    cells=cells, steps=steps, spatial_nonlocal=True,
                    field_source="native current stage, recomputed each step")
    return CaseSpec(case, layout, {subject: initial}, {"density": subject}, steps*dt, dt, invariants,
                    f"M26 periodic {dimension}D x-dependent nonlinear PDE, midpoint global kernel, explicit Euler; final density nonnegativity checked for this finite profile")


def build_m27(*, cells=16, dimension=2, steps=4, dt=None, kappa=.02):
    """Nonlinear double-well Cahn--Hilliard on actual spatial cells."""
    import numpy as np
    import pops
    from pops.fields import CellMidpoint
    from pops.time import FixedDt
    if type(steps) is not int or steps < 2 or kappa <= 0:
        raise ValueError("positive kappa and at least two steps required")
    dt = 1e-5*(16/cells)**4 if dt is None else float(dt)
    if not 0 < dt <= 1e-5*(16/cells)**4:
        raise ValueError("requested step exceeds this explicit fourth-order reception envelope")
    case, layout, subject, shape = _periodic_scalar("api_v02_m27", cells, dimension)
    p = pops.Program("nonlinear_mixed_explicit")
    c = p.state(subject)
    midpoint = CellMidpoint()
    lap_c = _laplacian(p, c.n, midpoint, "lap_c")
    mu = p.value("chemical_potential", (c.n[0]**3-c.n[0]-kappa*lap_c[0],))
    rate = _laplacian(p, mu, midpoint, "lap_mu")
    p.commit(c.next, p.value("c_next", c.n+p.dt*rate, at=c.next.point))
    p.step_strategy(FixedDt(dt))
    case.program(p)
    x = (np.arange(cells)+.5)/cells
    initial_line = .2*np.cos(2*pi*x)+.07*np.sin(4*pi*x)
    initial = initial_line.reshape((1,)+tuple(reversed(shape)))
    dx = 1/cells
    def lap(v):
        return (np.roll(v,-1)-2*v+np.roll(v,1))/dx**2
    expected = initial_line.copy()
    for _ in range(steps):
        expected += dt*lap(expected**3-expected-kappa*lap(expected))
    def energy(v):
        return dx*np.sum((v*v-1)**2/4)+kappa*np.sum((np.roll(v,-1)-v)**2)/(2*dx)
    def invariants(runtime):
        actual = state_array(runtime, "density", (cells,))
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        mass = float(abs(dx*np.sum(actual-initial_line)))
        energy_change = float(energy(actual)-energy(initial_line))
        return dict(passed=error<3e-10 and mass<2e-12 and energy_change<=1e-12,
                    native_discrete_error=error, mass_defect=mass, energy_change=energy_change,
                    cells=cells, steps=steps, nonlinear_double_well=True,
                    unconditional_energy_claim=False)
    return CaseSpec(case, layout, {subject: initial}, {"density": subject}, steps*dt, dt, invariants,
                    f"M27 periodic {dimension}D x-dependent nonlinear Cahn-Hilliard, actual fourth-order stencil, explicit Euler")


def build_m19(*, cells=16, velocity_cells=8, steps=4, dt=1e-4, collision_frequency=.5):
    """1x1v Vlasov--Poisson--BGK, periodic x, finite zero-flux v interval.

    The storage is (v component, y=1, x cell); this is a real multicell x PDE.
    Density is a declared weighted fibre contraction, then explicitly pulled
    back to each velocity component. The periodic mean-zero discrete Poisson
    inverse is authored as a finite Fourier Green kernel. Native global spatial
    interaction computes E at every step from the current distribution.
    """
    import numpy as np
    import pops
    from pops import math
    from pops._ir.quantity import PhysicalDimension
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian2D
    from pops.fibres import FiberAxis, FiberRepresentation
    from pops.fields import SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
    from pops.model.spaces import FieldSpace
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.time import FixedDt, FailRun
    if collision_frequency < 0:
        raise ValueError("collision frequency must be nonnegative")
    if type(cells) is not int or cells < 8 or cells % 2 or type(velocity_cells) is not int or velocity_cells < 4:
        raise ValueError("M19 needs an even spatial mesh >=8 and a velocity fibre >=4")
    if type(steps) is not int or steps < 2 or not 0 < dt <= 1e-4*(16/cells):
        raise ValueError("M19 reception requires multiple steps inside the declared CFL envelope")
    edges = tuple(np.linspace(-3., 3., velocity_cells+1))
    velocities = tuple(float((a+b)/2) for a,b in zip(edges,edges[1:]))
    weights = tuple(float(b-a) for a,b in zip(edges,edges[1:]))
    fibre = FiberRepresentation((FiberAxis("v", velocities, weights, PhysicalDimension(), "zero_flux", edges),))
    frame = CartesianDomain("phase_x", (0.,0.), (1.,1.)).frame(Cartesian2D())
    model = pops.Model("finite_velocity_transport", frame=frame)
    f = model.state("distribution", components=fibre.components, representation=fibre.representation,
                    sampling="cell_average")
    fibre.validate(f)
    flux = model.flux("spatial_transport", state=f, frame=frame,
        components={frame.axes[0]:tuple(v*f[j] for j,v in enumerate(velocities)),
                    frame.axes[1]:tuple(0.*f[j] for j in range(fibre.size))},
        waves={frame.axes[0]:velocities, frame.axes[1]:(0.,)*fibre.size})
    transport = model.rate("spatial_balance", equation=math.ddt(f) == -math.div(flux))
    case = pops.Case("M19_finite_fibre_Vlasov_Poisson")
    block = case.block("distribution", model)
    subject = block[f]
    case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    plan = DiscretizationPlan()
    plan.rates.add(transport, FiniteVolume(flux=flux, variables=variables.Conservative(f),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    case.numerics(plan,block=block)
    layout = Uniform(CartesianGrid(frame=frame,cells=(cells,1),periodic=PeriodicAxes(frame.axes)))
    p = pops.Program("finite_fibre_VP_Euler")
    state = p.state(subject)
    raw_values = fibre.values(state.n)
    raw_moments = tuple(fibre.moment(raw_values,(k,)) for k in range(3))
    margin = math.minimum(raw_moments[0],raw_moments[0]*raw_moments[2]-raw_moments[1]**2)
    for value in raw_values:
        margin = math.minimum(margin,value)
    margin_field = p.value("kinetic_domain_margin",fibre.pullback(margin))
    current = p.guard("positive_distribution_density_temperature",state.n,p.min(margin_field)>0.,action=FailRun())
    values = fibre.values(current)
    moments = tuple(fibre.moment(values,(k,)) for k in range(3))
    density = p.value("density_pullback", fibre.pullback(moments[0]))
    dx = 1/cells
    # -Lap(phi)=rho-mean(rho), E=-D0(phi); the k=0 gauge is omitted.
    # Nyquist's centered derivative is zero; conjugate modes are paired.
    coefficients = tuple((k,float(2*np.sin(2*pi*k/cells)/dx/(4*np.sin(pi*k/cells)**2/dx**2)))
                         for k in range(1,cells//2))
    kernel = SpatialInteractionKernel(2, lambda x,y: sum(
        coefficient*math.sin(2*pi*k*(x[0]-y[0])) for k,coefficient in coefficients))
    Espace = FieldSpace("electric_pullback", components=state.n.space.components,
        frame=state.n.space.frame, clock=state.n.space.clock, support=state.n.space.support,
        sampling="cell_center", representation_data=state.n.space.representation_data)
    electric = p.spatial_interaction(density,kernel,output_space=Espace,measure=CellVolumeMeasure(),
        quadrature=CellMidpoint(),realization=DirectSpatialInteraction(32 << 20),source_scope="issued")
    E = p.cell_mean(electric,template=state.n,quadrature=CellMidpoint(),name="electric_cell_mean")
    # E is constant on the fibre, so every velocity face uses the same sampled E.
    velocity_div = fibre.flux_divergence(values,axis=0,flux=lambda left,right,face:
        math.maximum(E[0],0.)*left+math.minimum(E[0],0.)*right)
    # Discrete BGK: sample the Maxwellian, then project its quadrature defects
    # onto span(1,v,v²). This realizes the three collision invariants exactly;
    # positivity is admitted explicitly, not guaranteed for arbitrary moments.
    rho,momentum,second = moments
    mean,temperature = momentum/rho,second/rho-(momentum/rho)**2
    sampled = tuple(rho/math.sqrt(2*pi*temperature)*math.exp(-(v-mean)**2/(2*temperature))
                    for v in velocities)
    gram = tuple(tuple(sum(w*v**(a+b) for w,v in zip(weights,velocities)) for b in range(3)) for a in range(3))
    a,b,c = gram[0]; _,d,e = gram[1]; _,_,h = gram[2]
    determinant = a*(d*h-e*e)-b*(b*h-c*e)+c*(b*e-c*d)
    inverse = ((d*h-e*e,c*e-b*h,b*e-c*d),
               (c*e-b*h,a*h-c*c,b*c-a*e),
               (b*e-c*d,b*c-a*e,a*d-b*b))
    defect = tuple(moments[k]-fibre.moment(sampled,(k,)) for k in range(3))
    coefficients_bgk = tuple(sum(inverse[k][l]*defect[l] for l in range(3))/determinant for k in range(3))
    equilibrium = p.value("conservative_discrete_Maxwellian",tuple(sampled[j]+sum(
        coefficients_bgk[k]*v**k for k in range(3)) for j,v in enumerate(velocities)))
    eq_margin = equilibrium[0]
    for j in range(1,fibre.size):
        eq_margin = math.minimum(eq_margin,equilibrium[j])
    eq_margin_field = p.value("equilibrium_margin",fibre.pullback(eq_margin))
    equilibrium = p.guard("positive_projected_Maxwellian",equilibrium,p.min(eq_margin_field)>=0.,action=FailRun())
    spatial_rate = transport(current)
    candidate = p.value("next_distribution",tuple(current[j]+p.dt*(spatial_rate[j]-velocity_div[j]
        +collision_frequency*(equilibrium[j]-current[j])) for j in range(fibre.size)),at=state.next.point)
    final_margin = candidate[0]
    for j in range(1,fibre.size):
        final_margin = math.minimum(final_margin,candidate[j])
    final_margin_field = p.value("candidate_margin",fibre.pullback(final_margin))
    candidate = p.guard("positive_candidate",candidate,p.min(final_margin_field)>0.,action=FailRun())
    p.commit(state.next,candidate)
    p.step_strategy(FixedDt(dt)); case.program(p)
    x = (np.arange(cells)+.5)/cells
    base = np.exp(-np.asarray(velocities)**2/2)
    base /= np.dot(np.asarray(weights),base)
    initial = base[:,None]*(1+.08*np.cos(2*pi*x))[None,:]
    green = sum(coefficient*np.sin(2*pi*k*(x[:,None]-x[None,:])) for k,coefficient in coefficients)*dx
    expected = initial.copy()
    speed = max(map(abs,velocities))
    def rhs(data):
        rho = np.asarray(weights)@data
        e = green@rho
        face = .5*(np.asarray(velocities)[:,None]*(data+np.roll(data,-1,axis=1))
                   -speed*(np.roll(data,-1,axis=1)-data))
        spatial = -(face-np.roll(face,1,axis=1))/dx
        vf = np.zeros((velocity_cells+1,cells))
        vf[1:-1] = np.maximum(e,0)*data[:-1]+np.minimum(e,0)*data[1:]
        target = np.stack([np.asarray(weights)*np.asarray(velocities)**k for k in range(3)])@data
        u = target[1]/target[0]
        T = target[2]/target[0]-u*u
        sampled0 = target[0][None,:]/np.sqrt(2*pi*T)[None,:]*np.exp(-(np.asarray(velocities)[:,None]-u[None,:])**2/(2*T)[None,:])
        basis = np.stack([np.asarray(velocities)**k for k in range(3)])
        delta = target-(basis*np.asarray(weights)[None,:])@sampled0
        equilibrium0 = sampled0+basis.T@np.linalg.solve(np.asarray(gram),delta)
        if np.min(equilibrium0)<0:
            raise ValueError("BGK reception falls outside the positive projected-Maxwellian domain")
        return spatial-(vf[1:]-vf[:-1])/np.asarray(weights)[:,None]+collision_frequency*(equilibrium0-data)
    for _ in range(steps):
        expected += dt*rhs(expected)
    def invariants(runtime):
        actual = state_array(runtime,"distribution",(velocity_cells,cells))
        if actual is None:
            return non_owner_metrics()
        error = float(np.max(np.abs(actual-expected)))
        mass = float(abs(np.sum(np.asarray(weights)[:,None]*(actual-initial))*dx))
        return dict(passed=error<4e-10 and mass<2e-12 and float(np.min(actual))>=0,
                    native_discrete_error=error,mass_defect=mass,minimum=float(np.min(actual)),
                    spatial_cells=cells,velocity_nodes=velocity_cells,steps=steps,collision_frequency=collision_frequency,
                    collision_invariants="weighted 1,v,v²; polynomial quadrature correction",
                    poisson_gauge="zero mean",velocity_boundary="zero flux on finite interval",
                    continuum_or_energy_conservation_claim=False)
    return CaseSpec(case,layout,{subject:initial[:,None,:]},{"distribution":subject},steps*dt,dt,invariants,
        "M19 1x1v finite-volume Vlasov-Poisson-BGK; typed fibre; native Poisson Green field; moment-conservative positive-domain projected Maxwellian")


def build_model(model_id, *, profile=None, require_complete=False, **kwargs):
    from .profile_dispatch import build_profile
    key = str(model_id).upper()
    if key not in {f"M{i:02}" for i in range(15, 29)}:
        raise MissingCapabilityError(key, required=("unknown model identifier; supported IDs are M15 through M28",),
                                     profile="selected closed M15-M28 physical model and numerical method")
    return build_profile(key, profile=profile, require_complete=require_complete, **kwargs)


build_case = build_model
