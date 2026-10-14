"""M14: spatial cold two-fluid sheath, Dirichlet Poisson lifting and wall capacitor.

Normalized one-dimensional profile embedded in genuine Dim2 Cartesian storage. Both species
have positive velocity throughout the short reception: the right outflow trace is consequently
an absorbing, emission-free wall. The left prescribed reservoir is an external charge source.
This is a transient cold-fluid sheath/circuit profile, not a thermal floating-wall equilibrium.
"""
from __future__ import annotations

import math
import numpy as np
import pops
from pops import math as pm
from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Inflow, Outflow
from pops.domain import Rectangle
from pops.fields import FieldDiscretization, CellCenteredSecondOrder, FieldOutput, GradientOutput, bcs
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid
from pops.model import PhysicalDimension
from pops.numerics import DiscretizationPlan, StateStorage, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.solvers.elliptic import CartesianCG
from pops.solvers.tolerances import Relative
from pops.time import FixedDt, FailRun, RejectAttempt
from .common import CaseSpec, state_array, non_owner_metrics
from .models_M01_M14 import _initial


CAPACITANCE = .5
INITIAL_WALL_CHARGE = -.2
ELECTRON_MASS = .1
RESERVOIR = np.array((1., 2., 1., 3.))


def _poisson_matrix(cells, spacing):
    """Independent cell-centered -dxx with physical Dirichlet faces at 0 and 1."""
    matrix = (2*np.eye(cells)-np.eye(cells,k=1)-np.eye(cells,k=-1))/spacing**2
    matrix[0,0] += 1/spacing**2
    matrix[-1,-1] += 1/spacing**2
    return matrix


def _reference(initial, *, dt, steps, capacitance, frozen_voltage=False):
    """NumPy face loop and dense Poisson solve, independent of PoPS kernels and providers."""
    state = initial.copy()
    cells = state.shape[-1]; spacing = 1/cells
    operator = _poisson_matrix(cells,spacing)
    wall = INITIAL_WALL_CHARGE; injected = 0.; work = 0.
    last_potential = last_state = None
    for _ in range(steps):
        last_state = state.copy()
        charge = state[0,0]-state[2,0]
        potential = np.linalg.solve(operator,charge)
        last_potential = potential.copy()
        extended = np.concatenate(([-potential[0]],potential,[-potential[-1]]))
        gradient = (extended[2:]-extended[:-2])/(2*spacing)
        voltage = (INITIAL_WALL_CHARGE if frozen_voltage else wall)/capacitance
        electric = -gradient-voltage
        # A face-prescribed reservoir uses the centered Dirichlet reconstruction:
        # (interior + exterior)/2 = prescribed face value.
        ghost = np.concatenate((2*RESERVOIR[:,None]-state[:,0,:1],state[:,0],state[:,0,-1:]),axis=1)
        faces = np.empty((4,cells+1))
        for face in range(cells+1):
            left,right = ghost[:,face],ghost[:,face+1]
            def physical(value):
                ni,pi,ne,pe = value
                return np.array((pi,pi*pi/ni,pe,pe*pe/ne))
            speed = max(abs(left[1]/left[0]),abs(right[1]/right[0]),
                        abs(left[3]/left[2]),abs(right[3]/right[2]))
            faces[:,face] = .5*(physical(left)+physical(right))-.5*speed*(right-left)
        update = -(faces[:,1:]-faces[:,:-1])/spacing
        update[1] += state[0,0]*electric
        update[3] -= state[2,0]*electric/ELECTRON_MASS
        state[:,0] += dt*update
        charge_in = dt*(faces[0,0]-faces[2,0])
        charge_out = dt*(faces[0,-1]-faces[2,-1])
        # The electrical work is a derived capacitor energy, not a second current accumulator.
        work += (wall+.5*charge_out)*charge_out/capacitance
        wall += charge_out
        injected += charge_in
    return state,wall,injected,work,last_potential,last_state


def build_m14(cells=(16,1), *, capacitance=CAPACITANCE, reject=False, receipt=False):
    """Cold ion/electron Euler-Poisson plus C dV/dt=Ji-Je on the absorbing wall.

    n_t+j_x=0, j_t+(j²/n)_x=(q/m)nE, -phi_xx=ni-ne.
    psi solves the homogeneous Dirichlet problem and phi=psi+(Q/C)x is the exact
    harmonic lifting of phi(0)=0, phi(1)=Q/C. All field and global reads use U^n,Q^n;
    coupled forward Euler updates the volume and the capacitor in one root transaction.
    """
    if (len(cells)!=2 or cells[1]!=1 or cells[0]<4
            or not math.isfinite(capacitance) or capacitance<=0):
        raise ValueError("M14 reception requires positive capacitance and cells=(N>=4,1)")
    frame = Rectangle("M14_sheath",lower=(0.,0.),upper=(1.,1.)).frame(Cartesian2D())
    x,y = frame.axes
    model = pops.Model("M14_cold_two_fluid_sheath",frame=frame)
    U = model.state("species",components=("ion_density","ion_current","electron_density","electron_current"),
                    sampling="cell_average")
    ni,ji,ne,je = U
    flux = model.flux("species_transport",state=U,frame=frame,
        components={x:(ji,ji*ji/ni,je,je*je/ne),y:(0*ni,0*ji,0*ne,0*je)},
        waves={x:(ji/ni,ji/ni,je/ne,je/ne),y:(0.,0.,0.,0.)})
    psi = model.field("homogeneous_potential")
    gradient = model.aux("grad_x")
    model.aux("grad_y")
    poisson_operator = model.field_operator("homogeneous_Dirichlet_Poisson",unknown=psi,
        equation=-pm.laplacian(psi)==ni-ne,
        outputs=(FieldOutput("homogeneous_potential",psi),GradientOutput("grad",psi,sign=1)))
    wall_port = model.global_quantity("wall_charge",units=PhysicalDimension())
    # Exact linear decomposition of -grad(phi)=-grad(psi)-Q/C. Each source keeps its
    # physical input port: a solved spatial gradient versus one true global charge.
    space_force = model.source("space_charge_force",on=U,
                               value=(0*ni,-ni*gradient,0*ne,ne*gradient/ELECTRON_MASS))
    wall_force = model.source("harmonic_wall_force",on=U,
                              value=(0*ni,-ni*wall_port/capacitance,0*ne,
                                     ne*wall_port/(capacitance*ELECTRON_MASS)))
    balance = model.rate("Euler_Poisson",equation=pm.ddt(U)==-pm.div(flux)+space_force+wall_force)
    transport = balance.select(flux)
    acceleration, lifting = balance.select(space_force),balance.select(wall_force)
    case = pops.Case("M14_sheath_capacitor"+("_reject" if reject else ""))
    block = case.block("fluid",model)
    numerical = DiscretizationPlan()
    numerical.rates.add(transport,FiniteVolume(flux=flux,variables=variables.Conservative(U),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.User(state=U,
            body=lambda left,right,fl,fr,speed:.5*(fl+fr)-.5*speed*(right-left),
            stability=lambda left,right,fl,fr,speed:speed)))
    numerical.rates.add(acceleration,StateStorage())
    numerical.rates.add(lifting,StateStorage())
    numerical.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min:Inflow(state=block[U],value=tuple(RESERVOIR)),
        frame.boundaries.x_max:Outflow(state=block[U]),
        frame.boundaries.y_min:Outflow(state=block[U]),
        frame.boundaries.y_max:Outflow(state=block[U]),
    }))
    case.numerics(numerical,block=block)
    field = case.field(poisson_operator,FieldDiscretization(method=CellCenteredSecondOrder(),
        boundaries=(bcs.BoundaryCondition(bcs.XMin(),bcs.Dirichlet(0.)),
                    bcs.BoundaryCondition(bcs.XMax(),bcs.Dirichlet(0.)),
                    bcs.BoundaryCondition(bcs.YMin(),bcs.Neumann(0.)),
                    bcs.BoundaryCondition(bcs.YMax(),bcs.Neumann(0.))),
        solver=CartesianCG(tolerance=Relative(1.e-12),max_iterations=1000)))
    p = pops.Program("M14_joint_explicit_window")
    q = p.state(block[U]); old = q.n
    wall = p.integral_state("wall_charge",initial=INITIAL_WALL_CHARGE,units=PhysicalDimension())
    reservoir = p.integral_state("injected_charge",initial=0.,units=PhysicalDimension())
    captured_charge = p.integral_value(wall,at=old.point,scope="candidate")
    context = field(old).consume(action=FailRun())
    space_value = p.source(model.module.operator_handle("space_charge_force"),old,context)
    wall_value = p.evaluate_source(lifting,old,global_inputs={block[wall_port]:captured_charge})
    flow = transport(old)
    candidate = p.value("joint_candidate",old+p.dt*flow+p.dt*space_value+p.dt*wall_value,at=q.next.point)
    p.store_history("source_state",old,depth=1)
    if reject:
        candidate = p.guard("reject_after_provisional_wall_current",candidate,
                            p.norm_inf(candidate)<0.,action=RejectAttempt())
    p.commit(q.next,candidate)
    # A ledger amount points into the volume. The right-wall capacitor receives its negative;
    # electron number current has negative charge. Reservoir accumulation uses the left sign.
    for component,charge_sign in ((0,1.),(2,-1.)):
        p.accept_external_trace(wall,rate=flow,axis=0,side=1,component=component,scale=-charge_sign)
        p.accept_external_trace(reservoir,rate=flow,axis=0,side=0,component=component,scale=charge_sign)
    dt = .0002; steps = 4
    p.step_strategy(FixedDt(dt));case.program(p)
    layout = Uniform(CartesianGrid(frame=frame,cells=cells))
    centers = (np.arange(cells[0])+.5)/cells[0]
    ions = 1+.02*np.sinc(.5/cells[0])*np.sin(np.pi*centers)
    electrons = .95+.03*np.sinc(1/cells[0])*np.sin(2*np.pi*centers)
    initial = np.stack((ions,2*ions,electrons,3*electrons))[:,None,:]
    reference = _reference(initial,dt=dt,steps=steps,capacitance=capacitance)
    frozen = _reference(initial,dt=dt,steps=steps,capacitance=capacitance,frozen_voltage=True)[0]
    initial_total = float(np.mean(initial[0]-initial[2])+INITIAL_WALL_CHARGE)
    def check(runtime):
        actual = state_array(runtime,"fluid",initial.shape)
        history_state = runtime.history_global("source_state",1)
        slots = runtime.field_provider_slots()
        if len(slots)!=1:raise ValueError("M14 requires exactly its authored Poisson provider")
        history_psi = runtime.field_potential_global(slots[0])
        charge_wall = runtime.integral_state(wall)
        charge_in = runtime.integral_state(reservoir)
        if actual is None:return non_owner_metrics()
        expected,expected_wall,expected_in,expected_work,_,_ = reference
        source_state = np.asarray(history_state).reshape(initial.shape)
        potential = np.asarray(history_psi).reshape(cells[0])
        poisson_error = float(np.max(abs(_poisson_matrix(cells[0],1/cells[0])@potential
                                        -(source_state[0,0]-source_state[2,0]))))
        state_error = float(np.max(abs(actual-expected)))
        charge_error = abs(charge_wall-expected_wall)+abs(charge_in-expected_in)
        total_error = abs(float(np.mean(actual[0]-actual[2]))+charge_wall-charge_in-initial_total)
        circuit_work = (charge_wall**2-INITIAL_WALL_CHARGE**2)/(2*capacitance)
        work_error = abs(circuit_work-expected_work)
        outgoing = float(min(np.min(actual[1]/actual[0]),np.min(actual[3]/actual[2])))
        feedback_gap = float(np.max(abs(expected-frozen)))
        passed = bool(np.isfinite(actual).all() and np.min(actual[[0,2]])>0 and outgoing>0
            and state_error<2.e-9 and charge_error<2.e-10 and total_error<2.e-10
            and poisson_error<2.e-8 and work_error<2.e-10 and feedback_gap>1.e-8
            and abs(charge_wall-INITIAL_WALL_CHARGE)>1.e-5)
        return dict(passed=passed,metrics_owner=True,full_FV_Poisson_oracle_defect=state_error,
            capacitor_and_reservoir_defect=charge_error,volume_surface_charge_defect=total_error,
            capacitor_work_defect=work_error,Poisson_residual=poisson_error,
            minimum_outgoing_velocity=outgoing,live_voltage_vs_frozen_defect=feedback_gap,
            wall_charge=float(charge_wall),wall_voltage=float(charge_wall/capacitance))
    spec = CaseSpec(case,layout,_initial(case,block[U],initial),{"fluid":block[U]},steps*dt,dt,check,
        "M14: open cold ion/electron Euler-Poisson sheath; prescribed left reservoir, absorbing right wall in checked strictly outgoing regime; homogeneous Dirichlet Green solve plus exact live Q/C harmonic lifting; four joint forward-Euler windows, accepted species face charges update capacitor and reservoir once; full independent FV/Poisson/circuit oracle, no thermal equilibrium or total-energy conservation claim")
    return (spec,wall,reservoir) if receipt else spec
