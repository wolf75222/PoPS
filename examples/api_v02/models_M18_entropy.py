"""Finite discrete M_N entropy PDE: authored Newton + spatial D0 + RK4.

Velocity quadrature is Gauss--Legendre on [-1,1]; f=exp(sum alpha_j*v**j).
This is the quadrature-defined entropy closure, not the exact continuous
velocity integral. No invariant-domain or entropy-stability theorem is claimed
for centered space + RK4. Reception is restricted to smooth interior data.
"""
from __future__ import annotations

import numpy as np
import pops
from pops import math as pm
from pops.fields import CellMidpoint, CellVolumeMeasure, DirectSpatialInteraction
from pops.model import FieldSpace
from pops.numerics import DiscretizationPlan, StateStorage
from pops.representations import Representation
from pops.time import FixedDt
from pops.time.expressions import ProgramExpression
from .common import CaseSpec, state_array, non_owner_metrics
from .models_M01_M14 import _initial
from .models_M12 import _central_kernel


def reference_closure(moments, nodes, weights, *, tolerance=2.e-14, max_steps=24):
    """Independent dense NumPy Newton; used only to assess generated execution."""
    width=moments.shape[0]
    basis=np.stack([nodes**j for j in range(width)],axis=1)
    targets=np.moveaxis(moments,0,-1)
    alpha=np.zeros_like(targets)
    for iteration in range(max_steps+1):
        f=np.exp(np.einsum("...j,kj->...k",alpha,basis))
        reproduced=np.einsum("k,...k,kj->...j",weights,f,basis)
        residual=reproduced-targets
        if np.max(abs(residual)) < tolerance:
            flux=np.einsum("k,...k,k,kj->...j",weights,f,nodes,basis)
            return np.moveaxis(alpha,-1,0),np.moveaxis(flux,-1,0),float(np.max(abs(residual)))
        if iteration==max_steps:break
        H=np.einsum("k,...k,ki,kj->...ij",weights,f,basis,basis)
        # Different elimination implementation from the emitted scalar algebra.
        alpha-=np.linalg.solve(H,residual[...,None])[...,0]
    raise ArithmeticError("reference entropy closure did not converge")


def reference_rk4(initial,nodes,weights,dx,dt,steps):
    def rhs(value):
        _,flux,_=reference_closure(value,nodes,weights)
        return -(np.roll(flux,-1,axis=-1)-np.roll(flux,1,axis=-1))/(2*dx)
    value=initial.copy()
    for _ in range(steps):
        a=rhs(value);b=rhs(value+.5*dt*a);c=rhs(value+.5*dt*b);d=rhs(value+dt*c)
        value+=dt*(a+2*b+2*c+d)/6
    return value


def _newton_closure(p,target,nodes,weights,*,label):
    """Pure authoring: finite quadrature, H elimination and iterate are PoPS IR."""
    width=len(target.space.components)
    powers=tuple(tuple(float(v**j) for j in range(2*width)) for v in nodes)
    seed=p.value(label+"_dual_seed",ProgramExpression((0.,)*width,target),at=target.point)
    initial=p.product(alpha=seed,target=target)
    def distribution(alpha):
        return tuple(pm.exp(sum(alpha[j]*power[j] for j in range(width))) for power in powers)
    def moment(f,power):
        return sum(float(w)*row[power]*value for w,row,value in zip(weights,powers,f,strict=True))
    def residual(builder,state):
        alpha=state["alpha"];wanted=state["target"];f=distribution(alpha)
        return builder.value(label+"_original_residual",tuple(moment(f,j)-wanted[j] for j in range(width)),at=alpha.point)
    def step(builder,state):
        alpha=state["alpha"];wanted=state["target"];f=distribution(alpha)
        right=builder.value(label+"_residual",tuple(moment(f,j)-wanted[j] for j in range(width)),at=alpha.point)
        matrix=[builder.value(label+"_H%d"%i,tuple(moment(f,i+j) for j in range(width)),at=alpha.point)
                for i in range(width)]
        # Positive-weight, distinct-node Hessian is SPD for finite alpha.
        # No clipping, determinant repair, regularization or hidden root switch.
        for k in range(width-1):
            factors=builder.value(label+"_factors%d"%k,tuple(
                matrix[i][k]/matrix[k][k] if i>k else 0. for i in range(width)),at=alpha.point)
            right=builder.value(label+"_eliminated_rhs%d"%k,tuple(
                right[i]-factors[i]*right[k] if i>k else right[i] for i in range(width)),at=alpha.point)
            for i in range(k+1,width):
                matrix[i]=builder.value(label+"_eliminated_H%d_%d"%(k,i),tuple(
                    matrix[i][j]-factors[i]*matrix[k][j] if j>k else 0.
                    for j in range(width)),at=alpha.point)
        delta=builder.value(label+"_delta_seed",ProgramExpression((0.,)*width,alpha),at=alpha.point)
        for k in range(width-1,-1,-1):
            solved=(right[k]-sum(matrix[k][j]*delta[j] for j in range(k+1,width)))/matrix[k][k]
            delta=builder.value(label+"_backsub%d"%k,tuple(solved if i==k else delta[i] for i in range(width)),at=alpha.point)
        updated=builder.value(label+"_Newton",tuple(alpha[j]-delta[j] for j in range(width)),at=alpha.point)
        return builder.product(alpha=updated,target=wanted)
    def stop(builder,state):
        defect=residual(builder,state)
        return builder.dot_all(defect,defect) < 1.e-24
    iteration=p.iterate(initial,step,stop,max_steps=24)
    alpha=p.require_success(iteration)["state"]["alpha"]
    f=distribution(alpha)
    defect=p.value(label+"_verified_moment_residual",tuple(moment(f,j)-target[j] for j in range(width)),at=target.point)
    flux=p.value(label+"_closed_flux",tuple(moment(f,j+1) for j in range(width)),at=target.point)
    return alpha,flux,defect


def build_m18_entropy(*,order=2,quadrature=6,cells=16,steps=2,dt=1.e-4):
    """Three/four moments, >=6 velocities, native Newton at each spatial RK4 stage."""
    if type(order) is not int or order not in (2,3):raise ValueError("reception supports entropy order2 or3")
    if type(quadrature) is not int or not 6<=quadrature<=12:raise ValueError("reception quadrature requires6..12 nodes")
    if type(cells) is not int or cells<8 or cells%2:raise ValueError("reception requires even cells>=8")
    if type(steps) is not int or not 2<=steps<=8 or not 0<dt<=1.e-4*16/cells:
        raise ValueError("smooth interior reception requires2..8 steps and small positive dt")
    nodes,weights=np.polynomial.legendre.leggauss(quadrature)
    width=order+1
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    frame=Rectangle("M18_entropy_torus",(0.,0.),(1.,1.)).frame(Cartesian2D())
    layout=Uniform(CartesianGrid(frame=frame,cells=(cells,1),periodic=PeriodicAxes(frame.axes)))
    representation=Representation("conservative",{
        "family":"discrete_entropy_moments",
        "order":order,"basis":"monomials","velocity_domain":(-1.,1.),
        "quadrature":"GaussLegendre","nodes":tuple(float(v) for v in nodes),"weights":tuple(float(w) for w in weights)})
    model=pops.Model("M18_discrete_entropy_order%d"%order,frame=frame)
    U=model.state("moments",components=tuple("M%d"%j for j in range(width)),
        representation=representation,sampling="cell_average")
    case=pops.Case("M18_entropy_M%d_Q%d"%(order,quadrature));block=case.block("moments",model)
    plan=DiscretizationPlan();plan.states.add(U,StateStorage());case.numerics(plan,block=block)
    p=pops.Program("entropy_Newton_centered_RK4");q=p.state(block[U])
    kernel=_central_kernel((cells,1),(1.,1.),0)
    def rhs(value,label):
        alpha,flux,defect=_newton_closure(p,value,nodes,weights,label=label)
        p.store_history(label+"_residual",defect,depth=1)
        space=FieldSpace(label+"_flux_derivative",components=value.space.components,
            representation="field",layout="cell",centering="cell",sampling="cell_center",
            frame=value.space.frame,clock=value.space.clock,support=value.space.support,
            units=value.space.units,representation_data=value.space.representation_data)
        derivative=p.spatial_interaction(flux,kernel,output_space=space,
            measure=CellVolumeMeasure(),quadrature=CellMidpoint(),
            realization=DirectSpatialInteraction(64<<20),source_scope="issued",name=label+"_D0flux")
        means=p.cell_mean(derivative,template=value,quadrature=CellMidpoint(),name=label+"_D0mean")
        return p.value(label+"_rhs",tuple(-means[j] for j in range(width)),at=value.point)
    half=p.stage("half",c=.5);end=p.stage("end",c=1.)
    a=rhs(q.n,"k1")
    b=rhs(p.value("stage2",q.n+.5*p.dt*a,at=half),"k2")
    c=rhs(p.value("stage3",q.n+.5*p.dt*b,at=half),"k3")
    d=rhs(p.value("stage4",q.n+p.dt*c,at=end),"k4")
    candidate=p.value("accepted",q.n+p.dt/6*(a+2*b+2*c+d),at=q.next.point)
    alpha,flux,defect=_newton_closure(p,candidate,nodes,weights,label="accepted")
    p.store_history("accepted_alpha",alpha,depth=1)
    p.store_history("accepted_flux",flux,depth=1)
    p.store_history("accepted_residual",defect,depth=1)
    p.commit(q.next,candidate);p.step_strategy(FixedDt(dt));case.program(p)
    x=(np.arange(cells)+.5)/cells
    dual=[-.7+.07*np.cos(2*np.pi*x),.12+.06*np.sin(2*np.pi*x),-.2+.04*np.cos(4*np.pi*x)]
    if order==3:dual.append(.08+.025*np.sin(4*np.pi*x))
    alpha0=np.stack(dual)[:,None,:]
    basis=np.stack([nodes**j for j in range(width)],axis=1)
    initial_f=np.exp(np.einsum("jyx,kj->kyx",alpha0,basis))
    initial=np.einsum("k,kyx,kj->jyx",weights,initial_f,basis)
    expected=reference_rk4(initial,nodes,weights,1/cells,dt,steps)
    def check(runtime):
        actual=state_array(runtime,"moments",initial.shape)
        # Every rank enters all history gathers before the owner-only analysis.
        labels=("k1_residual","k2_residual","k3_residual","k4_residual","accepted_residual","accepted_alpha","accepted_flux")
        raw={label:runtime.history_global(label,1) for label in labels}
        if actual is None:return non_owner_metrics()
        histories={label:np.asarray(value).reshape(initial.shape) for label,value in raw.items()}
        independent_alpha,independent_flux,oracle_residual=reference_closure(actual,nodes,weights)
        reconstructed_f=np.exp(np.einsum("jyx,kj->kyx",histories["accepted_alpha"],basis))
        reconstructed=np.einsum("k,kyx,kj->jyx",weights,reconstructed_f,basis)
        reconstructed_flux=np.einsum("k,kyx,k,kj->jyx",weights,reconstructed_f,nodes,basis)
        error=float(np.max(abs(actual-expected)))
        moment_error=float(np.max(abs(reconstructed-actual)))
        flux_error=float(max(np.max(abs(histories["accepted_flux"]-independent_flux)),
                             np.max(abs(reconstructed_flux-independent_flux))))
        alpha_error=float(np.max(abs(histories["accepted_alpha"]-independent_alpha)))
        stage_error=float(max(np.max(abs(histories[label])) for label in labels[:5]))
        inventory_error=float(np.max(abs(np.mean(actual-initial,axis=(-2,-1)))))
        nonuniform=float(np.max(np.ptp(actual,axis=-1)))
        highest_dual=float(np.min(abs(histories["accepted_alpha"][-1])))
        return dict(passed=bool(np.all(np.isfinite(actual)) and error<2.e-10 and moment_error<2.e-11
                    and flux_error<2.e-10 and alpha_error<2.e-9 and stage_error<2.e-11
                    and inventory_error<2.e-12 and nonuniform>1.e-3 and highest_dual>1.e-2),
            metrics_owner=True,native_discrete_error=error,all_moment_reconstruction_error=moment_error,
            independent_flux_error=flux_error,independent_dual_error=alpha_error,
            max_recorded_stage_residual=stage_error,inventory_error=inventory_error,
            reference_closure_residual=oracle_residual,spatial_nonuniformity=nonuniform,
            minimum_highest_dual_magnitude=highest_dual,order=order,quadrature_nodes=quadrature,
            spatial_cells=cells,steps=steps)
    return CaseSpec(case,layout,_initial(case,block[U],initial),{"moments":block[U]},steps*dt,dt,check,
        "M18 discrete entropy M%d on Gauss%d [-1,1], native full-Hessian Newton at every centered-space RK4 stage; smooth interior periodic profile, no continuous-integral, positivity or entropy-stability claim"%(order,quadrature))
