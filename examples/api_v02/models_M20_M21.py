"""Coupled kinetic profiles using only finite-fibre algebra and ordinary Program nodes."""
from __future__ import annotations

from math import pi
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics, MissingCapabilityError


def build_m20(*,cells=16,velocity_cells=8,steps=4,dt=1e-5,frequency=.7):
    """Two-species multispecies Fokker--Planck with spatial transport.

    Cross-mixture u12,T12 are the symmetric pair relation of S14, eqs(1--20).
    Velocity FV is centered with zero boundary flux; its discrete total P/E
    defects are projected out by four explicit weighted linear constraints.
    This is a selected method, not a claim of the article's entropy theorem.
    """
    import numpy as np
    import pops
    from pops import math
    from pops._ir.quantity import PhysicalDimension
    from pops.fibres import FiberAxis,FiberRepresentation
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
    from pops.numerics.spatial import FiniteVolume
    from pops.time import FixedDt,FailRun
    if type(cells) is not int or cells<8 or type(velocity_cells) is not int or velocity_cells<4:
        raise ValueError("M20 requires a genuine spatial grid and at least four velocity cells")
    if steps<2 or not 0<dt<=1e-5*(8/velocity_cells)**2 or frequency<=0:
        raise ValueError("M20 explicit positive-domain reception envelope exceeded")
    edges=tuple(float(v) for v in np.linspace(-4.,4.,velocity_cells+1))
    v=tuple((a+b)/2 for a,b in zip(edges,edges[1:]));w=tuple(b-a for a,b in zip(edges,edges[1:]))
    vaxis=FiberAxis("v",v,w,PhysicalDimension(),"zero_flux",edges)
    velocity=FiberRepresentation((vaxis,))
    fibre=FiberRepresentation((FiberAxis("species",(1.,2.),(1.,1.),PhysicalDimension(),"zero_flux"),vaxis))
    masses=(1.,2.);width=fibre.size
    frame=Rectangle("M20_periodic",(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model("two_species_FP",frame=frame)
    f=model.state("distribution",components=fibre.components,representation=fibre.representation,sampling="cell_average")
    flux=model.flux("spatial_transport",state=f,frame=frame,
        components={frame.axes[0]:tuple(v[j%velocity_cells]*f[j] for j in range(width)),
                    frame.axes[1]:tuple(0.*f[j] for j in range(width))},
        waves={frame.axes[0]:v+v,frame.axes[1]:(0.,)*width})
    rate=model.rate("spatial_balance",equation=math.ddt(f)==-math.div(flux))
    case=pops.Case("M20_two_species_FP");block=case.block("distribution",model);subject=block[f]
    case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    plan=DiscretizationPlan();plan.rates.add(rate,FiniteVolume(flux=flux,variables=variables.Conservative(f),
        reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()));case.numerics(plan,block=block)
    p=pops.Program("joint_pair_collision_Euler");state=p.state(subject)
    def guarded(value,label):
        minimum=value[0]
        for j in range(1,width):minimum=math.minimum(minimum,value[j])
        margin=p.value(label+"_margin",fibre.pullback(minimum))
        return p.guard(label,value,p.min(margin)>0.,action=FailRun())
    current=guarded(state.n,"positive_distributions")
    values=fibre.values(current)
    species=tuple(values[s*velocity_cells:(s+1)*velocity_cells] for s in range(2))
    moments=tuple(tuple(velocity.moment(values_s,(k,)) for k in range(3)) for values_s in species)
    number=tuple(M[0] for M in moments);rho=tuple(m*n for m,n in zip(masses,number))
    mean=tuple(M[1]/M[0] for M in moments)
    temperature=tuple(m*(M[2]/M[0]-u*u) for m,M,u in zip(masses,moments,mean))
    total_rho_frequency=frequency*(rho[0]+rho[1])
    mixed_u=frequency*(rho[0]*mean[0]+rho[1]*mean[1])/total_rho_frequency
    delta=rho[0]*rho[1]*frequency**2/total_rho_frequency
    number_frequency=frequency*(number[0]+number[1])
    mixed_T=(frequency*(number[0]*temperature[0]+number[1]*temperature[1])
             +delta*(mean[0]-mean[1])**2)/number_frequency
    raw=[]
    for s,values_s in enumerate(species):
        raw.extend(frequency*x for x in velocity.flux_divergence(values_s,axis=0,
            flux=lambda left,right,face,s=s:(face-mixed_u)*(left+right)/2
                +mixed_T/masses[s]*(right-left)/(w[0])))
    # Constant quadrature Gram: constraints are each species' number, total
    # mass-weighted momentum, and total kinetic energy. No fitted coefficient.
    basis=np.array([[float(s==0),float(s==1),masses[s]*vj,.5*masses[s]*vj*vj]
                    for s in range(2) for vj in v],dtype=float).T
    weights=np.asarray(w+w)
    gram=(basis*weights[None,:])@basis.T
    inverse=np.linalg.inv(gram)
    defect=tuple(sum(float(weights[j]*basis[k,j])*raw[j] for j in range(width)) for k in range(4))
    alpha=tuple(sum(float(inverse[k,l])*defect[l] for l in range(4)) for k in range(4))
    collision=tuple(raw[j]-sum(float(basis[k,j])*alpha[k] for k in range(4)) for j in range(width))
    spatial=rate(current)
    candidate=p.value("next_distribution",tuple(current[j]+p.dt*(spatial[j]+collision[j]) for j in range(width)),at=state.next.point)
    p.commit(state.next,guarded(candidate,"positive_candidate"));p.step_strategy(FixedDt(dt));case.program(p)
    x=(np.arange(cells)+.5)/cells;va=np.asarray(v);wa=np.asarray(w)
    populations=[]
    for s,(u,T,n) in enumerate(((.25,1.,1.),(-.2,.7,.8))):
        base=np.exp(-masses[s]*(va-u)**2/(2*T));base/=wa@base
        populations.append(base[:,None]*n*(1+.04*np.cos(2*pi*x+.5*s))[None,:])
    initial=np.concatenate(populations,axis=0);expected=initial.copy();dx=1/cells;speed=max(map(abs,v))
    def rhs(data):
        f0,f1=np.split(data,2);Ns=np.array([wa@f0,wa@f1]);Ps=np.array([(wa*va)@f0,(wa*va)@f1]);Es=np.array([(wa*va**2)@f0,(wa*va**2)@f1])
        rs=np.asarray(masses)[:,None]*Ns;us=Ps/Ns;Ts=np.asarray(masses)[:,None]*(Es/Ns-us*us)
        uc=(rs*us).sum(axis=0)/rs.sum(axis=0);dc=frequency*rs[0]*rs[1]/rs.sum(axis=0)
        Tc=(frequency*(Ns*Ts).sum(axis=0)+dc*(us[0]-us[1])**2)/(frequency*Ns.sum(axis=0))
        raw_rows=[]
        for s,frow in enumerate((f0,f1)):
            faces=np.zeros((velocity_cells+1,cells))
            faces[1:-1]=(np.asarray(edges[1:-1])[:,None]-uc)*(frow[:-1]+frow[1:])/2+Tc/masses[s]*(frow[1:]-frow[:-1])/w[0]
            raw_rows.append(frequency*(faces[1:]-faces[:-1])/wa[:,None])
        C=np.concatenate(raw_rows,axis=0)
        C-=basis.T@np.linalg.solve(gram,(basis*weights[None,:])@C)
        vr=np.tile(va,2)[:,None];right=np.roll(data,-1,axis=1)
        face=.5*(vr*(data+right)-speed*(right-data))
        return -cells*(face-np.roll(face,1,axis=1))+C
    for _ in range(steps):expected+=dt*rhs(expected)
    def invariants(runtime):
        actual=state_array(runtime,"distribution",(width,cells))
        if actual is None:return non_owner_metrics()
        error=float(np.max(abs(actual-expected)))
        conservation=(basis*weights[None,:])@np.sum(actual-initial,axis=1)/cells
        defect=float(np.max(abs(conservation)))
        return dict(passed=error<5e-10 and defect<3e-12 and float(actual.min())>0,
                    native_discrete_error=error,joint_number_momentum_energy_defect=defect,
                    minimum=float(actual.min()),spatial_cells=cells,velocity_cells=velocity_cells,
                    species=2,steps=steps,entropy_claim=False,
                    velocity_method="centered FV plus explicit weighted invariant projection")
    layout=Uniform(CartesianGrid(frame=frame,cells=(cells,1),periodic=PeriodicAxes(frame.axes)))
    return CaseSpec(case,layout,{subject:initial[:,None,:]},{"distribution":subject},steps*dt,dt,invariants,
        "M20 two-species inhomogeneous 1x1v cross Fokker-Planck, symmetric mixture, explicit conservative discrete projection, positive-domain Euler")



def build_m21(*,cells=16,velocity_cells=4,mu_cells=2,steps=4,dt=2e-5,kperp=2.,mu0=.2):
    """Closed electromagnetic slab with an explicitly fixed perpendicular symbol.

    Full positive f_s(z,v_parallel,mu), constant B=J=1, quadrature J*2pi/m,
    Boussinesq polarization coefficient sum(m_s*n0_s), no ExB/curvature/mirror
    or collisions. k_perp² is a chosen surrogate operator, not a multidimensional
    spatial solve. Parallel induction remains coupled: its derivative constraint
    is eliminated using exactly the same finite D_v as the kinetic update.
    """
    import numpy as np
    import pops
    from pops import math
    from pops._ir.quantity import PhysicalDimension
    from pops.fibres import FiberAxis,FiberRepresentation
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
    from pops.numerics.spatial import FiniteVolume
    from pops.fields import SpatialInteractionKernel, CellVolumeMeasure, CellMidpoint, DirectSpatialInteraction
    from pops.model.spaces import FieldSpace
    from pops.time import FixedDt,FailRun
    if type(cells) is not int or cells<8 or cells%2 or type(velocity_cells) is not int or velocity_cells<4 or type(mu_cells) is not int or mu_cells<2:
        raise ValueError("EM slab requires real z cells and explicit v_parallel,mu fibres")
    if steps<2 or not 0<dt<=2e-5*(16/cells) or kperp<=0 or mu0<=0:
        raise ValueError("M21 finite-symbol positive-domain reception envelope exceeded")
    masses=(1.,2.);charges=(-1.,1.);n0=(1.,1.);B0=1.;J=B0
    ve=tuple(float(x) for x in np.linspace(-3.,3.,velocity_cells+1));v=tuple((a+b)/2 for a,b in zip(ve,ve[1:]));wv=tuple(b-a for a,b in zip(ve,ve[1:]))
    me=tuple(float(x) for x in np.linspace(0.,1.,mu_cells+1));mus=tuple((a+b)/2 for a,b in zip(me,me[1:]));wm=tuple(b-a for a,b in zip(me,me[1:]))
    va=FiberAxis("v_parallel",v,wv,PhysicalDimension(),"zero_flux",ve)
    ma=FiberAxis("mu",mus,wm,PhysicalDimension(),"zero_flux",me)
    fibre=FiberRepresentation((FiberAxis("species",(1.,2.),tuple(J*2*pi/m for m in masses),PhysicalDimension(),"zero_flux"),va,ma))
    velocity=FiberRepresentation((va,ma));per_species=velocity.size;width=fibre.size
    nodes=tuple(vj for vj in v for _ in mus);local_weights=tuple(dv*dm for dv in wv for dm in wm)
    measure=tuple(J*2*pi/m for m in masses)
    polarization=kperp*kperp*sum(m*n/B0**2 for m,n in zip(masses,n0))
    frame=Rectangle("M21_parallel_z",(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model("fixed_perpendicular_symbol_EM_slab",frame=frame)
    f=model.state("distribution",components=fibre.components,representation=fibre.representation,sampling="cell_average")
    flux=model.flux("parallel_streaming",state=f,frame=frame,
        components={frame.axes[0]:tuple(nodes[j%per_species]*f[j] for j in range(width)),frame.axes[1]:tuple(0.*f[j] for j in range(width))},
        waves={frame.axes[0]:nodes+nodes,frame.axes[1]:(0.,)*width})
    transport=model.rate("streaming_balance",equation=math.ddt(f)==-math.div(flux))
    case=pops.Case("M21_EM_slab_fixed_kperp");block=case.block("distribution",model);subject=block[f]
    case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    plan=DiscretizationPlan();plan.rates.add(transport,FiniteVolume(flux=flux,variables=variables.Conservative(f),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()));case.numerics(plan,block=block)
    p=pops.Program("compatible_slab_Ampere_Euler");state=p.state(subject)
    def positive(value,label):
        minimum=value[0]
        for j in range(1,width):minimum=math.minimum(minimum,value[j])
        field=p.value(label+"_margin",fibre.pullback(minimum))
        return p.guard(label,value,p.min(field)>0.,action=FailRun())
    current=positive(state.n,"positive_distribution")
    values=fibre.values(current)
    blocks=tuple(values[s*per_species:(s+1)*per_species] for s in range(2))
    number=tuple(measure[s]*velocity.moment(row,(0,0)) for s,row in enumerate(blocks))
    phi_scalar=sum(q*(n-background) for q,n,background in zip(charges,number,n0))/polarization
    phi=p.value("polarization_pullback",fibre.pullback(phi_scalar))
    # The finite sine kernel realizes -D0 exactly on this even periodic mesh.
    # This is an intermediate derivative field, not a conservative accepted
    # flux delivery; it therefore uses the explicit spatial field port.
    dx=1/cells
    derivative_modes=tuple((k,float(2*np.sin(2*pi*k/cells)/dx)) for k in range(1,cells//2))
    derivative_kernel=SpatialInteractionKernel(2,lambda x,y:sum(
        coefficient*math.sin(2*pi*k*(x[0]-y[0])) for k,coefficient in derivative_modes))
    derivative_space=FieldSpace("negative_parallel_gradient",components=current.space.components,
        frame=current.space.frame,clock=current.space.clock,support=current.space.support,
        sampling="cell_center",representation_data=current.space.representation_data)
    gradient_field=p.spatial_interaction(phi,derivative_kernel,output_space=derivative_space,
        measure=CellVolumeMeasure(),quadrature=CellMidpoint(),
        realization=DirectSpatialInteraction(32 << 20),source_scope="issued")
    minus_grad_phi=p.cell_mean(gradient_field,template=current,quadrature=CellMidpoint(),name="negative_parallel_gradient_mean")
    streaming=transport(current)
    dv=[]
    for row in blocks:
        dv.extend(velocity.flux_divergence(row,axis=0,flux=lambda left,right,face:(left+right)/2))
    star=tuple(streaming[j]-charges[j//per_species]/masses[j//per_species]*minus_grad_phi[0]*dv[j] for j in range(width))
    def integral_by_species(data,s,power):
        return measure[s]*velocity.moment(tuple(data[s*per_species:(s+1)*per_species]),(power,0))
    numerator=mu0*sum(charges[s]*integral_by_species(star,s,1) for s in range(2))
    denominator=kperp*kperp-mu0*sum(charges[s]**2/masses[s]*integral_by_species(dv,s,1) for s in range(2))
    denominator_field=p.value("Ampere_derivative_denominator",fibre.pullback(denominator))
    denominator_field=p.guard("invertible_discrete_Ampere_derivative",denominator_field,p.min(denominator_field)>0.,action=FailRun())
    adot=p.value("parallel_induction",fibre.pullback(numerator/denominator_field[0]))
    kinetic_rate=tuple(star[j]+charges[j//per_species]/masses[j//per_species]*adot[0]*dv[j] for j in range(width))
    candidate=p.value("candidate",tuple(current[j]+p.dt*kinetic_rate[j] for j in range(width)),at=state.next.point)
    candidate=positive(candidate,"positive_candidate")
    # Ampere initializes A and constrains every accepted state by construction.
    # The *independent* rate residual tests the eliminated derivative relation.
    residual=kperp*kperp*adot[0]-mu0*sum(charges[s]*integral_by_species(kinetic_rate,s,1) for s in range(2))
    p.store_history("Ampere_rate_residual",p.value("original_derivative_residual",fibre.pullback(residual)),depth=1)
    p.store_history("A_dot",adot,depth=1)
    accepted_A=mu0*sum(charges[s]*integral_by_species(tuple(candidate[j] for j in range(width)),s,1) for s in range(2))/(kperp*kperp)
    p.store_history("A_parallel",p.value("Ampere_accepted_field",fibre.pullback(accepted_A),at=state.next.point),depth=1)
    p.commit(state.next,candidate);p.step_strategy(FixedDt(dt));case.program(p)
    z=(np.arange(cells)+.5)/cells;vflat=np.asarray(nodes);weights=np.asarray(local_weights)
    initial=[]
    for s in range(2):
        row=np.empty((per_species,cells))
        for j,zj in enumerate(z):
            u=(.03 if s==0 else -.02)*np.sin(2*pi*zj)
            base=np.array([np.exp(-masses[s]*(vj-u)**2/2-mu) for vj in v for mu in mus])
            base/=measure[s]*np.dot(weights,base)
            row[:,j]=base*(1+(.03 if s==0 else -.02)*np.cos(2*pi*zj))
        initial.append(row)
    initial=np.concatenate(initial);expected=initial.copy();speed=max(map(abs,nodes));dx=1/cells
    def rates(data):
        rows=np.split(data,2);N=np.array([measure[s]*(weights@row) for s,row in enumerate(rows)])
        phi0=np.sum(np.asarray(charges)[:,None]*(N-np.asarray(n0)[:,None]),axis=0)/polarization
        grad=(np.roll(phi0,-1)-np.roll(phi0,1))/(2*dx)
        vf=[]
        for row in rows:
            matrix=row.reshape(velocity_cells,mu_cells,cells)
            faces=np.zeros((velocity_cells+1,mu_cells,cells));faces[1:-1]=(matrix[:-1]+matrix[1:])/2
            vf.append(((faces[1:]-faces[:-1])/np.asarray(wv)[:,None,None]).reshape(per_species,cells))
        dv0=np.concatenate(vf);right=np.roll(data,-1,axis=1);vr=np.tile(vflat,2)[:,None]
        faces=.5*(vr*(data+right)-speed*(right-data));stream=-cells*(faces-np.roll(faces,1,axis=1))
        star0=stream+np.repeat(np.asarray(charges)/np.asarray(masses),per_species)[:,None]*grad*dv0
        def integ(arr,s):return measure[s]*((weights*vflat)@arr[s*per_species:(s+1)*per_species])
        num=mu0*sum(charges[s]*integ(star0,s) for s in range(2))
        den=kperp*kperp-mu0*sum(charges[s]**2/masses[s]*integ(dv0,s) for s in range(2))
        a=num/den
        r=star0+np.repeat(np.asarray(charges)/np.asarray(masses),per_species)[:,None]*a*dv0
        residual0=kperp*kperp*a-mu0*sum(charges[s]*integ(r,s) for s in range(2))
        return r,a,residual0
    for _ in range(steps):expected+=dt*rates(expected)[0]
    def invariants(runtime):
        actual=state_array(runtime,"distribution",(width,cells))
        native_residual=runtime.history_global("Ampere_rate_residual",1) if hasattr(runtime,"history_global") else None
        native_adot=runtime.history_global("A_dot",1) if hasattr(runtime,"history_global") else None
        native_A=runtime.history_global("A_parallel",1) if hasattr(runtime,"history_global") else None
        if actual is None:return non_owner_metrics()
        error=float(np.max(abs(actual-expected)))
        expected_A=mu0*sum(charges[s]*measure[s]*((weights*vflat)@actual[s*per_species:(s+1)*per_species]) for s in range(2))/(kperp*kperp)
        field_defect=float(np.max(abs(np.asarray(native_A).reshape(width,cells)[0]-expected_A))) if native_A is not None else float("inf")
        defect=float(np.max(abs(native_residual))) if native_residual is not None else float("inf")
        amplitude=float(np.max(abs(native_adot))) if native_adot is not None else 0.
        number_defects=[abs(measure[s]*np.sum(weights[:,None]*(actual[s*per_species:(s+1)*per_species]-initial[s*per_species:(s+1)*per_species]))/cells) for s in range(2)]
        return dict(passed=error<5e-10 and defect<3e-12 and field_defect<3e-12 and max(number_defects)<3e-12 and amplitude>1e-5 and actual.min()>0,
            native_discrete_error=error,discrete_Ampere_derivative_residual=defect,Ampere_constraint_residual=field_defect,induction_amplitude=amplitude,
            species_number_defect=float(max(number_defects)),spatial_cells=cells,velocity_cells=velocity_cells,
            mu_cells=mu_cells,full_f_3D_claim=False,perpendicular_operator="fixed positive k_perp squared symbol surrogate",
            Alfven_dispersion_qualification=False)
    layout=Uniform(CartesianGrid(frame=frame,cells=(cells,1),periodic=PeriodicAxes(frame.axes)))
    return CaseSpec(case,layout,{subject:initial[:,None,:]},{"distribution":subject},steps*dt,dt,invariants,
        "M21 closed EM slab f_s(z,v_parallel,mu), fixed k_perp symbol surrogate and Boussinesq polarization; same-discrete-Dv Ampere derivative elimination; not full-f 3D/tokamak or DG qualification")
