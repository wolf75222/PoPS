"""Finite closed moment-PDE profiles; scientific formula libraries elaborate to IR."""
from __future__ import annotations

from math import pi
from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics


def _periodic_case(name, model, state, flux, *, cells, initial, rhs, domain, steps, dt,
                   pointwise_source=None):
    import numpy as np
    import pops
    from pops import math
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume
    from pops.time import FixedDt, FailRun
    rate = model.rate("spatial_balance",equation=math.ddt(state)==-math.div(flux))
    case = pops.Case(name); block = case.block("moments",model); subject = block[state]
    case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    plan = DiscretizationPlan();plan.rates.add(rate,FiniteVolume(flux=flux,
        variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.HLL()))
    case.numerics(plan,block=block)
    p = pops.Program(name+"_Euler");u = p.state(subject)
    def guard(value,label):
        margins = tuple(domain(tuple(value[j] for j in range(len(state)))))
        margin = margins[0]
        for candidate in margins[1:]:
            margin = math.minimum(margin,candidate)
        field = p.value(label+"_margin",(margin,)*len(state))
        return p.guard(label,value,p.min(field)>0.,action=FailRun())
    current = guard(u.n,"current_domain")
    source = (0.,)*len(state) if pointwise_source is None else pointwise_source(tuple(current[j] for j in range(len(state))))
    k = rate(current)
    candidate = p.value("candidate",tuple(current[j]+p.dt*(k[j]+source[j]) for j in range(len(state))),at=u.next.point)
    p.commit(u.next,guard(candidate,"candidate_domain"));p.step_strategy(FixedDt(dt));case.program(p)
    layout = Uniform(CartesianGrid(frame=model.frame,cells=(cells,1),periodic=PeriodicAxes(model.frame.axes)))
    expected = initial.copy()
    for _ in range(steps):
        expected += dt*rhs(expected)
    def invariants(runtime):
        actual = state_array(runtime,"moments",initial.shape)
        if actual is None: return non_owner_metrics()
        error = float(np.max(abs(actual-expected)))
        inventory_error = float(np.max(abs(np.sum(actual-expected,axis=1)/cells)))
        return dict(passed=error<2e-8 and inventory_error<2e-10,
                    native_discrete_error=error,inventory_vs_discrete_reference=inventory_error,
                    spatial_cells=cells,steps=steps)
    return CaseSpec(case,layout,{subject:initial[:,None,:]},{"moments":subject},steps*dt,dt,invariants,name)


def _frame(name):
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    return Rectangle(name,(0.,0.),(1.,1.)).frame(Cartesian2D())


def build_m15(*,cells=16,steps=4,dt=1e-4):
    """The five-moment axial Appendix-B.1 HyQMOM variant; order is explicit."""
    import pops
    from pops.moments import hyqmom_b1_axial_flux
    from examples.migration.scientific.api040_m15_axial_oracle import initial_averages, rhs, domain_values
    if cells<8 or steps<2 or not 0<dt<=1e-4*(16/cells): raise ValueError("M15 reception envelope exceeded")
    frame = _frame("M15_interval")
    model = pops.Model("M15_axial_B1_order4",frame=frame)
    U = model.state("M",components=tuple("M%d"%j for j in range(5)),sampling="cell_average")
    physical = hyqmom_b1_axial_flux(tuple(U))
    flux = model.flux("transport",state=U,frame=frame,
        components={frame.axes[0]:tuple(physical),frame.axes[1]:tuple(0.*value for value in U)})
    model.wave_speeds_from_jacobian()
    def domain(values):
        r,a,b,c,d=values
        return r,r*b-a*a,r*(b*d-c*c)-a*(a*d-b*c)+b*(a*c-b*b)
    return _periodic_case("M15 five axial B1 moments, x-dependent genuine Dim2 FV HLL",model,U,flux,
        cells=cells,initial=initial_averages(cells),rhs=lambda value:rhs(value)[0],domain=domain,steps=steps,dt=dt)


def _m18_flux(values, backend):
    """Exact entropy inverse on nodes (-1,0,+1), positive quadrature weights 1.

    n=A(1+x+1/x), m=A(x-1/x), x=exp(alpha1)>0. The positive
    quadratic root determines x; A=exp(alpha0), hence flux=(m,n-A).
    """
    n,m=values
    r=m/n
    x=(r+backend.sqrt(4-3*r*r))/(2*(1-r))
    A=n/(1+x+1/x)
    return m,n-A


def build_m18(*,cells=16,steps=4,dt=1e-4):
    """Discrete M1 minimum-entropy transport, three velocity nodes, exact dual inverse."""
    import numpy as np
    import pops
    from pops import math
    if cells<8 or steps<2 or not 0<dt<=1e-4*(16/cells): raise ValueError("M18 reception envelope exceeded")
    frame = _frame("M18_entropy_interval")
    model = pops.Model("M18_three_node_entropy_M1",frame=frame)
    U = model.state("moments",components=("number","current"),sampling="cell_average")
    physical = _m18_flux(tuple(U),math)
    flux = model.flux("entropy_transport",state=U,frame=frame,
        components={frame.axes[0]:physical,frame.axes[1]:(0.*U[0],0.*U[1])})
    model.wave_speeds_from_jacobian()
    x=(np.arange(cells)+.5)/cells
    alpha0=-1+.08*np.cos(2*pi*x); alpha1=.1+.05*np.sin(2*pi*x)
    initial=np.stack((np.exp(alpha0)*(1+2*np.cosh(alpha1)),2*np.exp(alpha0)*np.sinh(alpha1)))
    def physical_np(value): return np.stack(_m18_flux(value,np))
    def speeds(value):
        columns=[]
        for j in range(2):
            shifted=value.astype(complex);shifted[j]+=1e-30j
            columns.append(physical_np(shifted).imag/1e-30)
        J=np.moveaxis(np.stack(columns,axis=1),(0,1),(-2,-1))
        spectrum=np.linalg.eigvals(J)
        if np.max(abs(spectrum.imag))>1e-11: raise ValueError("entropy spectrum nonreal")
        return spectrum.real.min(axis=-1),spectrum.real.max(axis=-1)
    def rhs(value):
        lower,upper=speeds(value);right=np.roll(value,-1,axis=1)
        lo=np.minimum(lower,np.roll(lower,-1));hi=np.maximum(upper,np.roll(upper,-1))
        fl,fr=physical_np(value),physical_np(right)
        middle=(hi*fl-lo*fr+lo*hi*(right-value))/(hi-lo)
        face=np.where(lo>=0,fl,np.where(hi<=0,fr,middle))
        return -cells*(face-np.roll(face,1,axis=1))
    return _periodic_case("M18 discrete three-node M1 entropy closure with exact positive dual inversion, periodic FV HLL",
        model,U,flux,cells=cells,initial=initial,rhs=rhs,domain=lambda q:(q[0],q[0]-abs(q[1])),steps=steps,dt=dt)


def build_m17(*,cells=16,steps=4,dt=1e-4):
    """Full DF+B Fan--Li15 with a declared raw-moment path and Gauss4 rule."""
    import numpy as np
    import pops
    from pops import math
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid, PeriodicAxes
    from pops.moments.fan_li import FAN_LI15_INDICES, fan_li15_expressions
    from pops.numerics import DiscretizationPlan, PathConservativeFiniteVolume, SymbolicPath, reconstruction, riemann, variables
    from pops.time import FixedDt, FailRun
    from examples.migration.scientific.api040_m17_oracle import gaussian_mixture, semidiscrete_rhs
    if cells<8 or steps<2 or not 0<dt<=1e-4*(16/cells): raise ValueError("M17 reception envelope exceeded")
    indices=tuple(FAN_LI15_INDICES);slot={pq:k for k,pq in enumerate(indices)}
    frame=_frame("M17_periodic")
    model=pops.Model("FanLi15_full_principal_system",frame=frame)
    U=model.state("moments",components=tuple("M%d%d"%pq for pq in indices),sampling="cell_average")
    physical=fan_li15_expressions(tuple(U))
    flux=model.flux("conservative_transport",state=U,frame=frame,
        components={frame.axes[0]:physical.x,frame.axes[1]:physical.y})
    product=model.nonconservative_product("temperature_regularization",state=U,
        matrices={frame.axes[0]:physical.directional_nonconservative_matrix((1,0)),
                  frame.axes[1]:physical.directional_nonconservative_matrix((0,1))},
        conservative_components=tuple("M%d%d"%pq for pq in indices if sum(pq)<4))
    def path_speed(left,right,axis):
        second=slot[(2,0) if axis==0 else (0,2)]
        return math.sqrt(6+math.sqrt(10))*math.sqrt(math.maximum(
            left[second]/left[0],right[second]/right[0]))
    quadrature=((.06943184420297371,.17392742256872692),(.33000947820757187,.32607257743127305),
                (.6699905217924281,.32607257743127305),(.9305681557970262,.17392742256872692))
    path=SymbolicPath(product,frame=frame,quadrature=quadrature,speed=path_speed)
    rate=model.rate("full_balance",equation=math.ddt(U)==-math.div(flux)-product)
    plan=DiscretizationPlan();plan.rates.add(rate,PathConservativeFiniteVolume(flux=flux,path=path,
        variables=variables.Conservative(U),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    case=pops.Case("M17_full_FanLi15");block=case.block("moments",model);subject=block[U]
    case.numerics(plan,block=block)
    case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    p=pops.Program("FanLi_raw_path_Euler");u=p.state(subject)
    def guard(value,label):
        rho_field=p.value(label+"_density",(value[0],)*15)
        value=p.guard(label+"_density_positive",value,p.min(rho_field)>0.,action=FailRun())
        rho=value[0]; ux=value[slot[1,0]]/rho;uy=value[slot[0,1]]/rho
        a=value[slot[2,0]]/rho-ux*ux;b=value[slot[1,1]]/rho-ux*uy;c=value[slot[0,2]]/rho-uy*uy
        margin=math.minimum(rho,math.minimum(a,a*c-b*b))
        margin_field=p.value(label+"_margin",(margin,)*15)
        return p.guard(label,value,p.min(margin_field)>0.,action=FailRun())
    current=guard(u.n,"positive_SPD_current")
    k=rate(current)
    candidate=p.value("next_moments",current+p.dt*k,at=u.next.point)
    p.commit(u.next,guard(candidate,"positive_SPD_candidate"))
    p.step_strategy(FixedDt(dt));case.program(p)
    x=(np.arange(cells)+.5)/cells
    initial=np.stack([gaussian_mixture(((.55+.02*np.cos(2*pi*xj),(-.2,.1),(.7,.05,.6)),
                                      (.45-.015*np.sin(2*pi*xj),(.3,-.15),(.5,-.04,.8)))) for xj in x],axis=1)
    expected=initial.copy()
    for _ in range(steps): expected += dt*semidiscrete_rhs(0.,expected.ravel()).reshape(15,cells)
    conserved=tuple(j for j,pq in enumerate(indices) if sum(pq)<4)
    def invariants(runtime):
        actual=state_array(runtime,"moments",(15,cells))
        if actual is None:return non_owner_metrics()
        error=float(np.max(abs(actual-expected)))
        inventory=float(np.max(abs(np.sum(actual[list(conserved)]-initial[list(conserved)],axis=1)/cells)))
        return dict(passed=error<3e-8 and inventory<3e-12,native_discrete_error=error,
                    conservative_lower_moment_defect=inventory,spatial_cells=cells,steps=steps,
                    physical_operator="DF+B",path="straight raw moments",native_path_quadrature="Gauss4",
                    independent_path_quadrature="Gauss24")
    layout=Uniform(CartesianGrid(frame=frame,cells=(cells,1),periodic=PeriodicAxes(frame.axes)))
    return CaseSpec(case,layout,{subject:initial[:,None,:]},{"moments":subject},steps*dt,dt,invariants,
        "M17 full Fan-Li15 D2M4 tensor-temperature system, raw path Gauss4 FV Rusanov, explicit Euler on genuine x cells")


def _b1_reference_flux(values,normal=(1.,0.)):
    """Numerical binomial translation oracle, no PoPS expression evaluator."""
    import numpy as np
    from math import comb
    indices=tuple((p,q) for q in range(5) for p in range(5-q))
    raw=dict(zip(indices,values));rho=raw[0,0];u,v=raw[1,0]/rho,raw[0,1]/rho
    central={(p,q):sum(comb(p,i)*comb(q,j)*(-u)**(p-i)*(-v)**(q-j)*raw[i,j]
                      for i in range(p+1) for j in range(q+1)) for p,q in indices}
    sx=np.sqrt(central[2,0]/rho);sy=np.sqrt(central[0,2]/rho)
    S={(p,q):value/(rho*sx**p*sy**q) for (p,q),value in central.items()}
    a,b,c=S[3,0],S[0,3],S[1,1];d,e,f,g,h,j,k=S[4,0],S[0,4],S[2,1],S[1,2],S[3,1],S[1,3],S[2,2]
    fifth={(5,0):a*(5*d-3*a*a-1)/2,
        (4,1):-a*(8*d-9*a*a-4)*c/4+(10*d-15*a*a-6)*f/4+2*a*h,
        (3,2):(2*d-3*a*a)*g/2+(3*k-1)*a/2,
        (2,3):(2*e-3*b*b)*f/2+(3*k-1)*b/2,
        (1,4):-b*(8*e-9*b*b-4)*c/4+(10*e-15*b*b-6)*g/4+2*b*j,
        (0,5):b*(5*e-3*b*b-1)/2}
    for (p,q),value in fifth.items():central[p,q]=rho*sx**p*sy**q*value
    closed=dict(raw)
    for p,q in fifth:
        closed[p,q]=sum(comb(p,i)*comb(q,j)*u**(p-i)*v**(q-j)*central[i,j]
                        for i in range(p+1) for j in range(q+1))
    return np.stack([normal[0]*closed[p+1,q]+normal[1]*closed[p,q+1] for p,q in indices])


def _m16_model(*,oblique_refusal=False):
    """The B.1 fifteen-moment model of M16; its signed wave speeds are the eigenvalues of the
    flux Jacobian along each axis, which have no closed form."""
    import pops
    from pops.moments import moment_flux_expressions,HyQMOM15Closure
    indices=tuple((p,q) for q in range(5) for p in range(5-q))
    frame=_frame("M16_periodic");model=pops.Model("HyQMOM15_original_B1",frame=frame)
    U=model.state("moments",components=tuple("M%d%d"%pq for pq in indices),sampling="cell_average")
    physical=moment_flux_expressions(model,tuple(U),4,HyQMOM15Closure(),robust=False)
    directions=((1.,-1.),(1.,1.)) if oblique_refusal else ((1.,0.),(0.,1.))
    flux=model.flux("B1_transport",state=U,frame=frame,components={axis:tuple(
        normal[0]*fx+normal[1]*fy for fx,fy in zip(physical.x,physical.y)) for axis,normal in zip(frame.axes,directions)})
    model.wave_speeds_from_jacobian()
    return model,U,flux,indices


def build_m16(*,cells=16,steps=4,dt=1e-5,omega=.2,oblique_refusal=False):
    """Complete B.1 fifteen-moment flux plus constant magnetic Lorentz source.

    The ordinary profile evaluates Cartesian x,y faces and checks the *actual*
    Jacobian spectrum in native HLL. It asserts no multidirectional theorem.
    oblique_refusal=True supplies the correlated Gaussian whose DFx-DFy has
    nonreal eigenvalues: a successful native evolution must fail reception.
    """
    import numpy as np
    from examples.migration.scientific.api040_m17_oracle import gaussian_mixture
    if cells<8 or steps<2 or not 0<dt<=1e-5*(16/cells):raise ValueError("M16 reception envelope exceeded")
    model,U,flux,indices=_m16_model(oblique_refusal=oblique_refusal);slot={pq:j for j,pq in enumerate(indices)}
    def lorentz(values):
        raw=dict(zip(indices,values))
        return tuple(omega*(p*raw.get((p-1,q+1),0)-q*raw.get((p+1,q-1),0)) for p,q in indices)
    x=(np.arange(cells)+.5)/cells
    if oblique_refusal:
        gaussian=np.array([1,0,1,0,3,0,.5,0,1.5,1,0,1.5,0,1.5,3],dtype=float)
        initial=gaussian[:,None]*(1+.01*np.cos(2*pi*x))[None,:]
    else:
        initial=np.stack([gaussian_mixture(((.55+.02*np.cos(2*pi*xx),(-.2,.1),(.5,0.,.7)),
            (.45-.015*np.sin(2*pi*xx),(.3,-.1),(.7,0.,.5)))) for xx in x],axis=1)
    def rhs(value):
        # This branch exists only in the test oracle. The rejected profile has
        # no accepted hyperbolic reference evolution; its invariant always fails.
        if oblique_refusal:return np.zeros_like(value)
        columns=[]
        for j in range(15):
            perturbed=value.astype(complex);perturbed[j]+=1e-25j
            columns.append(_b1_reference_flux(perturbed).imag/1e-25)
        jac=np.moveaxis(np.stack(columns,axis=1),(0,1),(-2,-1))
        spectrum=np.linalg.eigvals(jac)
        if np.max(abs(spectrum.imag))>2e-8:raise ValueError("selected B1 Cartesian reception state is nonhyperbolic")
        lower,upper=spectrum.real.min(axis=-1),spectrum.real.max(axis=-1)
        right=np.roll(value,-1,axis=1);lo=np.minimum(lower,np.roll(lower,-1));hi=np.maximum(upper,np.roll(upper,-1))
        fl,fr=_b1_reference_flux(value),_b1_reference_flux(right)
        middle=(hi*fl-lo*fr+lo*hi*(right-value))/(hi-lo)
        face=np.where(lo>=0,fl,np.where(hi<=0,fr,middle))
        return -cells*(face-np.roll(face,1,axis=1))+np.stack([np.broadcast_to(item,value.shape[1:]) for item in lorentz(value)])
    def domain(values):
        r=values[0];xx=r*values[slot[2,0]]-values[slot[1,0]]**2
        yy=r*values[slot[0,2]]-values[slot[0,1]]**2;xy=r*values[slot[1,1]]-values[slot[1,0]]*values[slot[0,1]]
        return r,xx,yy,xx*yy-xy*xy
    spec=_periodic_case("M16 original B1 fifteen-moment transport + Lorentz; conditional Cartesian spectral admission",
        model,U,flux,cells=cells,initial=initial,rhs=rhs,domain=domain,steps=steps,dt=dt,pointwise_source=lorentz)
    if oblique_refusal:
        from dataclasses import replace
        return replace(spec,profile="M16 negative: original B1 correlated Gaussian, nonreal DFx-DFy spectrum; native refusal required",
            invariants=lambda runtime:dict(passed=False,error="nonhyperbolic B1 oblique evolution was incorrectly accepted"))
    return spec
