"""S04 HyQMOM: an order-parametric algebraic closure and bounded Dim1 FV cases.

Primary source: Fox--Laurent, https://arxiv.org/html/2103.10138, Algorithm 2,
Eqs. (27)--(28), (37)--(46), and sections 5.1--5.2. ``order`` denotes the
highest retained raw moment (2 or 4 in this reception); there are order+1
unknowns. No distribution reconstruction is used to close the production flux.

The native profile admits strictly positive Hankel matrices only. Vacuum,
exact rank-deficient boundary states (Appendix E), arbitrary orders beyond the
two received profiles, collision terms and continuum convergence are not
qualified by a finite native/discrete-reference comparison.
"""
from __future__ import annotations

from itertools import permutations

from examples.api_v02.common import CaseSpec, state_array, non_owner_metrics


def hyqmom_recurrence(raw):
    """Algorithm 2 on raw moments; Python loops elaborate scalar expressions.

    Returns the closed terminal moment and recurrence coefficients. Scalar
    objects may be Python numbers, arrays or PoPS expressions. Strict Hankel
    positivity is a caller-owned precondition, not a clipping rule.
    """
    raw = tuple(raw)
    order = len(raw)-1
    if order < 2 or order % 2:
        raise ValueError("HyQMOM requires moments M0 through an even order >=2")
    n = order//2
    sigma = {(k,p): 0 for k in range(-1,n+1) for p in range(2*n+2)}
    for p, value in enumerate(raw):
        sigma[0,p] = value
    a, b = [0]*(n+1), [0]*(n+1)
    a[0] = raw[1]/raw[0]
    for k in range(1,n):
        for p in range(k,2*n-k+1):
            sigma[k,p] = sigma[k-1,p+1]-a[k-1]*sigma[k-1,p]-b[k-1]*sigma[k-2,p]
        a[k] = sigma[k,k+1]/sigma[k,k]-sigma[k-1,k]/sigma[k-1,k-1]
        b[k] = sigma[k,k]/sigma[k-1,k-1]
    sigma[n,n] = sigma[n-1,n+1]-a[n-1]*sigma[n-1,n]-b[n-1]*sigma[n-2,n]
    b[n] = sigma[n,n]/sigma[n-1,n-1]
    a[n] = sum(a[:n])/n
    sigma[n,n+1] = sigma[n,n]*(a[n]+sigma[n-1,n]/sigma[n-1,n-1])
    for k in range(n-1,-1,-1):
        sigma[k,2*n-k+1] = (sigma[k+1,2*n-k]+a[k]*sigma[k,2*n-k]
                              + b[k]*sigma[k-1,2*n-k])
    return sigma[0,2*n+1], tuple(a), tuple(b)


def hyqmom_flux(raw):
    values = tuple(raw)
    return (*values[1:], hyqmom_recurrence(values)[0])


def hankel_margins(raw):
    """Leading raw Hankel determinants: strict Hamburger realizability, S04 §2.4."""
    raw = tuple(raw)
    margins = []
    for width in range(1,(len(raw)+1)//2+1):
        determinant = 0
        for permutation in permutations(range(width)):
            inversions = sum(permutation[i] > permutation[j]
                             for i in range(width) for j in range(i+1,width))
            term = (-1)**inversions
            for i,j in enumerate(permutation):
                term = term*raw[i+j]
            determinant = determinant+term
        margins.append(determinant)
    return tuple(margins)


def reference_flux(raw):
    """Independent central-moment formulas, S04 (4), §5.1 and §5.2.

    Does not call the production Chebyshev recurrence or interpret its IR.
    """
    import numpy as np
    raw = np.asarray(raw)
    rho, m1, m2 = raw[:3]
    u = m1/rho
    c2 = m2/rho-u*u
    if len(raw) == 3:
        terminal = rho*(u**3+3*u*c2)
    elif len(raw) == 5:
        m3,m4 = raw[3:]
        c3 = m3/rho-3*u*m2/rho+2*u**3
        c4 = m4/rho-4*u*m3/rho+6*u*u*m2/rho-3*u**4
        c5 = .5*c3*(5*c4/c2-3*c3*c3/(c2*c2)-c2)
        terminal = rho*(u**5+10*u**3*c2+10*u*u*c3+5*u*c4+c5)
    else:
        raise ValueError("independent closed-form reference covers only orders 2 and 4")
    return np.concatenate((raw[1:],terminal[None,...]),axis=0)


def reference_spectrum(raw):
    """Gram-system polynomial construction followed by symmetric Jacobi spectra.

    Independent of production Algorithm 2 and of its differentiated Jacobian.
    S04 (28), (39), (45)--(46); raw b has velocity-squared dimensions.
    """
    import numpy as np
    raw = np.asarray(raw,dtype=float)
    if raw.ndim != 1 or len(raw) not in (3,5) or not np.isfinite(raw).all():
        raise ValueError("one finite order-2 or order-4 moment vector required")
    n = (len(raw)-1)//2
    full = np.array([[raw[i+j] for j in range(n+1)] for i in range(n+1)])
    if np.linalg.eigvalsh(full).min() <= 0:
        raise ValueError("strictly realizable Hankel matrix required")
    polynomials, norms, a = [], [], []
    for k in range(n+1):
        if k == 0:
            q = np.array([1.])
        else:
            gram = full[:k,:k]
            q = np.append(np.linalg.solve(gram,-raw[k:2*k]),1.)
        square = np.convolve(q,q)
        norm = np.dot(square,raw[:len(square)])
        polynomials.append(q); norms.append(norm)
        if k < n:
            a.append(np.dot(square,raw[1:len(square)+1])/norm)
    b = [0.]+[norms[k]/norms[k-1] for k in range(1,n+1)]
    j = np.diag(a)
    for k in range(1,n):
        j[k-1,k] = j[k,k-1] = np.sqrt(b[k])
    k_matrix = np.zeros((n+1,n+1)); k_matrix[:n,:n] = j
    k_matrix[n,n] = sum(a)/n
    k_matrix[n-1,n] = k_matrix[n,n-1] = np.sqrt((2*n+1)*b[n]/n)
    return np.sort(np.concatenate((np.linalg.eigvalsh(j),np.linalg.eigvalsh(k_matrix))))


def reference_rhs(raw):
    import numpy as np
    spectra = np.stack([reference_spectrum(raw[:,i]) for i in range(raw.shape[1])])
    low, high = spectra[:,0],spectra[:,-1]
    right = np.roll(raw,-1,axis=1)
    lower = np.minimum(low,np.roll(low,-1))
    upper = np.maximum(high,np.roll(high,-1))
    left_flux,right_flux = reference_flux(raw),reference_flux(right)
    middle = (upper*left_flux-lower*right_flux+lower*upper*(right-raw))/(upper-lower)
    face = np.where(lower >= 0,left_flux,np.where(upper <= 0,right_flux,middle))
    return -raw.shape[1]*(face-np.roll(face,1,axis=1))


def initial_moments(order,cells,regime):
    """Exact cell averages of a non-Gaussian positive four-atom initial measure."""
    import numpy as np
    nodes = np.array([-1.1,-.25,.6,1.4])
    weights = np.array([.2,.25,.35,.2])[:,None]*np.ones((1,cells))
    if regime == "smooth":
        phase = 2*np.pi*(np.arange(cells)+.5)/cells
        weights *= 1+.12*np.sinc(1/cells)*np.sin(phase[None,:]+np.arange(4)[:,None])
    elif regime == "riemann":
        # Two separated periodic interfaces; no whole-line exact kinetic claim.
        weights[:,cells//2:] *= np.array([.6,1.3,.8,1.5])[:,None]
    else:
        raise ValueError("regime must be smooth or riemann")
    return np.stack([np.sum(weights*nodes[:,None]**k,axis=0) for k in range(order+1)])


def build_m15_orders(*,order=4,regime="smooth",cells=16,steps=4,dt=1e-4):
    """True Dim1 FV/HLL/Euler on M0..Morder, strict current/candidate guards."""
    import numpy as np
    import pops
    from pops import math as pm
    from pops.domain import CartesianDomain
    from pops.frames import Cartesian1D
    from pops.initial import InitialCondition
    from pops.lib.initial import BindArray
    from pops.projection import ConservativeCellAverage
    from pops.layouts import Uniform
    from pops.mesh import CartesianGrid,PeriodicAxes
    from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
    from pops.numerics.spatial import FiniteVolume
    from pops.time import FixedDt,FailRun
    if order not in (2,4) or type(order) is not int:
        raise ValueError("received HyQMOM orders are exactly 2 and 4")
    if type(cells) is not int or cells < 8 or cells % 2 or not 2 <= steps <= 32 or not 0 < dt <= 1e-4*16/cells:
        raise ValueError("bounded HyQMOM reception grid/time envelope exceeded")
    frame = CartesianDomain("M15_S04_interval",(0.,),(1.,)).frame(Cartesian1D())
    model = pops.Model("M15_S04_order%d"%order,frame=frame)
    u = model.state("M",components=tuple("M%d"%k for k in range(order+1)),sampling="cell_average")
    flux = model.flux("free_transport",state=u,frame=frame,
                      components={frame.axes[0]:hyqmom_flux(tuple(u))})
    rate = model.rate("moment_balance",equation=pm.ddt(u)==-pm.div(flux))
    model.wave_speeds_from_jacobian()
    case = pops.Case("M15_S04_%d_%s"%(order,regime)); block=case.block("moments",model)
    subject = block[u]
    case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    plan=DiscretizationPlan();plan.rates.add(rate,FiniteVolume(flux=flux,
        variables=variables.Conservative(u),reconstruction=reconstruction.FirstOrder(),riemann=riemann.HLL()))
    case.numerics(plan,block=block)
    p=pops.Program("M15_S04_Euler");state=p.state(subject)
    def guard(value,name):
        margins=hankel_margins(tuple(value[k] for k in range(order+1)))
        margin=margins[0]
        for next_margin in margins[1:]: margin=pm.minimum(margin,next_margin)
        field=p.value(name+"_hankel_margin",(margin,)*(order+1))
        return p.guard(name,value,p.min(field)>0.,action=FailRun())
    current=guard(state.n,"strict_current_domain")
    rate_value=rate(current)
    candidate=p.value("candidate",tuple(current[k]+p.dt*rate_value[k] for k in range(order+1)),at=state.next.point)
    p.commit(state.next,guard(candidate,"strict_candidate_domain"));p.step_strategy(FixedDt(dt));case.program(p)
    layout=Uniform(CartesianGrid(frame=frame,cells=(cells,),periodic=PeriodicAxes(frame.axes)))
    initial=initial_moments(order,cells,regime);expected=initial.copy()
    for _ in range(steps): expected=expected+dt*reference_rhs(expected)
    def check(runtime):
        actual=state_array(runtime,"moments",initial.shape)
        if actual is None: return non_owner_metrics()
        error=float(np.max(abs(actual-expected)))
        inventory=float(np.max(abs(np.mean(actual-initial,axis=1))))
        margin=float(np.min(np.stack(hankel_margins(actual))))
        spectrum=np.stack([reference_spectrum(actual[:,i]) for i in range(cells)])
        change=float(np.max(abs(actual-initial)))
        return dict(passed=bool(np.isfinite(actual).all() and error<2e-9 and inventory<2e-12
                               and margin>0 and change>1e-8),order=order,regime=regime,
                    native_discrete_error=error,inventory_error=inventory,minimum_hankel_determinant=margin,
                    minimum_spectral_gap=float(np.min(np.diff(spectrum,axis=1))),state_change=change)
    profile=("M15 S04 Algorithm2 order%d (%d raw moments), %s positive four-atom data; "
             "Dim1 periodic FV/HLL/forward Euler, fresh algebraic flux and differentiated native Jacobian; "
             "strict Hankel guard. No vacuum/exact boundary, collisions or continuum convergence qualification"%(order,order+1,regime))
    return CaseSpec(case,layout,{subject:initial},{"moments":subject},steps*dt,dt,check,profile)
