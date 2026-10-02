"""Explicit six-atom closure and straight-raw B_rho path; Source preparation."""
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.moments import CartesianMonomialBasis
from pops.moments.atomic_cubature import AtomicCubature
from pops.moments.polynomial_path import (EndpointPathInputs, NormalizedPathInputs,
                                         endpoint_polynomial_path, square_root)
from pops.numerics import (DiscretizationPlan, NormalizedPolynomialPath,
                          PathConservativeFiniteVolume, reconstruction, riemann, variables)
from pops.time import AdaptiveCFL

INDICES=((1,1),(0,0),(0,2),(1,0),(2,0),(0,1))
NODES=((0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1))
NAMES=('cross','mass','vertical_energy','horizontal_impulse','horizontal_energy','vertical_impulse')


def declarations(*, nonconservative=True, raw_speed=False):
    basis=CartesianMonomialBasis(INDICES)
    closure=AtomicCubature(indices=basis.indices,nodes=NODES)
    frame=Rectangle('cubature_box',(0.,0.),(2.,1.)).frame(Cartesian2D())
    x,y=frame.axes
    model=pops.Model('atomic_transport',frame=frame)
    state=model.state('population',components=NAMES)
    values=tuple(state[k] for k in range(6))
    physical={axis:tuple(closure.moment(values,(p+(j==0),q+(j==1))) for p,q in basis.indices)
              for j,axis in enumerate(frame.axes)}
    flux=model.flux('atomic_flux',state=state,frame=frame,components=physical)
    rho=state[basis.index((0,0))]
    strength=1. if nonconservative else 0.
    matrices={axis:tuple(tuple(strength*factor*rho if i==j else 0.*rho for j in range(6)) for i in range(6))
              for axis,factor in ((x,1.),(y,-.5))}
    product=model.nonconservative_product('density_transport',state=state,matrices=matrices)
    endpoint=NormalizedPathInputs(basis)
    pair=EndpointPathInputs(basis)
    raw=tuple(endpoint.raw(index) for index in basis.indices)
    gx,gy=endpoint.direction(0),endpoint.direction(1)
    declared_flux=tuple(gx*closure.moment(raw,(p+1,q))+gy*closure.moment(raw,(p,q+1)) for p,q in basis.indices)
    coupling=pair.direction(0)-pair.direction(1)/2
    integral=tuple(strength*coupling*(pair.left((0,0))+pair.right((0,0)))/2
                   *(pair.right(index)-pair.left(index)) for index in basis.indices)
    # Each atom has |vx|,|vy|<=1. The maximum endpoint bound certifies the
    # complete raw straight path since rho interpolates affinely and is positive.
    density_speed=endpoint.raw((0,0)) if raw_speed else endpoint.density
    speed=square_root(gx**2)+square_root(gy**2)+strength*square_root((gx-gy/2)**2)*density_speed
    plan=endpoint_polynomial_path(basis,flux=declared_flux,integral=integral,speed=speed)
    path=NormalizedPolynomialPath(product,frame=frame,covectors={x:(1,0),y:(0,1)},
                                 plan=plan,flux={axis.name:physical[axis] for axis in frame.axes},
                                 matrices=tuple(matrices[axis] for axis in frame.axes))
    return model,state,flux,product,path,basis


def make_case(*, nonconservative=True):
    model,state,flux,product,path,basis=declarations(nonconservative=nonconservative)
    rate=model.rate('full_balance',equation=ddt(state)==-div(flux)-product)
    method=PathConservativeFiniteVolume(flux=flux,path=path,variables=variables.Conservative(state),
                                       reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov())
    numerics=DiscretizationPlan();numerics.rates.add(rate,method)
    case=pops.Case('atomic_case');block=case.block('population',model)
    case.numerics(numerics,block=block)
    program=ForwardEuler(block[state],rate=rate)
    program.step_strategy(AdaptiveCFL(cfl=.25,max_dt=1e-3));case.program(program)
    case.initials.add(InitialCondition(state=block[state],value=BindArray(),projection=ConservativeCellAverage()))
    layout=Uniform(CartesianGrid(frame=path.frame,cells=(8,4),periodic=PeriodicAxes(path.frame.axes)))
    return case,layout
