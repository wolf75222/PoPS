"""M18/W09: one declared quadrature, finite dual and explicit domain refusals."""
import pops
from pops.frames import Cartesian2D
from pops.domain import Rectangle
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.layouts import Uniform
from pops.initial import InitialCondition
from pops.lib.initial import BindArray
from pops.projection import ConservativeCellAverage
from pops.moments.closures import DiscreteEntropyQuadrature, DiscreteEntropyCertificate
from pops.solvers.nonlinear import LocalNewton
from pops.time import LocalResidual, FailRun, FixedDt

NODES = (-.5, 0., .5)
WEIGHTS = (1/3, 1/3, 1/3)
QUADRATURE = DiscreteEntropyQuadrature(NODES, WEIGHTS, ((1.,)*3,NODES))
CERTIFICATES = tuple(DiscreteEntropyCertificate(QUADRATURE,c,label=label) for c,label in (
    ((1.,0.),'positive_mass'),((.5,-1.),'upper_support'),((.5,1.),'lower_support')))

def make_case():
    frame = Rectangle('domain',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model = pops.Model('multipliers',frame=frame)
    dual = model.state('dual',components=('a','b'))
    data = pops.Model('moment_data',frame=frame)
    target = data.state('target',components=('u0','u1'))
    case = pops.Case('quadrature_domain_witness')
    dual_block = case.block('unknown',model)
    a = dual_block[dual]
    m = case.block('prescribed',data)[target]
    P = pops.Program('declared_domain_then_dual_residual')
    A, M = P.state(a), P.state(m)
    seed = P.value('seed',A.n,at=A.next.point)
    # These are domain halfspaces supplied by the library composition, not
    # model-dependent compiler logic or a repair to the prescribed moment.
    for certificate in CERTIFICATES:
        seed = certificate.guard_finite_dual(P,seed,M.n)
    def residual(_P,unknowns,*,target):
        return {'dual':QUADRATURE.residual(unknowns['dual'],target)}
    result = P.solve(LocalResidual(residual,{'dual':seed},captures={'target':M.n}),
        solver=LocalNewton(tolerance=2e-11,max_iterations=40,safeguard='backtracking',max_backtracks=16,minimum_step=2**-16)).consume(action=FailRun())
    P.commit(A.next,result[dual_block])
    P.step_strategy(FixedDt(.01)); case.program(P)
    for subject in (a,m):
        case.initials.add(InitialCondition(state=subject,value=BindArray(),projection=ConservativeCellAverage()))
    return case,Uniform(CartesianGrid(frame=frame,cells=(2,2),periodic=PeriodicAxes(frame.axes))),(a,m)
