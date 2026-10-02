"""Public two-State authoring; second State has explicit degree-five binding."""
import numpy as np
from pops._ir.expr import Const
from tests.python.integration.runtime import test_program_affine_moment_explicit_basis_runtime as a
DUMMY=('unrelated_left','unrelated_right')
PARTITION=('spectator','population')

def build():
    frame=a.Rectangle('two_state_monomial_box',(0.,0.),(1.,1.)).frame(a.Cartesian2D())
    model=a.pops.Model('two_state_explicit_polynomial',frame=frame)
    dummy=model.species('earlier_state',state=DUMMY)
    selected=model.species('selected_state',state=a.NAMES)
    case=a.pops.Case('second_state_affine_case');blocks={};rates={}
    blocks['spectator']=case.block('spectator',model,states=(dummy,))
    state,label=selected,'population'
    flux=model.flux('zero_'+label,state=state,frame=frame,
        components={axis:tuple(0*v for v in state) for axis in frame.axes},
        waves={axis:tuple(Const(0.) for _ in state.components) for axis in frame.axes})
    rate=model.rate('stationary_'+label,equation=a.ddt(state)==-a.div(flux));rates[label]=rate
    plan=a.DiscretizationPlan();plan.rates.add(rate,a.FiniteVolume(flux=flux,
        variables=a.variables.Conservative(state),reconstruction=a.reconstruction.FirstOrder(),riemann=a.riemann.Rusanov()))
    block=case.block(label,model,states=(state,));case.numerics(plan,block=block);blocks[label]=block
    matrix=[[0.]*len(a.NAMES) for _ in a.NAMES]
    x,y=a.NAMES.index(a.BINDING[(1,0)]),a.NAMES.index(a.BINDING[(0,1)])
    matrix[x][y],matrix[y][x]=a.OMEGA,-a.OMEGA
    operator=model.operator('selected_skew',returns=model.local_linear_operator('selected_skew',on=selected,matrix=tuple(map(tuple,matrix))))
    program=a.pops.Program('second_state_explicit_affine')
    spectator=program.state(blocks['spectator'][dummy]);q=program.state(blocks['population'][selected])
    midpoint=program.solve(a.LocalLinear(operator=program.I-(program.dt/2)*program.linear_source(operator),rhs=q.n),solver=a.DenseLU()).consume(action=a.FailRun())
    mean=program.value('endpoint',2*midpoint-q.n,at=q.n.point)
    pushed=program.affine_moment_update(q.n,mean,linear_operator=operator,theta_dt=program.dt/2,basis=a.BASIS,components=a.BINDING)
    program.commit(spectator.next,program.value('spectator_unchanged',spectator.n,at=spectator.next.point))
    program.commit(q.next,program.value('accepted',pushed+program.dt*rates['population'](pushed),at=q.next.point))
    program.step_strategy(a.FixedDt(a.DT));case.program(program)
    transfer=a.AMRTransfer();subjects={}
    for label,state in (('spectator',dummy),('population',selected)):
        subject=blocks[label][state];subjects[label]=subject
        case.initials.add(a.InitialCondition(state=subject,value=a.BindArray(),projection=a.ConservativeCellAverage()))
        transfer.state(subject,a.StateTransfer())
    threshold=case.param(a.RuntimeParam('unused_refinement_threshold',default=1000.))
    layout=a.AMR(grid=a.CartesianGrid(frame=frame,cells=(a.N,a.N),periodic=a.PeriodicAxes(frame.axes)),
        hierarchy=a.AMRHierarchy(max_levels=1,ratios=()),
        tagging=a.AMRTagging(rules=(a.Tag(a.ValueExpr(subjects['population'])[a.BINDING[(0,0)]]>case.value(threshold)),a.Buffer(cells=0)),
            hysteresis=a.Hysteresis(0,a.EqualityPolicy.HOLD),conflict_policy=a.ConflictPolicy.REFINE_WINS),
        regrid=a.AMRRegrid(schedule=a.every(100,clock=program.clock)),transfer=transfer,execution=a.AMRExecution.synchronous())
    return case,layout,subjects,model,operator

def initials():
    # Distinct nonconstant dyadic spectator values; signed zero remains observable.
    x=np.arange(a.N,dtype=np.float64)[None,:];y=np.arange(a.N,dtype=np.float64)[:,None]
    dummy=np.stack((np.broadcast_to(3.+x/8,(a.N,a.N)),np.broadcast_to(-2.+y/16,(a.N,a.N))))
    dummy[1,0,0]=-0.
    return {'spectator':dummy,'population':a.initial('success')}
