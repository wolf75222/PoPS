"""Public isolated self-species BGK; velocity moments are explicit inputs."""
from fractions import Fraction
import pops
from pops.model import Module, PhysicalSupport, PhysicalDimension
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.fields import AnalyticAux, AuxiliaryBoundary
from pops.analytic import coordinate
from pops.math import Var, exp
from pops.time import FixedDt
from pops.layouts import Uniform
from pops.mesh import (CartesianGrid, PeriodicAxes, PhysicalSupportMap, AxisQuadrature,
    LayoutPlanBuilder, LayoutMappingOperation, LayoutRepresentation, LayoutSynchronization,
    native_physical_mapping)

UNIT=PhysicalDimension()
PHASE=PhysicalSupport((('velocity','finite velocity fibre'),('position','periodic position')))
POSITION=PhysicalSupport((('position','periodic position'),))
DT=Fraction(1,128)
NU=Fraction(3,2)
NAMES=('distribution_data','number_data','momentum_data','energy_data',
       'reduced_number','reduced_momentum','reduced_energy','unrelated_data')

def collision_module(frame, *, name='declared isolated collision', nu=Fraction(3,2)):
    m=pops.Model(name,frame=frame)
    states=tuple(m.species(n,state=(c,),support=PHASE,units=(UNIT,),sampling='cell_average')
                 for n,c in (('distribution','population'),('number','density'),
                             ('momentum','first_moment'),('energy','second_moment')))
    f,n,p,e=(s[0] for s in states)
    module=m.module
    a=module.aux_field('velocity_coordinate','cell_scalar',frame=frame.canonical_id,unit='1')
    v=Var('velocity_coordinate','aux')
    u=p/n
    variance=e/n-u*u
    # Original self-BGK Maxwellian, not a discrete moment-fit or renormalized equilibrium.
    from math import pi
    equilibrium=n/(2*pi*variance)**Fraction(1,2)*exp(-(v-u)**2/(2*variance))
    collision=m.coupled_rate('self_collision',inputs=states,
                             outputs={states[0]:(nu*(equilibrium-f),)})
    module.aux_provider(AnalyticAux(module.aux_handle(a),
        coordinate(frame,frame.axes[0]),frame=frame,
        boundary=AuxiliaryBoundary(width=1,kind='foextrap')))
    return m,states,collision

def fixed_moment_counterexample(*, reverse=False):
    frame=Rectangle('phase velocity then position',(-8,0),(8,1)).frame(Cartesian2D())
    m,states,collision=collision_module(frame)
    case=pops.Case('isolated collision source with explicit moment inputs')
    keys=('distribution_data','number_data','momentum_data','energy_data')
    order=tuple(reversed(range(4))) if reverse else range(4)
    blocks={i:case.block(keys[i],m,states=(states[i],)) for i in order}
    subjects=tuple(blocks[i][s] for i,s in enumerate(states))
    program=pops.Program('explicit BGK relaxation')
    state=tuple(program.state(s) for s in subjects)
    rate=collision(*(s.n for s in state))[blocks[0]]
    program.commit(state[0].next,program.value('accepted population',state[0].n+program.dt*rate,
                                             at=state[0].next.point))
    program.step_strategy(FixedDt(1/128));case.program(program)
    layout=Uniform(CartesianGrid(frame=frame,cells=(32,4),periodic=PeriodicAxes(frame.axes)))
    return pops.resolve(pops.validate(case),layout=layout)


def build(directory, *, nx=32, nv=32, reverse=False, scope='isolated self species',
          provider_factory=native_physical_mapping):
    """Public product-space self-BGK, with three reduce/lift chains at EACH stage.

    There is no transport in this isolated relaxation. Finite velocity faces carry
    no transport term; periodic position does not enter the local collision law.
    """
    phase_frame=Rectangle('phase velocity then position',(-8,0),(8,1)).frame(Cartesian2D())
    position_frame=Rectangle('position then inactive',(0,0),(1,1)).frame(Cartesian2D())
    model,states,collision=collision_module(phase_frame,name=scope,nu=NU)
    definitions=[(NAMES[i],model,s,True) for i,s in enumerate(states)]
    for name,component in zip(NAMES[4:7],('density','first_moment','second_moment'),strict=True):
        module=Module(name+' arbitrary owner',frame=position_frame)
        space=module.state_space('physical moment',(component,),support=POSITION,units=(UNIT,),sampling='cell_average')
        definitions.append((name,module,module.state_handle(space),False))
    inert=Module('independent untouched species',frame=phase_frame)
    space=inert.state_space('sentinel',('first','second'),support=PHASE,units=(UNIT,UNIT),sampling='cell_average')
    definitions.append((NAMES[7],inert,inert.state_handle(space),True))
    case=pops.Case('isolated BGK reduce moments and lift each stage')
    blocks={}; subjects={}
    for name,module,state,_ in reversed(definitions) if reverse else definitions:
        blocks[name]=case.block(name,module,states=(state,))
        subjects[name]=blocks[name][state]
    p=pops.Program('SSPRK2 isolated relaxation recompute all velocity moments')
    storage={name:p.state(subjects[name]) for name in NAMES}
    dv=Fraction(16,nv)
    velocity=tuple(-8+(j+Fraction(1,2))*dv for j in range(nv))
    reductions=tuple(PhysicalSupportMap(PHASE,POSITION,
        source_axes=(0,1),target_axes=(0,),native_dimension=2,
        reductions=(AxisQuadrature(0,-8,8,nv,UNIT,
            weights=tuple(dv*v**power for v in velocity)),)) for power in range(3))
    lift=PhysicalSupportMap(POSITION,PHASE,source_axes=(0,),target_axes=(0,1),native_dimension=2)
    def evaluate(value):
        reduced=[]; extended=[]
        for power,mapping in enumerate(reductions):
            number=storage[NAMES[4+power]]
            moment=p.map(mapping,source=value,target=number.stage(value.point.name,point=value.point))
            target=storage[NAMES[1+power]]
            extended.append(p.map(lift,source=moment,target=target.stage(value.point.name,point=value.point)))
            reduced.append(moment)
        return collision(value,*extended)[blocks[NAMES[0]]],reduced,extended
    start=p.stage('accepted collision evaluation',c=0)
    initial=p.value('accepted distribution',storage[NAMES[0]].n,at=start)
    r0,_,_=evaluate(initial)
    end=p.stage('predictor collision evaluation',c=1)
    predictor=p.value('Euler collision predictor',initial+p.dt*r0,at=end)
    r1,reduced,extended=evaluate(predictor)
    accepted=p.value('SSPRK2 accepted distribution',
        Fraction(1,2)*initial+Fraction(1,2)*(predictor+p.dt*r1),at=storage[NAMES[0]].next.point)
    p.commit(storage[NAMES[0]].next,accepted)
    for name,value in zip(NAMES[1:7],(*extended,*reduced),strict=True):
        p.commit(storage[name].next,p.value('last predictor '+name,value,at=storage[name].next.point))
    p.commit(storage[NAMES[7]].next,p.value('unrelated unchanged',storage[NAMES[7]].n,at=storage[NAMES[7]].next.point))
    p.step_strategy(FixedDt(float(DT)));case.program(p)
    validated=pops.validate(case); layout_subjects=validated.layout_subjects()
    builder=LayoutPlanBuilder(validated.owner_path.canonical())
    grids=(Uniform(CartesianGrid(frame=phase_frame,cells=(nv,nx),periodic=PeriodicAxes(phase_frame.axes))),
           Uniform(CartesianGrid(frame=position_frame,cells=(nx,1),periodic=PeriodicAxes(position_frame.axes))))
    layouts=(builder.layout('phase fibre storage',grids[0]),builder.layout('physical moment storage',grids[1]))
    phase_names={name for name,_,_,phase in definitions if phase}
    for row in layout_subjects.blocks:builder.assign_block(row,layouts[row.local_id not in phase_names])
    for row in layout_subjects.states:builder.assign_state(row,layouts[row.block_ref.local_id not in phase_names])
    rows={row.block_ref.local_id:row for row in layout_subjects.states}
    requirements=[]
    for power,mapping in enumerate(reductions):
        for source,target,recipe in ((NAMES[0],NAMES[4+power],mapping),
                                     (NAMES[4+power],NAMES[1+power],lift)):
            req,=builder.require_mapping(layouts[source not in phase_names],layouts[target not in phase_names],
                source=rows[source],target=rows[target],operation=LayoutMappingOperation(recipe.operation_abi),
                synchronization=LayoutSynchronization.PROGRAM_POINT_V1,
                source_representation=LayoutRepresentation.CELL_AVERAGE_V1,
                target_representation=LayoutRepresentation.CELL_AVERAGE_V1,physical_map=recipe)
            requirements.append(req)
    providers=tuple(provider_factory(row,directory) for row in requirements)
    layout=builder.resolve(**layout_subjects.to_dict(),providers=providers)
    return pops.resolve(validated,layout=layout,layout_providers=dict(zip(layouts,grids,strict=True)),
        components=tuple(row.component for row in providers),compile_options={'model_source_policy':'require'})
