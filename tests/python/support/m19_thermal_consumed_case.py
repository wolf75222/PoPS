"""Non-kinetic declared Helmholtz output consumed through two physical maps."""
from fractions import Fraction
import pops
from pops.model import Module, Rate, FieldSpace, Handle, OwnerPath, PhysicalSupport, PhysicalDimension
from pops.fields import FieldProblem, FieldBoundary, FieldDiscretization, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import laplacian, Reaction
from pops._ir.elliptic import DivCoeffGrad
from pops.solvers import CG
from pops.time import FailRun, FixedDt
from pops.numerics.terms import SourceTerm
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import (CartesianGrid, PeriodicAxes, LayoutPlanBuilder, LayoutRepresentation,
    LayoutMappingOperation, LayoutSynchronization, PhysicalSupportMap, AxisQuadrature,
    native_physical_mapping)

WEIGHTS = ((Fraction(1,8),Fraction(1,8),Fraction(1,4),Fraction(1,2)),
           (Fraction(-1,4),Fraction(0),Fraction(1,4),Fraction(0)))
NAMES = ('reservoir_z', 'zeta_laminated_material', 'reservoir_a')

def build(directory, *, reverse=False, gradient=False, provider_factory=native_physical_mapping):
    unit = PhysicalDimension()
    slab = PhysicalSupport((('eta','normalized thickness'),('x','periodic rod')))
    rod = PhysicalSupport((('x','periodic rod'),))
    frame = Rectangle('storage x then eta',(0,0),(1,1)).frame(Cartesian2D())
    case = pops.Case('screened thermal laminate and independent reservoirs')
    source = Module('nonkinetic temperature load')
    space = source.state_space('thermal coefficients',('unused','heat_load'),support=slab,
                              units=(unit,unit),sampling='cell_average')
    destinations=[]
    for i,name in enumerate((NAMES[0],NAMES[2])):
        model=Module('thermal reservoir '+str(i))
        evolved=model.state_space('accumulation',('spectator','enthalpy','marker'),support=rod,
                                  units=(unit,unit,unit),sampling='cell_average')
        incoming=model.field_space('applied temperature',('delivered',),support=rod,units=(unit,),sampling='cell')
        factor=i+1
        @model.operator('consume_mapped_temperature',signature=(evolved,incoming)>>Rate(evolved),kind='local_source')
        def rhs(state,field,factor=factor):
            return (0*state[0],factor*field[0],0*state[2])
        block=case.block(name,model)
        destinations.append((block,block[model.state_handle(evolved)],block[model.field_handle(incoming)],model))
        if i==0:
            src=case.block(NAMES[1],source)
    source_state=src[source.state_handle(space)]
    phi=Handle('screened_temperature',kind='field',owner=OwnerPath.model('thermal potential'))
    physical=FieldProblem('screened thermal response',unknowns=(phi,),
        equations=(Reaction(phi,1)-DivCoeffGrad(phi,.125)==source.state_symbols(space)[1],),
        boundaries=(FieldBoundary(phi,bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(),bcs.Periodic())),),
        unknown_spaces={phi:FieldSpace('temperature',('temperature',),support=slab,units=(unit,),sampling='cell')},
        coordinate_units=(unit,unit))
    field=case.field(physical,FieldDiscretization(method=CellCenteredSecondOrder(),boundaries=(),
        observation_axes=(1,0),solver=CG(max_iter=100,rel_tol=1e-12,abs_tol=1e-12)))
    program=pops.Program('solve once and consume two mapped outputs')
    current=program.state(source_state)
    solution=field.observe(program.solve(field,values={source_state:current.n},at=current.n.point).consume(action=FailRun()))
    port=solution.mapping_port(field[phi],derivative_axis=1 if gradient else None,factor=-1 if gradient else 1)
    maps=[]
    order=tuple(reversed(tuple(enumerate(destinations)))) if reverse else tuple(enumerate(destinations))
    for i,(block,state,target,model) in order:
        stage=program.state(state)
        mapping=PhysicalSupportMap(slab,rod,source_axes=(1,0),target_axes=(1,),native_dimension=2,
            reductions=(AxisQuadrature(1,0,1,4,unit,weights=WEIGHTS[i]),))
        context=solution.publish_mapped(mapping,{(target,'delivered'):port},states={state:stage.n})
        rate=program.rhs(state=stage.n,fields=context,terms=[SourceTerm(block[model.operator_handle('consume_mapped_temperature')])])
        program.commit(stage.next,program.value('integrated '+block.local_id,stage.n+program.dt*rate,at=stage.next.point))
        maps.append((mapping,target,i))
    program.commit(current.next,program.value('unchanged load',1*current.n,at=current.next.point))
    program.step_strategy(FixedDt(1/64));case.program(program)
    validated=pops.validate(case);subjects=validated.layout_subjects()
    builder=LayoutPlanBuilder(validated.owner_path.canonical())
    grids=(Uniform(CartesianGrid(frame=frame,cells=(3,4),periodic=PeriodicAxes(frame.axes))),
           Uniform(CartesianGrid(frame=frame,cells=(1,3),periodic=PeriodicAxes(frame.axes))))
    layouts=(builder.layout('laminate storage',grids[0]),builder.layout('reservoir storage',grids[1]))
    for block in subjects.blocks:builder.assign_block(block,layouts[block.local_id!=NAMES[1]])
    for row in subjects.states:builder.assign_state(row,layouts[row.block_ref.local_id!=NAMES[1]])
    for row in subjects.fields:builder.assign_field(row,layouts[1] if row.block_ref else layouts[0])
    requirements=[]
    for mapping,target,i in maps:
        requirement,=builder.require_mapping(*layouts,source=field,target=target,
            operation=LayoutMappingOperation(mapping.operation_abi),synchronization=LayoutSynchronization.PROGRAM_POINT_V1,
            source_representation=LayoutRepresentation.CELL_FIELD_V1,target_representation=LayoutRepresentation.CELL_FIELD_V1,
            source_observation=port,target_component='delivered',physical_map=mapping)
        requirements.append(requirement)
    providers=tuple(provider_factory(r,directory) for r in requirements)
    layout=builder.resolve(**subjects.to_dict(),providers=providers)
    return pops.resolve(validated,layout=layout,layout_providers=dict(zip(layouts,grids)),
        components=tuple(p.component for p in providers),compile_options={'model_source_policy':'require'})

def reference(*,gradient=False,steps=2):
    """Independent periodic centered stencil; not a solve/provider call."""
    import numpy as np
    eta=(np.arange(4)+.5)/4
    phi=2+.1*np.cos(2*np.pi*eta)
    h=.25
    lap=(np.roll(phi,-1)-2*phi+np.roll(phi,1))/h**2
    load=phi-lap/8
    output=-(np.roll(phi,-1)-np.roll(phi,1))/(2*h) if gradient else phi
    initial={NAMES[1]:np.stack((np.full((4,3),-19.),np.broadcast_to(load[:,None],(4,3))))}
    expected={}
    for i,name in enumerate((NAMES[0],NAMES[2])):
        initial[name]=np.stack((np.full((3,1),11.+i),np.full((3,1),-101.-i),np.full((3,1),37.+i)))
        value=initial[name].copy()
        integral=sum(float(w)*float(v) for w,v in zip(WEIGHTS[i],output,strict=True))
        value[1]+=steps*(i+1)*integral/64
        expected[name]=value
    expected[NAMES[1]]=initial[NAMES[1]].copy()
    return initial,expected,phi,load
