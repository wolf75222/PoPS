"""SOURCE and real-header host checks; no Native pipeline qualification."""
import pytest
from pops.boundary.transport import Inflow, Outflow, TransportBoundarySet
from pops.boundary import interior_trace
from tests.python.unit.boundary.test_transport_authoring import _authoring


def test_public_interior_trace_expression_automatically_composes():
    frame,template,_,_,numerics,case,block,state=_authoring()
    (u,)=template
    inflow=Inflow(state=state,value=interior_trace(state,"u")+1.0)
    assert inflow.values
    numerics.boundaries.add(TransportBoundarySet({
        face:(inflow if face==frame.boundaries.x_min else Outflow(state=state))
        for face in frame.boundaries.all}))
    case.numerics(numerics,block=block)
    authority=case._resolved_numerics_for('tracer').boundaries[0]
    pairs=authority.inferred_component_bindings()
    assert len(pairs)==1
    binding,component=pairs[0]
    assert binding.component_id==component.component_manifest.component_id
    assert len(component.component_type.package.payloads)==1


def test_public_helmholtz_field_inflow_auto_composes_source_plan():
    import pops
    from tests.python.support.public_inflow_field_case import build
    case,layout=build(ghost=True)
    plan=pops.resolve(pops.validate(case),layout=layout)
    assert len(plan.component_inputs)==1
    ghost=plan.blocks[0].numerics.boundaries[0]
    assert len(ghost.component_bindings)==1
    ghost.require_component_inputs(plan.component_inputs)
    component=plan.component_inputs[0]
    assert component.component_type.interface.name=='ghost_boundary'
    signature=component.component_manifest.signature['inferred_boundary_expression']
    assert signature['schema']=='inferred-boundary-expression-component@1'
    assert len(signature['expressions'])==2
    assert {row['name'] for row in signature['interfaces']}=={'ghost_boundary','accepted_initial_ghost'}


def test_generated_public_callback_actual_header_syntax(tmp_path):
    import subprocess,shutil
    import pops
    from pathlib import Path
    from tests.python.support.public_inflow_field_case import build
    case,layout=build(ghost=True);plan=pops.resolve(pops.validate(case),layout=layout)
    source=plan.component_inputs[0].component_type.package.payloads[0].content
    path=tmp_path/'actual.cpp';path.write_bytes(source)
    subprocess.run([shutil.which('clang++'),'-std=c++20','-Wall','-Wextra','-Werror','-fsyntax-only','-DPOPS_NATIVE_DIM=2',
        '-I',str(Path(__file__).resolve().parents[2]/'include'),str(path)],check=True,capture_output=True)


def multi_component():
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D
    from pops.math import ValueExpr,ddt,div
    from pops.params import RuntimeParam
    from pops.fields.boundary_values import logical_time
    from pops.numerics import DiscretizationPlan,reconstruction,riemann,variables
    from pops.numerics.spatial import FiniteVolume
    frame=Rectangle('renamed domain',(0.,0.),(1.,1.)).frame(Cartesian2D())
    model=pops.Model('arbitrary three-output model',frame=frame)
    state=model.state('inventory',components=('last','middle','first'),sampling='cell_average')
    a,b,c=state
    left=model.field('different left');right=model.field('different right')
    coefficient=model.param(RuntimeParam('gain',default=.25))
    flux=model.flux('null',frame=frame,state=state,components={axis:(0*a,0*b,0*c) for axis in frame.axes},waves={axis:(0*a,0*b,0*c) for axis in frame.axes})
    rate=model.rate('rate',equation=ddt(state)==-div(flux))
    case=pops.Case('renamed case');block=case.block('second arbitrary block',model)
    v=ValueExpr(block[state])
    from pops.boundary import interior_trace
    expr=interior_trace(block[state],'middle')+ValueExpr(block[left])+2*ValueExpr(block[right])+logical_time()+model.value(coefficient)
    inflow=Inflow(state=block[state],value=(expr,interior_trace(block[state],'first'),interior_trace(block[state],'last')))
    numerics=DiscretizationPlan();numerics.rates.add(rate,FiniteVolume(flux=flux,variables=variables.Conservative(state),reconstruction=reconstruction.FirstOrder(),riemann=riemann.Rusanov()))
    numerics.boundaries.add(TransportBoundarySet({face:inflow if face==frame.boundaries.x_min else Outflow(state=block[state]) for face in frame.boundaries.all}))
    case.numerics(numerics,block=block)
    authority=case._resolved_numerics_for('second arbitrary block').boundaries[0]
    _,component=authority.inferred_component_bindings()[0]
    return component


def test_multioutput_two_fields_state_time_param_qualified_identity():
    component=multi_component()
    evidence=component.component_manifest.signature['inferred_boundary_expression']
    assert len(evidence['expressions'])==3
    assert len(evidence['dependencies']['fields'])==2
    assert not evidence['dependencies']['states']
    source=component.component_type.package.payloads[0].content.decode()
    assert 'physical_time' in source and 'parameter(r,' in source
    # Exact named components determine their independent indices, not declaration position zero.
    assert 'point,1)' in source and 'point,2)' in source and 'point,0)' in source


def test_ambiguous_multicomponent_value_and_unlowerable_operation_refused():
    from pops.codegen.inferred_boundary_expression import lower_expression
    from pops.math import ValueExpr
    from pops._ir.expr import Var
    import pops
    from tests.python.support.public_inflow_field_case import build
    case,_=build(ghost=False)
    state=case._resolved_numerics_for('marker').boundaries[0].conditions[0].state
    with pytest.raises(ValueError,match='ambiguous'):lower_expression(ValueExpr(state))
    with pytest.raises(NotImplementedError,match='unsupported Expr node Var'):lower_expression(Var('free name','state'))


def test_time_only_inflow_selects_component_without_manual_binding():
    from pops.fields.boundary_values import logical_time
    frame,_,_,_,numerics,case,block,state=_authoring()
    numerics.boundaries.add(TransportBoundarySet({face:Inflow(state=state,value=logical_time()+1) if face==frame.boundaries.x_min else Outflow(state=state) for face in frame.boundaries.all}))
    case.numerics(numerics,block=block)
    authority=case._resolved_numerics_for('tracer').boundaries[0]
    assert len(authority.inferred_component_bindings())==1


def test_actual_generated_math_host_two_fields_permuted_components(tmp_path):
    import re,json,subprocess,shutil
    from pathlib import Path
    component=multi_component()
    source=component.component_type.package.payloads[0].content.decode()
    dependencies=list(dict.fromkeys(re.findall(r'dependency\(r,("[^"]+"),(\d+)\)',source)))
    assert sorted(int(n) for _,n in dependencies)==[1,1]
    state_identity=re.search(r'interior\(r,("[^"]+"),3\)',source).group(1)
    parameter=re.search(r'parameter\(r,("[^"]+")\)',source).group(1)
    rows=[];scalar=0
    for identity,count in dependencies:
        if count=='3':array='state_values'
        else:array=['left_values','right_values'][scalar];scalar+=1
        rows.append('{sizeof(PopsQualifiedConstFieldV1),1,%s,view(%s,%s)}'%(identity,array,count))
    probe='''
int main(){
 double state_values[]={2,3,5,7,11,13},left_values[]={17,19},right_values[]={23,29},output[6]={};
 auto view=[](const double* p,std::size_t n){PopsConstFieldViewV1 v{};v.struct_size=sizeof(v);v.data=p;v.dimension=2;v.extents[0]=1;v.extents[1]=2;v.axis_strides[0]=2;v.axis_strides[1]=1;v.component_count=n;v.component_stride=2;v.scalar_type=POPS_SCALAR_FLOAT64_V1;v.memory_space=POPS_MEMORY_SPACE_HOST_V1;return v;};
 PopsQualifiedConstFieldV1 dependencies[]={ROWS};
 PopsQualifiedScalarV1 parameters[]={{sizeof(PopsQualifiedScalarV1),PARAMETER,.25}};
 PopsGhostBoundaryRequestV1 r{};r.struct_size=sizeof(r);r.dependency_count=2;r.dependencies=dependencies;r.state_identity=STATE_IDENTITY;r.interior=view(state_values,3);r.parameter_count=1;r.parameters=parameters;r.logical_time.dt=.125;r.logical_time.physical_time=.5;
 r.ghosts.struct_size=sizeof(r.ghosts);r.ghosts.data=output;r.ghosts.dimension=2;r.ghosts.extents[0]=1;r.ghosts.extents[1]=2;r.ghosts.axis_strides[0]=2;r.ghosts.axis_strides[1]=1;r.ghosts.component_count=3;r.ghosts.component_stride=2;r.ghosts.scalar_type=POPS_SCALAR_FLOAT64_V1;r.ghosts.memory_space=POPS_MEMORY_SPACE_HOST_V1;
 PopsComponentStatusV1 status{};
 if(ghost.apply_region_batch(nullptr,&r,&status)!=0)return 1;
 if(output[0]!=68.75||output[1]!=84.75||output[2]!=11||output[3]!=13||output[4]!=2||output[5]!=3)return 2;
 r.logical_time.dt=0;if(ghost.apply_region_batch(nullptr,&r,&status)==0)return 3;
 PopsAcceptedInitialGhostRequestV1 initial_request{};initial_request.struct_size=sizeof(initial_request);initial_request.point_contract_version=1;initial_request.region_request=r;initial_request.region_request.logical_time.fraction_denominator=1;
 if(initial.apply_initial_region_batch(nullptr,&initial_request,&status)!=0)return 4;
 initial_request.region_request.logical_time.tick=1;if(initial.apply_initial_region_batch(nullptr,&initial_request,&status)==0)return 5;
 r.logical_time.dt=.125;dependencies[0].values.component_count=99;if(ghost.apply_region_batch(nullptr,&r,&status)==0)return 6;
}
'''.replace('ROWS',','.join(rows)).replace('PARAMETER',parameter).replace('STATE_IDENTITY',state_identity)
    path=tmp_path/'actual_math.cpp';path.write_text(source+probe);binary=tmp_path/'probe'
    result=subprocess.run([shutil.which('clang++'),'-std=c++20','-Wall','-Wextra','-Werror','-DPOPS_NATIVE_DIM=2','-I',str(Path(__file__).resolve().parents[2]/'include'),str(path),'-o',str(binary)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(binary)],capture_output=True,text=True)
    assert result.returncode==0,(result.returncode,result.stderr)


def test_companion_manifest_exact_loader_set_and_old_primary_unchanged():
    from pops import interfaces
    from pops.external.artifacts import ComponentRuntimeContract
    component=multi_component();contract=ComponentRuntimeContract.from_manifest(component.component_manifest)
    tables=interfaces.required_native_interface_tables(contract.manifest_data['signature'],interfaces.GhostBoundary)
    assert tables==((interfaces.GhostBoundary.abi_id,1,interfaces.GhostBoundary.cpp_table),
                    (interfaces.AcceptedInitialGhost.abi_id,1,interfaces.AcceptedInitialGhost.cpp_table))
    assert interfaces.required_native_interface_tables({},interfaces.GhostBoundary)==tables[:1]


@pytest.mark.parametrize('mutation',['version','boolversion','id','catalog','duplicate','primary'])
def test_forged_companion_manifest_is_refused_before_loading(mutation):
    from pops import interfaces
    signature=multi_component().component_manifest.to_data()['signature']
    extension=signature['native_interface_extensions']
    primary=interfaces.GhostBoundary
    if mutation=='version':extension['schema_version']=2
    if mutation=='boolversion':extension['schema_version']=True
    if mutation=='id':extension['interfaces'][0]['id']=999
    if mutation=='catalog':extension['interfaces'][0]['catalog_sha256']='0'*64
    if mutation=='duplicate':extension['interfaces']*=2
    if mutation=='primary':extension['interfaces'][0]=interfaces.GhostBoundary.signature_declaration()
    with pytest.raises(ValueError,match='native-interface-extensions@1'):
        interfaces.required_native_interface_tables(signature,primary)


def test_own_output_ghost_read_is_not_silently_an_interior_trace():
    from pops.codegen.inferred_boundary_expression import lower_expression
    from pops.math import ValueExpr
    from tests.python.support.public_inflow_field_case import build
    case,_=build(ghost=False)
    state=case._resolved_numerics_for("marker").boundaries[0].conditions[0].state
    with pytest.raises(ValueError,match="own output ghost"):
        lower_expression(ValueExpr(state)["c"],output_state=state)

def test_interior_trace_rejects_field_and_foreign_primary_state():
    from pops.codegen.inferred_boundary_expression import lower_expression
    _,_,_,_,_,_,_,state=_authoring()
    with pytest.raises(ValueError,match="exact primary"):
        lower_expression(interior_trace(state,"u"),output_state=object())
    with pytest.raises(TypeError,match="requires a State"):
        interior_trace(object())

@pytest.mark.parametrize("mutation",["truncated","unknown","extra","signature"])
def test_extension_incomplete_unknown_extra_or_drift_rejected(mutation):
    from pops import interfaces
    signature=multi_component().component_manifest.to_data()["signature"]
    extension=signature["native_interface_extensions"]
    row=extension["interfaces"][0]
    if mutation=="truncated":del row["version"]
    if mutation=="unknown":row["name"]="not-a-catalog-interface"
    if mutation=="extra":extension["hidden_table"]=11
    if mutation=="signature":row["cpp_table"]="PopsWrongTable"
    with pytest.raises(ValueError,match="native-interface-extensions@1"):
        interfaces.required_native_interface_tables(signature,interfaces.GhostBoundary)


@pytest.mark.parametrize("auto_ghost",[False,True])
def test_explicit_component_duplicates_remain_rejected(auto_ghost):
    import pops
    from tests.python.support.public_inflow_field_case import build
    case,layout=build(ghost=auto_ghost)
    component=multi_component()
    with pytest.raises(ValueError,match="resolve components contain duplicate component_id"):
        pops.resolve(pops.validate(case),layout=layout,components=(component,component))


@pytest.mark.parametrize("auto_ghost",[False,True])
def test_frozen_prerequisite_resolve_refuses_same_explicit_duplicates(auto_ghost):
    import subprocess
    import pops
    from tests.python.support.public_inflow_field_case import build
    # Execute the exact frozen pre-extension resolve module. The failure is in
    # unchanged layout validation, before any new boundary composition is reached.
    source=subprocess.run(["git","show","6c595c04:python/pops/codegen/_phases.py"],check=True,capture_output=True,text=True).stdout
    namespace={"__name__":"pops.codegen._frozen_duplicate_probe","__package__":"pops.codegen"}
    exec(compile(source,"6c595c04/python/pops/codegen/_phases.py","exec"),namespace)
    case,layout=build(ghost=auto_ghost);component=multi_component()
    with pytest.raises(ValueError,match="resolve components contain duplicate component_id"):
        namespace["resolve"](pops.validate(case),layout=layout,components=(component,component))


def compiled_public_ghost_plan():
    import pops
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
    from tests.python.support.public_inflow_field_case import build
    case,layout=build(ghost=True)
    plan=pops.resolve(pops.validate(case),layout=layout)
    return CompiledBoundaryPlan.from_resolved(plan.blocks[0].numerics.boundaries[0])


def test_public_component_values_bind_via_actual_detached_install_contract():
    compiled=compiled_public_ghost_plan()
    data=compiled.runtime_boundary_data({})
    assert data["faces"][0]["type"]=="external"
    assert data["faces"][0]["values"]==[0.,0.]
    assert len(data["component_regions"])==1


@pytest.mark.parametrize("mutation",["missing","manifest","target","region","type","protocol"])
def test_forged_component_value_delegation_refused(mutation):
    from pops.mesh.boundaries.compiled_plan import CompiledBoundaryPlan
    data=compiled_public_ghost_plan().canonical_identity()["compile_data"]
    face=data["faces"][0]
    if mutation=="missing":del face["value_delegate"]
    if mutation=="manifest":face["value_delegate"]["component_manifest_identity"]="forged"
    if mutation=="target":face["value_delegate"]["target"]["qualified_id"]="foreign"
    if mutation=="region":data["component_region_templates"][0]["region"]["axes"]=[1]
    if mutation=="type":face["type"]="dirichlet"
    if mutation=="protocol":face["value_protocol"]="unknown"
    with pytest.raises(ValueError,match="component value delegation"):
        CompiledBoundaryPlan(data).runtime_boundary_data({})
