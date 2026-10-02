"""Pure, content-authenticated public Expr -> native Ghost component lowering (@1)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from pops.identity import canonical_bytes

SCHEMA='inferred-boundary-expression-component@1'


def _components(handle):
    declaration=getattr(handle,'declaration_ref',None) or handle
    space=getattr(declaration,'space',None)
    if space is not None and getattr(space,'layout',None) not in (None,'cell'):
        raise NotImplementedError(SCHEMA+' requires a cell-centered ghost-region support; Face/Facet values need an explicit versioned trace conversion')
    result=tuple(getattr(space,'components',())) or tuple(getattr(declaration,'components',()))
    if not result and handle.kind=='field':
        # Existing scalar Field boundary-value contract; FieldSpace arrays above
        # retain their complete manifest and require explicit component selection.
        from pops.fields.boundary_values import _component_index
        _component_index(handle,None)
        result=(None,)
    if not result:raise ValueError('Inflow expression carrier lacks an authoritative component manifest')
    return result


def lower_expression(expression,*,output_state=None):
    """Return scalar C++ and exact leaf manifest; never execute/sample an Expr."""
    from pops._ir.expr import Const,_Bin,Neg,Sqrt,Exp,Abs,Sign,Pow
    from pops._ir.handle_expr import ValueExpr
    from pops._ir.quantity import QuantityRef
    from pops._ir.values import RuntimeParamRef
    from pops.fields.boundary_values import BoundaryValue,LogicalTimeValue
    from pops.identity.scalar import scalar_cpp
    from pops.boundary.interior_trace import InteriorTrace
    leaves=[]
    def carrier(handle,index):
        if not handle.is_resolved:raise ValueError('Inflow expression requires an owner-qualified carrier')
        if handle.kind not in ('state','field'):raise TypeError('Inflow expression requires state/Field carrier')
        if output_state is not None and handle==output_state:
            raise ValueError('Inflow reads its own output ghost without an upstream producer; use explicit interior_trace for the primary State')
        components=_components(handle)
        if type(index) is not int or not 0<=index<len(components):raise ValueError('Inflow expression component outside authoritative manifest')
        leaves.append({'handle':handle.canonical_identity(),'component':index,'components':list(components)})
        return 'read(dependency(r,%s,%d),point,%d)'%(json.dumps(handle.qualified_id),len(components),index)
    def visit(node):
        if isinstance(node,Const):return scalar_cpp(node.literal)
        if isinstance(node,RuntimeParamRef):
            handle=node.handle
            if not handle.is_resolved:raise ValueError('Inflow parameter requires an owner-qualified Handle')
            leaves.append({'parameter':handle.canonical_identity()})
            return 'parameter(r,%s)'%json.dumps(handle.qualified_id)
        if isinstance(node,InteriorTrace):
            if output_state is not None and node.handle!=output_state:
                raise ValueError('InteriorTrace must name the exact primary output State')
            components=_components(node.handle)
            leaves.append({'handle':node.handle.canonical_identity(),'component':node.component,
                'components':list(components),'support':'clamped-primary-state-interior-trace@1'})
            return 'read(interior(r,%s,%d),point,%d)'%(json.dumps(node.handle.qualified_id),len(components),node.component)
        if isinstance(node,QuantityRef):return carrier(node.handle,node.index)
        if isinstance(node,BoundaryValue):return carrier(node.handle,node.component)
        if isinstance(node,ValueExpr):
            if node.handle.kind=='parameter':
                leaves.append({'parameter':node.handle.canonical_identity()})
                return 'parameter(r,%s)'%json.dumps(node.handle.qualified_id)
            components=_components(node.handle)
            if len(components)!=1:raise ValueError('ambiguous Inflow ValueExpr: select an exact named component')
            return carrier(node.handle,0)
        if isinstance(node,LogicalTimeValue):
            coordinate=node.coordinate
            fields={'time':'physical_time','dt':'dt','step':'tick','substep':'substep','stage':'stage'}
            if coordinate not in fields:raise NotImplementedError(SCHEMA+' does not lower logical-time coordinate '+coordinate)
            leaves.append({'logical_time':coordinate})
            return 'double(r.logical_time.%s)'%fields[coordinate]
        if isinstance(node,Pow):return 'std::pow(%s,%s)'%(visit(node.a),visit(node.b))
        if isinstance(node,_Bin):
            if node.op not in ('+','-','*','/'):raise NotImplementedError(SCHEMA+' unsupported scalar operation '+node.op)
            return '(%s %s %s)'%(visit(node.a),node.op,visit(node.b))
        if isinstance(node,Neg):return '(-%s)'%visit(node.a)
        functions={Sqrt:'sqrt',Exp:'exp',Abs:'abs'}
        for kind,name in functions.items():
            if isinstance(node,kind):return 'std::%s(%s)'%(name,visit(node.a))
        if isinstance(node,Sign):
            value=visit(node.a);return '((%s>0.0)-(%s<0.0))'%(value,value)
        raise NotImplementedError(SCHEMA+' unsupported Expr node '+type(node).__name__)
    cpp=visit(expression)
    return cpp,tuple(leaves)


def inferred_component(condition,state,*,frame_id,dimension):
    from pops import interfaces
    from pops.model import ComponentManifest
    from pops.external.packages import SourceComponentPackage,PackagePayload,build_source_package_manifest
    from pops.external._package_data import content_identity,package_identity
    from pops.boundary.transport import _expression_data
    ghost=interfaces.GhostBoundary;initial=interfaces.AcceptedInitialGhost
    lowered=[lower_expression(row,output_state=state) for row in condition.values]
    evidence={'schema':SCHEMA,'frame':frame_id,'dimension':dimension,
        'state':state.canonical_identity(),'geometry':condition.geometry.canonical_identity(),
        'representation':condition.provider.outputs[0].representation.canonical_identity(),
        'expressions':[_expression_data(row,qualified=True) for row in condition.values],
        'leaves':[list(row[1]) for row in lowered],
        'dependencies':condition.provider.dependencies.canonical_identity(),
        'stencil':condition.requirement.canonical_identity(),
        'floating_point':'binary64-ordered-ast-no-reassociation@1',
        'point_support':{'schema':'PointwiseGhostValues@1','carriers':'exact-ghost-region-cell-centered',
            'time':'authenticated-native-ghost-point','stencil_extent':0,'interior_trace_substitution':False,'explicit_interior_trace':'clamped-primary-state-interior-trace@1'},
        'supported_interface_extensions':[initial.signature_declaration()],
        'interfaces':[ghost.to_data(),initial.to_data()]}
    from pops.identity.semantic import semantic_value
    evidence=semantic_value(evidence,where=SCHEMA)
    token=hashlib.sha256(canonical_bytes(evidence)).hexdigest()
    manifest=ComponentManifest(uri='pops://generated/inferred-boundary-expression/'+token,
        component_type=ghost.name,version='1.0.0',facets=ghost.facets,
        signature={'generic':True,'native_interface':ghost.signature_declaration(),
            'native_interface_extensions':{'schema_version':1,'interfaces':(initial.signature_declaration(),)},
            'inferred_boundary_expression':evidence},
        interfaces=ghost.manifest_declarations(),
        target={'variants':[{'dimension':dimension,'scalar':'float64','device':'cpu','features':[]}]},
        entry_points={'interface_table':'pops_component_interface_v1'})
    source=emit_source(manifest,[row[0] for row in lowered],dimension)
    row=build_source_package_manifest(components={'inflow':manifest},payloads={'inflow.cpp':('source',source)})
    package=SourceComponentPackage((manifest,),MappingProxyType({'inflow':manifest.component_id}),
        (PackagePayload('inflow.cpp','source',content_identity('source',source),source),),
        package_identity(row),Path('inferred-boundary-expression-'+token+'.pops.json'))
    return package.require('inflow',interface=ghost)()


def emit_source(manifest,expressions,dimension):
    writes='\n'.join('write(r.ghosts,point,%d,%s);'%(i,expr) for i,expr in enumerate(expressions))
    return ('''#include <pops/runtime/config/generated_component_abi.hpp>
#include <pops/core/foundation/types.hpp>
#include <type_traits>
static_assert(std::is_same_v<pops::Real,double>,"inferred Inflow@1 requires binary64");
#include <pops/runtime/dynamic/abi_key.hpp>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstddef>
#include <limits>
namespace {
int prepare(const PopsComponentPrepareRequestV1*,void** output,PopsComponentStatusV1* status){
 if(!output||!status)return 1;*output=nullptr;*status={sizeof(*status),0,POPS_COMPONENT_CONTINUE_V1,nullptr};return 0;}
void destroy(void*){}
std::ptrdiff_t offset(const auto& v,std::size_t point,std::size_t component){
 if(!v.data||v.scalar_type!=POPS_SCALAR_FLOAT64_V1||v.memory_space!=POPS_MEMORY_SPACE_HOST_V1||component>=v.component_count)throw std::invalid_argument("Inflow view contract");
 std::ptrdiff_t result=component*v.component_stride;
 for(int axis=v.dimension-1;axis>=0;--axis){if(!v.extents[axis])throw std::invalid_argument("Inflow empty extent");result+=(point%v.extents[axis])*v.axis_strides[axis];point/=v.extents[axis];}
 if(point)throw std::invalid_argument("Inflow point extent");return result;}
[[maybe_unused]] double read(const PopsConstFieldViewV1& v,std::size_t point,std::size_t component){return static_cast<const double*>(v.data)[offset(v,point,component)];}
void write(const PopsFieldViewV1& v,std::size_t point,std::size_t component,double value){if(!std::isfinite(value))throw std::invalid_argument("Inflow nonfinite expression");static_cast<double*>(v.data)[offset(v,point,component)]=value;}
[[maybe_unused]] const PopsConstFieldViewV1& interior(const PopsGhostBoundaryRequestV1& r,const char* id,std::size_t components){
 if(!r.state_identity||std::strcmp(r.state_identity,id)!=0||r.interior.component_count!=components||r.interior.dimension!=r.ghosts.dimension)throw std::invalid_argument("Inflow interior trace authority");
 for(int a=0;a<r.ghosts.dimension;++a)if(r.interior.extents[a]!=r.ghosts.extents[a])throw std::invalid_argument("Inflow interior trace support");return r.interior;}
[[maybe_unused]] const PopsConstFieldViewV1& dependency(const PopsGhostBoundaryRequestV1& r,const char* id,std::size_t components){
 const PopsConstFieldViewV1* found=nullptr;for(std::size_t i=0;i<r.dependency_count;++i){const auto& d=r.dependencies[i];if(d.qualified_id&&std::strcmp(d.qualified_id,id)==0){if(found||!d.present)throw std::invalid_argument("Inflow ambiguous dependency");found=&d.values;}}
 if(!found||found->component_count!=components||found->dimension!=r.ghosts.dimension)throw std::invalid_argument("Inflow dependency manifest");for(int a=0;a<r.ghosts.dimension;++a)if(found->extents[a]!=r.ghosts.extents[a])throw std::invalid_argument("Inflow dependency support");return *found;}
[[maybe_unused]] double parameter(const PopsGhostBoundaryRequestV1& r,const char* id){
 bool found=false;double value=0;for(std::size_t i=0;i<r.parameter_count;++i){const auto& p=r.parameters[i];if(p.qualified_id&&std::strcmp(p.qualified_id,id)==0){if(found)throw std::invalid_argument("Inflow ambiguous parameter");found=true;value=p.value;}}
 if(!found||!std::isfinite(value))throw std::invalid_argument("Inflow missing parameter");return value;}
int evaluate(const PopsGhostBoundaryRequestV1* request,PopsComponentStatusV1* status){
 if(!request||!status)return 1;try{const auto& r=*request;
 if(r.struct_size!=sizeof(r)||r.ghosts.dimension!=DIMENSION||r.ghosts.component_count!=COMPONENTS)throw std::invalid_argument("Inflow output contract");
 std::size_t points=1;for(int a=0;a<r.ghosts.dimension;++a){if(r.ghosts.extents[a]>std::numeric_limits<std::size_t>::max()/points)throw std::invalid_argument("Inflow region overflow");points*=r.ghosts.extents[a];if(!points)throw std::invalid_argument("Inflow empty region");}
 for(std::size_t point=0;point<points;++point){WRITES}
 *status={sizeof(*status),0,POPS_COMPONENT_CONTINUE_V1,nullptr};return 0;
 }catch(...){*status={sizeof(*status),1,POPS_COMPONENT_REJECT_STEP_V1,"inferred Inflow expression contract failed"};return 1;}}
int apply(void*,const PopsGhostBoundaryRequestV1* r,PopsComponentStatusV1* status){
 if(!r||!std::isfinite(r->logical_time.dt)||r->logical_time.dt<=0)return 1;return evaluate(r,status);}
int apply_initial(void*,const PopsAcceptedInitialGhostRequestV1* r,PopsComponentStatusV1* status){
 if(!r||r->struct_size!=sizeof(*r)||r->point_contract_version!=1)return 1;
 const auto& t=r->region_request.logical_time;
 if(!std::isfinite(t.physical_time)||t.tick!=0||t.stage!=0||t.substep!=0||t.fraction_numerator!=0||t.fraction_denominator!=1||t.dt!=0||std::signbit(t.dt))return 1;return evaluate(&r->region_request,status);}
const PopsGhostBoundaryApiV1 ghost={{sizeof(PopsGhostBoundaryApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,1,prepare,destroy},apply};
const PopsAcceptedInitialGhostApiV1 initial={{sizeof(PopsAcceptedInitialGhostApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_NATIVE_INTERFACE_ACCEPTED_INITIAL_GHOST_V1,1,prepare,destroy},apply_initial};
const PopsComponentInterfaceEntryV1 entries[]={
 {POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,1,sizeof(ghost),&ghost},
 {POPS_NATIVE_INTERFACE_ACCEPTED_INITIAL_GHOST_V1,1,sizeof(initial),&initial}};
const PopsComponentApiV1 api={sizeof(PopsComponentApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_ABI_KEY_LITERAL,POPS_COMPONENT_CATALOG_SHA256_V1,COMPONENT_ID,SEMANTIC_ID,MANIFEST_ID,2,entries};
}
extern "C" const PopsComponentApiV1* pops_component_interface_v1(){return &api;}
'''.replace('DIMENSION',str(dimension)).replace('COMPONENTS',str(len(expressions))).replace('WRITES',writes).replace('COMPONENT_ID',json.dumps(manifest.component_id)).replace('SEMANTIC_ID',json.dumps(manifest.semantic_digest.token)).replace('MANIFEST_ID',json.dumps(manifest.manifest_digest.token))).encode()
