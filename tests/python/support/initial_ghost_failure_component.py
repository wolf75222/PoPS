"""Test-owned real source package; never an installed-runtime replacement."""
from dataclasses import replace
import json
from pops import interfaces
from pops.external import build_source_package_manifest,load
from pops.model import ComponentManifest
from pops.mesh.boundaries import BoundaryComponentBinding

class ManualInitialFaceExecution:
    """Exact manual component delegate, not an unused inferred implementation."""
    def __init__(self,base,binding,component):
        inferred=base.inferred_component_bindings()
        assert len(inferred)==1 and inferred[0][0].target==binding.target
        binding.require_component(component)
        self.base=base;self.binding=binding
    def canonical_identity(self):return {'schema_version':1,'authority_type':'manual-initial-failure-delegate@1','base':self.base.canonical_identity(),'binding':self.binding.canonical_identity()}
    def inferred_component_bindings(self):return ()
    def _externalize(self,data):
        from copy import deepcopy
        result=deepcopy(data);rows=[face for face in result['faces'] if face['producer']==self.binding.target.qualified_id]
        assert len(rows)==1 and rows[0].get('value_protocol')=='native-boundary-component-values@1'
        rows[0]['type']='external';rows[0]['values']=[];rows[0]['value_delegate']=self.binding.canonical_identity()
        return result
    def compile_boundary_data(self):return self._externalize(self.base.compile_boundary_data())
    def runtime_boundary_data(self,params):return self._externalize(self.base.runtime_boundary_data(params))

class InitialFailureBoundary:
    def __init__(self,base,component):self.base=base;self.component=component
    def inspect(self):return {'schema_version':1,'authority_type':'initial-failure-board@1','base':self.base.inspect(),'component':self.component.component_manifest.manifest_digest.token}
    def resolve_for_numerics(self,context):return ResolvedInitialFailure(self.base.resolve_for_numerics(context),self.component)

class ResolvedInitialFailure:
    def __init__(self,base,component):self.base=base;self.component=component
    def amr_boundary_requirement(self,*args,**kwargs):return self.base.amr_boundary_requirement(*args,**kwargs)
    def canonical_identity(self):return {'schema_version':1,'authority_type':'initial-failure-board@1','base':self.base.canonical_identity(),'component':self.component.component_manifest.manifest_digest.token}
    def ghost_plan_composer_capability(self):return {'schema_version':1,'scope':'self'}
    def compose_ghost_plan(self,context):
        from pops.mesh.boundaries.composition import compose_transport_boundary
        boundary=compose_transport_boundary(self.base,context=context)
        productions=[p for p in boundary.productions if p.region.boundary is not None and p.region.boundary.orientation.axis==0 and p.region.boundary.orientation.outward_sign==-1]
        assert len(productions)==1
        providers=productions[0].producer.boundary_providers
        assert len(providers)==1 and providers[0].dependencies.fields
        target=providers[0].handle
        # Preserve the exact physical producer/dependencies; only implementation is supplied.
        return replace(boundary,execution_authority=ManualInitialFaceExecution(boundary.execution_authority,BoundaryComponentBinding(target,self.component),self.component),component_bindings=(BoundaryComponentBinding(target,self.component),))

def package_data():
    ordinary=interfaces.GhostBoundary;initial=interfaces.AcceptedInitialGhost
    manifest=ComponentManifest(uri='pops://external.test/initial-field-ghost/failure',component_type='ghost_boundary',version='1.0.0',facets=ordinary.facets,
      signature={'generic':True,'state_components':2,'native_interface':ordinary.signature_declaration(),'native_interface_extensions':{'schema_version':1,'interfaces':[initial.signature_declaration()]}},interfaces=ordinary.manifest_declarations(),
      target={'variants':[{'dimension':2,'scalar':'float64','device':'cpu','features':[]}]},entry_points={'interface_table':'pops_component_interface_v1'})
    source=r'''#include <pops/runtime/config/generated_component_abi.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <cerrno>
#include <climits>
#include <string>
namespace {
void event(const char* kind,int rank,int size,double value,double time){
 const char* base=std::getenv("POPS_TEST_INITIAL_GHOST_LOG");if(!base)return;
 char path[4096];std::snprintf(path,sizeof path,"%s-rank%d.log",base,rank);
 FILE* f=std::fopen(path,"a");if(f){std::fprintf(f,"%s rank=%d size=%d field=%a time=%a\n",kind,rank,size,value,time);std::fclose(f);}
}
struct State{int rank;int size;};
int rejected(PopsComponentStatusV1* status,int code,const char* reason);
bool diagnostic_rank(const char* text,int* value,bool positive){
 if(!text||!*text||(text[0]=='0'&&text[1])||text[0]<'0'||text[0]>'9')return false;
 for(const char* at=text;*at;++at)if(*at<'0'||*at>'9')return false;
 errno=0;char* end=nullptr;const long parsed=std::strtol(text,&end,10);
 if(errno==ERANGE||!end||*end||parsed<0||parsed>INT_MAX||(positive&&parsed==0))return false;
 *value=static_cast<int>(parsed);return true;
}
bool canonical_epoch(const std::string& text){
 if(text.empty()||(text[0]=='0'&&text.size()!=1))return false;
 for(char value:text)if(value<'0'||value>'9')return false;
 errno=0;char* end=nullptr;std::strtoull(text.c_str(),&end,10);
 return errno!=ERANGE&&end&&!*end;
}
bool exact_world_package_descendant(const char* identity,const char* execution){
 if(!identity||!execution||!*execution)return false;
 const std::string root=std::string("MPI_COMM_WORLD/pops.amr.package-assembly/")+execution;
 const std::string candidate(identity);
 if(candidate==root)return true;
 const std::string hierarchy=root+"/pops.amr-system.multiblock/pops.generated-amr-levels/";
 if(candidate.compare(0,hierarchy.size(),hierarchy)!=0)return false;
 const std::string suffix=candidate.substr(hierarchy.size());
 const auto split=suffix.find('/');
 return split!=std::string::npos&&suffix.find('/',split+1)==std::string::npos&&
        canonical_epoch(suffix.substr(0,split))&&canonical_epoch(suffix.substr(split+1));
}
int prepare(const PopsComponentPrepareRequestV1* r,void** out,PopsComponentStatusV1* status){
 if(!status)return 51;
 if(out)*out=nullptr;
 if(!r||!out||r->struct_size<sizeof(PopsComponentPrepareRequestV1))
  return rejected(status,51,"initial Ghost prepare requires a complete request/output");
 int rank=0,size=1;
 const auto& execution=r->execution;
 // Logging and fault rank selection come from the authenticated Native profile,
 // never from a process-global MPI communicator inside provider code.
 if(!diagnostic_rank(std::getenv("POPS_TEST_INITIAL_GHOST_RANK"),&rank,false)||
    !diagnostic_rank(std::getenv("POPS_TEST_INITIAL_GHOST_SIZE"),&size,true)||rank>=size)
  return rejected(status,51,"initial Ghost prepare lacks canonical Native diagnostic rank authority");
 const char* base=std::getenv("POPS_TEST_INITIAL_GHOST_LOG");
 if(base){char path[4096];std::snprintf(path,sizeof path,"%s-rank%d.log",base,rank);
  FILE* f=std::fopen(path,"a");if(f){
   std::fprintf(f,"prepare-authority rank=%d size=%d context=%u identity=%s communicator=%s datatype=%s comm_handle=%lld datatype_handle=%lld backend=%s\n",
    rank,size,execution.context_version,execution.execution_identity?execution.execution_identity:"null",
    execution.communicator_identity?execution.communicator_identity:"null",
    execution.communicator_datatype_identity?execution.communicator_datatype_identity:"null",
    static_cast<long long>(execution.communicator_f_handle),static_cast<long long>(execution.communicator_datatype_f_handle),
    execution.backend_identity?execution.backend_identity:"null");std::fclose(f);}}
 if(!execution.communicator_identity||!execution.communicator_datatype_identity)
  return rejected(status,51,"initial Ghost prepare lacks execution authority identities");
 const char* declared=std::getenv("POPS_TEST_INITIAL_GHOST_EXECUTION_ID");
 if(!declared||!execution.execution_identity||std::strcmp(declared,execution.execution_identity)!=0)
  return rejected(status,51,"initial Ghost prepare belongs to another declared execution context");
 const char* identity=execution.communicator_identity;
 const bool serial=std::strcmp(identity,"serial")==0;
 const bool local=std::strcmp(identity,POPS_EXECUTION_NONCOLLECTIVE_IDENTITY_V1)==0;
 const std::string package_identity=std::string("pops.amr.package-assembly/")+execution.execution_identity;
 const bool package_lane=identity==package_identity;
 if(serial||local){
  if(execution.communicator_f_handle||execution.communicator_datatype_f_handle||
     std::strcmp(execution.communicator_datatype_identity,"none")!=0)
   return rejected(status,51,"initial Ghost local preparation cannot carry collective authority");
 }else if((std::strcmp(identity,"MPI_COMM_WORLD")==0||package_lane||
           exact_world_package_descendant(identity,execution.execution_identity))&&
          std::strcmp(execution.communicator_datatype_identity,"MPI_DOUBLE")==0){
  // PoPS already authenticated this exact owned lane; no MPI routine is called here.
 }else return rejected(status,51,"initial Ghost prepare requires an exact Serial/World/package/local authority");
 *out=new State{rank,size};event("prepare",rank,size,0.,0.);
 *status={sizeof(PopsComponentStatusV1),0,POPS_COMPONENT_CONTINUE_V1,nullptr};return 0;
}
void destroy(void* p){auto* s=static_cast<State*>(p);if(s){event("destroy",s->rank,s->size,0.,0.);delete s;}}
int ordinary(void*,const PopsGhostBoundaryRequestV1*,PopsComponentStatusV1* status){
 *status={sizeof(PopsComponentStatusV1),61,POPS_COMPONENT_ABORT_RUN_V1,"failure board forbids a positive callback before initial fault"};return 0;
}
int rejected(PopsComponentStatusV1* status,int code,const char* reason){
 if(status)*status={sizeof(PopsComponentStatusV1),code,POPS_COMPONENT_ABORT_RUN_V1,reason};
 return code;
}
int initial(void* p,const PopsAcceptedInitialGhostRequestV1* r,PopsComponentStatusV1* status){
 auto* s=static_cast<State*>(p);
 if(s&&r){
  const char* base=std::getenv("POPS_TEST_INITIAL_GHOST_LOG");
  if(base){char path[4096];std::snprintf(path,sizeof path,"%s-rank%d.log",base,s->rank);
   FILE* f=std::fopen(path,"a");if(f){const auto& point=r->region_request.logical_time;
    std::fprintf(f,"initial-entry rank=%d size=%d dt=%a time=%a target=%s\n",s->rank,s->size,point.dt,point.physical_time,
      std::getenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK")?"present":"absent");std::fclose(f);}}
 }
 if(!s||!r||!status)return rejected(status,52,"initial Ghost callback missing prepared state/request/status");
 const char* text=std::getenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK");
 if(!text||!*text)return rejected(status,55,"initial Ghost fault target is not armed before callback");
 // Canonical decimal integer only; no sign, whitespace, partial parse or overflow.
 if((text[0]=='0'&&text[1])||text[0]<'0'||text[0]>'9')return rejected(status,55,"initial Ghost fault target has invalid canonical rank authority");
 for(const char* at=text;*at;++at)if(*at<'0'||*at>'9')return rejected(status,55,"initial Ghost fault target has invalid canonical rank authority");
 errno=0;char* end=nullptr;const long target=std::strtol(text,&end,10);
 if(errno==ERANGE||!end||*end||target<0||target>=s->size)return rejected(status,55,"initial Ghost fault target has invalid canonical rank authority");
 const auto& q=r->region_request;
 if(r->point_contract_version!=1||q.logical_time.tick!=0||q.logical_time.dt!=0.||!q.dependencies||!q.ghosts.data)return rejected(status,53,"initial Ghost callback point/dependency/output contract is invalid");
 const PopsQualifiedConstFieldV1* field=nullptr;
 for(std::size_t i=0;i<q.dependency_count;++i)if(q.dependencies[i].present&&q.dependencies[i].values.component_count==1)field=&q.dependencies[i];
 if(!field||!field->values.data)return rejected(status,54,"initial Ghost callback has no scalar Field dependency");
 std::ptrdiff_t offset=0;for(int axis=0;axis<field->values.dimension;++axis)offset+=field->values.ghost_lower[axis]*field->values.axis_strides[axis];
 const double phi=static_cast<const double*>(field->values.data)[offset];
 event("field-before-write",s->rank,s->size,phi,q.logical_time.physical_time);
 // Dirty genuine ABI scratch before the outer owner makes its collective decision.
 *static_cast<double*>(q.ghosts.data)=-1234.;event("tentative-write",s->rank,s->size,phi,q.logical_time.physical_time);
 if(s->rank==target){event("injected-failure",s->rank,s->size,phi,q.logical_time.physical_time);
 *status={sizeof(PopsComponentStatusV1),73,POPS_COMPONENT_ABORT_RUN_V1,"independent initial Ghost rank-local failure after Field read/write"};}
 else *status={sizeof(PopsComponentStatusV1),0,POPS_COMPONENT_CONTINUE_V1,nullptr};
 return 0;
}
const PopsGhostBoundaryApiV1 ghost={{sizeof(PopsGhostBoundaryApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,1,prepare,destroy},ordinary};
const PopsAcceptedInitialGhostApiV1 init={{sizeof(PopsAcceptedInitialGhostApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_NATIVE_INTERFACE_ACCEPTED_INITIAL_GHOST_V1,1,prepare,destroy},initial};
const PopsComponentInterfaceEntryV1 entries[]={{POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1,1,sizeof ghost,&ghost},{POPS_NATIVE_INTERFACE_ACCEPTED_INITIAL_GHOST_V1,1,sizeof init,&init}};
const PopsComponentApiV1 api={sizeof(PopsComponentApiV1),POPS_COMPONENT_PROTOCOL_ABI_V1,POPS_ABI_KEY_LITERAL,POPS_COMPONENT_CATALOG_SHA256_V1,COMPONENT_ID,SEMANTIC_ID,MANIFEST_ID,2,entries};
}
extern "C" const PopsComponentApiV1* pops_component_interface_v1(){return &api;}
'''
    for key,value in [('COMPONENT_ID',manifest.component_id),('SEMANTIC_ID',manifest.semantic_digest.token),('MANIFEST_ID',manifest.manifest_digest.token)]:source=source.replace(key,json.dumps(value))
    return manifest,source.encode()

def load_component(directory):
    directory.mkdir(parents=True,exist_ok=False)
    manifest,source=package_data();name='initial_failure.cpp'
    (directory/name).write_bytes(source)
    data=build_source_package_manifest(components={'ghost':manifest},payloads={name:('source',source)})
    path=directory/'initial-failure.pops.json';path.write_text(json.dumps(data)+'\n')
    return load(path).require('ghost',interface=interfaces.GhostBoundary)()
