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
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
namespace {
void event(const char* kind,int rank,int size,double value,double time){
 const char* base=std::getenv("POPS_TEST_INITIAL_GHOST_LOG");if(!base)return;
 char path[4096];std::snprintf(path,sizeof path,"%s-rank%d.log",base,rank);
 FILE* f=std::fopen(path,"a");if(f){std::fprintf(f,"%s rank=%d size=%d field=%a time=%a\n",kind,rank,size,value,time);std::fclose(f);}
}
struct State{int rank;int size;};
int prepare(const PopsComponentPrepareRequestV1* r,void** out,PopsComponentStatusV1* status){
 if(!r||!out||!status)return 51;
 int rank=0,size=1;MPI_Comm comm=MPI_Comm_f2c(static_cast<MPI_Fint>(r->execution.communicator_f_handle));
 MPI_Comm_rank(comm,&rank);MPI_Comm_size(comm,&size);*out=new State{rank,size};event("prepare",rank,size,0.,0.);
 *status={sizeof(PopsComponentStatusV1),0,POPS_COMPONENT_CONTINUE_V1,nullptr};return 0;
}
void destroy(void* p){auto* s=static_cast<State*>(p);if(s){event("destroy",s->rank,s->size,0.,0.);delete s;}}
int ordinary(void*,const PopsGhostBoundaryRequestV1*,PopsComponentStatusV1* status){
 *status={sizeof(PopsComponentStatusV1),61,POPS_COMPONENT_ABORT_RUN_V1,"failure board forbids a positive callback before initial fault"};return 0;
}
int initial(void* p,const PopsAcceptedInitialGhostRequestV1* r,PopsComponentStatusV1* status){
 auto* s=static_cast<State*>(p);if(!s||!r||!status)return 52;
 const auto& q=r->region_request;
 if(r->point_contract_version!=1||q.logical_time.tick!=0||q.logical_time.dt!=0.||!q.dependencies||!q.ghosts.data)return 53;
 const PopsQualifiedConstFieldV1* field=nullptr;
 for(std::size_t i=0;i<q.dependency_count;++i)if(q.dependencies[i].present&&q.dependencies[i].values.component_count==1)field=&q.dependencies[i];
 if(!field||!field->values.data)return 54;
 std::ptrdiff_t offset=0;for(int axis=0;axis<field->values.dimension;++axis)offset+=field->values.ghost_lower[axis]*field->values.axis_strides[axis];
 const double phi=static_cast<const double*>(field->values.data)[offset];
 event("field-before-write",s->rank,s->size,phi,q.logical_time.physical_time);
 // Dirty genuine ABI scratch before the outer owner makes its collective decision.
 *static_cast<double*>(q.ghosts.data)=-1234.;event("tentative-write",s->rank,s->size,phi,q.logical_time.physical_time);
 const char* target=std::getenv("POPS_TEST_INITIAL_GHOST_TARGET_RANK");
 if(!target)return 55;
 if(s->rank==std::atoi(target)){event("injected-failure",s->rank,s->size,phi,q.logical_time.physical_time);
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
