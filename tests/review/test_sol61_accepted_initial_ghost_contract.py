"""Source/host initial-point contract, not Native qualification."""
from pathlib import Path
import shutil
import subprocess
import pytest
from pops import interfaces

ROOT=Path(__file__).resolve().parents[2]

def test_distinct_initial_interface_preserves_v1():
    assert (interfaces.GhostBoundary.abi_id,interfaces.GhostBoundary.version)==(1,1)
    initial=interfaces.AcceptedInitialGhost
    assert (initial.abi_id,initial.version)==(11,1)
    assert initial.cpp_table=='PopsAcceptedInitialGhostApiV1'
    assert initial.operations==('apply_initial_region_batch',)

@pytest.mark.parametrize('dimension',(1,2,3))
def test_actual_initial_point_validator_host(tmp_path,dimension):
    compiler=shutil.which('clang++') or shutil.which('c++')
    if compiler is None:pytest.skip('C++20 host compiler unavailable')
    source=tmp_path/'point.cpp';binary=tmp_path/'point'
    # Actual complete consumer header, not an extracted substitute.
    source.write_text('#include <pops/runtime/dynamic/component_consumers.hpp>\n'
        '#include "tests/cpp/support/component_abi_test_helpers.hpp"\n'
        '#include <cassert>\n#include <cmath>\n#include <numeric>\n#include <stdexcept>\n'
        'using namespace pops::component;\n'+r'''
int main(){
 PopsLogicalTimeV1 initial{sizeof(PopsLogicalTimeV1),"primary",0,0,0,0,0,1,0.,0.};
 for(int level=0;level<5;++level){initial.level=level;validate_accepted_initial_ghost_point(initial,1);}
 initial.physical_time=7.5;validate_accepted_initial_ghost_point(initial,1);
 for(int n=0;n<14;++n){auto bad=initial;unsigned version=1;
  switch(n){case 0:version=2;break;case 1:bad.tick=1;break;case 2:bad.level=-1;break;
   case 3:bad.stage=1;break;case 4:bad.substep=1;break;case 5:bad.fraction_numerator=1;break;
   case 6:bad.fraction_denominator=2;break;case 7:bad.dt=0.1;break;
   case 8:bad.physical_time=std::nan("");break;case 9:bad.clock_identity="";break;
   case 10:bad.struct_size=0;break;case 11:bad.dt=std::nan("");break;
   case 12:bad.dt=-0.;break;case 13:bad.physical_time=INFINITY;break;}
  bool refused=false;try{validate_accepted_initial_ghost_point(bad,version);}
  catch(const std::invalid_argument&){refused=true;}assert(refused);
 }
 assert(pops::component::generated_native_interface_version(POPS_NATIVE_INTERFACE_GHOST_BOUNDARY_V1)==1);
 assert(pops::component::generated_native_interface_version(POPS_NATIVE_INTERFACE_ACCEPTED_INITIAL_GHOST_V1)==1);
#if POPS_NATIVE_DIM == 2
 namespace abi=pops::component::test_support;
 double q[2]{1.,2.},phi[2]{2.,2.},ghosts[2]{},coordinates[2]{};
 int axis=0,side=-1;
 PopsQualifiedConstFieldV1 dependency{sizeof(PopsQualifiedConstFieldV1),1,"case::phi",abi::const_field_view(phi,1,1,2)};
 PopsGhostBoundaryRequestV1 region{sizeof(PopsGhostBoundaryRequestV1),"case::provider","case::Q","case::ghost",
   abi::const_field_view(q,1,1,2),abi::field_view(ghosts,1,1,2),abi::const_field_view(coordinates,1,1,2),
   {sizeof(PopsBoundaryRegionV1),POPS_BOUNDARY_FACE_V1,2,1,1,&axis,&side,"x-low"},1,&dependency,0,nullptr,
   {sizeof(PopsLogicalTimeV1),"primary",0,0,0,0,0,1,0.,0.},abi::noncollective_host_execution_context()};
 PopsGhostBoundaryApiV1 ordinary{};
 ordinary.apply_region_batch=+[](void*,const PopsGhostBoundaryRequestV1*,PopsComponentStatusV1*){assert(false);return 1;};
 PopsAcceptedInitialGhostApiV1 api{};
 api.apply_initial_region_batch=+[](void*,const PopsAcceptedInitialGhostRequestV1* request,PopsComponentStatusV1*){
   assert(request->point_contract_version==1 && request->region_request.logical_time.dt==0.);
   const auto& region=request->region_request;
   auto* output=static_cast<double*>(region.ghosts.data);
   const auto* input=static_cast<const double*>(region.dependencies[0].values.data);
   output[0]=1.;output[1]=input[0];return 0;
 };
 require_initial_ghost_provider_lifecycle(api,ordinary);
 auto mismatch=api;mismatch.header.destroy=+[](void*){};
 bool refused=false;try{require_initial_ghost_provider_lifecycle(mismatch,ordinary);}
 catch(const std::invalid_argument&){refused=true;}assert(refused);
 PopsComponentStatusV1 status{};
 refused=false;try{apply_ghost_boundary(ordinary,nullptr,region,status);}
 catch(const std::invalid_argument&){refused=true;}assert(refused && ghosts[0]==0. && ghosts[1]==0.);
 PopsAcceptedInitialGhostRequestV1 request{sizeof(PopsAcceptedInitialGhostRequestV1),1,region};
 assert(apply_accepted_initial_ghost(api,nullptr,request,status)==0 && ghosts[0]==1. && ghosts[1]==2.);
 auto bad=request;bad.region_request.logical_time.tick=1;
 refused=false;try{apply_accepted_initial_ghost(api,nullptr,bad,status);}
 catch(const std::invalid_argument&){refused=true;}assert(refused);
#endif
}
''')
    subprocess.run([compiler,'-std=c++20',f'-DPOPS_NATIVE_DIM={dimension}','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),'-I',str(ROOT),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True,capture_output=True)

def test_genuine_initial_field_then_freshness_then_ghost_order():
    source=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    start=source.index('void AmrSystem<Dim>::prepare_accepted_halo_candidates(')
    body=source[start:source.index('void AmrSystem<Dim>::publish_prepared_amr_program_candidates(',start)]
    field=body.index('p_->prepare_accepted_halo_field_dependencies(block, level, points[block][level]);')
    freshness=body.index('accepted_halo_boundary_preflight[block][level](points[block][level]);')
    ghost=body.index('prepare_generated_amr_block_level_state(',freshness)
    assert field<freshness<ghost<body.index('accepted halo candidate preparation/copy/fence failed collectively',ghost)
    assert 'accepted_halo_initial_point_preflight[block][level](point)' in body[:field]
    assert 'restore_accepted();' in body[ghost:]
    preflight=source[source.index('// Capability/point-only preflight'):source.index('prepared_fluxes.reserve',source.index('// Capability/point-only preflight'))]
    assert preflight.index('accepted_halo_initial_point_preflight')<preflight.index('accepted_halo_boundary_preflight')
    assert 'produced.clock != point.clock' in preflight and 'produced.dt != point.dt' in preflight

def test_public_screened_field_board_and_separate_dynamic_inflow_gap():
    import pops
    from tests.python.support.accepted_initial_field_ghost_case import build
    case,layout=build()
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert len(resolved.field_plans)==1
    assert layout.execution.runtime_execution_data()['mode']=='synchronous'
    dynamic,layout=build(ghost=True)
    with pytest.raises(pops._report.DiagnosticError,match='requires a compiled boundary component'):
        pops.resolve(pops.validate(dynamic),layout=layout)
