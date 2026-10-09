"""Actual header and actual Field admission fragment, Source/host only."""
from pathlib import Path
import shutil,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('dim',(1,2,3))
def test_actual_initial_field_authority_and_exact_field_gate(tmp_path,dim):
    source=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    method=source[source.index('SolveOutcome AmrSystem<Dim>::solve_program_field_from_blocks_on_prepared_lane('):]
    fragment=method[method.index('    const auto& topology_point'):method.index('    if (provider_slot.empty())')]
    cpp=r'''#include <pops/runtime/accepted_initial_field_point.hpp>
#include <optional>
#include <cassert>
#include <functional>
using pops::runtime::multiblock::BoundaryEvaluationPoint;
using pops::runtime::PreparedAcceptedInitialFieldPointV1;
struct Owner { std::optional<BoundaryEvaluationPoint> active_topology_rematerialization_point;
 const PreparedAcceptedInitialFieldPointV1* active_initial_field_point=nullptr;
 bool bootstrap_transaction=false; };
void gate(Owner* p_,const BoundaryEvaluationPoint& point,int active_level){
''' + fragment + r'''}
bool refuses(const std::function<void()>& f){try{f();return false;}catch(const std::invalid_argument&){return true;}}
int main(){
 BoundaryEvaluationPoint p; p.clock="actual.primary";p.dt=0.;p.physical_time=7.5;
 for(int level=0;level<5;++level){
  p.level=level;
  PreparedAcceptedInitialFieldPointV1 authority(p,"actual.primary",0,7.5,level,true);
  Owner owner;assert(refuses([&]{gate(&owner,p,level);}));
  owner.active_initial_field_point=&authority;
  assert(refuses([&]{gate(&owner,p,level);}));
  owner.bootstrap_transaction=true;gate(&owner,p,level);
  auto stale=p;stale.physical_time=7.6;assert(refuses([&]{gate(&owner,stale,level);}));
  stale=p;stale.clock="other";assert(refuses([&]{gate(&owner,stale,level);}));
  stale=p;stale.tick=1;assert(refuses([&]{gate(&owner,stale,level);}));
  stale=p;stale.dt=-0.;assert(refuses([&]{gate(&owner,stale,level);}));
  owner.active_initial_field_point=nullptr;auto positive=p;positive.dt=.125;gate(&owner,positive,level);
  owner.active_topology_rematerialization_point=p;gate(&owner,p,level); // existing exact topology route preserved
 }
 p.level=0;
 assert(refuses([&]{PreparedAcceptedInitialFieldPointV1 a(p,"foreign",0,7.5,0,true);}));
 assert(refuses([&]{PreparedAcceptedInitialFieldPointV1 a(p,"actual.primary",1,7.5,0,true);}));
 assert(refuses([&]{PreparedAcceptedInitialFieldPointV1 a(p,"actual.primary",0,0.,0,true);}));
 assert(refuses([&]{PreparedAcceptedInitialFieldPointV1 a(p,"actual.primary",0,7.5,1,true);}));
 assert(refuses([&]{PreparedAcceptedInitialFieldPointV1 a(p,"actual.primary",0,7.5,0,false);}));
 p.dt=.1;assert(refuses([&]{PreparedAcceptedInitialFieldPointV1 a(p,"actual.primary",0,7.5,0,true);}));
}
'''
    path=tmp_path/'actual.cpp';path.write_text(cpp);binary=tmp_path/'probe'
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler,'actual host compiler required'
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',f'-DPOPS_NATIVE_DIM={dim}','-I',str(ROOT/'include'),str(path),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True)

def test_real_scope_vote_solver_witness_and_wire_versions():
    s=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    body=s[s.index('  void prepare_accepted_halo_field_dependencies('):s.index('  using AcceptedHaloPointPack')]
    assert body.index('initial_authority.emplace')<body.index('accepted halo Field simultaneous source allocation failed collectively')<body.index('active_initial_field_point = &*initial_authority')<body.index('auto outcome = facade->solve_program_field_from_blocks_on_prepared_lane')
    assert 'macro_step, accepted_time' in body and 'bool(bootstrap_transaction)' in body
    assert '~InitialFieldPointReset() { owner.active_initial_field_point = previous; }' in body
    assert body.index('validate_field_candidate();')<body.index('plan.accepted_halo_producer_point = point')
    method=s[s.index('SolveOutcome AmrSystem<Dim>::solve_program_field_from_blocks_on_prepared_lane('):]
    assert 'authenticated_initial_point ? std::uint32_t{3} : std::uint32_t{2}' in method
    assert 'pops.amr.accepted-initial-field-point@1' in method
    assert method.index('request_contract = std::move(request).release()')<method.index('all_ranks_agree_exact_ordered_byte_pairs')<method.index('report = p_->solve_field_candidate')
