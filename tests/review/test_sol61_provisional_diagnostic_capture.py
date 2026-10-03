"""Actual C++ method probes and explicit Source-only Native boundary orchestration."""
from pathlib import Path
from types import SimpleNamespace
import shutil,struct,subprocess
import pytest
from tests.review.test_sol61_program_diagnostic_checkpoint_native import _method
ROOT=Path(__file__).resolve().parents[2]

def test_actual_diagnostic_capture_guards_preserve_idle_observer(tmp_path):
    source=(ROOT/'src/runtime/system/system_program.cpp').read_text()
    methods='\n'.join(_method(source,s) for s in (
        'std::vector<std::uint8_t> System<Dim>::checkpoint_program_diagnostics() const',
        'std::vector<std::uint8_t> System<Dim>::checkpoint_capture_program_diagnostics() const'))
    code=r'''
#include <pops/runtime/program/program_diagnostics_checkpoint.hpp>
#include <atomic>
#include <cassert>
#include <memory>
namespace Kokkos {int fences=0;void fence(){++fences;}}
namespace pops {
struct Lane{int rank()const{return 0;}int size()const{return 1;}};
struct Impl {
 struct Program{std::map<std::string,Real> diagnostics_;}program_;
 struct Lifecycle {std::string phase="bound";std::string state(int)const{return phase;}}lifecycle_;
 int depth=0,macro_step_=0;bool external_step_transaction_=false,external_step_transaction_committed_=false,external_restart_transaction_=false;
};
struct Authority{std::atomic<int>pending{0};};
template<int Dim>struct System {
 std::unique_ptr<Impl>p_=std::make_unique<Impl>();std::unique_ptr<Authority>solve_outcome_authority_=std::make_unique<Authority>();Lane lane;
 int step_transaction_depth()const{return p_->depth;}
 const Lane&prepared_boundary_execution_lane()const{return lane;}
 std::vector<std::uint8_t>checkpoint_program_diagnostics()const;
 std::vector<std::uint8_t>checkpoint_capture_program_diagnostics()const;
};
METHODS
}
template<class F>void rejects(F f){int before=Kokkos::fences;bool fail=false;try{f();}catch(const std::logic_error&){fail=true;}assert(fail&&Kokkos::fences==before);}
int main(){
 pops::System<2>s;s.p_->program_.diagnostics_["opaque-current-value"]=-0.;
 auto accepted=s.checkpoint_program_diagnostics();assert(s.checkpoint_capture_program_diagnostics()==accepted);
 s.p_->depth=1;rejects([&]{s.checkpoint_capture_program_diagnostics();});
 s.p_->external_step_transaction_=true;
 for(auto phase:{"bound","running","checkpointed"}){s.p_->lifecycle_.phase=phase;assert(s.checkpoint_capture_program_diagnostics()==accepted);rejects([&]{s.checkpoint_program_diagnostics();});}
 s.p_->external_step_transaction_committed_=true;rejects([&]{s.checkpoint_capture_program_diagnostics();});s.p_->external_step_transaction_committed_=false;
 s.p_->depth=2;rejects([&]{s.checkpoint_capture_program_diagnostics();});s.p_->depth=1;
 s.p_->external_restart_transaction_=true;rejects([&]{s.checkpoint_capture_program_diagnostics();});s.p_->external_restart_transaction_=false;
 s.solve_outcome_authority_->pending=1;rejects([&]{s.checkpoint_capture_program_diagnostics();});s.solve_outcome_authority_->pending=0;
 for(auto phase:{"unbound","failed","destroyed"}){s.p_->lifecycle_.phase=phase;rejects([&]{s.checkpoint_capture_program_diagnostics();});}
 assert(s.p_->program_.diagnostics_.size()==1);
}
'''.replace('METHODS',methods)
    cpp=tmp_path/'actual.cpp';cpp.write_text(code)
    for real in ('double','float'):
        exe=tmp_path/real
        subprocess.run([shutil.which('clang++'),'-std=c++20','-Wall','-Wextra','-Werror','-DPOPS_REAL_TYPE='+real,'-I',str(ROOT/'include'),str(cpp),'-o',str(exe)],check=True,capture_output=True)
        subprocess.run([str(exe)],check=True,capture_output=True)

@pytest.mark.parametrize('provisional',[False,True])
def test_source_bridge_selects_exact_route_and_preserves_codec(monkeypatch,provisional):
    from pops.runtime import _checkpoint_program_diagnostics as codec
    from pops.runtime import _checkpoint_resource_budget as budget
    from pops.output import _checkpoint_collective as collective
    image=b'POPSDIA1'+struct.pack('<QQQQ',64,0,1,0);calls=[]
    def capture(route):calls.append(route);return image
    owner=SimpleNamespace(_s=SimpleNamespace(_checkpoint_program_diagnostics=lambda:capture('idle'),_checkpoint_capture_program_diagnostics=lambda:capture('capture')),
        _checkpoint_program_diagnostic_capacity_per_rank=40,_checkpoint_program_diagnostic_byte_capacity=40)
    monkeypatch.setattr(budget,'_require_checkpoint_diagnostic_authority',lambda owner:None)
    monkeypatch.setattr(collective,'checkpoint_topology',lambda owner:collective.CheckpointTopology(0,1,None))
    def consensus(topology,phase,*,error=None,**kw):
        if error is not None:raise error
    monkeypatch.setattr(collective,'consensus',consensus)
    payload={};codec.capture_checkpoint_program_diagnostics(owner,payload,provisional_capture=provisional)
    assert calls==['capture' if provisional else 'idle']
    assert payload['program_diagnostics_state'].tobytes()==image
    marker=RuntimeError('genuine capture failure')
    def fail():raise marker
    name='_checkpoint_capture_program_diagnostics' if provisional else '_checkpoint_program_diagnostics'
    setattr(owner._s,name,fail)
    with pytest.raises(RuntimeError) as caught:codec.capture_checkpoint_program_diagnostics(owner,{},provisional_capture=provisional)
    assert caught.value is marker
    delattr(owner._s,name)
    with pytest.raises(TypeError,match='exact Program diagnostic capture'):codec.capture_checkpoint_program_diagnostics(owner,{},provisional_capture=provisional)

def test_uniform_calls_capture_route_amr_remains_idle():
    assert 'capture_checkpoint_program_diagnostics(self, out, provisional_capture=True)' in (ROOT/'python/pops/runtime/_system_io.py').read_text()
    assert 'provisional_capture=True' not in (ROOT/'python/pops/runtime/_amr_checkpoint_v3.py').read_text()
