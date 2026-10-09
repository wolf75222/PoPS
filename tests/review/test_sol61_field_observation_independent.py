"""Independent adversaries on exact production getter/codec; SOURCE/host only."""
from pathlib import Path
import ast, shutil, subprocess
ROOT=Path(__file__).resolve().parents[2]

def test_actual_getter_qualified_output_and_detached_return(tmp_path):
    # Reuse explicit owner stand-ins, not the author's assertions or execution.
    tree=ast.parse((ROOT/'tests/review/test_sol61_field_candidate_observation_source.py').read_text())
    scaffolds=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('#include <vector>')]
    assert len(scaffolds)==1
    cpp=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    body=cpp.split('std::vector<FieldCandidateObservation> AmrSystem<Dim>::field_candidate_observations() const',1)[1].split('template <int Dim>',1)[0].strip()
    code=scaffolds[0]+body+r'''
int main(){AmrSystem<2> s;
 FieldCandidateObservation a{3,4,{.25,999},{1,2,3}}, b=a;
 b.provider_slot="other";b.output_key="second";
 Plan other;other.output_key="second";s.p_->field_plans.emplace("other",other);
 s.p_->field_candidate_observations={{0,a},{1,b}};
 auto rows=s.field_candidate_observations();assert(rows.size()==2);
 rows[1].carrier_bytes[0]=17;rows[1].provider_slot="tampered";
 auto original=s.field_candidate_observations();assert(original[1].carrier_bytes[0]==1&&original[1].provider_slot=="other");
 auto no=[&]{bool refused=false;try{(void)s.field_candidate_observations();}catch(const std::invalid_argument&){refused=true;}assert(refused);};
 s.p_->field_plans["other"].output_key="first";no();s.p_->field_plans["other"].output_key="second";
 s.p_->field_plans["other"].plan_identity="foreign";no();s.p_->field_plans["other"].plan_identity="i";
 s.p_->field_plans.erase("other");no();s.p_->field_plans.emplace("other",other);
 s.p_->field_candidate_observations[1].owner_macro_step=3;no();s.p_->field_candidate_observations[1].owner_macro_step=2;
 s.p_->field_candidate_observations.clear();assert(s.field_candidate_observations().empty());
}
'''
    source=tmp_path/'independent.cpp';source.write_text(code)
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    binary=tmp_path/'probe'
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True)

def test_actual_real32_codec_multicomponent_permuted_levels(tmp_path):
    code=r'''#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <cassert>
#include <algorithm>
using namespace pops::runtime::checkpoint;
int main(){StateCarrierArchive<2> a;a.real_bits=32;a.ranks=2;a.shard=1;a.levels=3;a.blocks={"qualified/z","qualified/a"};
 for(unsigned l:{2u,0u,1u})for(unsigned b:{1u,0u}){StateCarrierPatch<2> p;p.block=b;p.level=l;p.patch=9+l;p.components=3;p.owner=1;
 p.lo={-2,4};p.hi={-1,5};p.grown_lo={-3,4};p.grown_hi={0,6};p.bits.resize(36);for(unsigned i=0;i<36;++i)p.bits[i]=i%2?0x80000000u:0x00000001u;a.patches.push_back(p);}
 bool order_refused=false;try{(void)encode_state_carriers(a);}catch(const std::invalid_argument&){order_refused=true;}assert(order_refused);
 std::sort(a.patches.begin(),a.patches.end(),[](const auto& x,const auto& y){return std::tie(x.block,x.level,x.patch)<std::tie(y.block,y.level,y.patch);});
 auto bytes=encode_state_carriers(a);auto decoded=decode_state_carriers<2>(bytes);assert(decoded.patches==a.patches&&decoded.blocks==a.blocks);
 bytes[0]='X';bool no=false;try{(void)decode_state_carriers<2>(bytes);}catch(const std::invalid_argument&){no=true;}assert(no);
}
'''
    source=tmp_path/'codec.cpp';source.write_text(code)
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    binary=tmp_path/'codec'
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True)


def test_activation_prebootstrap_contract_has_distinct_transaction_guards():
    cpp=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    activation=cpp.split('void AmrSystem<Dim>::enable_field_candidate_observation(',1)[1].split('template <int Dim>',1)[0]
    preflight=activation.split('catch (...)',1)[0]
    assert 'p_->bootstrap_transaction' in preflight, 'active bootstrap is not a pre-bootstrap owner'
    assert 'p_->restart_transaction' in preflight, 'initial-cursor restart remains an active transaction'
    assert activation.index('rethrow_collective_failure') < activation.index('all_ranks_agree_exact_ordered_byte_pairs') < activation.index('field_candidate_observation_enabled = true')


def test_real_bootstrap_constructor_and_actual_fixture_activation_wrapper(monkeypatch):
    # Execute the genuine Python lifecycle methods with explicitly Source-only engine spies.
    from types import SimpleNamespace
    from pops.runtime._amr_bootstrap_execution import NativeAMRBootstrapConsumer
    fixture=ast.parse((ROOT/'tests/python/integration/amr/test_public_initial_field_ghost.py').read_text())
    wrappers=[n for n in ast.walk(fixture) if isinstance(n,ast.FunctionDef) and any(isinstance(x,ast.Attribute) and x.attr=='_enable_field_candidate_observation' for x in ast.walk(n))]
    # The outer test also contains that attribute; retain only the innermost wrapper.
    wrappers=[n for n in wrappers if not any(isinstance(x,ast.FunctionDef) and x is not n for x in ast.walk(n))]
    assert len(wrappers)==1
    events=[]
    class State:
        active=False
        def _enable_field_candidate_observation(self,version):
            assert not self.active, 'activation is inside real bootstrap transaction'
            assert version==1;events.append('enable')
        def _begin_bootstrap_plan(self):self.active=True;events.append('begin')
    state=State()
    class Engine:
        _s=state
        def _commit_bootstrap_level(self):assert state.active;state.active=False;events.append('commit')
    original_init=NativeAMRBootstrapConsumer.__init__
    original_finalize=NativeAMRBootstrapConsumer.finalize_bootstrap
    scope={'original_init':original_init,'original_finalize':original_finalize,'world':None,'collective_call':lambda world,fn:fn()}
    exec(compile(ast.Module(body=wrappers,type_ignores=[]),'<actual fixture wrapper>','exec'),scope)
    wrapper=scope[wrappers[0].name]
    method='__init__' if 'init' in wrappers[0].name else 'finalize_bootstrap'
    monkeypatch.setattr(NativeAMRBootstrapConsumer,method,wrapper)
    plan=SimpleNamespace(identity=SimpleNamespace(to_data=lambda:{'schema':'source-only-plan'}))
    owner=NativeAMRBootstrapConsumer(Engine(),plan,[])
    owner.finalize_bootstrap()
    assert events==['enable','begin','commit']
