"""Extracted actual adapter + actual codec, with serial storage/transport stubs.
No real MultiFab/Kokkos/MPI execution is claimed by this bounded host probe.
"""
import shutil,subprocess
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('dim',(1,2,3))
def test_actual_adapter_preserves_grown_bits_and_refuses_nonaccepted(tmp_path,dim):
    text=(ROOT/'src/runtime/system/system.cpp').read_text()
    start=text.index('template <int Dim>\nstd::vector<std::vector<std::uint8_t>> System<Dim>::capture_state_storage_(bool provisional_capture)')
    end=text.index('\ntemplate <int Dim>\nstd::uint64_t System<Dim>::checkpoint_state_carriers_capacity',start)
    body=text[start:end]
    source=tmp_path/'adapter.cpp'
    source.write_text(r"""
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <pops/core/identity/prepared_provider.hpp>
#include <cassert>
#include <cmath>
#include <exception>
#include <memory>
#include <string_view>
#include <type_traits>
namespace Kokkos {void fence(){}}
namespace pops {
using Real=double;
using ExactOrderedBytePair=std::pair<std::string_view,std::string_view>;
struct Lane {int rank()const{return 0;} int size()const{return 1;}};
void collectively_rethrow_exception(std::exception_ptr e,const Lane&,const char*){if(e)std::rethrow_exception(e);}
bool all_ranks_agree_exact_ordered_byte_pairs(const std::vector<ExactOrderedBytePair>& pairs,const Lane&){assert(!pairs[0].second.empty());return true;}
void broadcast_bytes_inplace(char*,std::size_t,const Lane&,int){}
template<int D>struct Box{std::array<int,D>lo{},hi{};};
struct Mirror{std::vector<Real> data;std::size_t size()const{return data.size();}Real operator()(std::size_t i)const{return data[i];}};
template<int D>struct Fab{Box<D>valid,grown;Mirror storage;const Box<D>&box()const{return valid;}const Box<D>&grown_box()const{return grown;}Mirror create_host_mirror()const{return storage;}void copy_to_host(Mirror&h)const{h=storage;}};
struct Distribution{bool replicated()const{return false;}};
template<int D>struct Field{Fab<D>f;int components=0;std::size_t local_size()const{return 1;}const Fab<D>&fab(std::size_t)const{return f;}std::size_t global_index(std::size_t)const{return 0;}int ncomp()const{return components;}Distribution distribution()const{return {};}};
struct Lifecycle{std::string phase="bound";std::string state(int)const{return phase;}};
struct Names{std::vector<std::string>names()const{return {"vector-first","scalar-second"};}};
template<int D>class System{public:
struct Block{Field<D>U;};struct Impl{Lifecycle lifecycle_;int macro_step_=0;double t=7.5;bool external_restart_transaction_=false;bool external_step_transaction_=false;bool external_step_transaction_committed_=false;Names blocks_;std::array<Block,2>blocks;const Block&find(const std::string&name)const{return blocks[name=="vector-first"?0:1];}};
std::unique_ptr<Impl>p_=std::make_unique<Impl>();std::size_t depth=0;
std::size_t step_transaction_depth()const{return depth;}const Lane&prepared_boundary_execution_lane()const{static Lane lane;return lane;}
std::vector<std::vector<std::uint8_t>> observe_accepted_state_storage()const;
std::vector<std::vector<std::uint8_t>> capture_state_storage_(bool provisional_capture)const;
std::vector<std::uint8_t> checkpoint_state_carriers()const;
};
"""+body+r"""
}
int main(){using namespace pops;using namespace pops::runtime::checkpoint;constexpr int D=TEST_DIM;System<D>system;
for(int b=0;b<2;++b){auto&field=system.p_->blocks[b].U;field.components=b?1:3;std::size_t count=field.components;for(int a=0;a<D;++a){field.f.valid.lo[a]=0;field.f.valid.hi[a]=1;field.f.grown.lo[a]=-(a+1);field.f.grown.hi[a]=2+a;count*=4+2*a;}field.f.storage.data.resize(count);for(std::size_t i=0;i<count;++i)field.f.storage.data[i]=double(100*b+i)/7.;}
auto first=system.observe_accepted_state_storage();auto local=decode_state_carriers<D>(first[0]);auto complete=decode_state_carriers<D>(first[1]);assert(local.shard==0&&complete.shard==-1&&complete.blocks.size()==2);for(int b=0;b<2;++b){auto&patch=complete.patches[b];assert(patch.components==std::uint64_t(b?1:3));for(std::size_t i=0;i<patch.bits.size();++i)assert(patch.bits[i]==std::bit_cast<std::uint64_t>(system.p_->blocks[b].U.f.storage.data[i]));}
system.p_->blocks[0].U.f.storage.data[0]=-1234.;assert(system.observe_accepted_state_storage()[1]!=first[1]);assert(decode_state_carriers<D>(first[1]).patches[0].bits[0]!=std::bit_cast<std::uint64_t>(-1234.));
for(int mode=0;mode<3;++mode){system.depth=mode==0;system.p_->external_restart_transaction_=mode==1;system.p_->lifecycle_.phase=mode==2?"assembling":"bound";bool refused=false;try{(void)system.observe_accepted_state_storage();}catch(const std::logic_error&){refused=true;}assert(refused);}
system.depth=1;system.p_->external_restart_transaction_=false;system.p_->lifecycle_.phase="running";system.p_->external_step_transaction_=true;system.p_->external_step_transaction_committed_=false;assert(!system.checkpoint_state_carriers().empty());
for(int mode=0;mode<3;++mode){system.depth=mode==0?2:1;system.p_->external_step_transaction_=mode!=1;system.p_->external_step_transaction_committed_=mode==2;bool refused=false;try{(void)system.checkpoint_state_carriers();}catch(const std::logic_error&){refused=true;}assert(refused);}

}
""")
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    executable=tmp_path/'adapter'
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-DTEST_DIM='+str(dim),'-I'+str(ROOT/'include'),str(source),'-o',str(executable)],check=True,capture_output=True,text=True)
    subprocess.run([str(executable)],check=True)
