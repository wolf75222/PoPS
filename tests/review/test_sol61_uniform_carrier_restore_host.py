"""Actual native adapter bodies + actual POPSCAR1 codec, explicit host storage seam.
No Kokkos/device/MPI or installed Native execution is claimed.
"""
from pathlib import Path
import shutil,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
@pytest.mark.parametrize('dim',(1,2,3))
def test_actual_restore_adapter_exact_ghosts_and_projection_refusal(tmp_path,dim):
    text=(ROOT/'src/runtime/system/system.cpp').read_text()
    start=text.index('template <int Dim>\nvoid System<Dim>::validate_checkpoint_state_carriers')
    end=text.index('\ntemplate <int Dim>\nvoid System<Dim>::declare_program_integral',start)
    body=text[start:end]
    code=r'''
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <pops/mesh/index/box.hpp>
#include <cassert>
#include <bit>
#include <cmath>
#include <exception>
#include <memory>
#include <string_view>
namespace Kokkos {void fence(){} void deep_copy(const std::shared_ptr<std::vector<double>>&dst,const std::shared_ptr<std::vector<double>>&src){*dst=*src;}}
namespace pops {
using Real=double;struct Lane{};
using ExactOrderedBytePair=std::pair<std::string_view,std::string_view>;
void collectively_rethrow_exception(std::exception_ptr e,const Lane&,const char*){if(e)std::rethrow_exception(e);}
bool all_ranks_agree_exact_ordered_byte_pairs(std::initializer_list<ExactOrderedBytePair>,const Lane&){return true;}
struct Mirror {std::vector<double> data;std::size_t size()const{return data.size();}double&operator()(std::size_t i){return data[i];}double operator()(std::size_t i)const{return data[i];}};
struct Distribution{};
template<int D>struct Fab {
 Box<D> valid,grown;std::shared_ptr<std::vector<double>> values=std::make_shared<std::vector<double>>();
 const Box<D>&box()const{return valid;}const Box<D>&grown_box()const{return grown;}
 Mirror create_host_mirror()const{return Mirror{*values};}void copy_to_host(Mirror&h)const{h.data=*values;}
 void copy_from_host(const Mirror&h){*values=h.data;}const auto&storage()const{return values;}
};
template<int D>struct MultiFab {
 std::vector<Box<D>>boxes;Extent<D>g;int nc;std::vector<Fab<D>>fabs;
 MultiFab(std::vector<Box<D>>layout,Distribution,int,int ncomp,Extent<D>ghosts):boxes(layout),g(ghosts),nc(ncomp){for(auto b:boxes){Fab<D>f;f.valid=b;f.grown=b;for(int a=0;a<D;++a){f.grown.lo[a]-=g[a];f.grown.hi[a]+=g[a];}f.values->resize(f.grown.numPts()*nc);fabs.push_back(f);}}
 const auto&layout()const{return boxes;}const auto&ghosts()const{return g;}Distribution distribution()const{return {};}
 int ncomp()const{return nc;}int local_rank()const{return 0;}std::size_t local_size()const{return fabs.size();}std::size_t global_index(std::size_t i)const{return i;}
 Fab<D>&fab(std::size_t i){return fabs[i];}const Fab<D>&fab(std::size_t i)const{return fabs[i];}
};
struct Names {std::vector<std::string>names()const{return {"unrelated-vector"};}};
template<int D>class System {public:
 struct Block{MultiFab<D>U;};struct Impl{Names blocks_;bool external_restart_transaction_=true,external_step_transaction_committed_=false;Block block;
  Impl():block{make()}{}static MultiFab<D>make(){Box<D>b;Extent<D>g;for(int a=0;a<D;++a){b.lo[a]=0;b.hi[a]=1;g[a]=a+1;}return MultiFab<D>({b},{},0,2,g);}const Block&find(const std::string&)const{return block;}Block&find(const std::string&){return block;}};
 std::unique_ptr<Impl>p_=std::make_unique<Impl>();const Lane&prepared_boundary_execution_lane()const{static Lane l;return l;}
 void validate_checkpoint_state_carriers(std::span<const std::uint8_t>)const;void restore_checkpoint_state_carriers(std::span<const std::uint8_t>);
};
'''+body+r'''
}
int main(){using namespace pops;using namespace pops::runtime::checkpoint;constexpr int D=TEST_DIM;System<D>s;auto&f=s.p_->block.U.fab(0);
StateCarrierArchive<D>a;a.real_bits=64;a.ranks=1;a.levels=1;a.blocks={"unrelated-vector"};StateCarrierPatch<D>r;r.components=2;r.owner=0;
for(int d=0;d<D;++d){r.lo[d]=f.box().lo[d];r.hi[d]=f.box().hi[d];r.grown_lo[d]=f.grown_box().lo[d];r.grown_hi[d]=f.grown_box().hi[d];}
r.bits.assign(f.values->size(),0x8000000000000000ULL);const auto stride=f.grown_box().numPts();
for(int c=0;c<2;++c)for(std::size_t cell=0;cell<std::size_t(f.box().numPts());++cell){auto rem=cell;std::size_t addr=c*stride,mult=1;for(int d=0;d<D;++d){auto n=f.box().length(d);auto coord=f.box().lo[d]+rem%n;rem/=n;addr+=(coord-f.grown_box().lo[d])*mult;mult*=f.grown_box().length(d);}r.bits[addr]=std::bit_cast<std::uint64_t>(double(c+1));(*f.values)[addr]=double(c+1);}
a.patches={r};auto bytes=encode_state_carriers(a);auto original=*f.values;
// Same valid projection, different ghosts: actual native restore must preserve -0 bits.
s.restore_checkpoint_state_carriers(bytes);for(std::size_t i=0;i<r.bits.size();++i)assert(std::bit_cast<std::uint64_t>((*f.values)[i])==r.bits[i]);
a.patches[0].bits[0]=0x7ff8000000000042ULL;bytes=encode_state_carriers(a);s.restore_checkpoint_state_carriers(bytes);assert(std::bit_cast<std::uint64_t>((*f.values)[0])==0x7ff8000000000042ULL);
a.patches[0].bits.assign(r.bits.size(),std::bit_cast<std::uint64_t>(99.));bytes=encode_state_carriers(a);original=*f.values;bool refused=false;try{s.restore_checkpoint_state_carriers(bytes);}catch(const std::invalid_argument&){refused=true;}assert(refused);for(std::size_t i=0;i<original.size();++i)assert(std::bit_cast<std::uint64_t>((*f.values)[i])==std::bit_cast<std::uint64_t>(original[i]));
s.p_->external_restart_transaction_=false;bytes=encode_state_carriers(StateCarrierArchive<D>{64,1,1,-1,{"unrelated-vector"},{r}});refused=false;try{s.restore_checkpoint_state_carriers(bytes);}catch(const std::logic_error&){refused=true;}assert(refused);
}
'''
    source=tmp_path/'restore.cpp';source.write_text(code)
    compiler=shutil.which('clang++') or shutil.which('c++')
    assert compiler,'host compiler required'
    subprocess.run([compiler,'-std=c++20','-Werror','-DTEST_DIM='+str(dim),'-I'+str(ROOT/'include'),str(source),'-o',str(tmp_path/'probe')],check=True,capture_output=True)
    subprocess.run([str(tmp_path/'probe')],check=True,capture_output=True)
