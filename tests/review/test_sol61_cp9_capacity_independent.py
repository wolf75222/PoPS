"""Exact System method + real noniterable BoxArray; explicit host owner seam."""
from pathlib import Path
import subprocess,shutil
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('dim',(1,2,3))
def test_real_layout_global_capacity_all_blocks_and_empty_local_rank(tmp_path,dim):
    text=(ROOT/'src/runtime/system/system.cpp').read_text()
    begin=text.index('template <int Dim>\nstd::uint64_t System<Dim>::checkpoint_state_carriers_capacity() const')
    end=text.index('\ntemplate <int Dim>',begin+20)
    body=text[begin:end]
    code=r'''
#include <pops/mesh/layout/box_array.hpp>
#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <cassert>
#include <exception>
#include <map>
#include <memory>
namespace pops {
struct Lane{};
void collectively_rethrow_exception(std::exception_ptr error,const Lane&,const char*) {if(error)std::rethrow_exception(error);}
template<int D>struct Field {
 mesh::BoxArray<D> global;Extent<D> halo;int components;
 const mesh::BoxArray<D>&layout()const{return global;}
 const Extent<D>&ghosts()const{return halo;}
 int ncomp()const{return components;}
 // No local cells on one rank does not erase global layout authority.
 std::size_t local_size()const{return 0;}
};
struct Names {std::vector<std::string> ordered;const auto&names()const{return ordered;}};
template<int D>class System {public:
 struct Block {Field<D> U;};
 struct Impl {Names blocks_;std::map<std::string,Block> entries;const Block&find(const std::string&key)const{return entries.at(key);}};
 std::unique_ptr<Impl> p_=std::make_unique<Impl>();
 const Lane&prepared_boundary_execution_lane()const{static Lane lane;return lane;}
 std::uint64_t checkpoint_state_carriers_capacity()const;
};
'''+body+r'''
}
int main() {
 using namespace pops;using namespace pops::runtime::checkpoint;constexpr int D=TEST_DIM;
 System<D> system;std::uint64_t largest=0;
 for(int b=0;b<2;++b) {
  std::vector<Box<D>> boxes;std::uint64_t count=0;Extent<D> halo;
  for(int a=0;a<D;++a)halo[a]=b?(a+1):0;
  for(int p=0;p<b+2;++p) {
   Box<D> box;for(int a=0;a<D;++a){box.lo[a]=a?0:p*8;box.hi[a]=box.lo[a]+a+b;}
   count+=box.numPts();boxes.push_back(box);
  }
  largest=std::max(largest,count);auto name=b?"later-wide":"first-small";
  system.p_->blocks_.ordered.push_back(name);
  system.p_->entries.emplace(name,typename System<D>::Block{Field<D>{mesh::BoxArray<D>(boxes),halo,b?7:2}});
 }
 // Independent capacity arithmetic: every global valid cell may be a one-cell patch.
 std::uint64_t expected=64;
 for(const auto&name:system.p_->blocks_.ordered) {
  const auto&field=system.p_->find(name).U;assert(field.local_size()==0);
  std::uint64_t scalars=field.ncomp();for(int a=0;a<D;++a)scalars*=1+2*field.ghosts()[a];
  expected+=8+name.size()+largest*((6+4*D)*8+scalars*8);
 }
 assert(system.checkpoint_state_carriers_capacity()==expected);
 std::reverse(system.p_->blocks_.ordered.begin(),system.p_->blocks_.ordered.end());
 assert(system.checkpoint_state_carriers_capacity()==expected);
 // The actual codec worst-case one-cell fragmentation fits the global envelope exactly.
 StateCarrierArchive<D> image;image.real_bits=64;image.ranks=3;image.levels=1;image.shard=-1;
 image.blocks=system.p_->blocks_.ordered;
 for(std::size_t b=0;b<image.blocks.size();++b) {
  const auto&field=system.p_->find(image.blocks[b]).U;
  for(std::uint64_t p=0;p<largest;++p) {
   StateCarrierPatch<D> row;row.block=b;row.patch=p;row.components=field.ncomp();row.owner=p%2;
   std::uint64_t scalars=row.components;
   for(int a=0;a<D;++a){row.lo[a]=row.hi[a]=a?0:p;row.grown_lo[a]=row.lo[a]-field.ghosts()[a];row.grown_hi[a]=row.hi[a]+field.ghosts()[a];scalars*=1+2*field.ghosts()[a];}
   row.bits.resize(scalars,0x8000000000000000ULL);image.patches.push_back(row);
  }
 }
 assert(encode_state_carriers(image).size()==expected);
 system.p_->blocks_.ordered.clear();bool refused=false;
 try{system.checkpoint_state_carriers_capacity();}catch(const std::invalid_argument&){refused=true;}
 assert(refused);
}
'''
    source=tmp_path/'probe.cpp';source.write_text(code)
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    result=subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',f'-DTEST_DIM={dim}','-I'+str(ROOT/'include'),str(source),'-o',str(tmp_path/'probe')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(tmp_path/'probe')],check=True,capture_output=True)
