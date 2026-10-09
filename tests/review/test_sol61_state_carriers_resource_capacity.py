"""Actual codec structural capacity and writer inventory; Source/host only."""
from pathlib import Path
import shutil,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('dim',(1,2,3))
def test_actual_codec_maximal_fragmentation_typed_capacity(tmp_path,dim):
    source=tmp_path/'capacity.cpp';binary=tmp_path/'capacity'
    source.write_text(r'''#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <cassert>
#include <functional>
using namespace pops::runtime::checkpoint;
constexpr int D=TEST_DIM;
bool refuses(const std::function<void()>& fn){try{fn();return false;}catch(const std::exception&){return true;}}
int main(){
 std::vector<std::uint64_t> cells;
 for(int side:{2,4}){std::uint64_t count=1;for(int a=0;a<D;++a)count*=side;cells.push_back(count);}
 std::vector<StateCarrierStorageCapacity<D>> blocks(2);
 blocks[0].name="scalar";blocks[0].components=1;
 blocks[1].name="vector";blocks[1].components=3;
 for(int a=0;a<D;++a){blocks[0].ghosts[a]=0;blocks[1].ghosts[a]=a+1;}
 const auto bound=state_carriers_byte_capacity<D>(cells,blocks);
 for(int ranks:{1,3}){
  StateCarrierArchive<D> image;image.real_bits=64;image.ranks=ranks;image.shard=-1;image.levels=2;
  for(auto& b:blocks)image.blocks.push_back(b.name);
  for(int b=0;b<2;++b)for(int level=0;level<2;++level){
   int side=level==0?2:4;
   for(std::uint64_t index=0;index<cells[level];++index){
    StateCarrierPatch<D> row;row.block=b;row.level=level;row.patch=index;row.components=blocks[b].components;row.owner=index%ranks;
    auto flat=index;std::uint64_t size=row.components;
    for(int axis=0;axis<D;++axis){auto x=flat%side;flat/=side;row.lo[axis]=row.hi[axis]=x;
     row.grown_lo[axis]=row.lo[axis]-blocks[b].ghosts[axis];row.grown_hi[axis]=row.hi[axis]+blocks[b].ghosts[axis];size*=1+2*blocks[b].ghosts[axis];}
    row.bits.resize(size,0);image.patches.push_back(std::move(row));
   }
  }
  assert(encode_state_carriers(image).size()==bound); // worst case is tight, no safety multiplier
 }
 auto bad=blocks;bad[1].name="scalar";assert(refuses([&]{state_carriers_byte_capacity<D>(cells,bad);}));
 bad=blocks;bad[0].components=0;assert(refuses([&]{state_carriers_byte_capacity<D>(cells,bad);}));
 bad=blocks;bad[0].ghosts[0]=UINT64_MAX;assert(refuses([&]{state_carriers_byte_capacity<D>(cells,bad);}));
 auto huge=cells;huge[0]=UINT64_MAX;assert(refuses([&]{state_carriers_byte_capacity<D>(huge,blocks);}));
 auto empty=cells;empty[0]=0;assert(refuses([&]{state_carriers_byte_capacity<D>(empty,blocks);}));
}
''')
    compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
    subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',f'-DTEST_DIM={dim}','-I',str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True)
    subprocess.run([str(binary)],check=True)

@pytest.mark.parametrize('blocks,fields,levels',[(1,0,1),(1,1,2),(2,2,3)])
def test_exact_writer_families_include_carriers_not_field_padding(blocks,fields,levels):
    from pops.runtime._checkpoint_resource_budget import _checkpoint_member_names
    names=_checkpoint_member_names(runtime_kind='amr',block_names=tuple(f'b{i}' for i in range(blocks)),field_names=tuple(f'f{i}' for i in range(fields)),history_names=('history_names',),cache_names=(),levels=levels,rank_capacity=2,has_amr_legacy_phi=bool(fields))
    assert names.count('state_carriers_checkpoint')==1
    assert len([n for n in names if n.startswith('state_b')])==blocks*levels
    assert len([n for n in names if n.startswith('field_provider_phi_')])==fields*levels
    assert len([n for n in names if n.startswith('phi_')])==(levels if fields else 0)
    assert len(names)==len(set(names))

def test_unknown_codec_refused_and_native_capacity_vote_before_consensus():
    from pops.runtime._checkpoint_state_carriers import _native_image
    with pytest.raises(ValueError):_native_image(b'POPSCAR2'+bytes(100))
    s=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
    body=s[s.index('std::uint64_t AmrSystem<Dim>::checkpoint_state_carriers_byte_capacity() const'):s.index('void AmrSystem<Dim>::validate_checkpoint_state_carriers(')]
    assert 'p_->cfg.transition_ratios' in body and 'block.ghosts[axis]' in body and 'block.ncomp' in body
    assert body.index('catch (...)')<body.index('rethrow_collective_failure')<body.index('all_ranks_agree_exact_ordered_byte_pairs')
    assert body.index('contract = std::move(exact).release()') < body.index('named_pairs.emplace_back') < body.index('catch (...)')
    assert 'all_ranks_agree_exact_ordered_byte_pairs(named_pairs, lane)' in body
    assert 'copy_to_host' not in body and 'materialize_field' not in body and 'checkpoint_state_carriers()' not in body
    assert 'template std::uint64_t AmrSystem<kNativeDimension>::checkpoint_state_carriers_byte_capacity() const;' in s


def test_actual_zip_guard_refuses_incomplete_writer_budget_and_unowned_member():
    import io
    import numpy as np
    from types import SimpleNamespace
    from pops.runtime._checkpoint_resource_budget import _checkpoint_member_names
    from pops.output._checkpoint_collective import _checkpoint_central_directory_preflight
    names=_checkpoint_member_names(runtime_kind='amr',block_names=('first','second'),field_names=('left','right'),history_names=('history_names',),cache_names=(),levels=3,rank_capacity=2,has_amr_legacy_phi=True)
    def wire(extra=False):
        stream=io.BytesIO();payload={n:np.array(0) for n in names}
        if extra:payload['unowned']=np.array(0)
        np.savez_compressed(stream,**payload);return stream.getvalue()
    # Exact missing-carrier inventory causes the same actual central-directory refusal.
    with pytest.raises(ValueError,match='member count exceeds'):
        _checkpoint_central_directory_preflight(wire(),SimpleNamespace(max_members=len(names)-1,max_manifest_characters=100000))
    assert _checkpoint_central_directory_preflight(wire(),SimpleNamespace(max_members=len(names),max_manifest_characters=100000))==len(names)
    with pytest.raises(ValueError,match='member count exceeds'):
        _checkpoint_central_directory_preflight(wire(True),SimpleNamespace(max_members=len(names),max_manifest_characters=100000))


def test_common_budget_refuses_missing_or_foreign_carrier_capacity_before_owner_reads():
    from pops.runtime._checkpoint_resource_budget import _common_budget
    kwargs=dict(cells=(8,),shape=(8,),rank_capacity=1,auxiliary_metadata_bytes=0,
        auxiliary_components=0,accepted_program_bytes=0,source_authority_bytes=0,
        history_flux_snapshot_bytes=0,structural_bytes=0,field_provider_manifest_characters=0,
        program=None,block_nvars_by_name={},field_names=())
    with pytest.raises(ValueError):_common_budget(None,None,runtime_kind='amr',**kwargs)
    with pytest.raises(ValueError):_common_budget(None,None,runtime_kind='uniform',state_carriers_bytes=-1,**kwargs)
