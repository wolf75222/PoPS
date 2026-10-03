"""Real archive + real common-header CPU probes; existing Native seams metadata-only."""
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO
from copy import deepcopy
import struct, subprocess, hashlib
import numpy as np
import pytest
from pops.codegen import _checkpoint_migration_uniform_v2 as migration
ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=ROOT/'tests/review/fixtures/sdk24-migration732309-authority9.npz'

@pytest.fixture(scope='module')
def host(tmp_path_factory):
    directory=tmp_path_factory.mktemp('migration-common-header')
    cpp=directory/'probe.cpp'; binary=directory/'probe'
    cpp.write_text(r"""#include <pops/runtime/checkpoint/uniform_migration_authority.hpp>
#include <pops/runtime/program/program_diagnostics_checkpoint.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
template<int D> void rectangular() {
 using namespace pops::runtime::checkpoint;
 StateCarrierArchive<D> a;a.real_bits=64;a.ranks=3;a.levels=1;a.blocks={"renamed-vector"};
 std::array<std::uint64_t,D> shape{};shape.fill(2);shape[0]=3;
 std::uint64_t cells=1;for(auto n:shape)cells*=n;
 for(int j=0;j<2;++j) {
  StateCarrierPatch<D> p;p.block=0;p.level=0;p.patch=j;p.components=2;p.owner=j+1;
  for(int d=0;d<D;++d){p.lo[d]=0;p.hi[d]=shape[d]-1;}
  p.lo[0]=j==0?0:1;p.hi[0]=j==0?0:2;p.grown_lo=p.lo;p.grown_hi=p.hi;
  std::uint64_t size=1;for(int d=0;d<D;++d)size*=p.hi[d]-p.lo[d]+1;
  for(std::uint64_t c=0;c<2;++c)for(std::uint64_t i=0;i<size;++i) {
   auto rem=i;std::uint64_t global=0,stride=1;
   for(int d=0;d<D;++d){global+=(p.lo[d]+rem%(p.hi[d]-p.lo[d]+1))*stride;rem/=p.hi[d]-p.lo[d]+1;stride*=shape[d];}
   p.bits.push_back(c==0&&global==0?0x8000000000000000ULL:std::bit_cast<std::uint64_t>(double(100*c+global)));
  }
  a.patches.push_back(p);
 }
 auto result=uniform_migration_valid_projection<D>(encode_state_carriers(a),shape,a.blocks,{2});
 for(std::uint64_t c=0;c<2;++c)for(std::uint64_t i=0;i<cells;++i)
  if(result[0][c*cells+i]!=(c==0&&i==0?0x8000000000000000ULL:std::bit_cast<std::uint64_t>(double(100*c+i))))throw std::runtime_error("rectangular partition projection differs");
 shape[0]++;try{uniform_migration_valid_projection<D>(encode_state_carriers(a),shape,a.blocks,{2});}catch(const std::invalid_argument&){return;}
 throw std::runtime_error("geometry hole accepted");
}
int main(int argc,char**argv) {try {
 if(std::string(argv[1])=="self"){rectangular<1>();rectangular<2>();rectangular<3>();return 0;}
 std::ifstream f(argv[2],std::ios::binary);std::vector<std::uint8_t>b((std::istreambuf_iterator<char>(f)),{});
 if(std::string(argv[1])=="diag") {
   std::cout<<pops::runtime::program::read_program_diagnostics_checkpoint(b,std::stoi(argv[3]),std::stoi(argv[4])).size();return 0;
 }
 const std::array<std::uint64_t,2> shape{std::stoull(argv[3]),std::stoull(argv[4])};
 std::vector<std::string> names;std::vector<std::uint64_t> comps;
 for(int a=5;a<argc;a+=2){names.emplace_back(argv[a]);comps.push_back(std::stoull(argv[a+1]));}
 auto projections=pops::runtime::checkpoint::uniform_migration_valid_projection<2>(b,shape,names,comps);
 for(auto&p:projections)std::cout.write(reinterpret_cast<char*>(p.data()),p.size()*8);
 }catch(const std::exception&e){std::cerr<<e.what();return 2;} }
""")
    subprocess.run(['clang++','-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(cpp),'-o',str(binary)],check=True,capture_output=True)
    counter=0
    def invoke(mode,image,*args):
        nonlocal counter
        path=directory/f'image{counter}.bin';counter+=1;path.write_bytes(image)
        result=subprocess.run([str(binary),mode,str(path),*map(str,args)],capture_output=True)
        if result.returncode:raise ValueError(result.stderr.decode())
        return result.stdout
    def carrier(image,shape,blocks,components):
        assert type(image) is bytes, "Native binding requires exact bytes"
        raw=invoke('carrier',image,*shape,*[v for pair in zip(blocks,components) for v in pair])
        cells=int(np.prod(shape));rows=[];at=0
        for count in components:
            size=cells*count*8;rows.append(raw[at:at+size]);at+=size
        assert at==len(raw)
        return {'contract':'pops.uniform-migration-valid-projection@1','dimension':2,
                'blocks':list(blocks),'valid_state_double_bytes':tuple(rows)}
    return SimpleNamespace(invoke=invoke,carrier=carrier,diagnostic=lambda image,rank,ranks:int(invoke('diag',image,rank,ranks)))

@pytest.fixture
def payload():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()=='3841c3079edba2fdea4dd3c38c37a4775708d111c2ca0140fbc9fb78ed2c557f'
    with np.load(ARCHIVE,allow_pickle=False) as archive:return {k:archive[k].copy() for k in archive.files}

@pytest.fixture
def seams(monkeypatch,host):
    # Existing aux and Real-width Native boundaries are explicitly metadata-only.
    monkeypatch.setattr('pops.runtime._history_sample_identity.native_real_bytes',lambda:8)
    monkeypatch.setattr(migration,'_attest_empty_auxiliary_checkpoint',lambda image,dimension:b'source-only-aux-registry')
    native=SimpleNamespace(_attest_uniform_migration_state_carriers=host.carrier,
                          _attest_uniform_migration_program_diagnostics=host.diagnostic)
    monkeypatch.setattr('pops._native_selector.selected_native_module',lambda **kw:native)
    return native


def reseal(payload):
    from pops.runtime._checkpoint_manifest import (MANIFEST_KEY,IDENTITY_KEY,_identity_from_json,
        _seal_checkpoint_payload_with_identities)
    import json
    manifest=json.loads(str(payload[MANIFEST_KEY]))
    payload.pop(MANIFEST_KEY);payload.pop(IDENTITY_KEY)
    _seal_checkpoint_payload_with_identities(payload,runtime_kind='uniform',**{
        name:_identity_from_json(manifest[name+'_identity']) for name in ('semantic','artifact','bind','run')})


def test_actual_current9_authority_passes_without_copying_storage(payload,seams):
    authority=migration._current_authority(payload)
    assert authority.spatial.shape==(4,4) and authority.blocks==('blk',)

@pytest.mark.parametrize('attack',['pair','offsets','diagbody','valid','carriertrunc','blocks','extra','v8carrier'])
def test_resealed_invalid_current_authority_refused(payload,seams,attack):
    if attack=='pair':payload.pop('program_diagnostics_offsets')
    if attack=='offsets':payload['program_diagnostics_offsets'][-1]-=1
    if attack=='diagbody':payload['program_diagnostics_state']=np.append(payload['program_diagnostics_state'],np.uint8(0));payload['program_diagnostics_offsets'][-1]+=1
    if attack=='valid':payload['state_blk'].flat[0]+=1
    if attack=='carriertrunc':payload['state_carriers_checkpoint']=payload['state_carriers_checkpoint'][:-1]
    if attack=='blocks':payload['state_carriers_checkpoint'][64+8]^=1
    if attack=='extra':payload['unknown_storage']=np.zeros(1)
    if attack=='v8carrier':payload['pops_checkpoint_version']=np.array(8,dtype=np.int64)
    reseal(payload)
    with pytest.raises((ValueError,TypeError,RuntimeError)):migration._current_authority(payload)


def diagnostic(entries):
    word=lambda value:struct.pack('<Q',value)
    return b'POPSDIA1'+b''.join(word(v) for v in (64,0,1,len(entries)))+b''.join(
        word(len(name))+name+word(bits) for name,bits in entries)


def test_nonempty_current_diagnostics_are_validated_not_limited_to_demo(payload,seams):
    image=diagnostic([(b'retained-current-name',0x8000000000000000)])
    payload['program_diagnostics_state']=np.frombuffer(image,np.uint8).copy()
    payload['program_diagnostics_offsets']=np.array([0,len(image)],np.int64)
    reseal(payload)
    assert migration._current_authority(payload).blocks==('blk',)


def test_common_diagnostic_codec_rejects_duplicates_rank_and_tail(host):
    image=diagnostic([(b'one',0),(b'one',1)])
    with pytest.raises(ValueError,match='duplicate'):host.diagnostic(image,0,1)
    with pytest.raises(ValueError,match='rank'):host.diagnostic(diagnostic([]),1,2)
    with pytest.raises(ValueError,match='trailing'):host.diagnostic(diagnostic([])+b'x',0,1)


def test_native_attestation_fault_remains_original_and_prepublication(payload,seams,tmp_path):
    marker=ValueError('actual attestation failure')
    def fail(*a):raise marker
    seams._attest_uniform_migration_state_carriers=fail
    with pytest.raises(ValueError) as caught:migration._current_authority(payload)
    assert caught.value is marker and not list(tmp_path.iterdir())


@pytest.mark.parametrize("nonempty",[False,True])
def test_public_migration_omits_current_diagnostics_and_grown(payload,seams,tmp_path,nonempty):
    """Full offline publication from genuine v2/current9; old Native aux boundary labelled."""
    import importlib.util
    from pops.runtime._checkpoint_manifest import inspect_checkpoint_payload_integrity, _identity_from_json
    from pops.runtime._checkpoint_resource_budget import _producer_checkpoint_resource_budget
    spec=importlib.util.spec_from_file_location('migration_fixture',ROOT/'tests/python/unit/codegen/test_checkpoint_migration.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    source,legacy=fixture._write_source(tmp_path)
    authority_path=ARCHIVE
    if nonempty:
        image=diagnostic([(b'current-only-record',0x8000000000000000)])
        payload['program_diagnostics_state']=np.frombuffer(image,np.uint8).copy()
        payload['program_diagnostics_offsets']=np.array([0,len(image)],np.int64)
        reseal(payload)
        authority_path=tmp_path/'current-nonempty.npz';authority_path.write_bytes(migration._encode(payload))
    seams._attest_empty_uniform_auxiliary_checkpoint=lambda image:{'registry_contract':b'source-only-aux-registry','accepted_generation':0}
    manifest,restart=inspect_checkpoint_payload_integrity(payload,runtime_kind='uniform')
    owner=SimpleNamespace(_checkpoint_resource_budget=_producer_checkpoint_resource_budget(payload,runtime_kind='uniform',authority=restart.token),
        _checkpoint_identities=lambda:tuple(_identity_from_json(manifest[name+'_identity']) for name in ('semantic','artifact','bind')),
        last_run_identity=_identity_from_json(manifest['run_identity']))
    mapping=fixture._mapping(legacy,owner,restart,authority_path)
    destination=tmp_path/'migrated-v8.npz'
    migration.migrate_uniform_v2_checkpoint(source,destination,current_authority=authority_path,mapping=mapping)
    with np.load(destination,allow_pickle=False) as migrated:
        assert int(migrated['pops_checkpoint_version'])==8
        assert not {'state_carriers_checkpoint','program_diagnostics_state','program_diagnostics_offsets'} & set(migrated.files)
        np.testing.assert_array_equal(migrated['state_blk'],legacy['state_blk'])
    marker=ValueError('native attestation failed before atomic publication')
    def fail(*args):raise marker
    seams._attest_uniform_migration_state_carriers=fail
    rejected=tmp_path/'never-published.npz'
    with pytest.raises(ValueError) as caught:
        migration.migrate_uniform_v2_checkpoint(source,rejected,current_authority=authority_path,mapping=mapping)
    assert caught.value is marker and not rejected.exists()
    assert not tuple(tmp_path.glob('*.tmp'))


def test_common_carrier_projection_rectangular_partition_dimensions_1_2_3(host):
    assert host.invoke("self",b"")==b""
