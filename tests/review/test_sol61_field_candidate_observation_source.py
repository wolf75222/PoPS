"""Source/host qualification only: actual archive codec, ordering and persistence."""
from pathlib import Path
import shutil
import subprocess
import pytest
from tests.python.support.field_candidate_observation import save_rank_observations,SCHEMA
ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize("dim",[1,2,3])
def test_actual_full_carrier_codec_arbitrary_fields_and_grown(dim,tmp_path):
    compiler=shutil.which("clang++") or shutil.which("c++")
    assert compiler,"host C++ compiler required (no synthetic Native result)"
    source=tmp_path/"probe.cpp"
    source.write_text(r'''#include <pops/runtime/checkpoint/state_carriers.hpp>
#include <cassert>
#include <cmath>
using namespace pops::runtime::checkpoint;
template<int D> void run(){
 StateCarrierArchive<D> a; a.real_bits=64;a.ranks=2;a.shard=1;a.levels=3;
 a.blocks={"renamed/providerA","another/providerB"};
 for(unsigned b=0;b<2;++b)for(unsigned l=0;l<3;++l){
 StateCarrierPatch<D> p;p.block=b;p.level=l;p.patch=l+7;p.components=2;p.owner=1;
 std::size_t n=2;for(int d=0;d<D;++d){p.lo[d]=d;p.hi[d]=d+1;p.grown_lo[d]=d-1;p.grown_hi[d]=d+2;n*=4;}
 for(std::size_t i=0;i<n;++i)p.bits.push_back(std::bit_cast<std::uint64_t>(i%3==0?-0.0:(i%3==1?std::numeric_limits<double>::denorm_min():double(i+b*31+l))));
 a.patches.push_back(p);}
 auto bytes=encode_state_carriers(a);auto restored=decode_state_carriers<D>(bytes);
 assert(restored.blocks==a.blocks&&restored.patches==a.patches&&restored.shard==1);
 auto bad=a;bad.patches[0].bits.pop_back();bool refused=false;try{(void)encode_state_carriers(bad);}catch(const std::invalid_argument&){refused=true;}assert(refused);
 bad=a;bad.patches[0].grown_lo[0]=bad.patches[0].lo[0]+1;refused=false;try{(void)encode_state_carriers(bad);}catch(const std::invalid_argument&){refused=true;}assert(refused);
 bad=a;bad.patches[0].owner=0;refused=false;try{(void)encode_state_carriers(bad);}catch(const std::invalid_argument&){refused=true;}assert(refused);
 bytes.pop_back();refused=false;try{(void)decode_state_carriers<D>(bytes);}catch(const std::invalid_argument&){refused=true;}assert(refused);
}
int main(){run<REPLACE_DIM>();}
'''.replace("REPLACE_DIM",str(dim)))
    output=tmp_path/"probe"
    subprocess.run([compiler,"-std=c++20","-Wall","-Wextra","-Werror","-I",str(ROOT/"include"),str(source),"-o",str(output)],check=True,capture_output=True,text=True)
    subprocess.run([str(output)],check=True,capture_output=True,text=True)

def test_source_producer_fence_capture_then_real_ghost_and_readonly_getter():
    source=(ROOT/"src/runtime/amr/amr_system.cpp").read_text()
    body=source.split("void prepare_accepted_halo_field_dependencies(",1)[1].split("using AcceptedHaloPointPack",1)[0]
    assert body.index("copy_full_field_in_place")<body.index("Kokkos::fence();")<body.index("capture_field_candidate_observation(slot, block, level, point)")<body.index("publication->accept()")
    halo=source.split("void AmrSystem<Dim>::prepare_accepted_halo_candidates(",1)[1].split("void AmrSystem<Dim>::publish_prepared_amr_program_candidates",1)[0]
    assert halo.index("prepare_accepted_halo_field_dependencies")<halo.index("prepare_generated_amr_block_level_state")
    assert "staged_field_candidate_observations.clear();" in halo
    assert halo.rindex("restore_accepted();")<halo.index("field_candidate_observations.swap")
    getter=source.split("std::vector<FieldCandidateObservation> AmrSystem<Dim>::field_candidate_observations() const",1)[1].split("template <int Dim>",1)[0]
    assert "solve_program" not in getter and "ensure_engine" not in getter and "require_inspectable_hierarchy" not in getter
    for guard in ("topology_epoch", "materialization_generation", "owner_time != p_->accepted_time", "owner_macro_step != p_->macro_step"):
        assert guard in getter
    assert "FieldObservationKey{slot, witness.consumer_block, level}" in source
    checkpoint=source.split("checkpoint_state_carriers() const",1)[1].split("validate_checkpoint_state_carriers",1)[0]
    assert "field_candidate_observations" not in checkpoint

def test_persistence_requires_producer_and_unambiguous_invocation(tmp_path):
    # Explicit Source-only persistence adversaries; these bytes are not Native evidence.
    import json,hashlib
    row={"schema":SCHEMA,"accepted_publication":False,"provider_slot":"arbitrary/slot","consumer_block":"arbitrary/block","consumer_level":2,"carrier_bytes":b"POPSCAR1-source-only-test"}
    path=save_rank_observations(tmp_path,"accepted",1,[row])
    saved=json.loads(path.read_text())["observations"][0]
    assert saved["carrier"]["sha256"]==hashlib.sha256(row["carrier_bytes"]).hexdigest()
    assert Path(saved["carrier"]["path"]).read_bytes()==row["carrier_bytes"]
    with pytest.raises(ValueError,match="duplicate"):save_rank_observations(tmp_path,"accepted",1,[row,row])
    with pytest.raises(ValueError,match="contract"):save_rank_observations(tmp_path,"accepted",1,[dict(row,accepted_publication=True)])
    with pytest.raises(ValueError,match="missing executed"):save_rank_observations(tmp_path,"accepted",1,[])


def test_actual_readonly_getter_body_refuses_new_points_and_owner_epochs(tmp_path):
    """Compile the unchanged getter body with explicit SOURCE-only owner stand-ins."""
    compiler=shutil.which("clang++") or shutil.which("c++")
    assert compiler
    cpp=(ROOT/"src/runtime/amr/amr_system.cpp").read_text()
    signature="std::vector<FieldCandidateObservation> AmrSystem<Dim>::field_candidate_observations() const"
    body=cpp.split(signature,1)[1].split("template <int Dim>",1)[0].strip()
    src=tmp_path/"getter.cpp"
    src.write_text(r'''#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <cstdint>
struct Point { double physical_time; std::int64_t tick; };
struct FieldCandidateObservation {std::uint64_t topology_epoch,materialization_generation;Point point;std::vector<unsigned char> carrier_bytes;std::int64_t owner_macro_step=2;double owner_time=.25;std::string provider_slot="s",provider_identity="p",plan_identity="i",output_owner_identity="o",output_block="b",output_key="k",configuration_identity="cfg";};
struct Plan{std::string provider_identity="p",plan_identity="i",output_owner_identity="o",output_block="b",output_key="k",configuration="cfg";};
std::string prefixed_sha256(const char*,const std::string& text){return text;} // Explicit SOURCE stand-in, no crypto/runtime claim.
struct Engine {std::uint64_t epoch=3,generation=4;auto topology_epoch()const{return epoch;}auto materialization_generation()const{return generation;}};
struct Owner {std::map<std::string,Plan> field_plans={{"s",Plan{}}};static std::string exact_field_plan_contract(const std::string&,const Plan& p,bool){return p.configuration;}std::unique_ptr<Engine> engine=std::make_unique<Engine>();bool prepared_hierarchy=true,bootstrap_transaction=false,accepted_transaction_active=false,restart_transaction=false,field_candidate_observation_enabled=true;double accepted_time=.25;std::int64_t macro_step=2;std::map<int,FieldCandidateObservation> field_candidate_observations;void require_no_native_package_callback(const char*)const{};};
template<int Dim>struct AmrSystem {using Impl=Owner;std::unique_ptr<Owner> p_=std::make_unique<Owner>();int depth=0;int step_transaction_depth()const{return depth;}std::vector<FieldCandidateObservation> field_candidate_observations()const;};
template<int Dim>std::vector<FieldCandidateObservation> AmrSystem<Dim>::field_candidate_observations() const
'''+body+r'''
template<int D> void test(){AmrSystem<D> s;s.p_->field_candidate_observations={{0,{3,4,{.25,2},{1,2,3}}},{1,{3,4,{.25,2},{9,8}}}};
 auto rows=s.field_candidate_observations();assert(rows.size()==2&&rows[1].carrier_bytes==std::vector<unsigned char>({9,8}));rows[0].carrier_bytes[0]=99;assert(s.field_candidate_observations()[0].carrier_bytes[0]==1);
 auto refuses=[&]{bool no=false;try{(void)s.field_candidate_observations();}catch(const std::invalid_argument&){no=true;}assert(no);};
 s.p_->field_plans["s"].configuration="changed";refuses();s.p_->field_plans["s"].configuration="cfg";s.p_->field_plans["s"].provider_identity="foreign";refuses();s.p_->field_plans["s"].provider_identity="p";
 s.p_->engine->epoch=5;refuses();s.p_->engine->epoch=3;s.p_->engine->generation=6;refuses();s.p_->engine->generation=4;
 s.p_->accepted_time=.5;refuses();s.p_->accepted_time=.25;s.p_->macro_step=3;refuses();s.p_->macro_step=2;
 s.p_->field_candidate_observations[1].point.tick=71;assert(s.field_candidate_observations().size()==2); // Foreign clock tick is not the owner macro counter.
 s.p_->field_candidate_observations[1].point.physical_time=.125;refuses();s.p_->field_candidate_observations[1].point.physical_time=.25;
 s.p_->bootstrap_transaction=true;refuses();s.p_->bootstrap_transaction=false;s.p_->accepted_transaction_active=true;refuses();s.p_->accepted_transaction_active=false;s.depth=1;refuses();s.depth=0;
 s.p_->restart_transaction=true;refuses();s.p_->restart_transaction=false;s.p_->field_candidate_observation_enabled=false;refuses();s.p_->field_candidate_observation_enabled=true;s.p_->engine.reset();refuses();}
int main(){test<1>();test<2>();test<3>();}
''')
    out=tmp_path/"getter"
    subprocess.run([compiler,"-std=c++20","-Wall","-Wextra","-Werror",str(src),"-o",str(out)],check=True,capture_output=True,text=True)
    subprocess.run([str(out)],check=True,capture_output=True,text=True)


def test_binding_invalid_version_reaches_native_vote_instead_of_local_throw():
    source=(ROOT/"python/bindings/core/init/init_amr.cpp").read_text()
    body=source.split('.def("_enable_field_candidate_observation",',1)[1].split('.def("_field_candidate_observations",',1)[0]
    assert not any(line.lstrip().startswith("throw ") for line in body.splitlines())
    assert "std::uint32_t parsed = 0" in body
    assert "PyLong_CheckExact" in body and "PyBool_Check" in body
    assert "PyErr_Clear()" in body and "s.enable_field_candidate_observation(parsed)" in body


def test_source_outer_rollback_invalidates_even_same_time_observations():
    cpp=(ROOT/"src/runtime/amr/amr_system.cpp").read_text()
    restore=cpp.split("void restore(Impl& owner) {",1)[1].split("if (!owner.multiblock_hierarchy)",1)[0]
    assert "owner.field_candidate_observations.clear();" in restore
    assert "owner.staged_field_candidate_observations.clear();" in restore
    complete=cpp.split("void AmrSystem<Dim>::complete_program_step_()",1)[1].split("template <int Dim>",1)[0]
    assert complete.index("refresh_hierarchy_state")<complete.index("witness.owner_macro_step = p_->macro_step")<complete.index("return;")


def test_default_disabled_and_clock_restore_invalidates_without_solve():
    cpp=(ROOT/"src/runtime/amr/amr_system.cpp").read_text()
    assert "bool field_candidate_observation_enabled = false;" in cpp
    capture=cpp.split("void capture_field_candidate_observation(",1)[1].split("void prepare_accepted_halo_field_dependencies",1)[0]
    assert capture.index("if (!field_candidate_observation_enabled) return;")<capture.index("create_host_mirror")
    restore=cpp.split("void AmrSystem<Dim>::set_clock(",1)[1].split("template <int Dim>",1)[0]
    assert "field_candidate_observations.clear();" in restore and "staged_field_candidate_observations.clear();" in restore
    assert "solve_program" not in restore


def test_runtime_parameter_mutation_invalidates_observations():
    cpp=(ROOT/"src/runtime/amr/amr_system.cpp").read_text()
    for operation in ("set_program_params", "seed_program_params"):
        body=cpp.split("void AmrSystem<Dim>::"+operation+"(",1)[1].split("template <int Dim>",1)[0]
        assert "field_candidate_observations.clear();" in body and "staged_field_candidate_observations.clear();" in body
