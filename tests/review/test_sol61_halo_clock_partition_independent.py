"""Actual clock header and topology duration loop; Source/host only."""
from pathlib import Path
import shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_actual_topology_duration_partition_remainder_and_multilevel(tmp_path):
 source=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
 start=source.index('std::vector<double> durations',source.index('AcceptedHaloPointPack accepted_halo_topology_points()'))
 end=source.index('AcceptedHaloPointPack result',start)
 loop=source[start:end]
 cpp=tmp_path/'probe.cpp'
 cpp.write_text(r"""
#include <pops/numerics/time/amr/levels/amr_clock.hpp>
#include <cmath>
#include <cassert>
#include <vector>
using namespace pops::amr;
struct Config {int level_count;};
std::vector<double> probe(double macro_dt, std::vector<ParentChildClockRelation> temporal_relations) {
 Config cfg{static_cast<int>(temporal_relations.size()+1)};int macro_step=17;
 LOOP
 return durations;
}
int main(){
 auto v=probe(.125,{{0,1,{1,1},RemainderPolicy::IntegralOnly},{1,2,{2,1},RemainderPolicy::IntegralOnly},{2,3,{3,1},RemainderPolicy::IntegralOnly}});
 assert(v.size()==4&&v[0]==.125&&v[1]==.125&&v[2]==.0625&&v[3]==.125/6.);
 auto r=probe(.125,{{0,1,{5,2},RemainderPolicy::ExplicitFinalSubstep},{1,2,{3,1},RemainderPolicy::IntegralOnly}});
 assert(r[1]==.125*.2&&r[2]==.125/15.);
 auto z=probe(0.,{{0,1,{2,1},RemainderPolicy::IntegralOnly}});assert(z[0]==0.&&z[1]==0.);
 // Subtracting absolute clocks loses the real interval at this origin.
 double origin=1e30;assert(origin+.125==origin);assert(v[3]>0.);
 bool refused=false;try{(void)probe(.125,{{0,1,{5,2},RemainderPolicy::IntegralOnly}});}catch(const std::runtime_error&){refused=true;}assert(refused);
}
""".replace('LOOP',loop))
 compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
 exe=tmp_path/'probe'
 subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror','-I',str(ROOT/'include'),str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
 subprocess.run([str(exe)],check=True,capture_output=True,text=True)

def test_actual_primary_export_loader_preserves_nonfirst_and_legacy(tmp_path):
 source=(ROOT/'include/pops/runtime/program/module_metadata.hpp').read_text()
 start=source.index('using PrimaryClockFn =')
 end=source.index('using TemporalProviderFn',start)
 fragment=source[start:end]
 cpp=tmp_path/'primary.cpp'
 cpp.write_text(r"""
#include <set>
#include <string>
#include <stdexcept>
#include <cassert>
const char* exported=nullptr;bool present=false;
const char* primary(){return exported;}
namespace pops::dynlib {void* sym(void*,const char*){return present?reinterpret_cast<void*>(&primary):nullptr;}}
struct Metadata{std::string primary_clock_identity;};
std::string load(){void* dl_handle=nullptr;Metadata metadata;std::set<std::string> logical_clocks{"a.secondary","z.primary"};
 FRAGMENT
 return metadata.primary_clock_identity;}
int main(){assert(load().empty());present=true;exported="z.primary";assert(load()=="z.primary");
 bool refused=false;exported="foreign";try{(void)load();}catch(const std::runtime_error&){refused=true;}assert(refused);
 refused=false;exported=nullptr;try{(void)load();}catch(const std::runtime_error&){refused=true;}assert(refused);
 refused=false;exported="";try{(void)load();}catch(const std::runtime_error&){refused=true;}assert(refused);}
""".replace('FRAGMENT',fragment))
 compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
 exe=tmp_path/'primary'
 subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
 subprocess.run([str(exe)],check=True,capture_output=True,text=True)

def test_actual_required_field_closure_order_and_admission(tmp_path):
 source=(ROOT/'src/runtime/amr/amr_system.cpp').read_text()
 start=source.index('  std::set<std::string> accepted_halo_field_closure(')
 end=source.index('  void prepare_accepted_halo_field_dependencies(',start)
 methods=source[start:end]
 cpp=tmp_path/'dag.cpp'
 cpp.write_text(r"""
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <stdexcept>
#include <cassert>
struct Provider {std::string identity() const{return "provider";}};
struct Registry {Provider provider_for_key(const std::string&)const{return {};}
 std::vector<std::string> dependent_provider_identities(const std::vector<std::string>&)const{return {};}};
struct Rhs{unsigned read_contract_version=1;std::array<unsigned,2> state_read_cells{};std::string read_authority="compiler.cell-state-ast@1",declared_input_identity="q";};
struct Binding{std::string identity;};
struct Plan{std::string output_block,output_key;bool output=false;std::vector<std::string> output_keys,boundary_field_blocks,boundary_field_keys;std::vector<Binding> providers;
 bool use_prepared_level_rhs=false;std::vector<std::vector<Rhs>> rhs_by_block{{Rhs{}}};};
struct Hierarchy{std::vector<Registry> auxiliary_registries{Registry{}};std::vector<std::vector<std::vector<std::string>>> accepted_halo_boundary_fields{{{"a"}}};};
struct Fixture{Hierarchy h;Hierarchy* prepared_hierarchy=&h;std::map<std::string,Plan> field_plans;
 METHODS
};
int main(){Fixture f;auto& a=f.field_plans["a"];a.output_block="a";a.output_key="ka";a.boundary_field_blocks={"b"};a.boundary_field_keys={"kb"};
 auto& b=f.field_plans["b"];b.output_block="b";b.output_key="kb";
 auto& unused=f.field_plans["unused"];unused.output_block="unused";unused.output_key="ku";unused.boundary_field_blocks={"unused"};unused.boundary_field_keys={"ku"};unused.use_prepared_level_rhs=true;
 assert((f.accepted_halo_field_order({"a"})==std::vector<std::string>{"b","a"}));f.validate_accepted_halo_field_state_reads();
 bool refused=false;b.rhs_by_block[0][0].read_contract_version=0;try{f.validate_accepted_halo_field_state_reads();}catch(const std::invalid_argument&){refused=true;}assert(refused);b.rhs_by_block[0][0].read_contract_version=1;
 refused=false;b.boundary_field_blocks={"a"};b.boundary_field_keys={"ka"};try{(void)f.accepted_halo_field_order({"a"});}catch(const std::invalid_argument&){refused=true;}assert(refused);
}
""".replace('METHODS',methods))
 compiler=shutil.which('clang++') or shutil.which('c++');assert compiler
 exe=tmp_path/'dag'
 subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
 subprocess.run([str(exe)],check=True,capture_output=True,text=True)
