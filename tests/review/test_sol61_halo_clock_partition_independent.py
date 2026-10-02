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
