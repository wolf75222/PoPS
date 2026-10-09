"""Source-only actual dispatch lambda; no Native proof."""
from pathlib import Path
import shutil
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_actual_callback_whole_tower_and_legacy_dispatch(tmp_path):
 source=(ROOT/'include/pops/runtime/program/amr_program_context_subcycling_runtime.inc').read_text()
 start=source.index('[&](auto&& first, auto&& second, auto&&... history)')
 callback=source[start:source.index('},',start)+1]
 cpp=tmp_path/'probe.cpp'
 cpp.write_text(r"""
#include <vector>
#include <type_traits>
#include <utility>
#include <cassert>
using M=std::vector<std::vector<int>>;
struct Probe {
 int legacy=0,whole=0;
 void stage_prepared_publication_candidates_(size_t l,std::vector<int>& r){assert(l<5&&r.size()==3);++legacy;}
 void prepare_accepted_publication_halos_(double dt,M& m,const M& c,const M& h){
 assert(dt==.125&&m.size()==3&&c.size()==3&&h.size()==3);
 for(auto& r:m)assert(r.size()==5);++whole;}
 void run(){double dt=.125;auto callback=CALLBACK;
 static_assert(std::is_invocable_v<decltype(callback)&,M&,const M&,const M&>);
 M m(3,std::vector<int>(5)),c=m,h=m;
 for(size_t l=0;l<5;++l){std::vector<int> r(3);callback(l,r);}
 callback(m,std::as_const(c),std::as_const(h));assert(legacy==5&&whole==1);}
};int main(){Probe{}.run();}
""".replace('CALLBACK',callback).replace('#include <vector>','#include <vector>\n#include <cstddef>\nusing std::size_t;'))
 compiler=shutil.which('clang++') or shutil.which('c++')
 assert compiler, 'host compiler required'
 exe=tmp_path/'probe'
 subprocess.run([compiler,'-std=c++20','-O0',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
 subprocess.run([str(exe)],check=True,capture_output=True,text=True)
