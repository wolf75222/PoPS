"""Actual frozen accumulator/transport bodies with explicit math/MPI scaffolding."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PIN = "9471bf8ebceda549e75355ed0b9f2d77133d8a06"


def body(text, signature):
    begin = text.index(signature)
    opening = text.index("{", begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[begin:end]


def frozen(path):
    result = subprocess.run(["git", "show", PIN + ":" + path], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    assert (ROOT / path).read_text() == result, "review source drifted from frozen SHA"
    return result


def test_actual_frozen_pair_and_two_word_transport(tmp_path):
    iteration = frozen("include/pops/mesh/execution/for_each.hpp")
    mesh = frozen("include/pops/mesh/storage/mf_arith.hpp")
    accumulator = body(iteration, "struct FiniteCompensatedSum {") + ";"
    transport = body(mesh, "inline Real collective_finite_compensated_sum(")
    kernel = body(mesh, "template <int Dim>\nstruct FiniteOwnedDotKernel {") + ";"
    receipt = json.loads((ROOT / "docs/development/api_040/dot_all_precision_saved_oracle_f361_sol61.json").read_text())
    compiler = shutil.which("clang++")
    assert compiler, "independent host reception requires clang++; no masked skip"
    cpp = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define POPS_HD
#define POPS_HAS_MPI 1
namespace Kokkos { using std::isfinite; }
namespace pops {
using Real=double;
template<int N> using Index=std::array<int,N>;
template<class T,int N> struct FieldView {
 const double* data{};
 double operator()(const Index<N>&,int component) const { return data[component]; }
};
struct CommunicatorView {
 int count=2; bool enabled=true;
 bool active() const {return enabled;} int size() const {return count;}
 int native_handle() const {return 187;}
};
int votes=0,calls=0,inject=0;
std::vector<double> packets;
long all_reduce_max(long local,const CommunicatorView& c) {
 if(c.native_handle()!=187) throw std::runtime_error("foreign vote lane");
 ++votes; return (votes==inject)?1:local;
}
int mpi_real_datatype() {return 37;}
int MPI_Allgather(const double* send,int send_count,int send_type,double* receive,
                 int receive_count,int receive_type,int handle) {
 if(handle!=187||send_count!=2||receive_count!=2||send_type!=37||receive_type!=37)
   throw std::runtime_error("foreign transport lane/type/count");
 if(send[0]!=packets[0]||send[1]!=packets[1])
   throw std::runtime_error("local low word was not sent exactly");
 ++calls; std::copy(packets.begin(),packets.end(),receive); return 0;
}
namespace detail { void require_mpi_success(int result,const char*) {
 if(result) throw std::runtime_error("MPI failure"); } }
'''+accumulator+"\n"+kernel+"\n"+transport+r'''
}
using Sum=pops::FiniteCompensatedSum;
int checks=0;
void check(bool valid) {++checks;if(!valid) throw std::runtime_error("independent check failed");}
Sum summarize(const std::vector<double>& products,int grain) {
 Sum result;
 for(std::size_t begin=0;begin<products.size();begin+=grain) {
  Sum patch;
  for(std::size_t end=begin;end<products.size()&&end<begin+grain;++end) patch.add(products[end]);
  result.join(patch);
 }
 return result;
}
double gathered(const std::vector<Sum>& ranks,int fail_vote=0) {
 pops::packets.clear();pops::votes=0;pops::calls=0;pops::inject=fail_vote;
 for(const auto& rank:ranks) {pops::packets.push_back(rank.high);pops::packets.push_back(rank.low);}
 return pops::collective_finite_compensated_sum(ranks[0],{int(ranks.size()),true});
}
template<class Exception,class F> void refuse(F f,const char* message,int votes,int calls) {
 try {f();check(false);} catch(const Exception& error) {check(std::string(error.what())==message);}
 check(pops::votes==votes);check(pops::calls==calls);
}
int main() {
 std::vector<double> values{-1e16,1.,1e16};
 do {
  for(int grain:{1,2,3}) {auto pair=summarize(values,grain);check(pair.finite());check(pair.value()==1.);}
  Sum a;a.add(values[0]);Sum b;b.add(values[1]);b.add(values[2]);a.join(b);check(a.value()==1.);
  std::vector<Sum> ranks(3);
  for(int i=0;i<3;++i) ranks[i].add(values[i]);
  check(gathered(ranks)==1.);check(pops::votes==3&&pops::calls==1);
 } while(std::next_permutation(values.begin(),values.end()));
 Sum high_low;high_low.add(1e16);high_low.add(1.);Sum negative;negative.add(-1e16);
 check(high_low.high==1e16&&high_low.low==1.);
 check(gathered({high_low,negative})==1.);
 check(gathered({Sum{},high_low,Sum{},negative})==1.);
 Sum bad;bad.add(std::numeric_limits<double>::quiet_NaN());
 Sum propagated;propagated.join(bad);check(!propagated.finite());
 refuse<std::overflow_error>([&]{gathered({bad,Sum{}});},"Program dot_all local summary is nonfinite",1,0);
 refuse<std::overflow_error>([&]{gathered({Sum{},Sum{}},1);},"Program dot_all local summary is nonfinite",1,0);
 refuse<std::runtime_error>([&]{gathered({Sum{},Sum{}},2);},"Program dot_all could not allocate collective summaries",2,0);
 refuse<std::overflow_error>([&]{gathered({Sum{},Sum{}},3);},"Program dot_all collective sum is nonfinite",3,1);
 Sum huge;huge.add(std::numeric_limits<double>::max());
 refuse<std::overflow_error>([&]{gathered({huge,huge});},"Program dot_all collective sum is nonfinite",3,1);
 pops::votes=0;pops::calls=0;pops::inject=0;
 check(pops::collective_finite_compensated_sum(high_low,{1,true})==high_low.value());
 check(pops::votes==1&&pops::calls==0);
 double a[1]={std::numeric_limits<double>::quiet_NaN()},b[1]={1},active[1]={0},coverage[1]={1};
 pops::FiniteOwnedDotKernel<1> kernel{{a},{b},{active},{coverage},0,true,true};
 check(kernel({0})==0.);active[0]=1;coverage[0]=0;check(kernel({0})==0.);
 active[0]=std::numeric_limits<double>::quiet_NaN();check(!std::isfinite(kernel({0})));
 active[0]=1;coverage[0]=1;check(!std::isfinite(kernel({0})));
 a[0]=1e308;b[0]=1e308;check(!std::isfinite(kernel({0})));
 a[0]=2;b[0]=3;check(kernel({0})==6.);
'''
    for row in receipt["cases"]:
        lhs, rhs = row["products"]
        cpp += "{const std::vector<double> numerator{%s},denominator{%s};\n" % (",".join(lhs), ",".join(rhs))
        cpp += "const double start=%s,dt=%s,target=%s;\n" % (row["start"], row["requested"], row["target"])
        cpp += r'''
 for(int grain:{1,2,7,32,64,128}) {
  const auto n=summarize(numerator,grain),d=summarize(denominator,grain);
  check(n.finite()&&d.finite());
  const double reached=start+(-2.*n.value()/d.value())*dt;
  double lo=target,hi=target;for(int i=0;i<4;++i){lo=std::nextafter(lo,-INFINITY);hi=std::nextafter(hi,INFINITY);}
  check(reached>=lo&&reached<=hi);
 }
}
'''
    cpp += 'std::cout<<"actual frozen pair/transport checks="<<checks<<"\\n";return 0;}\n'
    path = tmp_path / "independent.cpp"
    path.write_text(cpp)
    binary = tmp_path / "independent"
    built = subprocess.run([compiler, "-std=c++20", "-O0", str(path), "-o", str(binary)],
                           capture_output=True, text=True, timeout=30)
    assert built.returncode == 0, built.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stderr
    assert executed.stdout == "actual frozen pair/transport checks=105\n"


def test_summary_is_kept_across_every_authored_reduction_boundary():
    mesh = frozen("include/pops/mesh/storage/mf_arith.hpp")
    uniform = frozen("include/pops/runtime/program/program_context.hpp")
    amr = frozen("include/pops/runtime/program/amr_program_context_spatial_operations.inc")
    local = body(mesh, "FiniteCompensatedSum dot_owned_active_all_finite_sum_local(")
    assert "result.join(patch);" in local and ".value()" not in local
    for source in (uniform, amr):
        provider = body(source, "Real dot_all(int program_block,")
        assert "dot_owned_active_all_finite_sum_local" in provider
        assert "collective_finite_compensated_sum" in provider
        assert "lane.communicator()" in provider and "all_reduce_sum(" not in provider
    assert "accumulated.join(checked);" in body(amr, "Real dot_all(int program_block,")
