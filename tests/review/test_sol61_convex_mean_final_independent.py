"""Independent SOURCE_ONLY final convex-mean probe on the actual header body."""
from pathlib import Path
import subprocess
import tempfile


def test_actual_cpp_convex_mean_weight_bookkeeping_extrema_and_signed_zero():
    root = Path(__file__).resolve().parents[2]
    source = (root/"include/pops/runtime/analytic/initial_materialization.hpp").read_text()
    body = source[source.index("POPS_HD inline Real gauss_node"):source.index("template <int Dim>\nstruct AnalyticInitialKernel")]
    probe = r"""
#include <array>
#include <cmath>
#include <limits>
#include <iostream>
#include <random>
#include <algorithm>
#define POPS_HD
using Real=double;
template<int D>using Index=std::array<int,D>;
template<int D>using RealVector=std::array<double,D>;
namespace Kokkos {using std::isfinite;}
enum class AnalyticOp {Constant, Other};
struct Instruction {AnalyticOp op=AnalyticOp::Other;};
struct Eval {double value;bool valid;};
constexpr double nodes[4]={-.861136311594052575223946488892809505,-.339981043584856264802665759103244687,.339981043584856264802665759103244687,.861136311594052575223946488892809505};
constexpr long double weights[4]={.347854845137453857373063949221999408L,.652145154862546142626936050778000593L,.652145154862546142626936050778000593L,.347854845137453857373063949221999408L};
struct AnalyticProgramView {
 int instruction_count=2;Instruction instructions[2];std::array<double,64> values{};
 template<std::size_t D>double eval(const std::array<double,D>& p)const{
   int sample=0,factor=1;
   for(std::size_t axis=0;axis<D;++axis){
     int node=0;for(int i=1;i<4;++i)if(std::abs(p[axis]-nodes[i])<std::abs(p[axis]-nodes[node]))node=i;
     if(std::abs(p[axis]-nodes[node])>1e-15)throw "unexpected quadrature node";
     sample+=factor*node;factor*=4;
   }
   return values[sample];
 }
 template<std::size_t D>Eval eval_checked(const std::array<double,D>& p,const double*,int)const{return {eval<D>(p),true};}
};
template<int D>struct Geometry {
 double spacing(int)const{return 2.;}
 double face_coordinate(int,int i)const{return -1.+2.*i;}
 RealVector<D>cell_center(const Index<D>&)const{RealVector<D>p{};return p;}
};
"""+body+r"""
template<int D>void check(){
 constexpr int count=1<<(2*D);Index<D>index{};Geometry<D>geometry;AnalyticProgramView p;
 auto compare=[&](){
   double scale=0;for(int sample=0;sample<count;++sample)scale=std::max(scale,std::abs(p.values[sample]));
   long double numerator=0,denominator=0;
   for(int sample=0;sample<count;++sample){
     int encoded=sample;long double weight=1;
     for(int axis=0;axis<D;++axis){weight*=weights[encoded&3];encoded>>=2;}
     // Scaled independent full quadrature reference, not online interpolation.
     numerator+=weight*(scale ? static_cast<long double>(p.values[sample])/scale:0);
     denominator+=weight;
   }
   double actual=AnalyticCellAverage<D>{p,geometry}(index);
   if(!std::isfinite(actual))throw "finite convex samples produced nonfinite mean";
   long double normalized=scale ? static_cast<long double>(actual)/scale:0;
   long double error=std::abs(normalized-numerator/denominator);
   if(scale>=std::numeric_limits<double>::min()&&error>1e-13L)throw "weighted mean differs from independent quadrature";
   return actual;
 };
 // All skipped equal samples must still contribute to the denominator.
 p.values.fill(.6);p.values[count-1]=-.2;compare();
 p.values.fill(std::numeric_limits<double>::max());p.values[count-1]=-std::numeric_limits<double>::max();compare();
 for(int i=0;i<count;++i)p.values[i]=(i&1) ? -std::numeric_limits<double>::max():std::numeric_limits<double>::max();compare();
 for(int i=0;i<count;++i)p.values[i]=(i&1) ? std::nextafter(std::numeric_limits<double>::max(),0.):std::numeric_limits<double>::max();compare();
 for(double value:{.1,-.7,std::numeric_limits<double>::max(),-std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min(),-std::numeric_limits<double>::denorm_min(),0.,-0.}){
   p.values.fill(value);double result=compare();
   if(result!=value||std::signbit(result)!=std::signbit(value))throw "constant bit pattern changed";
 }
 p.values.fill(0.);p.values[0]=-0.;if(compare()!=0.)throw "mixed signed zeros changed numerical zero";
 p.values.fill(-0.);p.values[0]=0.;if(compare()!=0.)throw "mixed signed zeros changed numerical zero";
 std::mt19937_64 rng(20261002);std::uniform_real_distribution<double>fraction(-1.,1.);
 for(int trial=0;trial<1000;++trial){
   double amplitude=(trial&1) ? std::numeric_limits<double>::max():.7;
   for(int sample=0;sample<count;++sample)p.values[sample]=amplitude*fraction(rng);
   compare();
 }
}
int main(){check<1>();check<2>();check<3>();std::cout<<"SOURCE_ONLY independent convex reference passed\n";}
"""
    with tempfile.TemporaryDirectory(prefix="pops-convex-mean-independent-") as directory:
        cpp=Path(directory)/"probe.cpp";binary=Path(directory)/"probe"
        cpp.write_text(probe)
        subprocess.run(["c++","-std=c++20","-Wall","-Wextra","-Werror",str(cpp),"-o",str(binary)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
        assert "SOURCE_ONLY" in result.stdout
