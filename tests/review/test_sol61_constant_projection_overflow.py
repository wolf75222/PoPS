"""Independent SOURCE_ONLY actual-body overflow regression, never native evidence."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HEADER = "include/pops/runtime/analytic/initial_materialization.hpp"

def _body(source):
    return source[source.index("POPS_HD inline Real gauss_node"):source.index("template <int Dim>\nstruct AnalyticInitialKernel")]

def test_actual_body_opposed_finite_samples_have_finite_mean():
    old = subprocess.check_output(["git","show","a2408d6f:"+HEADER], cwd=ROOT, text=True)
    new = (ROOT/HEADER).read_text()
    assert old[old.index("POPS_HD inline Real gauss_node"):old.index("template <int Dim>\nstruct AnalyticCellAverage")] == new[new.index("POPS_HD inline Real gauss_node"):new.index("template <int Dim>\nstruct AnalyticCellAverage")]
    common = r"""
#include <array>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#define POPS_HD
using Real=double;
template<int D>using Index=std::array<int,D>;
template<int D>using RealVector=std::array<double,D>;
namespace Kokkos {using std::isfinite;}
enum class AnalyticOp {Constant, Other};
struct Instruction {AnalyticOp op=AnalyticOp::Other;};
struct Eval {double value;bool valid;};
struct AnalyticProgramView {
 int instruction_count=2;Instruction instructions[2];double amplitude=1.5e308;int degree=1,mode=0;
 template<std::size_t D>double eval(const std::array<double,D>& p)const{
   if(degree==0)return amplitude;
   if(mode==1)return amplitude*(p[0]+.25)-amplitude*.25;
   return amplitude*std::pow(p[0],degree);
 }
 template<std::size_t D>Eval eval_checked(const std::array<double,D>& p,const double*,int)const{return {eval<D>(p),true};}
};
template<int D>struct Geometry {
 double spacing(int)const{return 2.;}
 double face_coordinate(int,int i)const{return -1.+2.*i;}
 RealVector<D>cell_center(const Index<D>&)const{RealVector<D>p{};return p;}
};
"""
    main = r"""
template<int D>void check(){
 Index<D> index{};Geometry<D>geometry;AnalyticProgramView p;
 for(int mode=0;mode<2;++mode){
   p.mode=mode;
   double before=baseline::AnalyticCellAverage<D>{p,geometry}(index);
   double after=candidate::AnalyticCellAverage<D>{p,geometry}(index);
   // Independent antiderivative: mean(A*x) on [-1,1] is exactly zero.
   if(!std::isfinite(before)||std::abs(before)>1e294)throw "baseline not finite odd mean";
   std::cout<<D<<" "<<mode<<" "<<std::setprecision(17)<<before<<" "<<after<<"\n";
   if(!std::isfinite(after)||std::abs(after)>1e294)throw "finite sample convex mean failed";
 }
 p.mode=0;p.degree=0;
 for(double value:{.1,-.7,std::numeric_limits<double>::max(),-std::numeric_limits<double>::max(),std::numeric_limits<double>::denorm_min(),0.,-0.}){
   p.amplitude=value;
   double result=candidate::AnalyticCellAverage<D>{p,geometry}(index);
   if(result!=value||std::signbit(result)!=std::signbit(value))throw "candidate constant bits changed";
 }
 p.amplitude=2.7;
 for(int degree=1;degree<=7;++degree){
   p.degree=degree;
   // Independent antiderivative on [-1,1], not another quadrature algorithm.
   double expected=degree%2 ? 0. : p.amplitude/(degree+1);
   double result=candidate::AnalyticCellAverage<D>{p,geometry}(index);
   if(!std::isfinite(result)||std::abs(result-expected)>64*std::numeric_limits<double>::epsilon()*p.amplitude)
     throw "candidate changed polynomial mean";
 }
}
int main(){check<1>();check<2>();check<3>();}
"""
    source=common+"namespace baseline {\n"+_body(old)+"}\nnamespace candidate {\n"+_body(new)+"}\n"+main
    with tempfile.TemporaryDirectory(prefix="pops-independent-projection-overflow-") as folder:
        cpp=Path(folder)/"probe.cpp";binary=Path(folder)/"probe"
        cpp.write_text(source)
        subprocess.run(["c++","-std=c++20","-Wall","-Wextra","-Werror",str(cpp),"-o",str(binary)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
        rows=result.stdout.splitlines()
        assert len(rows)==6
        print("SOURCE_ONLY actual old/candidate body (dimension mode old new):\n"+result.stdout)
