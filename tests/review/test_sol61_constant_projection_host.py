"""SOURCE_ONLY exact C++ quadrature-body host probe; no PoPS/DSO/Kokkos build."""
from pathlib import Path
import subprocess
import tempfile


def test_actual_cpp_projection_body_preserves_arbitrary_constants_and_gauss_moments():
    root=Path(__file__).resolve().parents[2]
    source=(root/"include/pops/runtime/analytic/initial_materialization.hpp").read_text()
    body=source[source.index("POPS_HD inline Real gauss_node"):source.index("template <int Dim>\nstruct AnalyticInitialKernel")]
    # POD test geometry/evaluator substitute only their interfaces. Quadrature
    # nodes, weights, loop and averaging are copied directly from actual source.
    probe=r"""
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>
#define POPS_HD
using Real=double;
template<int D>using Index=std::array<int,D>;
template<int D>using RealVector=std::array<double,D>;
namespace Kokkos {using std::isfinite;}
enum class AnalyticOp {Constant, Other};
struct Instruction {AnalyticOp op=AnalyticOp::Other;};
struct Eval {double value;bool valid;};
struct AnalyticProgramView {
 int instruction_count=2;Instruction instructions[2];double constant=0;int degree=0,axis=0;
 template<std::size_t D>double eval(const std::array<double,D>& p)const {return degree ? constant+std::pow(p[axis],degree):constant;}
 template<std::size_t D>Eval eval_checked(const std::array<double,D>& p,const double*,int)const{return {eval<D>(p),true};}
};
template<int D>struct Geometry {
 double spacing(int)const{return .125;}
 double face_coordinate(int,int i)const{return i*.125;}
 RealVector<D>cell_center(const Index<D>& i)const{RealVector<D>p{};for(int a=0;a<D;++a)p[a]=(i[a]+.5)*.125;return p;}
};
"""+body+r"""
template<int D>void check(){
 Geometry<D>geometry;Index<D>index{};index.fill(3);
 for(double c:{.1,-.7,1.,3.25,1e20,std::numeric_limits<double>::min(),-0.}){
   AnalyticProgramView program;program.constant=c;
   AnalyticCellAverage<D>average{program,geometry};double actual=average(index);
   if(actual!=c || std::signbit(actual)!=std::signbit(c))throw std::runtime_error("constant mode changed");
 }
 for(int degree=1;degree<=7;++degree)for(int axis=0;axis<D;++axis){
   AnalyticProgramView program;program.constant=.3;program.degree=degree;program.axis=axis;
   AnalyticCellAverage<D>average{program,geometry};
   double lo=geometry.face_coordinate(axis,index[axis]),hi=geometry.face_coordinate(axis,index[axis]+1);
   double expected=.3+(std::pow(hi,degree+1)-std::pow(lo,degree+1))/((degree+1)*(hi-lo));
   if(std::abs(average(index)-expected)>2e-14)throw std::runtime_error("Gauss polynomial moment changed");
 }
}
int main(){check<1>();check<2>();check<3>();std::cout<<"SOURCE_ONLY host constants and degree1..7 moments passed\n";}
"""
    with tempfile.TemporaryDirectory(prefix="pops-constant-projection-host-") as directory:
        cpp=Path(directory)/"probe.cpp";binary=Path(directory)/"probe"
        cpp.write_text(probe)
        subprocess.run(["c++","-std=c++20","-Wall","-Wextra","-Werror",str(cpp),"-o",str(binary)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
        assert "SOURCE_ONLY" in result.stdout
