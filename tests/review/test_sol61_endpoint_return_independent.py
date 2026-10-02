"""New emitted headers only; never a reconstruction of the failed Native brick."""
from pathlib import Path
import subprocess
from pops.codegen.moment_path_kernel import emit_moment_path_kernel
from pops.moments import CartesianMonomialBasis
from pops.moments.polynomial_path import NormalizedPathInputs,EndpointPathInputs,endpoint_polynomial_path


def test_distinct_density_last_storage_return_and_missing_return_counterexample(tmp_path):
    basis=CartesianMonomialBasis(((2,0),(0,1),(1,1),(0,2),(1,0),(0,0)))
    inputs,pair=NormalizedPathInputs(basis),EndpointPathInputs(basis)
    plan=endpoint_polynomial_path(basis,
        flux=tuple(inputs.direction(0)*inputs.raw(i) for i in basis.indices),
        integral=tuple((pair.direction(0)-pair.direction(1)/2)*
            (pair.left((0,0))+pair.right((0,0)))/2*(pair.right(i)-pair.left(i)) for i in basis.indices),
        speed=1+inputs.density)
    emitted=emit_moment_path_kernel(plan,'Law')
    body=r'''
int main(){
 std::array<double,6> a={2,0,0,2,0,2},b={3.03,0,0,3,.3,3};
 for(auto g:{std::array<double,2>{-.5,.75},std::array<double,2>{.25,-.5},std::array<double,2>{0,0}}){
  auto x=Law{}.path_integral(a,b,g),y=Law{}.path_integral(b,a,g),same=Law{}.path_integral(a,a,g);
  if(!x.succeeded()||!y.succeeded()||!same.succeeded())return 1;
  for(int k=0;k<6;++k){double ref=(g[0]-g[1]/2)*2.5*(b[k]-a[k]);
   if(std::abs(x.integral[k]-ref)>1e-14||y.integral[k]!=-x.integral[k]||same.integral[k]!=0)return 2;}
 }
 b[5]=std::numeric_limits<double>::infinity();
 if(Law{}.path_integral(a,b,{1,0}).succeeded())return 3;
 return 0;
}
'''
    root=Path(__file__).resolve().parents[2]
    def compile(lines,name):
        cpp=tmp_path/(name+'.cpp');exe=tmp_path/name
        cpp.write_text('#include <pops/numerics/moments/normalized_moment_path.hpp>\n#include <array>\n'+ '\n'.join(lines)+body)
        result=subprocess.run(['c++','-std=c++20','-O2','-fno-fast-math','-ffp-contract=off',
            '-Werror=return-type','-I'+str(root/'include'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
        return result,exe
    missing=list(emitted)
    last=max(index for index,line in enumerate(missing) if line=='    return result;')
    del missing[last]
    red,_=compile(missing,'missing')
    assert red.returncode!=0 and 'return' in red.stderr
    green,exe=compile(emitted,'fixed')
    assert green.returncode==0,green.stderr
    subprocess.run([str(exe)],check=True)
