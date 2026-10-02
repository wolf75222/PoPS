"""Actual generated endpoint body: strict compile and independent Raw integral."""
from pathlib import Path
import shutil, subprocess
from pops.codegen.moment_path_kernel import emit_moment_path_kernel
from pops.moments import CartesianMonomialBasis
from pops.moments.polynomial_path import NormalizedPathInputs,EndpointPathInputs,endpoint_polynomial_path

def test_actual_endpoint_header_returns_finite_raw_integral(tmp_path):
    basis=CartesianMonomialBasis(((1,1),(0,0),(0,2),(1,0),(2,0),(0,1)))
    endpoint,pair=NormalizedPathInputs(basis),EndpointPathInputs(basis)
    plan=endpoint_polynomial_path(basis,flux=tuple(endpoint.direction(0)*endpoint.raw(i) for i in basis.indices),
        integral=tuple((pair.direction(0)-pair.direction(1)/2)*(pair.left((0,0))+pair.right((0,0)))/2*(pair.right(i)-pair.left(i)) for i in basis.indices),
        speed=1+endpoint.density)
    source='#include <pops/numerics/moments/normalized_moment_path.hpp>\n#include <array>\n'+ '\n'.join(emit_moment_path_kernel(plan,'Law'))+r'''
int main(){
 double a[6]={0,2,2,0,2,0},b[6]={0,3,3,.3,3.03,0};
 auto x=Law{}.path_integral(a,b,{.7,-.2});
 auto y=Law{}.path_integral(b,a,{.7,-.2});
 auto z=Law{}.path_integral(a,a,{.7,-.2});
 auto zero=Law{}.path_integral(a,b,{0,0});
 if(!x.succeeded()||!y.succeeded()||!z.succeeded()||!zero.succeeded())return 1;
 for(int k=0;k<6;++k){
   double ref=.8*2.5*(b[k]-a[k]);
   if(std::abs(x.integral[k]-ref)>1e-14||y.integral[k]!=-x.integral[k]||z.integral[k]!=0||zero.integral[k]!=0)return 2;
 }
 b[1]=std::numeric_limits<double>::infinity();
 if(Law{}.path_integral(a,b,{1,0}).succeeded())return 3;
}
'''
    cpp=tmp_path/'endpoint.cpp';cpp.write_text(source)
    root=Path(__file__).resolve().parents[2];exe=tmp_path/'endpoint'
    result=subprocess.run([shutil.which('clang++') or 'c++','-std=c++20','-DPOPS_NATIVE_DIM=2','-O2','-fno-fast-math','-ffp-contract=off','-Werror=return-type','-I',str(root/'include'),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    subprocess.run([str(exe)],check=True)
