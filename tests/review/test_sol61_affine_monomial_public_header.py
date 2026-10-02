"""Actual public header CPU probe, with no mock body or Native/DSO qualification."""
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('real',['double','float'])
def test_real_header_discrete_measure_affine_push_forward(tmp_path,real):
    source=tmp_path/'probe.cpp'; binary=tmp_path/'probe'
    source.write_text(r'''
#include <pops/numerics/moments/affine_velocity.hpp>
#include <algorithm>
#include <iostream>
template<int Degree> bool check() {
  using B=pops::moments::CartesianMomentBasis<Degree>;
  using R=pops::Real;
  R old[B::size]{}, endpoint[B::size]{}, out[B::size]{};
  const long double x[]={-.3L,.4L,1.2L}, y[]={.6L,-.2L,.8L}, w[]={.2L,.3L,.5L};
  for(int q=0;q<=Degree;++q)for(int p=0;p+q<=Degree;++p){
    long double sum=0;for(int k=0;k<3;++k)sum+=w[k]*std::pow(x[k],p)*std::pow(y[k],q);
    old[B::index(p,q)]=R(sum); endpoint[B::index(p,q)]=old[B::index(p,q)];
  }
  const R omega=.75,half=.125;
  // Independent rational Cayley rotation and specified endpoint translation.
  const long double a=static_cast<long double>(omega)*half,c=(1-a*a)/(1+a*a),s=2*a/(1+a*a);
  endpoint[B::index(1,0)]=R(.23);endpoint[B::index(0,1)]=R(-.17);
  const long double bx=static_cast<long double>(endpoint[B::index(1,0)])-(c*old[B::index(1,0)]+s*old[B::index(0,1)]);
  const long double by=static_cast<long double>(endpoint[B::index(0,1)])-(-s*old[B::index(1,0)]+c*old[B::index(0,1)]);
  if(!pops::moments::affine_velocity_push_forward<Degree>(old,endpoint,omega,half,out))return false;
  const long double tol=sizeof(R)==8?2e-12L:3e-5L;
  for(int q=0;q<=Degree;++q)for(int p=0;p+q<=Degree;++p){
    long double expected=0;for(int k=0;k<3;++k)expected+=w[k]*std::pow(bx+c*x[k]+s*y[k],p)*std::pow(by-s*x[k]+c*y[k],q);
    if(std::fabs(out[B::index(p,q)]-expected)>tol*std::max(1.L,std::fabs(expected)))return false;
  }
  if(out[0]!=old[0]||out[B::index(1,0)]!=endpoint[B::index(1,0)]||out[B::index(0,1)]!=endpoint[B::index(0,1)])return false;
  for(auto& v:out)v=R(7);
  endpoint[0]=R(2);
  if(pops::moments::affine_velocity_push_forward<Degree>(old,endpoint,omega,half,out))return false;
  for(auto v:out)if(v!=R(7))return false;
  return true;
}
int main(){if(!check<2>()||!check<4>()||!check<5>())return 1; std::cout<<"actual header host PASS\n";}
''')
    subprocess.run(['c++','-std=c++20','-Wall','-Wextra','-Werror',f'-DPOPS_REAL_TYPE={real}','-I'+str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    result=subprocess.run([str(binary)],check=True,capture_output=True,text=True)
    assert result.stdout=='actual header host PASS\n'
