"""Public M17 arithmetic composition; Native deliberately unexecuted."""
import sys
from pathlib import Path
import pytest
import pops
from pops.codegen.module_codegen import _emit_bricks
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from tests.python.support.atomic_cubature_path_case import make_case


@pytest.mark.parametrize('nonconservative',[False,True])
def test_public_atomic_path_declarations_resolve_and_emit(nonconservative):
    case,layout=make_case(nonconservative=nonconservative)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    block=resolved.blocks[0]
    lowered,_=lower_and_validate(block.model,state_space=block.state_spaces[0],
                                resolved_operations=block.resolved_operations,numerics=block.numerics)
    cpp=emit_cpp_program(resolved.time,model=lowered,target='system')
    brick=_emit_bricks(lowered._m)[1]
    assert 'ctx.path_rhs_into(' in cpp
    assert 'raw_left[' in brick and 'raw_right[' in brick
    assert 'hermite' not in brick.lower()
    assert 'fan_li' not in brick.lower()
    assert lowered._m._path_conservative['identity'].startswith('pops.numerics.normalized-polynomial-path.v1:sha256:')
    assert 'pops._pops' not in sys.modules
    assert Path(pops.__file__).resolve()==Path(__file__).resolve().parents[2]/'python/pops/__init__.py'


@pytest.mark.parametrize('nonconservative,raw_speed',[(False,False),(True,False),(True,True)])
def test_actual_generated_atomic_kernel_against_particle_flux_and_raw_integral(tmp_path,nonconservative,raw_speed):
    import subprocess
    from pops.codegen.moment_path_kernel import emit_moment_path_kernel
    from tests.python.support.atomic_cubature_path_case import declarations
    *_,path,basis=declarations(nonconservative=nonconservative,raw_speed=raw_speed)
    kernel='\n'.join(emit_moment_path_kernel(path.native_kernel()['plan'],'AtomicKernel'))
    # The values below are obtained by independent atom summation in C++,
    # not by executing the Vandermonde closure that authored the Python graph.
    source='''#include <pops/numerics/moments/normalized_moment_path.hpp>
#include <array>
#include <cmath>
#include <limits>
'''+kernel+f'\nconstexpr double strength={int(nonconservative)};'+r'''
int main(){double nodes[6][2]={{0,0},{1,0},{-1,0},{0,1},{0,-1},{1,1}};
int indices[6][2]={{1,1},{0,0},{0,2},{1,0},{2,0},{0,1}};
double wl[6]={1,2,3,4,5,6},wr[6]={2,1,4,2,3,5};
std::array<double,6> left{},right{};
for(int i=0;i<6;++i)for(int k=0;k<6;++k){double mon=std::pow(nodes[k][0],indices[i][0])*std::pow(nodes[k][1],indices[i][1]);left[i]+=wl[k]*mon;right[i]+=wr[k]*mon;}
for(auto g: {std::array<double,2>{2,-3},std::array<double,2>{-.75,.25},std::array<double,2>{0,0}}){
auto f=AtomicKernel{}.path_directional_flux(left,g);auto r=AtomicKernel{}.path_integral(left,right,g);
auto reverse=AtomicKernel{}.path_integral(right,left,g);auto same=AtomicKernel{}.path_integral(left,left,g);
if(!f.succeeded()||!r.succeeded()||!reverse.succeeded()||!same.succeeded())return 1;
for(int i=0;i<6;++i){double expectedflux=0;for(int k=0;k<6;++k)expectedflux+=wl[k]*(g[0]*nodes[k][0]+g[1]*nodes[k][1])*std::pow(nodes[k][0],indices[i][0])*std::pow(nodes[k][1],indices[i][1]);
double expectedintegral=strength*(g[0]-g[1]/2)*(left[1]+right[1])/2*(right[i]-left[i]);
if(std::abs(f.flux.values[i]-expectedflux)>1e-12||std::abs(r.integral[i]-expectedintegral)>1e-12)return 2;
if(r.integral[i]!=-reverse.integral[i]||same.integral[i]!=0)return 3;}
for(double s:{0.,.375,1.}){double rho=left[1]+s*(right[1]-left[1]);for(auto& node:nodes)if(std::abs(g[0]*node[0]+g[1]*node[1]+strength*(g[0]-g[1]/2)*rho)>r.speed_bound)return 4;}}
left[1]=-1;if(AtomicKernel{}.path_integral(left,right,{1,0}).succeeded())return 5;
return 0;}
'''
    cpp,binary=tmp_path/'atomic.cpp',tmp_path/'atomic'
    cpp.write_text(source)
    root=Path(__file__).resolve().parents[2]
    subprocess.run(['c++','-std=c++20','-O2','-fno-fast-math','-ffp-contract=off','-I'+str(root/'include'),str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
    subprocess.run([str(binary)],check=True,capture_output=True,text=True)
