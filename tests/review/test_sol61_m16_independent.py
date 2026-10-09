"""Nonauthor order-five public composition and actual-header CPU checks."""
from fractions import Fraction as F
from pathlib import Path
import subprocess
import sys
import pytest
import pops
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.moments import CartesianMonomialBasis, affine_push_forward
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program

ROOT = Path(__file__).resolve().parents[2]


def test_order_five_translated_atomic_measure_and_public_binding():
    canonical = tuple((p,q) for q in range(6) for p in range(6-q))
    indices = canonical[7:] + canonical[:7]
    basis = CartesianMonomialBasis(indices)
    atoms = ((F(1,4),F(-3,4),F(2)),(F(-1,2),F(1,8),F(3)),(F(3,2),F(5,4),F(1)))
    old = tuple(sum(w*x**p*y**q for x,y,w in atoms) for p,q in indices)
    matrix=((F(3,5),F(4,5)),(F(-4,5),F(3,5)))
    offset=(F(1,4),F(-1,8))
    mapped=affine_push_forward(old,indices=indices,matrix=matrix,offset=offset)
    for value,(p,q) in zip(mapped,indices):
        assert value == sum(w*(offset[0]+matrix[0][0]*x+matrix[0][1]*y)**p
                           *(offset[1]+matrix[1][0]*x+matrix[1][1]*y)**q for x,y,w in atoms)
    names=tuple('storage_'+str((i*8)%21) for i in range(21))
    binding=dict(zip(indices,names))
    assert names.index(binding[(0,0)]) != 0
    frame=Rectangle('rectangle',(0,0),(2,1)).frame(Cartesian2D())
    model=pops.Model('atomic_source',frame=frame)
    state=model.state('population',components=names)
    a=[[0.]*21 for _ in range(21)]
    x,y=names.index(binding[(1,0)]),names.index(binding[(0,1)])
    a[x][y],a[y][x]=2.,-2.
    operator=model.operator('rotation',returns=model.local_linear_operator('rotation',on=state,matrix=tuple(map(tuple,a))))
    case=pops.Case('independent'); block=case.block('population_block',model=model)
    program=pops.Program('atomic_affine'); value=program.state(block[state])
    result=program.affine_moment_update(value.n,value.n,linear_operator=operator,theta_dt=program.dt/2,basis=basis,components=binding)
    program.commit(value.next,program.value('mapped',result,at=value.next.point))
    lowered,_=lower_and_validate(model,facade=model)
    cpp=emit_cpp_program(program,model=lowered)
    assert program._serialize(include_provenance=False)['version']==23
    for k,index in enumerate(canonical):
        assert f'old_moments[{k}] = u0A(index, {names.index(binding[index])});' in cpp
    assert 'pops._pops' not in sys.modules
    assert Path(pops.__file__).resolve()==ROOT/'python/pops/__init__.py'


def test_real_header_order_five_overflow_refuses_without_publication(tmp_path):
    source=tmp_path/'overflow.cpp'; binary=tmp_path/'overflow'
    source.write_text('''
#include <pops/numerics/moments/affine_velocity.hpp>
#include <limits>
#include <iostream>
int main(){using B=pops::moments::CartesianMomentBasis<5>; using R=pops::Real;
R old[B::size]{}, endpoint[B::size]{}, out[B::size];old[0]=endpoint[0]=1;
for(auto& v:out)v=7; endpoint[B::index(1,0)]=std::numeric_limits<R>::max()/2;
if(pops::moments::affine_velocity_push_forward<5>(old,endpoint,R(0),R(0),out))return 1;
for(auto v:out)if(v!=7)return 2;
static_assert(pops::moments::CartesianMomentBasis<65534>::cardinality==2147450880LL);
std::cout<<"overflow refused without publication\\n";}
''')
    subprocess.run(['c++','-std=c++20','-Wall','-Wextra','-Werror','-I'+str(ROOT/'include'),str(source),'-o',str(binary)],check=True,capture_output=True,text=True)
    assert subprocess.run([str(binary)],check=True,capture_output=True,text=True).stdout=='overflow refused without publication\n'


def test_real_header_cardinality_exceeds_int_refused(tmp_path):
    source=tmp_path/'cardinality.cpp'
    source.write_text('#include <pops/numerics/moments/affine_velocity.hpp>\nstatic_assert(pops::moments::CartesianMomentBasis<65535>::size>0);\n')
    result=subprocess.run(['c++','-std=c++20','-fsyntax-only','-I'+str(ROOT/'include'),str(source)],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'affine moment cardinality exceeds native component index representation' in result.stderr
