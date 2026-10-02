"""SOURCE_ONLY genuine public fixture preparation; no compile/bind/run."""
import sys
from pathlib import Path
import numpy as np
import pytest
import pops
from tests.python.integration.runtime import test_program_affine_moment_explicit_basis_runtime as fixture

@pytest.mark.parametrize('mode',('success','nonfinite','overflow'))
def test_authentic_public_fixture_is_resolvable_without_native(mode):
    assert Path(pops.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2]/'python')
    assert 'pops._pops' not in sys.modules
    case,layout=fixture.build(mode)
    resolved=pops.resolve(pops.validate(case),layout=layout)
    assert resolved.initial_condition_plan.bindings
    values=fixture.initial(mode)
    assert values.shape==(21,4,4) and np.isfinite(values).all()
    assert fixture.BASIS.order==5 and fixture.BASIS.indices!=fixture.INDICES
    assert set(fixture.BINDING.values())==set(fixture.NAMES)


def test_atomic_oracle_exact_density_and_first_moments():
    before,after=fixture.atom_moments(),fixture.atom_moments(1)
    positions={index:fixture.NAMES.index(name) for index,name in fixture.BINDING.items()}
    assert before[positions[(0,0)]]==after[positions[(0,0)]]==2
    x,y=positions[(1,0)],positions[(0,1)]
    np.testing.assert_allclose(after[[x,y]],((4*before[x]+3*before[y])/5,(-3*before[x]+4*before[y])/5),rtol=0,atol=2e-16)


def test_overflow_seed_actual_header_refuses_before_output(tmp_path):
    import subprocess
    root=Path(__file__).resolve().parents[2]
    slots={index:fixture.NAMES.index(name) for index,name in fixture.BINDING.items()}
    values=fixture.initial('overflow')[:,0,0]
    ordered=[float(values[slots[index]]) for index in fixture.INDICES]
    cpp=tmp_path/'overflow.cpp'; executable=tmp_path/'overflow'
    literals=','.join(value.hex() for value in ordered)
    cpp.write_text('#include <pops/numerics/moments/affine_velocity.hpp>\n'
        'int main(){double old[21]={'+literals+'};double mean[21],out[21];'
        'for(int i=0;i<21;++i){mean[i]=old[i];out[i]=7;}'
        'mean[1]=.8*old[1]+.6*old[6];mean[6]=-.6*old[1]+.8*old[6];'
        'if(pops::moments::affine_velocity_push_forward<5>(old,mean,16./3.,.0625,out))return 1;'
        'for(double v:out)if(v!=7)return 2;return 0;}')
    subprocess.run(['c++','-std=c++20','-I'+str(root/'include'),str(cpp),'-o',str(executable)],check=True,capture_output=True,text=True)
    subprocess.run([str(executable)],check=True,capture_output=True,text=True)
