"""Independent Source/host contract probes; no native extension/JIT."""
import pathlib
import subprocess
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[2]

def test_actual_header_refuses_computed_threshold_overflow(tmp_path):
    src=tmp_path/'overflow.cpp'
    src.write_text(r'''#include <pops/numerics/elliptic/interface/field_nonlinear.hpp>
#include <limits>
#include <cassert>
int main(){
 pops::FieldNewtonOptions o;
 o.convergence={pops::FieldNewtonConvergenceKind::kRelative,std::numeric_limits<pops::Real>::max(),0};
 bool refused=false;
 try{(void)pops::field_newton_stop_tolerance(o,2);}catch(const std::invalid_argument&){refused=true;}
 assert(refused && "finite coefficients must not publish an infinite stopping threshold");
}
''')
    exe=tmp_path/'overflow'
    subprocess.run(['c++','-std=c++20','-I'+str(ROOT/'include'),str(src),'-o',str(exe)],check=True,capture_output=True)
    subprocess.run([str(exe)],check=True,capture_output=True)

def test_install_identity_preflight_precedes_native_mutation():
    from pops.solvers import Newton
    from pops.solvers.tolerances import Relative,AbsoluteFloor
    p=Newton(tolerance=Relative(1e-10,floor=AbsoluteFloor(1e-11))).lower_field_nonlinear(target='amr_system',layout=None)
    class R:
        calls=0
        def set_field_newton_convergence_plan(self,*args):self.calls+=1
    runtime=R()
    p.convergence['absolute']['value']=(2e-11).hex()
    with pytest.raises(ValueError):p.install(runtime,'existing-slot')
    assert runtime.calls==0

def test_missing_typed_capability_never_calls_legacy_setter():
    from pops.solvers import Newton
    from pops.solvers.tolerances import Relative
    p=Newton(tolerance=Relative(1e-10)).lower_field_nonlinear(target='system',layout=None)
    class LegacyOnly:
        calls=0
        def set_field_newton_plan(self,*args):self.calls+=1
    runtime=LegacyOnly()
    with pytest.raises(TypeError,match='install protocol'):p.install(runtime,'existing-slot')
    assert runtime.calls==0
