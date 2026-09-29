"""Execute the public product's emitted residual with the real native provider.

This host probe does not qualify MultiFab, MPI, device execution or publication.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess

import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.math import minimum, sqrt, where
from test_local_residual_product import product_case


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler unavailable")
    code = """#include <cmath>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
namespace Kokkos { using std::isfinite; using std::sqrt; using std::fmin; }
struct Capture { const double* data; double operator()(int,int c)const{return data[c];} };
"""
    for mode in ("normal", "nonlinear", "invalid", "inactive", "unsolvable"):
        def residual(P, z, *, old_a, old_b, mode=mode):
            tail = z["b"][2] - old_b[2]
            if mode == "invalid":
                tail = minimum(sqrt(-old_b[2]), 0) + tail
            elif mode == "inactive":
                tail = where(old_b[2] > 0, lambda: tail, lambda: sqrt(-old_b[2]))
            elif mode == "unsolvable":
                tail = 1
            first = (z["a"][0]*z["b"][0] if mode == "nonlinear"
                     else z["a"][0]+z["b"][0])
            return {"a": (z["b"][0]-old_a[0], z["b"][1]-old_a[1]),
                    "b": (first-old_b[0],
                          z["a"][1]+z["b"][1]-old_b[1], tail)}
        program, _ = product_case(body=residual)
        source = emit_cpp_program(program)
        start = source.index("auto residual_eval =")
        marker = "pops::solve_prepared_local_nonlinear(prepared_, G_);"
        end = source.index(marker, start) + len(marker)
        code += 'extern "C" int '+mode+'(const double* guess,const double* capture,double* out){\n'
        code += "const int index=0; Capture u0A{capture},u1A{capture+2}; pops::Real G_[5];\n"
        code += "for(int i=0;i<5;++i)G_[i]=guess[i];\n" + source[start:end]
        code += "\nfor(int i=0;i<5;++i)out[i]=solved_.value[i]; out[5]=solved_.residual_norm;\n"
        code += "return static_cast<int>(solved_.status);}\n"
    folder = tmp_path_factory.mktemp("local_product_host")
    source, library = folder/"probe.cpp", folder/"probe.so"
    source.write_text(code)
    subprocess.run([compiler,"-std=c++20","-O2","-fno-fast-math","-shared","-fPIC",
                    "-I"+str(Path(__file__).resolve().parents[4]/"include"),
                    str(source),"-o",str(library)],check=True)
    loaded=ctypes.CDLL(str(library))
    def run(mode, guess, capture):
        function=getattr(loaded,mode)
        function.argtypes=[ctypes.POINTER(ctypes.c_double)]*3
        function.restype=ctypes.c_int
        output=(ctypes.c_double*6)()
        status=function((ctypes.c_double*5)(*guess),(ctypes.c_double*5)(*capture),output)
        return status,tuple(output)
    return run


@pytest.mark.parametrize("guess", [(0,0,0,0,0), (10,-4,8,2,-7)])
@pytest.mark.parametrize("capture", [(2,3,7,11,5), (-1,4,2,9,3)])
def test_joint_full_system_singular_diagonal_and_rebind(provider, guess, capture):
    status, output=provider("normal",guess,capture)
    assert status == 0
    a,b,c,d,e=capture
    assert output[:5] == pytest.approx((c-a,d-b,a,b,e),abs=1e-10)
    x,y,p,q,r=output[:5]
    assert max(abs(p-a),abs(q-b),abs(x+p-c),abs(y+q-d),abs(r-e)) < 1e-11


def test_all_original_rows_and_intermediate_domain_are_authoritative(provider):
    assert provider("unsolvable",(0,)*5,(2,3,7,11,5))[0] != 0
    assert provider("invalid",(0,)*5,(2,3,7,11,5))[0] != 0
    assert provider("inactive",(0,)*5,(2,3,7,11,5))[0] == 0


@pytest.mark.parametrize("capture", [(2,3,7,11,5), (-1,4,2,9,3)])
def test_actual_cross_unknown_monomial(provider, capture):
    status,output=provider("nonlinear",(0,0,1,1,1),capture)
    assert status == 0
    a,b,c,d,e=capture
    x,y,p,q,r=output[:5]
    assert (x,y,p,q,r) == pytest.approx((c/a,d-b,a,b,e),abs=1e-10)
    assert max(abs(p-a),abs(q-b),abs(x*p-c),abs(y+q-d),abs(r-e)) < 1e-11
