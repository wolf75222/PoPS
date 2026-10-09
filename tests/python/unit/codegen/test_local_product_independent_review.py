"""Independent 1+2+4 packing/capture probe; host execution is not MPI acceptance."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
from pops import model, time
from pops.codegen.program_emit_local_product import product_components, product_residual_lines
from pops.solvers.nonlinear import LocalNewton
from typed_program_support import typed_state


# Dense nonsymmetric matrix, independent of PoPS's differentiation/codegen.
MATRIX = np.diag(np.arange(5., 12.)) + np.outer(np.arange(1., 8.),
                                                        np.arange(7., 0., -1.)) / 50


def authored_product(reverse=False, bad=False):
    module = model.Module("independent_product")
    program = time.Program("independent_product")
    values = {}
    for key, width in (("a", 1), ("b", 2), ("c", 4)):
        space = module.state_space(key, tuple("v%d" % c for c in range(width)))
        values[key] = typed_state(program, key, space=space, model=module,
                                  state=module.state_handle(space))
    shifted = program.value("separate_capture", (1 + values["a"][0],), at=values["a"].point)
    before = program._ir_hash()

    def body(P, unknowns, *, old, shifted):
        flat = [unknowns[k][c] for k, width in (("a",1),("b",2),("c",4))
                for c in range(width)]
        residual = [sum(float(MATRIX[i,j])*flat[j] for j in range(7))
                    - old[0] - (i+1)*shifted[0] for i in range(7)]
        return {"c": tuple(residual[3:]), "a": tuple(residual[:1]),
                "b": tuple(residual[1:2] if bad else residual[1:3])}

    initial = dict(reversed(tuple(values.items()))) if reverse else values
    captures = {"shifted": shifted, "old": values["a"]}
    if bad:
        with pytest.raises(ValueError, match="width"):
            program.solve(time.LocalResidual(body, initial, captures=captures), solver=LocalNewton())
        assert program._ir_hash() == before
        return
    program.solve(time.LocalResidual(body, initial, captures=captures), solver=LocalNewton())
    return next(v for v in program._values if v.op == "solve_coupled_implicit")


def test_failed_product_authoring_keeps_exact_program_image():
    authored_product(bad=True)


@pytest.fixture(scope="module")
def dense_product(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = """#include <cmath>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
namespace Kokkos { using std::isfinite; }
struct Capture { double value; double operator()(int,int)const{return value;} };
"""
    for reverse in (False, True):
        value = authored_product(reverse)
        assert tuple(map(len, product_components(value).values())) == (1,2,4)
        assert value.inputs[3].block == value.inputs[4].block
        assert value.inputs[3].id != value.inputs[4].id
        variables = {item.id: "input%d" % index for index,item in enumerate(value.inputs)}
        body = "\n".join(product_residual_lines(value, variables))
        source += 'extern "C" int solve%d(double old,double shifted,double* out){\n' % reverse
        source += "int index=0; Capture input3A{old},input4A{shifted};\n"
        source += "auto residual=[&](const pops::Real (&Ueval)[7],pops::Real (&rout)[7]){\n"+body+"\n};\n"
        source += """pops::PreparedLocalNonlinearControls controls;
controls.absolute_tolerance=1e-11;
const auto prepared=pops::prepare_local_nonlinear_problem<7>(residual,
 pops::FiniteDifferenceLocalJacobian<7>{},pops::AcceptAllLocalCandidates<7>{},controls);
pops::Real seed[7]={3,-2,1,-4,2,0,7};
const auto result=pops::solve_prepared_local_nonlinear(prepared,seed);
for(int i=0;i<7;++i)out[i]=result.value[i];
return static_cast<int>(result.status);}
"""
    folder = tmp_path_factory.mktemp("independent_local_product")
    cpp, library = folder/"probe.cpp", folder/"probe.so"
    cpp.write_text(source)
    subprocess.run([compiler,"-std=c++20","-O2","-fno-fast-math","-shared","-fPIC",
                    "-I"+str(Path(__file__).resolve().parents[4]/"include"),
                    str(cpp),"-o",str(library)], check=True)
    return ctypes.CDLL(str(library))


@pytest.mark.parametrize("reverse", [False,True])
@pytest.mark.parametrize("old,shifted", [(2.,7.),(-3.,.25)])
def test_dense_seven_unknowns_preserve_same_block_distinct_captures(dense_product, reverse, old, shifted):
    function = getattr(dense_product,"solve%d" % reverse)
    function.argtypes = [ctypes.c_double,ctypes.c_double,ctypes.POINTER(ctypes.c_double)]
    function.restype = ctypes.c_int
    output = (ctypes.c_double*7)()
    assert function(old,shifted,output) == 0
    rhs = old + np.arange(1.,8.)*shifted
    np.testing.assert_allclose(output,np.linalg.solve(MATRIX,rhs),rtol=0,atol=2e-11)
    assert np.max(np.abs(MATRIX@np.asarray(output)-rhs)) < 1e-11
