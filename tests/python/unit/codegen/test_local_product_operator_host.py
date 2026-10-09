"""Real emitted original equations and prepared Newton; host, not MPI acceptance."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import pops
from pops.codegen.program_models import ProgramModelGraph
from pops.codegen.program_emit_kernels import ProgramProviderPlans
from pops.codegen.program_emit_local_product import (
    product_operator_environments, product_residual_lines,
)
from tests.python.support.local_product_operator_case import make_case


@pytest.fixture(scope="module")
def compiled_equations(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    text = """#include <cmath>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
namespace Kokkos { using std::isfinite; }
struct Capture { const double* values; double operator()(int,int c)const{return values[c];} };
struct Params { double value; double get(int)const{return value;} };
"""
    routes = []
    for reverse in (False, True):
        case, layout, _ = make_case((1,2,4), reverse=reverse, derived=True)
        plan = pops.resolve(pops.validate(case), layout=layout)
        graph = ProgramModelGraph.from_resolved_blocks(plan.blocks)
        value = next(v for v in plan.time._values if v.op == "solve_coupled_implicit")
        variables = {v.id:"arg%d" % i for i,v in enumerate(value.inputs)}
        indices = plan.time._block_indices()
        environment, _, _ = product_operator_environments(
            value, graph, ProgramProviderPlans(), indices, variables)
        body = "\n".join(product_residual_lines(value, variables, environment))
        text += 'extern "C" int solve%d(const double* gains,const double* old,const double* seed,' % reverse
        text += 'double* out,double* residual,int evaluate_only){int index=0;\n'
        offset = 0
        for i, width in enumerate((1,2,4)):
            text += "Capture arg%dA{old+%d};\n" % (3+i, offset)
            offset += width
        for node in value.attrs["residual_block"]:
            if node.op in ("source", "apply"):
                text += "const int product_providers_%d=0;\n" % node.id
                text += "const Params product_params_%d{gains[%d]};\n" % (node.id, indices[node.block])
        text += "auto residual_eval=[&](const pops::Real (&Ueval)[7],pops::Real (&rout)[7]){\n"
        text += body + "\n};\n"
        text += """pops::Real initial[7];for(int i=0;i<7;++i)initial[i]=seed[i];
if(evaluate_only){pops::Real r[7];residual_eval(initial,r);
 for(int i=0;i<7;++i)residual[i]=r[i];return 0;}
pops::PreparedLocalNonlinearControls controls; controls.absolute_tolerance=1e-11;
const auto prepared=pops::prepare_local_nonlinear_problem<7>(residual_eval,
 pops::FiniteDifferenceLocalJacobian<7>{},pops::AcceptAllLocalCandidates<7>{},controls);
const auto result=pops::solve_prepared_local_nonlinear(prepared,initial);
pops::Real r[7];residual_eval(result.value,r);
for(int i=0;i<7;++i){out[i]=result.value[i];residual[i]=r[i];}
return static_cast<int>(result.status);}
"""
        routes.append(tuple(indices[v.block] for v in value.inputs[:3]))
    folder = tmp_path_factory.mktemp("local_product_operators")
    cpp, lib = folder/"operators.cpp", folder/"operators.so"
    cpp.write_text(text)
    subprocess.run([compiler,"-std=c++20","-O2","-fno-fast-math","-shared","-fPIC",
        "-I"+str(Path(__file__).resolve().parents[4]/"include"),str(cpp),"-o",str(lib)],check=True)
    return ctypes.CDLL(str(lib)), routes


def call(compiled, reverse, gain, old, seed, *, evaluate=False):
    library, routes = compiled
    reordered = np.empty(3)
    for i, slot in enumerate(routes[reverse]):
        reordered[slot] = gain[i]
    array = lambda values: (ctypes.c_double*len(values))(*values)
    out, residual = array(np.zeros(7)), array(np.zeros(7))
    function = getattr(library,"solve%d" % reverse)
    function.argtypes = [ctypes.POINTER(ctypes.c_double)]*5 + [ctypes.c_int]
    function.restype = ctypes.c_int
    status = function(array(reordered),array(old),array(seed),out,residual,int(evaluate))
    return status, np.array(out), np.array(residual)


def original_residual(x, old, gain):
    return 4*np.repeat(gain,(1,2,4))*x*x + np.array([1,1,2,1,2,3,4])*x + .1*x.sum()-old


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize("gain", ((.5,.75,1.), (1.25,.2,.8)))
def test_seven_local_operator_unknowns_use_current_iterates_and_qualified_params(compiled_equations, reverse, gain):
    target = np.linspace(.2,1.4,7)
    old = original_residual(target,np.zeros(7),gain)
    for seed in (np.full(7,.1), np.full(7,2.)):
        status, solution, residual = call(compiled_equations,reverse,gain,old,seed)
        assert status == 0
        np.testing.assert_allclose(solution,target,rtol=0,atol=2e-11)
        np.testing.assert_allclose(residual,original_residual(solution,old,gain),rtol=0,atol=3e-15)
        assert np.max(np.abs(residual)) <= 1e-11


@pytest.mark.parametrize("reverse", (False, True))
def test_original_equation_is_not_frozen_at_the_seed_or_an_operator_result(compiled_equations, reverse):
    old, gain = np.arange(1.,8.), (.5,.75,1.)
    for candidate in (np.ones(7), np.arange(1.,8.)/3):
        _, _, residual = call(compiled_equations,reverse,gain,old,candidate,evaluate=True)
        np.testing.assert_allclose(residual,original_residual(candidate,old,gain),rtol=0,atol=3e-15)
    bad = np.ones(7); bad[5] = np.nan
    status, _, residual = call(compiled_equations,reverse,gain,old,bad)
    assert status != 0
    assert np.isnan(residual).all()
