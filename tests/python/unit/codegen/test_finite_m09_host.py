"""Host execution of the actual common emitted residual and PoPS provider.

This is neither a native loader test nor MPI/GPU qualification.
"""
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pops
import pytest
from pops.codegen.program_emit_local_product import product_residual_lines
from pops.codegen.program_emit_expressions import checked_expression_dag, checked_pointwise_rows
from pops.time.expressions import encode_expressions
from pops.linalg import FiniteSupport, FiniteLinearMap
from pops.math import where
from pops._ir.expr import Compare
from tests.python.support.finite_m09_case import case_module, oracle


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("c++")
    if compiler is None:
        pytest.skip("host compiler unavailable")
    kokkos = Path(sys.prefix)/"include"
    if not (kokkos/"Kokkos_Array.hpp").exists():
        pytest.skip("Kokkos headers unavailable for real host instantiation")
    source = """#include <pops/numerics/linalg/finite_linear.hpp>
#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>
struct Capture { const double* data; double operator()(int,int c)const{return data[c];} };
"""
    for condensed in (False, True):
        for permuted in (False, True):
            case, _, _, _ = case_module.build_case(oracle.witness(), condensed=condensed, permuted=permuted)
            program = case._time
            token = next(v for v in program._values if v.op == "solve_coupled_implicit")
            variables = {item.id:"input%d" % i for i,item in enumerate(token.inputs)}
            captures = []
            for i,item in enumerate(token.inputs):
                if i < token.attrs["output_count"]:
                    continue
                captures.append("Capture %sA{%s};" % (variables[item.id], "velocity" if len(item.space.components)==8 else "potential"))
            n = 4 if condensed else 12
            body = "\n".join(product_residual_lines(token,variables))
            source += 'extern "C" int solve_%d_%d(const double* velocity,const double* potential,double* out){\n' % (condensed,permuted)
            source += "int index=0;\n"+"\n".join(captures)+"\n"
            source += "auto residual=[&](const pops::Real (&Ueval)[%d],pops::Real (&rout)[%d]){\n%s\n};\n" % (n,n,body)
            source += """pops::PreparedLocalNonlinearControls controls;
controls.absolute_tolerance=1e-13;
const auto problem=pops::prepare_local_nonlinear_problem<%d>(residual,
 pops::FiniteDifferenceLocalJacobian<%d>{},pops::AcceptAllLocalCandidates<%d>{},controls);
pops::Real seed[%d]={};
const auto result=pops::solve_prepared_local_nonlinear(problem,seed);
for(int i=0;i<%d;++i)out[i]=result.value[i];
return static_cast<int>(result.status);}
""" % (n,n,n,n,n)
    for permuted in (False, True):
        case,_,_,_=case_module.build_case(oracle.witness(),condensed=True,permuted=permuted)
        value=next(v for v in case._time._values if v.name=="reconstructed_velocity")
        rows=[["%s[%d]" % ("old_v" if len(item.space.components)==8 else "phi",c)
               for c in range(len(item.space.components))] for item in value.inputs]
        lines,rendered,invalid=checked_pointwise_rows(value,rows)
        source+='extern "C" int reconstruct_%d(const double* old_v,const double* phi,double* out){\n'%permuted
        source+="\n".join(lines)+"\nif (%s) return 1;\n"%invalid
        source+="\n".join("out[%d]=%s;"%(i,x) for i,x in enumerate(rendered))+"\nreturn 0;}\n"
    # Singular elimination pivot, invertible coupled monolithic system.
    from dataclasses import replace
    singular_data=replace(oracle.witness(),a=np.diag([0.]*4+[1.]*4))
    for condensed in (False,True):
        case,_,_,_=case_module.build_case(singular_data,condensed=condensed)
        value=next(v for v in case._time._values if v.op=="solve_coupled_implicit")
        variables={item.id:"input%d"%i for i,item in enumerate(value.inputs)}
        declarations=["Capture %sA{%s};"%(variables[item.id],"velocity" if len(item.space.components)==8 else "potential")
                      for i,item in enumerate(value.inputs) if i>=value.attrs["output_count"]]
        n=4 if condensed else 12
        body="\n".join(product_residual_lines(value,variables))
        source+='extern "C" int singular_%d(const double* velocity,const double* potential,double* out){\n'%condensed
        source+='int index=0;\n'+"\n".join(declarations)+"\n"
        source+='auto residual=[&](const pops::Real (&Ueval)[%d],pops::Real (&rout)[%d]){\n%s\n};\n'%(n,n,body)
        source+='pops::PreparedLocalNonlinearControls controls; controls.absolute_tolerance=1e-13;\n'
        source+='const auto problem=pops::prepare_local_nonlinear_problem<%d>(residual,pops::FiniteDifferenceLocalJacobian<%d>{},pops::AcceptAllLocalCandidates<%d>{},controls);\n'%(n,n,n)
        source+='pops::Real seed[%d]={}; const auto result=pops::solve_prepared_local_nonlinear(problem,seed);\n'%n
        source+='if(result.status==pops::LocalNonlinearStatus::kConverged) for(int i=0;i<%d;++i)out[i]=result.value[i];\n'%n
        source+='return static_cast<int>(result.status); }\n'
    support=FiniteSupport("pivot", ("a", "b"))
    singular=FiniteLinearMap(support,support,((0.,0.),(0.,0.))).solve(support.bind((1.,2.)))
    for active in (False, True):
        expr=where(Compare("gt", 1., 0.) if active else Compare("lt", 1., 0.), lambda: singular[0], lambda: 7.)
        roots,nodes,inputs=encode_expressions((expr,), pops.Program("lazy"))
        assert not inputs
        lines,rendered,invalid=checked_expression_dag(roots,nodes,[])
        source+='extern "C" double lazy_%d(){\n%s\nreturn (%s)?-99.:%s;\n}\n' % (active,"\n".join(lines),invalid,rendered[0])
    folder=tmp_path_factory.mktemp("finite_m09_host")
    cpp,library=folder/"probe.cpp",folder/"probe.so"
    cpp.write_text(source)
    command=[compiler,"-std=c++20","-O2","-fno-fast-math","-shared","-fPIC",
             "-I"+str(Path(__file__).resolve().parents[4]/"include"),"-I"+str(kokkos),str(cpp),"-o",str(library)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    return ctypes.CDLL(str(library))


@pytest.mark.parametrize("condensed",(False,True))
@pytest.mark.parametrize("permuted",(False,True))
def test_real_original_residual_and_native_condensation(host,condensed,permuted):
    data=oracle.witness()
    vo=tuple(reversed(range(8))) if permuted else tuple(range(8))
    po=(2,0,3,1) if permuted else tuple(range(4))
    v,p=data.velocity_old[list(vo)],data.potential_old[list(po)]
    output=(ctypes.c_double*(4 if condensed else 12))()
    pointer=ctypes.POINTER(ctypes.c_double)
    solve=getattr(host,"solve_%d_%d"%(condensed,permuted))
    solve.argtypes=[pointer,pointer,pointer]
    assert solve(v.ctypes.data_as(pointer),p.ctypes.data_as(pointer),output)==0
    expected_v,expected_p=oracle.monolithic(data)
    if condensed:
        np.testing.assert_allclose(output,expected_p[list(po)],rtol=0,atol=1e-11)
        reconstruct=getattr(host,"reconstruct_%d"%permuted)
        reconstruct.argtypes=[pointer,pointer,pointer]
        reconstructed=(ctypes.c_double*8)()
        assert reconstruct(v.ctypes.data_as(pointer),output,reconstructed)==0
        actual_v=np.asarray(reconstructed)[np.argsort(vo)]
        actual_p=np.asarray(output)[np.argsort(po)]
        assert oracle.original_residual(data,actual_v,actual_p)<oracle.MAX_ERROR
    else:
        # Named product packing sorts by exact block identity: velocity/potential
        # offsets are read from the authored residual's product declaration below.
        case,_,_,_=case_module.build_case(data,permuted=permuted)
        token=next(v for v in case._time._values if v.op=="solve_coupled_implicit")
        expected=np.concatenate([expected_v[list(vo)] if len(item.space.components)==8 else expected_p[list(po)]
                                 for item in token.inputs[:token.attrs["output_count"]]])
        np.testing.assert_allclose(output,expected,rtol=0,atol=1e-11)


def test_inactive_singular_solve_is_lazy_and_active_solve_is_rejected(host):
    for active,expected in ((False,7.),(True,-99.)):
        fn=getattr(host,"lazy_%d"%active)
        fn.restype=ctypes.c_double
        assert fn()==expected


def test_singular_pivot_refuses_while_full_native_system_solves(host):
    from dataclasses import replace
    data=replace(oracle.witness(),a=np.diag([0.]*4+[1.]*4))
    pointer=ctypes.POINTER(ctypes.c_double)
    for condensed in (False,True):
        function=getattr(host,"singular_%d"%condensed)
        function.argtypes=[pointer,pointer,pointer]
        result=(ctypes.c_double*(4 if condensed else 12))(*([77.]*(4 if condensed else 12)))
        status=function(data.velocity_old.ctypes.data_as(pointer),data.potential_old.ctypes.data_as(pointer),result)
        if condensed:
            assert status!=0
            assert all(x==77. for x in result)
        else:
            assert status==0
            case,_,_,_=case_module.build_case(data)
            token=next(v for v in case._time._values if v.op=="solve_coupled_implicit")
            offset=0
            rows={}
            for item in token.inputs[:token.attrs["output_count"]]:
                width=len(item.space.components)
                rows[width]=np.asarray(result)[offset:offset+width]
                offset+=width
            assert oracle.original_residual(data,rows[8],rows[4])<oracle.MAX_ERROR
