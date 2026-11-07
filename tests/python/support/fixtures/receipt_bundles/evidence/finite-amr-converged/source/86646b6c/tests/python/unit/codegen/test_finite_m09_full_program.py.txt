"""Whole emitted Program syntax and output allocation, not extracted cell lambdas."""
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pops
import pytest
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.support.finite_m09_case import case_module, oracle


def emitted(condensed):
    case,layout,_,_=case_module.build_case(oracle.witness(),condensed=condensed)
    plan=pops.resolve(pops.validate(case),layout=layout)
    source=emit_cpp_program(plan.time,model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    return plan,source


@pytest.mark.parametrize("condensed",(False,True))
def test_every_finite_output_and_status_use_the_explicit_template(condensed):
    plan,source=emitted(condensed)
    values=[v for v in plan.time._values if "finite_support_v1" in v.attrs]
    for value in values:
        allocation=re.search(r"& u%d = ctx.scratch_state\(%d, 0, ([^)]+)\);"%(value.id,value.id),source)
        assert allocation
        body=source[allocation.end():]
        end=body.index("if (ctx.pointwise_status_max")
        body=body[:end]
        template=re.search(r"outA\(index, 0\) = (\w+)A\(index, 0\);",body)
        assert template
        assert allocation[1]==template[1]
        assert "ctx.scalar_scratch(%d, 1, %s, 1, 0)"%(value.id,template[1]) in body


@pytest.mark.parametrize("condensed",(False,True))
def test_complete_generated_program_compiles_with_real_kokkos_headers(tmp_path,condensed):
    compiler=shutil.which("clang++") or shutil.which("c++")
    kokkos=Path(sys.prefix)/"include"
    if compiler is None or not (kokkos/"Kokkos_Core.hpp").exists():
        pytest.skip("C++ compiler and actual Kokkos headers required")
    _,source=emitted(condensed)
    cpp=tmp_path/"complete_program.cpp"
    cpp.write_text(source)
    flags=["-std=c++20","-fsyntax-only","-fno-fast-math","-DPOPS_NATIVE_DIM=2",
           "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI","-DPOPS_HAS_KOKKOS","-DKOKKOS_DEPENDENCE",
           "-I"+str(Path(__file__).resolve().parents[4]/"include"),"-I"+str(kokkos)]
    if (kokkos/"mpi.h").exists():
        flags += ["-DPOPS_HAS_MPI"]
    config=(kokkos/"KokkosCore_config.h").read_text()
    if re.search(r"^#define KOKKOS_ENABLE_OPENMP\b",config,re.MULTILINE):
        if sys.platform=="darwin":
            omp=Path("/opt/homebrew/opt/libomp/include")
            if not (omp/"omp.h").exists():
                pytest.skip("installed Kokkos requires OpenMP headers")
            flags += ["-Xpreprocessor","-fopenmp","-I"+str(omp)]
        else:
            flags += ["-fopenmp"]
    result=subprocess.run([compiler,*flags,str(cpp)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
