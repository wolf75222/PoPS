"""Independent T3 probe: division-by-zero masked by fmin or lazy where."""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
import re
import subprocess

import pops
import pytest
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.program_codegen import emit_cpp_program
from pops.frames import Cartesian2D
from pops.math import minimum, where
from pops.solvers.nonlinear import LocalNewton
from pops.time import FailRun, LocalResidual


def _source(kind):
    model = pops.Model("division_domain", frame=Cartesian2D())
    state = model.state("state", components=("quantity",))
    u = state[0]
    invalid = 1. / (u-u)
    if kind == "masked":
        body = minimum(invalid, 0*u)
    elif kind == "inactive":
        body = where(u > 0, lambda: 0*u, lambda: invalid)
    elif kind == "active":
        body = where(u > 0, lambda: minimum(invalid, 0*u), lambda: 0*u)
    else:
        raise AssertionError(kind)
    model.source("named", on=state, value=(body,))
    source = model.module.operator_handle("named")
    case = pops.Case("division_case")
    block = case.block("material", model)
    program = pops.Program("local_division")
    temporal = program.state(block[state])
    seed = program.value("seed", temporal.n, at=temporal.next.point)

    def residual(p, unknown, frozen):
        rate = p.source(source, unknown)
        return p.value("original", unknown-frozen-p.dt*rate, at=unknown.point)

    candidate = program.solve(LocalResidual(residual, seed,
                                            captures={"frozen": temporal.n}),
                              solver=LocalNewton(tolerance=1.e-12)).consume(action=FailRun())
    program.commit(temporal.next, candidate)
    assert program.validate()
    lowered, _ = lower_and_validate(model, facade=model)
    return emit_cpp_program(program, model=lowered)


def _lambda(source):
    start = source.index("auto residual_eval = ")
    opening = source.index("{", start)
    depth, end = 1, opening+1
    while depth:
        depth += (source[end] == "{")-(source[end] == "}")
        end += 1
    return source[start:end+1]


def test_public_source_division_keeps_evaluated_domain_and_lazy_branch(tmp_path):
    sources = {kind: _source(kind) for kind in ("masked", "inactive", "active")}
    for source in sources.values():
        assert "quiet_NaN" in source and "Kokkos::isfinite" in source
        assert not re.search(r"Kokkos::fmin\(\s*\(?\s*pops::Real\(1", source)
    code = ("#include <cmath>\n#include <limits>\n#include <cfenv>\n"
            "#pragma STDC FENV_ACCESS ON\n"
            "#include <pops/numerics/nonlinear/prepared_local_nonlinear.hpp>\n"
            "namespace Kokkos { using std::fmin; using std::isfinite; }\n")
    for kind, source in sources.items():
        code += (f'extern "C" double residual_{kind}(double unknown) {{\n'
                 'double Ueval[1]={unknown},Gval[1]={unknown},Cval0[1]={unknown},rout[1],dt=.1;\n'
                 + _lambda(source) + '\nstd::feclearexcept(FE_ALL_EXCEPT);'
                 'residual_eval(Ueval,rout);return rout[0];}\n')
        code += (f'extern "C" int flags_{kind}() {{'
                 'return std::fetestexcept(FE_INVALID|FE_DIVBYZERO);}\n')
    source_file, library = tmp_path / "division.cpp", tmp_path / "division.so"
    source_file.write_text(code)
    result = subprocess.run(["c++", "-std=c++20", "-shared", "-fPIC", "-O3",
                             "-fno-fast-math", "-ffp-contract=off", "-I",
                             str(Path(__file__).resolve().parents[4]/"include"),
                             str(source_file), "-o", str(library)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compiled = ctypes.CDLL(str(library))
    for kind in sources:
        evaluation = getattr(compiled, "residual_"+kind)
        evaluation.argtypes = [ctypes.c_double]
        evaluation.restype = ctypes.c_double
        flags = getattr(compiled, "flags_"+kind)
        flags.restype = ctypes.c_int
        actual = evaluation(1.)
        if kind == "inactive":
            assert actual == 0.
            assert flags() == 0
        else:
            assert math.isnan(actual)
