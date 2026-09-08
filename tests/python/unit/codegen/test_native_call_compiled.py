"""Native primitive execution witness; separate from integrated Case qualification."""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import time

import pytest

from pops._ir.expr import Var
from pops._ir.lowering import diff
from pops.codegen.cpp_writer import _cse_emit
from pops.model import Module, Signature
from pops.model.bundles import ProductSpace
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain
from pops.native_components import (PreparedNativeComponent, compiler_include_roots,
                                    verify_prepared_native_dependencies)


HEADER = r'''#pragma once
#include <pops/core/model/native_call.hpp>
#include <atomic>
namespace imported {
struct Exchange {
  static inline std::atomic<long> calls{0};
  static pops::NativeCallResult<5> evaluate(double p, double e, double q, double f, double m) {
    ++calls;
    const double force = q/m - p;
    const double power = 0.5*(p + q/m)*force;
    pops::NativeCallResult<5> result;
    result.status = pops::EvaluationStatus::kOk;
    result.values = {force, power, -force, -power, 0};
    return result;
  }
  static pops::NativeCallResult<25> jacobian(double p, double e, double q, double f, double m) {
    pops::NativeCallResult<25> result;
    result.status = pops::EvaluationStatus::kOk;
    result.values[0] = -1; result.values[2] = 1/m; result.values[4] = -q/(m*m);
    result.values[5] = -p; result.values[7] = q/(m*m); result.values[9] = -q*q/(m*m*m);
    for (int j=0; j<5; ++j) { result.values[10+j]=-result.values[j]; result.values[15+j]=-result.values[5+j]; }
    return result;
  }
};
}
'''


@pytest.mark.compiler
@pytest.mark.native_loader
def test_imported_native_function_executes_shared_projections_and_exact_derivative(tmp_path, record_property):
    from pops.codegen.toolchain import pops_loader_build_flags, _probe_cxx_std, loader_cxx_std
    root = Path(__file__).resolve().parents[4]
    library = tmp_path / "library"
    library.mkdir()
    (library / "exchange.hpp").write_text(HEADER)
    component = PreparedNativeComponent.header_only("imported.exchange", include_root=library,
                                                   entry_headers=("exchange.hpp",))
    module = Module("external_exchange")
    a = module.state_space("a", ("p", "E"))
    b = module.state_space("b", ("p", "E", "m"))
    function = NativeFunction(component, "imported::Exchange::evaluate",
        Signature((a, b), ProductSpace({"a": a, "b": b})),
        domains=(NativeInputDomain(1, 2, lower=0, lower_open=True),),
        derivatives=(NativeDerivative("exact", "imported::Exchange::jacobian"),))
    p, e, q, f, m = (Var(name, "cons") for name in ("p", "e", "q", "f", "m"))
    call = function((p, e), (q, f, m))
    lines, values, statuses = _cse_emit(
        [call.a[0] + call.a[0], call.a[1], *call.b], "pops::Real", "  ",
        return_native_statuses=True)
    derivative_lines, derivatives, derivative_statuses = _cse_emit(
        [diff(call.a[0], "p"), diff(call.a[1], "p")], "pops::Real", "  ",
        return_native_statuses=True)
    body = "\n".join(lines)
    assert body.count("imported::Exchange::evaluate(") == 1
    source = '#include <exchange.hpp>\nextern "C" int evaluate(double p, double e, double q, double f, double m, double* output) {\n'
    source += body + '\n  if (static_cast<int>(%s.status) != 0) return static_cast<int>(%s.status);\n' % (statuses[0], statuses[0])
    source += "\n".join("  output[%d] = %s;" % (i, value) for i, value in enumerate(values))
    source += '\n  return 0;\n}\nextern "C" long count() { return imported::Exchange::calls.load(); }\n'
    source += 'extern "C" int jacobian(double p, double e, double q, double f, double m, double* output) {\n'
    source += "\n".join(derivative_lines) + '\n  if (static_cast<int>(%s.status) != 0) return 2;\n' % derivative_statuses[0]
    source += "\n".join("  output[%d] = %s;" % (i, value) for i, value in enumerate(derivatives))
    source += "\nreturn 0;\n}\n"
    source_path, binary, depfile = tmp_path / "kernel.cpp", tmp_path / "kernel.so", tmp_path / "kernel.d"
    source_path.write_text(source)
    staged = component.stage_verified(tmp_path / "staged")
    compiler, cflags, lflags = pops_loader_build_flags()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    command = [compiler, "-std=" + standard, "-shared", "-fPIC", "-O2", *cflags,
               "-I", str(root / "include"), "-I", staged,
               "-MMD", "-MF", str(depfile), str(source_path), "-o", str(binary), *lflags]
    start = time.perf_counter()
    built = subprocess.run(command, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    assert built.returncode == 0, built.stderr
    verify_prepared_native_dependencies(depfile, generated_source=source_path,
        pops_include_root=root / "include", staged_components=((component, staged),),
        toolchain_include_roots=compiler_include_roots(cflags))
    loaded = ctypes.CDLL(str(binary))
    for symbol in ("evaluate", "jacobian"):
        getattr(loaded, symbol).argtypes = [ctypes.c_double] * 5 + [ctypes.POINTER(ctypes.c_double)]
        getattr(loaded, symbol).restype = ctypes.c_int
    loaded.count.restype = ctypes.c_long
    output = (ctypes.c_double * 5)(77, 77, 77, 77, 77)
    assert loaded.evaluate(1, 2, -1, 3, 2, output) == 0
    assert tuple(output) == pytest.approx((-3, -0.375, 1.5, 0.375, 0), abs=1e-12)
    assert loaded.count() == 1
    assert loaded.jacobian(1, 2, -1, 3, 2, output) == 0
    assert tuple(output)[:2] == (-1, -1)
    before = tuple(output)
    assert loaded.evaluate(1, 2, -1, 3, -1, output) == 2
    assert tuple(output) == before and loaded.count() == 1
    record_property("native_joint_calls", 1)
    record_property("mathematical_first_projection_uses", 2)
    record_property("native_domain_rejections", 1)
    record_property("compile_seconds", elapsed)
    record_property("source_bytes", len(source.encode()))
    record_property("binary_bytes", binary.stat().st_size)
