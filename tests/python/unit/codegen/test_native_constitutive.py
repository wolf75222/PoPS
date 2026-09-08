"""A joint imported constitutive law at one consumer-selected endpoint."""
import pytest
from pops._ir.expr import Var
from pops._ir.lowering import diff
from pops.codegen.native_constitutive import emit_native_constitutive
from pops.model import FieldSpace, Module, Signature
from pops.model.bundles import ProductSpace
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain
from pops.native_components import PreparedNativeComponent


def constitutive_function(directory):
    directory.mkdir()
    (directory / "constitutive.hpp").write_text(r'''#pragma once
#include <pops/core/model/native_call.hpp>
#include <atomic>
namespace constitutive {
static inline std::atomic<long> calls{0}, jacobian_calls{0};
inline pops::NativeCallResult<3> evaluate(double q) {
  ++calls;
  pops::NativeCallResult<3> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values = {q+q*q, 1+q*q, 2+q*q};
  return result;
}
inline pops::NativeCallResult<3> jacobian(double q) {
  ++jacobian_calls;
  pops::NativeCallResult<3> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values = {1+2*q, 2*q, 2*q};
  return result;
}
}
''')
    component = PreparedNativeComponent.header_only("constitutive.joint", include_root=directory,
        entry_headers=("constitutive.hpp",))
    module = Module("constitutive")
    state = module.state_space("scalar", ("q",))
    output = ProductSpace({"transform": FieldSpace("W", components=("value",)),
                           "diffusivity": FieldSpace("A", components=("x", "y"))})
    function = NativeFunction(component, "constitutive::evaluate", Signature((state,), output),
        derivatives=(NativeDerivative("exact", "constitutive::jacobian"),),
        domains=(NativeInputDomain(0, 0, lower=0, lower_open=True),),
        effects=("fallible", "diagnostic_counter"))
    return function


def test_endpoint_joint_W_and_diagonal_A_preserve_exact_derivative_and_status(tmp_path):
    function = constitutive_function(tmp_path / "external")
    q = Var("q", "cons")
    call = function((q,), occurrence="constitutive-endpoint")
    expressions = (*call.transform, *call.diffusivity, diff(call.transform[0], q))
    emitted = emit_native_constitutive(expressions)
    text = "\n".join(emitted.lines)
    assert text.count("constitutive::evaluate(") == 1
    assert text.count("constitutive::jacobian(") == 1
    assert len(emitted.values) == 4
    assert len(emitted.functions) == 2
    assert "constitutive_status_" == emitted.status
    assert "constitutive_reason_" == emitted.reason
    assert "selected_ > constitutive_status_" in text
    assert ".reason > constitutive_reason_" in text
    assert all(".read(" in value for value in emitted.values)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_native_constitutive_primal_derivative_and_domain_status_execute(tmp_path):
    import ctypes
    from pathlib import Path
    import subprocess
    from pops.codegen.toolchain import pops_loader_build_flags, _probe_cxx_std, loader_cxx_std
    from pops.native_components import compiler_include_roots, verify_prepared_native_dependencies
    function = constitutive_function(tmp_path / "external")
    q = Var("q", "cons")
    call = function((q,), occurrence="endpoint")
    emitted = emit_native_constitutive((*call.transform, *call.diffusivity, diff(call.transform[0], q)))
    source = '#include <constitutive.hpp>\nextern "C" int evaluate(double q, double* output, unsigned* reason) {\n'
    source += "\n".join(emitted.lines)
    source += "\n*reason = %s; if (%s != 0) return %s;\n" % (emitted.reason, emitted.status, emitted.status)
    source += "\n".join("output[%d] = %s;" % (i, value) for i, value in enumerate(emitted.values))
    source += '\nreturn 0;\n}\nextern "C" long count() { return constitutive::calls.load(); }\n'
    source += 'extern "C" long jacobians() { return constitutive::jacobian_calls.load(); }\n'
    source_path, binary, depfile = tmp_path / "law.cpp", tmp_path / "law.so", tmp_path / "law.d"
    source_path.write_text(source)
    include = Path(__file__).resolve().parents[4] / "include"
    staged = function.component.stage_verified(tmp_path / "stage")
    compiler, cflags, lflags = pops_loader_build_flags()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    built = subprocess.run([compiler, "-std="+standard, "-shared", "-fPIC", "-O2", *cflags,
        "-I", str(include), "-I", staged, "-MMD", "-MF", str(depfile), str(source_path),
        "-o", str(binary), *lflags], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    verify_prepared_native_dependencies(depfile, generated_source=source_path,
        pops_include_root=include, staged_components=((function.component, staged),),
        toolchain_include_roots=compiler_include_roots(cflags))
    native = ctypes.CDLL(str(binary))
    native.evaluate.argtypes = (ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint))
    native.evaluate.restype = ctypes.c_int
    native.count.restype = native.jacobians.restype = ctypes.c_long
    output, reason = (ctypes.c_double*4)(99,99,99,99), ctypes.c_uint(99)
    assert native.evaluate(2, output, ctypes.byref(reason)) == 0 and reason.value == 0
    assert tuple(output) == (6,5,6,5) and native.count() == native.jacobians() == 1
    before = tuple(output)
    assert native.evaluate(-1, output, ctypes.byref(reason)) == 2 and reason.value == 1
    assert tuple(output) == before and native.count() == native.jacobians() == 1
