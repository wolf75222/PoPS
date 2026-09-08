"""Scalar Boolean C++ predicates preserve their selected evaluation regions."""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess

import pytest

from pops._ir.expr import Sqrt, Var
from pops.codegen.cpp_writer import _cse_emit


def test_boolean_predicates_use_logical_operators_and_keep_guarded_cse_scoped():
    q = Var("q", "cons")
    root = ((q > -.5) & (q < 1.5)) | ~(q > 4)
    lines, roots = _cse_emit([root], "double", "  ", materialize_all=True)
    source = "\n".join(lines + roots)
    assert " && " in source and " || " in source and "(!" in source
    assert " & " not in source and " | " not in source
    guarded = (q > 0) & ((Sqrt(q) + Sqrt(q)) > 0)
    lines, _roots, names = _cse_emit([guarded], "double", "  ",
                                    materialize_all=True, return_names=True)
    source = "\n".join(lines)
    assert source.index("&&") < source.index("std::sqrt")
    assert source.count("std::sqrt") == 1
    assert any(name.startswith("guard") for name in names)


@pytest.mark.compiler
@pytest.mark.native_loader
def test_compiled_scalar_guards_skip_invalid_rhs_and_report_selected_native_failure(tmp_path):
    from pops.codegen.toolchain import pops_loader_build_flags, _probe_cxx_std, loader_cxx_std
    from pops.model import FieldSpace, Module, Signature
    from pops.native_calls import NativeFunction
    from pops.native_components import (PreparedNativeComponent, compiler_include_roots,
                                        verify_prepared_native_dependencies)
    include = Path(__file__).resolve().parents[4] / "include"
    library = tmp_path / "library"
    library.mkdir()
    (library / "guard.hpp").write_text(r'''#pragma once
#include <pops/core/model/native_call.hpp>
namespace guard_test {
static int calls = 0;
inline pops::NativeCallResult<1> evaluate(double q) {
  ++calls;
  if (q <= 0) return pops::NativeCallResult<1>::rejected(73);
  pops::NativeCallResult<1> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values[0] = q;
  return result;
}
}
''')
    component = PreparedNativeComponent.header_only("guard_test", include_root=library,
                                                   entry_headers=("guard.hpp",))
    module = Module("guard_test")
    state = module.state_space("state", ("q",))
    function = NativeFunction(component, "guard_test::evaluate",
        Signature((state,), FieldSpace("answer", components=("value",))))
    q = Var("q", "cons")
    call = function((q,))
    output = call[function.output_entries[0][0]][0]
    predicates = [((q > -.5) & (q < 1.5)) | (q > 4),
                  (q > 0) & ((Sqrt(q) + Sqrt(q)) > 0)]
    pure_lines, pure_values, pure_names = _cse_emit(predicates, "pops::Real", "  ",
        materialize_all=True, return_names=True)
    native_lines, native_values, statuses = _cse_emit([(q > 0) & (output > 0),
        (q > 0) | (output > 0)], "pops::Real", "  ", return_native_statuses=True)
    assert len(statuses) == 2
    source = '#include <guard.hpp>\nextern "C" int pure(double q, double* output) {\n'
    source += "\n".join(pure_lines)
    source += "\n" + "\n".join("output[%d] = %s;" % (i, value) for i, value in enumerate(pure_values))
    source += "\nreturn " + " && ".join("Kokkos::isfinite(%s)" % name for name in pure_names) + ";\n}\n"
    source += 'extern "C" int native(double q, double* output) {\n' + "\n".join(native_lines)
    source += "\n" + "\n".join("output[%d] = %s;" % (i, value) for i, value in enumerate(native_values))
    source += "\nreturn " + " | ".join("static_cast<int>(%s.status)" % name for name in statuses) + ";\n}\n"
    source += 'extern "C" int count() { return guard_test::calls; }\n'
    source_path, binary, depfile = tmp_path / "guard.cpp", tmp_path / "guard.so", tmp_path / "guard.d"
    source_path.write_text(source)
    staged = component.stage_verified(tmp_path / "staged")
    compiler, cflags, lflags = pops_loader_build_flags()
    standard = _probe_cxx_std(compiler, loader_cxx_std())
    built = subprocess.run([compiler, "-std=" + standard, "-shared", "-fPIC", "-O2", *cflags,
        "-I", str(include), "-I", staged, "-MMD", "-MF", str(depfile), str(source_path),
        "-o", str(binary), *lflags], capture_output=True, text=True, check=False)
    assert built.returncode == 0, built.stderr
    verify_prepared_native_dependencies(depfile, generated_source=source_path,
        pops_include_root=include, staged_components=((component, staged),),
        toolchain_include_roots=compiler_include_roots(cflags))
    loaded = ctypes.CDLL(str(binary))
    for name in ("pure", "native"):
        getattr(loaded, name).argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        getattr(loaded, name).restype = ctypes.c_int
    values = (ctypes.c_double * 2)()
    for q_value, expected in ((-1, (0, 0)), (.5, (1, 1)), (2, (0, 1)), (5, (1, 1))):
        assert loaded.pure(q_value, values) == 1
        assert tuple(values) == expected
    assert loaded.native(1, values) == 0
    assert tuple(values) == (1, 1) and loaded.count() == 1
    assert loaded.native(-1, values) == 2
    assert tuple(values) == (0, 0) and loaded.count() == 2
