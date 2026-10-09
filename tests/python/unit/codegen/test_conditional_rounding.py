"""C09: one scientific expression algebra with guarded floating evaluation."""
import math
import pytest


def test_where_builds_both_branches_once_and_evaluates_only_selected_branch():
    from pops.math import Var, where
    q = Var("q", "cons")
    calls = []
    def yes():
        calls.append("yes")
        return 1 / q
    def no():
        calls.append("no")
        return 7
    expression = where(q > 0, yes, no)
    assert calls == ["yes", "no"]
    assert expression.eval({"q": 0}) == 7
    assert expression.eval({"q": 2}) == .5
    assert expression.deps() == {"q"}


def test_where_rejects_eager_branch_values():
    from pops.math import Var, where
    q = Var("q", "cons")
    with pytest.raises(TypeError, match="callable"):
        where(q > 0, 1 / q, 7)


def test_rounding_keeps_identity_signed_zero_and_formal_derivative():
    from pops.math import Var, rounded
    from pops._ir.lowering import diff
    from pops._ir.visitors import _key
    q = Var("q", "cons")
    assert _key(rounded(q)) != _key(q)
    assert rounded(q).deps() == {"q"}
    assert math.copysign(1, rounded(-0.).eval({})) == -1
    assert diff(rounded(q * q), q).eval({"q": 3.}) == 6.
    assert (rounded(q + 1.) - q).eval({"q": 1.e16}) == 0.


def test_common_cse_does_not_hoist_invalid_branch_operations():
    from pops.math import Var, where, rounded
    from pops.codegen.cpp_writer import _cse_emit
    q = Var("q", "cons")
    expression = where(q > 0, lambda: rounded(1 / q), lambda: 7)
    lines, values, names = _cse_emit([expression], "double", "", materialize_all=True,
                                    return_names=True)
    source = "\n".join(lines + values)
    assert source.index("?") < source.index(" / ")
    assert "volatile double" in source
    assert any(name.startswith("guard") for name in names)


@pytest.mark.parametrize("flag", ["-ffast-math", "-Ofast", "-fassociative-math",
                                  "-ffinite-math-only", "-fno-signed-zeros",
                                  "-freciprocal-math", "-ffp-contract=fast",
                                  "-ffp-contract=on"])
def test_scientific_profile_refuses_options_that_erase_floating_semantics(monkeypatch, flag):
    from pops.codegen.cache import _dsl_optflags
    monkeypatch.setenv("POPS_DSL_OPTFLAGS", "-O3 " + flag)
    with pytest.raises(ValueError, match="floating|strict"):
        _dsl_optflags()


def test_compiled_common_ir_guards_rounding_and_floating_observations(tmp_path):
    """Compile the actual common emitter's scalar output, independently of the runtime."""
    import ctypes
    import shutil
    import subprocess
    from pops.math import Var, where, rounded, minimum, maximum
    from pops.codegen.cpp_writer import _cse_emit
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    q = Var("q", "cons")
    zero = q - q
    expressions = [where(q > 0, lambda: rounded(1 / zero), lambda: rounded(-0.)),
                   rounded(q + 1.) - q,
                   where(q > 0, lambda: minimum(zero / zero, q), lambda: 7.),
                   where(q > 0, lambda: maximum(zero / zero, q), lambda: 7.),
                   where(q > 0, lambda: 1 / q, lambda: 7.)]
    source = ('#include <cmath>\n#include <cfenv>\n#include <limits>\n'
              'namespace pops { using Real = double; }\n'
              'namespace Kokkos { using std::fmin; using std::fmax; }\n')
    for index, expression in enumerate(expressions):
        lines, values, names = _cse_emit([expression], "double", "", materialize_all=True,
                                       return_names=True)
        finite = " && ".join("std::isfinite(%s)" % name for name in names) or "true"
        source += ('extern "C" int expression_%d(double q, double* out) {\n' % index
                   + "std::feclearexcept(FE_ALL_EXCEPT);\n" + "\n".join(lines)
                   + "\n*out = %s;\nreturn (%s) ? 1 : 0;\n}\n" % (values[0], finite))
    source += 'extern "C" int invalid_flags() { return std::fetestexcept(FE_INVALID | FE_DIVBYZERO); }\n'
    path, library = tmp_path / "expressions.cpp", tmp_path / "expressions.so"
    path.write_text(source)
    result = subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-O3",
                             "-fno-fast-math", "-ffp-contract=off", str(path), "-o", str(library)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compiled = ctypes.CDLL(str(library))
    for index in range(len(expressions)):
        function = getattr(compiled, "expression_%d" % index)
        function.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        function.restype = ctypes.c_int
    output = ctypes.c_double()
    assert compiled.expression_0(0., ctypes.byref(output)) == 1
    assert math.copysign(1., output.value) == -1.
    assert compiled.invalid_flags() == 0
    assert compiled.expression_0(1., ctypes.byref(output)) == 0
    assert compiled.expression_1(1.e16, ctypes.byref(output)) == 1
    assert output.value == 0.
    for index in (2, 3):
        function = getattr(compiled, "expression_%d" % index)
        assert function(0., ctypes.byref(output)) == 1 and output.value == 7.
        assert compiled.invalid_flags() == 0
        assert function(1., ctypes.byref(output)) == 0
    assert compiled.expression_4(1.e-250, ctypes.byref(output)) == 1
    assert output.value == pytest.approx(1.e250)
    assert compiled.expression_4(1.e308, ctypes.byref(output)) == 1
    assert output.value > 0.


def test_physical_expression_emitter_propagates_selected_invalid_intermediates():
    from pops.math import Var, where, minimum
    from pops.codegen.cpp_writer import _cse_emit
    q = Var("q", "cons")
    zero = q - q
    expression = where(q > 0, lambda: minimum(zero / zero, q), lambda: 7.)
    lines, (value,) = _cse_emit([expression], "double", "")
    assert "std::isfinite(guard" in value
    assert "quiet_NaN" in value
    source = "\n".join(lines)
    assert source.index("?") < source.index(" / ")
