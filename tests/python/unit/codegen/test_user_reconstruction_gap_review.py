"""Independent C09 observations in the actual emitted User policy.

This small C++ harness tests the emitted scalar evaluator, not the PoPS runtime.
The sample counter makes branch-local reads observable independently of values.
"""
import ctypes
import math
import re
import shutil
import struct
import subprocess
from fractions import Fraction
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def compiled_user_policies(tmp_path_factory):
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy
    from pops.math import maximum, minimum, sqrt, where
    from pops.numerics import reconstruction
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    bodies = (
        lambda sample: minimum(sample(0), sample(1)),
        lambda sample: maximum(sample(0), sample(1)),
        lambda sample: where(sample(0) > 0, lambda: sample(1), lambda: sample(2)),
        lambda sample: where(minimum(sqrt(-sample(0)), 1.) > 0.,
                             lambda: sample(1), lambda: sample(2)),
        lambda sample: where(minimum(sqrt(-sample(0)), 1.) > 2.,
                             lambda: sample(1), lambda: sample(2)),
    )
    source = ["#include <cmath>", "#include <limits>", "#define POPS_HD",
              "namespace Kokkos { using std::fmin; using std::fmax; }",
              "namespace pops { using Real=double;",
              "template<class T> concept ReconstructionPolicy = requires { T::n_ghost; };",
              "template<class T> constexpr bool stencil_envelope_fits_storage = true; }"]
    for index, body in enumerate(bodies):
        descriptor = reconstruction.User(body, formal_order=1)
        # Only the owning compiler-view carrier is stubbed; the retained body is
        # authored and authenticated by the production User API and emitter.
        emitted = emit_user_reconstruction_policy(SimpleNamespace(_user_reconstruction=descriptor))
        source.append(emitted.replace("namespace pops_generated", "namespace policy_%d" % index))
        source.append('extern "C" double evaluate_%d(const double* values, int* calls) {' % index)
        source.append("  auto sample = [&](int offset) { ++calls[offset]; return values[offset]; };")
        source.append("  return policy_%d::UserReconstructionPolicy{}.stencil_face_value(sample); }" % index)
    directory = tmp_path_factory.mktemp("user_policy_observation")
    cpp, library = directory / "policy.cpp", directory / "policy.so"
    cpp.write_text("\n".join(source))
    result = subprocess.run([compiler, "-std=c++20", "-shared", "-fPIC", "-O3",
                             "-fno-fast-math", "-ffp-contract=off", str(cpp), "-o", str(library)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    compiled = ctypes.CDLL(str(library))
    functions = [getattr(compiled, "evaluate_%d" % index) for index in range(len(bodies))]
    for function in functions:
        function.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]
        function.restype = ctypes.c_double
    return functions


def evaluate(function, values):
    samples = (ctypes.c_double * 3)(*values)
    counts = (ctypes.c_int * 3)()
    return function(samples, counts), tuple(counts)


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -float("inf")),
                         ids=("nan", "positive_inf", "negative_inf"))
@pytest.mark.parametrize("index", (0, 1), ids=("minimum", "maximum"))
def test_nonfinite_active_sample_cannot_be_masked_by_minmax(compiled_user_policies, index, invalid):
    value, counts = evaluate(compiled_user_policies[index], (invalid, 5., 0.))
    assert counts == (1, 1, 0)
    assert math.isnan(value), "active non-finite sample was silently replaced by its finite neighbor"


def test_where_reads_only_samples_in_selected_branch(compiled_user_policies):
    value, counts = evaluate(compiled_user_policies[2], (1., 7., float("nan")))
    assert value == 7.
    assert counts == (1, 1, 0), "the inactive branch sample was evaluated before its guard"


def test_active_nonfinite_conditional_sample_is_rejected(compiled_user_policies):
    value, _ = evaluate(compiled_user_policies[2], (1., float("nan"), 7.))
    assert math.isnan(value)


def test_where_false_branch_does_not_read_inactive_sample(compiled_user_policies):
    value, counts = evaluate(compiled_user_policies[2], (-1., float("nan"), 9.))
    assert value == 9.
    assert counts == (1, 0, 1)


def test_nonfinite_predicate_sample_cannot_choose_a_finite_branch(compiled_user_policies):
    value, _ = evaluate(compiled_user_policies[2], (float("nan"), 7., 9.))
    assert math.isnan(value)


@pytest.mark.parametrize("index,read_counts", [(3, (1, 1, 0)), (4, (1, 0, 1))])
def test_invalid_intermediate_preserves_later_selected_read(compiled_user_policies, index, read_counts):
    value, counts = evaluate(compiled_user_policies[index], (1., 7., 9.))
    assert counts == read_counts
    assert math.isnan(value)


@pytest.mark.parametrize("precision", ("float", "double"))
def test_bounded_product_keeps_the_order_and_bits_of_the_checked_scalar_policy(
        tmp_path, monkeypatch, precision):
    from pops._ir.expr import Const
    from pops.codegen import cpp_writer
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy
    from pops.identity.scalar import scalar_literal
    from pops.math import minimum, where
    from pops.numerics import reconstruction

    compiler = shutil.which("c++")
    assert compiler is not None
    typed = Const(scalar_literal(Fraction(10**50 + 1, 10**51 + 3), target="float"))
    scalar, bits, largest, smallest = (
        (ctypes.c_float, "f", float.fromhex("0x1.fffffep127"), float.fromhex("0x1p-149"))
        if precision == "float" else
        (ctypes.c_double, "d", float.fromhex("0x1.fffffffffffffp1023"), float.fromhex("0x1p-1074"))
    )
    inf = math.inf
    cases = {
        "bounded": (lambda s: minimum(0.5 * s(0), s(1)), 1,
                    [(x, largest, 0.) for x in (0., -0., smallest, 2 * smallest,
                                                 -2 * smallest, largest, -largest,
                                                 inf, -inf, math.nan)]),
        "branches": (lambda s: where(s(0) > 0, lambda: 0.5 * s(1), lambda: s(2) * -0.5), 2,
                     [(1., 8., inf), (-1., inf, 8.), (1., inf, 8.), (-1., 8., inf)]),
        "and_then": (lambda s: where((s(0) > 0) & (0.5 * s(1) > 0), lambda: s(2), lambda: 1.), 1,
                     [(-1., inf, 9.), (1., inf, 9.)]),
        "or_else": (lambda s: where((s(0) > 0) | (0.5 * s(1) > 0), lambda: s(2), lambda: 1.), 1,
                    [(1., inf, 9.), (-1., inf, 9.)]),
        "masked": (lambda s: where(minimum(0.5 * s(0), 1.) > 0., lambda: s(1), lambda: s(2)), 1,
                   [(inf, 8., 9.), (-inf, 8., 9.), (math.nan, 8., 9.)]),
        "sum": (lambda s: minimum(s(0) + 0.5 * s(0), s(1)), 1,
                [(largest, 1., 0.), (-largest, 1., 0.)]),
        "typed": (lambda s: minimum(typed * s(0), s(1)), 0, [(largest, 1., 0.)]),
        "overflow": (lambda s: minimum(2.0 * s(0), s(1)), 0, [(largest, 1., 0.)]),
        "zero": (lambda s: minimum(-0.0 * s(0), s(1)), 0, [(inf, 1., 0.)]),
        "inherited": (lambda s: where(s(0) > 0, lambda: 0.5 * s(0), lambda: 0.), 0,
                      [(inf, 1., 0.)]),
        "chain": (lambda s: minimum(0.5 * (0.5 * s(0)), s(1)), 1, [(inf, 1., 0.)]),
    }
    source = ["#include <cmath>", "#include <limits>", "#define POPS_HD",
              "namespace Kokkos { using std::fmin; }",
              "namespace pops { using Real=" + precision + ";",
              "template<class T> concept ReconstructionPolicy = requires { T::n_ghost; };",
              "template<class T> constexpr bool stencil_envelope_fits_storage = true; }"]
    emit_cse = cpp_writer._cse_emit

    def emit_stock(*args, **kwargs):
        return emit_cse(*args, **dict(kwargs, omit_checked_contraction=False))

    for label, (body, removed, _) in cases.items():
        carrier = SimpleNamespace(_user_reconstruction=reconstruction.User(body, formal_order=1))
        head = emit_user_reconstruction_policy(carrier)
        with monkeypatch.context() as patch:
            patch.setattr(cpp_writer, "_cse_emit", emit_stock)
            stock = emit_user_reconstruction_policy(carrier)
        # The source byte difference is entirely the authorized observation(s).
        observer = re.compile(r"(?:    )?reconstruction_finite &= std::isfinite\(\w+\);[\n ]")
        head_observers = set(observer.findall(head))
        omitted = [match for match in observer.findall(stock) if match not in head_observers]
        assert len(omitted) == removed, label
        expected = stock
        for observation in omitted:
            expected = expected.replace(observation, "", 1)
        assert expected == head, label
        for arm, emitted in (("head", head), ("stock", stock)):
            name = label + "_" + arm
            source.append(emitted.replace("namespace pops_generated", "namespace policy_" + name))
            source.append(
                'extern "C" pops::Real %s(const pops::Real* values, int* count, int* order) {'
                ' auto sample = [&](int offset) { order[(*count)++] = offset; return values[offset]; };'
                " return policy_%s::UserReconstructionPolicy{}.stencil_face_value(sample); }"
                % (name, name))
    cpp, library = tmp_path / "policy.cpp", tmp_path / "policy.so"
    cpp.write_text("\n".join(source))
    run = subprocess.run([compiler, "-std=c++20", "-shared", "-fPIC", "-O3", "-fno-fast-math",
                          "-ffp-contract=off", str(cpp), "-o", str(library)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    dylib = ctypes.CDLL(str(library))
    for label, (_, _, inputs) in cases.items():
        for values in inputs:
            observed = []
            for arm in ("stock", "head"):
                fn = getattr(dylib, label + "_" + arm)
                fn.argtypes = [ctypes.POINTER(scalar), ctypes.POINTER(ctypes.c_int),
                               ctypes.POINTER(ctypes.c_int)]
                fn.restype = scalar
                order, count = (ctypes.c_int * 8)(), ctypes.c_int()
                value = fn((scalar * 3)(*values), ctypes.byref(count), order)
                observed.append((None if math.isnan(value) else struct.pack(bits, value),
                                 tuple(order[:count.value])))
            assert observed[0] == observed[1], (label, values, observed)
            if label in ("typed", "overflow", "masked", "sum"):
                assert observed[1][0] is None
            if label == "masked":
                assert observed[1][1] == (0, 2 if values[0] == -inf else 1)
            if label == "branches":
                assert observed[1][1] == ((0, 1) if values[0] > 0 else (0, 2))
            if label in ("and_then", "or_else"):
                active = (values[0] > 0) == (label == "and_then")
                assert observed[1][1] == ((0, 1, 2) if active else
                                          (0,) if label == "and_then" else (0, 2))
