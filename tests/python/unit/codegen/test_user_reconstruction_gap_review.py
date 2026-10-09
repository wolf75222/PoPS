"""Independent C09 observations in the actual emitted User policy.

This small C++ harness tests the emitted scalar evaluator, not the PoPS runtime.
The sample counter makes branch-local reads observable independently of values.
"""
import ctypes
import math
import shutil
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def compiled_user_policies(tmp_path_factory):
    from pops.codegen.user_reconstruction_lowering import emit_user_reconstruction_policy
    from pops.math import minimum, maximum, where
    from pops.numerics import reconstruction
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")
    bodies = (
        lambda sample: minimum(sample(0), sample(1)),
        lambda sample: maximum(sample(0), sample(1)),
        lambda sample: where(sample(0) > 0, lambda: sample(1), lambda: sample(2)),
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
