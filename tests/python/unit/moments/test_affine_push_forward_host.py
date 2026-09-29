"""Common C++ expression emission against particles and the retained native recipe."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from pops.codegen.cpp_writer import _cse_emit
from pops.math import Var, rounded, where
from pops.moments import affine_push_forward


def test_polynomial_host_matches_historical_rotation_and_preserves_lazy_domains(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    indices = tuple((i, j) for j in range(5) for i in range(5 - j))
    raw = tuple(Var(f"m{i}", "cons") for i in range(15))
    c, s, bx, by = (Var(name, "cons") for name in ("c", "s", "bx", "by"))
    expressions = affine_push_forward(raw, indices=indices, matrix=((c, s), (-s, c)), offset=(bx, by))
    lines, names = _cse_emit(expressions, "double", "")
    guard = Var("guard", "cons")
    lazy = affine_push_forward(raw[:3], indices=((0,), (1,), (2,)), matrix=((1.,),),
        offset=(rounded(where(guard > 0., lambda: 1. / (guard - guard), lambda: .2)),))
    lazy_lines, lazy_names = _cse_emit(lazy, "double", "")
    source = "\n".join([
        '#include <pops/numerics/moments/affine_velocity.hpp>',
        '#include <limits>',
        'extern "C" void body(const double* input, double c, double s, double bx, double by, double* output) {',
        *(f"const double m{i} = input[{i}];" for i in range(15)),
        *lines, *(f"output[{i}] = {name};" for i, name in enumerate(names)), '}',
        'extern "C" int legacy(const double* input, double omega, double theta, double* output) {',
        'double old[15], endpoint[15], mapped[15];',
        'for(int i=0;i<15;++i) old[i]=endpoint[i]=input[i];',
        'bool ok=pops::moments::affine_velocity_push_forward<4>(old,endpoint,omega,theta,mapped);',
        'if(ok) for(int i=0;i<15;++i) output[i]=mapped[i]; return ok; }',
        'extern "C" void lazy(const double* input, double guard, double* output) {',
        *(f"const double m{i} = input[{i}];" for i in range(3)),
        *lazy_lines, *(f"output[{i}] = {name};" for i, name in enumerate(lazy_names)), '}',
    ])
    cpp, library = tmp_path / "affine.cpp", tmp_path / "affine.so"
    assert "volatile double rounded_value_" in source
    cpp.write_text(source)
    root = Path(__file__).resolve().parents[4]
    built = subprocess.run([compiler, '-std=c++20', '-O2', '-fno-fast-math', '-ffp-contract=off',
                            '-shared', '-fPIC', '-DPOPS_NATIVE_DIM=2', '-I', str(root / 'include'),
                            str(cpp), '-o', str(library)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    loaded = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    loaded.body.argtypes = [pointer] + [ctypes.c_double] * 4 + [pointer]
    loaded.legacy.argtypes = [pointer, ctypes.c_double, ctypes.c_double, pointer]
    loaded.legacy.restype = ctypes.c_int
    loaded.lazy.argtypes = [pointer, ctypes.c_double, pointer]
    particles = np.array([[-.5, .2], [.7, -.3], [.1, .8], [-.2, -.6]])
    weights = np.array([.2, .5, .7, .3])
    moments = np.array([np.dot(weights, particles[:, 0]**i * particles[:, 1]**j) for i, j in indices])
    mean = np.average(particles, axis=0, weights=weights)
    for omega, theta in ((-.7, .1), (1.e12, .02)):
        # The independent oracle moves each particle. Both test angles remain
        # within the direct formula's finite range; no all-range claim is made.
        a = omega * theta
        c = (1 - a*a)/(1 + a*a)
        s = 2*a/(1 + a*a)
        matrix = np.array([[c, s], [-s, c]])
        shift = mean - matrix @ mean
        moved = particles @ matrix.T + shift
        expected = [np.dot(weights, moved[:, 0]**i * moved[:, 1]**j) for i, j in indices]
        actual, historical = np.empty(15), np.empty(15)
        loaded.body(moments, c, s, *shift, actual)
        assert loaded.legacy(moments, omega, theta, historical) == 1
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(actual, historical, rtol=2e-13, atol=2e-14)
        assert actual[0] == moments[0]
    output = np.empty(3)
    loaded.lazy(moments, -1., output)
    np.testing.assert_allclose(output, [moments[0], moments[1]+.2*moments[0],
                                      moments[2]+.4*moments[1]+.04*moments[0]], atol=2e-15)
    loaded.lazy(moments, 1., output)
    assert not np.isfinite(output[1:]).any()
