"""Signed Newton RHS counterexample; independent math, no native/Pops import."""
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def callback_sign():
    source = (ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    body = source.split("auto residual =", 1)[1].split("auto derivative =", 1)[0]
    assert "evaluate(q, result, evaluation);" in body
    return -1 if re.search(r"scale\(level,\s*Real\(-1\)\)", body) else 1


@pytest.mark.parametrize("width", (1, 3, 5))
def test_original_q_minus_target_moves_toward_root_with_actual_callback_sign(width):
    q = np.zeros(width)
    target = np.linspace(.15, .35, width)
    original = q-target
    # The true workspace solves +J_F delta=RHS and then trial=q+alpha*delta.
    delta = np.linalg.solve(np.eye(width), callback_sign()*original)
    assert np.array_equal(q+delta, target)
    for alpha in (1., .5, .125):
        assert np.linalg.norm(q+alpha*delta-target) < np.linalg.norm(original)


def test_old_positive_rhs_is_a_counterexample_at_every_admitted_step():
    target = np.array([.15, .25, .18])
    residual = -target
    for exponent in range(21):
        alpha = 2.**-exponent
        assert np.linalg.norm(alpha*residual-target) > np.linalg.norm(residual)


@pytest.mark.parametrize("permutation", ((0, 1, 2), (2, 0, 1)))
def test_nonconstant_signed_cross_diffusion_cubic_original_equation(permutation):
    # Independent nonconstant physical equation. No grid solves substitute for
    # the production composite AMR witness; this verifies the direction algebra.
    n, dx = 8, 1/8
    d = np.array([[2., -.35, .2], [-.35, 1.6, -.25], [.2, -.25, 1.3]])
    r = np.array([[2., .2, -.1], [.2, 1.8, .15], [-.1, .15, 2.2]])
    d, r = d[np.ix_(permutation, permutation)], r[np.ix_(permutation, permutation)]
    lap = (2*np.eye(n)-np.roll(np.eye(n), 1, axis=1)-np.roll(np.eye(n), -1, axis=1))/dx**2
    a = np.kron(d, lap)+np.kron(r, np.eye(n))
    x = (np.arange(n)+.5)*dx
    target = (np.array([.15, .25, .18])[list(permutation), None]
              + np.array([.02, .01, -.015])[list(permutation), None]*np.cos(2*np.pi*x)).ravel()
    forcing = a@target + .2*target**3
    q = np.zeros(3*n)
    def f(value):
        return a@value + .2*value**3-forcing
    residual = f(q)
    delta = np.linalg.solve(a, callback_sign()*residual)
    assert np.linalg.norm(f(q+delta)) < np.linalg.norm(residual)
    old_delta = np.linalg.solve(a, residual)
    assert np.linalg.norm(f(q+old_delta)) > np.linalg.norm(residual)


def test_original_rhs_negation_is_collective_and_jvp_recheck_keep_physical_sign():
    source = (ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp").read_text()
    residual = source.split("auto residual =", 1)[1].split("auto derivative =", 1)[0]
    assert "local_phase_(lane" in residual and callback_sign() == -1
    derivative = source.split("auto derivative =", 1)[1].split("std::vector<field_type*>", 1)[0]
    assert "Real(0.5) / h, plus_[level], -Real(0.5) / h, minus_[level]" in derivative
    assert "evaluate(candidate_, recheck_, 0);" in source


def test_real_cmake_gtest_scan_keeps_old_four_and_all_original_fragment_cases(tmp_path):
    cmake = (ROOT / "tests/CMakeLists.txt").read_text()
    assert "SOURCES ${ARG_SOURCES} ${ARG_DISCOVERY_SOURCES}" in cmake
    assert 'elseif(name STREQUAL "test_composite_general_field")' in cmake
    cpp = ROOT / "tests/cpp/unit/elliptic/test_composite_general_field.cpp"
    fragment = cpp.with_name("amr_original_field_residual.inc")
    expected = set(re.findall(r"TEST\((CompositeGeneralField),\s*(\w+)\)", cpp.read_text()+fragment.read_text()))
    expected = {suite+"."+name for suite, name in expected}
    old_fragment = subprocess.check_output(
        ["git", "show", "9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6:" + str(fragment.relative_to(ROOT))],
        cwd=ROOT, text=True)
    old_cpp = subprocess.check_output(
        ["git", "show", "9fb7e8fb64c3e3de289d9c139ba310e9e47e43a6:" + str(cpp.relative_to(ROOT))],
        cwd=ROOT, text=True)
    historical = {suite+"."+name for suite, name in re.findall(
        r"TEST\((CompositeGeneralField),\s*(\w+)\)", old_cpp+old_fragment)}
    assert len(historical) == 11
    assert historical <= expected
    (tmp_path / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(InventoryOnly LANGUAGES NONE)
enable_testing()
include(GoogleTest)
add_executable(inventory IMPORTED)
set_target_properties(inventory PROPERTIES IMPORTED_LOCATION "/not-executed/inventory")
gtest_add_tests(TARGET inventory SOURCES "%s" "%s" TEST_LIST cases)
set_tests_properties(${cases} PROPERTIES LABELS "unit;elliptic;fast;cpp-target:test_composite_general_field")
file(WRITE "${CMAKE_BINARY_DIR}/cases.txt" "${cases}")
''' % (cpp, fragment))
    result = subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "inventory")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout+result.stderr
    found = set((tmp_path / "inventory/cases.txt").read_text().split(";"))
    assert found == expected
    ctest = (tmp_path / "inventory/CTestTestfile.cmake").read_text()
    assert ctest.count("cpp-target:test_composite_general_field") == len(expected)
