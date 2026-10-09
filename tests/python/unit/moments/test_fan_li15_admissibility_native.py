"""Exact binary-input H1 counterexamples and strict floating certificate scope."""

from fractions import Fraction
import json
import math
import subprocess

import numpy as np
import pytest

from tests.python.unit.moments import test_fan_li15_path_native as path_witnesses

path_bridge = path_witnesses.path_bridge


def raw_state(rho, mx, my, mxx, mxy, myy):
    raw = [0.0] * 15
    for k, value in zip((0, 1, 5, 2, 6, 9), (rho, mx, my, mxx, mxy, myy), strict=True):
        raw[k] = value
    return tuple(raw)


def exact_h1_minors(raw):
    r, x, y, a, b, c = (Fraction(raw[k]) for k in (0, 1, 5, 2, 6, 9))
    return r, r * a - x * x, r * (a * c - b * b) - x * (x * c - y * b) + y * (x * b - y * a)


@pytest.mark.compiler
@pytest.mark.parametrize(
    "raw",
    [
        raw_state(1.0, 1024.1, -1023.7, 1048782.81, -1048370.1699999999, 1047962.1900000001),
        raw_state(
            1.5,
            0.5,
            0.21428571428571427,
            0.16666816666666667,
            0.07143007142857144,
            0.030613744897959184,
        ),
        raw_state(1.5, 6.0, -4.5, 24.0000015, -17.9999985, 13.5000015),
    ],
)
def test_exact_nonpositive_raw_determinants_previously_accepted_are_refused(path_bridge, raw):
    minors = exact_h1_minors(raw)
    assert minors[0] > 0 and minors[1] > 0 and minors[2] <= 0
    assert path_bridge.admissibility([raw]) == (8,)
    result = path_bridge.interface(raw, raw, (1, 0))
    assert result["status"] == 8 and result["side"] == 0
    for key in ("flux", "left_ncp", "right_ncp"):
        np.testing.assert_array_equal(result[key], np.zeros(15))


@pytest.mark.compiler
def test_subnormal_overflow_and_resolved_indefinite_inputs_are_classified(path_bridge):
    tiny = math.ulp(0.0)
    cases = [
        (
            raw_state(tiny, 0, 0, tiny, 0, tiny),
            0,
        ),  # Tiny density, exact normalized unit covariance.
        (
            raw_state(1.0, 0, 0, tiny, 0, tiny),
            0,
        ),  # Subnormal covariance, rescaled before determinant.
        (raw_state(1.0, 0, 0, 1e-300, 0, 1e300), 8),  # Exact SPD; unrepresentable scaled minor.
        (raw_state(tiny, 0, 0, 1, 0, 1), 3),  # Normalization overflows.
        (raw_state(1.0, 1e200, 0, 1e300, 0, 1), 3),  # Covariance calculation overflows.
        (raw_state(1.0, 0, 0, 1, 2, 1), 4),  # Resolved negative determinant.
        (raw_state(1.0, 0, 0, -1, 0, 1), 4),  # Resolved negative first covariance minor.
    ]
    actual = path_bridge.admissibility([state for state, status in cases])
    assert actual == tuple(status for state, status in cases)
    for (state, _status), observed in zip(cases, actual, strict=True):
        if observed == 0:
            assert all(minor > 0 for minor in exact_h1_minors(state))
    fp = subprocess.run(
        [str(path_bridge.executable)],
        input="fp_environment\n",
        text=True,
        capture_output=True,
        check=True,
    )
    assert fp.stdout.strip() == "1 1", (
        "witness requires round-to-nearest and gradual subnormal arithmetic"
    )


@pytest.mark.compiler
def test_fast_math_compilation_is_explicitly_refused(path_bridge, tmp_path):
    receipt = json.loads((path_bridge.executable.parent / "compile-receipt.json").read_text())
    command = receipt["command"].copy()
    command[-1] = str(tmp_path / "forbidden_fast_math")
    command.insert(1, "-ffast-math")
    command.remove("-fno-fast-math")
    compiled = subprocess.run(command, capture_output=True, text=True)
    assert compiled.returncode != 0
    assert (
        "Raw moment interval certification requires strict floating-point semantics" in compiled.stderr
    )
