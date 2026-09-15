"""Header-only formula witnesses, not PoPS native/runtime qualification.

The retained bridge compiles this header without the official native library.
Expected integrals use high-precision quadrature of an independent generating
function / primary multi-index formula, never the production I_k recurrence.
"""

from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import subprocess

import mpmath as mp
import numpy as np
import pytest

from tests.python.support.fan_li15_oracle import TOP, gaussian_mixture, quadrature_path


SOURCE = r"""
#include <pops/numerics/moments/fan_li15_interface.hpp>
#include <cstdlib>
#include <cfenv>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
using pops::Real;
static_assert(!std::is_same_v<pops::moments::FanLi15ConservativeFlux,
                             pops::moments::FanLi15NcpResidual>);
static Real value() {
  std::string text; std::cin >> text;
  char* end = nullptr;
  const Real result = std::strtod(text.c_str(), &end);
  if (end != text.c_str()+text.size()) std::exit(3);
  return result;
}
int main() {
  std::cout << std::setprecision(17);
  std::string command;
  while (std::cin >> command) {
    if (command == "fp_environment") {
      volatile Real tiny=std::numeric_limits<Real>::denorm_min(), two=Real(2);
      std::cout << (std::fegetround()==FE_TONEAREST) << ' '
                << (tiny>Real(0) && tiny*two>tiny) << '\n';
      continue;
    }
    if (command == "admissibility") {
      Real raw[15]; for (Real& x : raw) x=value();
      std::cout << static_cast<int>(pops::moments::fan_li15_admissibility(raw)) << '\n';
      continue;
    }
    if (command == "weights") {
      const Real high=value(), low=value();
      Real weights[5]{}, scale=0;
      const bool valid=pops::moments::fan_li15_detail::density_integrals(high,low,weights,scale);
      std::cout << valid << ' ' << scale;
      for (Real x : weights) std::cout << ' ' << x;
      std::cout << '\n';
      continue;
    }
    if (command == "flux") {
      Real raw[15]; for (Real& x : raw) x=value();
      const Real gx=value(), gy=value();
      const auto result=pops::moments::fan_li15_grad_directional_flux(raw,gx,gy);
      std::cout << static_cast<int>(result.status);
      for (Real x : result.flux.values) std::cout << ' ' << x;
      std::cout << '\n';
      continue;
    }
    if (command != "path" && command != "interface") return 2;
    Real left[15], right[15], left_copy[15], right_copy[15];
    for (Real& x : left) x=value();
    for (Real& x : right) x=value();
    const Real gx=value(), gy=value();
    std::memcpy(left_copy,left,sizeof left); std::memcpy(right_copy,right,sizeof right);
    if (command == "interface") {
      const auto result=pops::moments::fan_li15_rusanov_interface(left,right,gx,gy);
      std::cout << static_cast<int>(result.status) << ' ' << result.input_side << ' ' << result.speed_bound;
      for (Real x : result.conservative_flux.values) std::cout << ' ' << x;
      for (Real x : result.left_ncp.values) std::cout << ' ' << x;
      for (Real x : result.right_ncp.values) std::cout << ' ' << x;
      std::cout << '\n';
      if (std::memcmp(left_copy,left,sizeof left) || std::memcmp(right_copy,right,sizeof right)) return 4;
      continue;
    }
    const auto result=pops::moments::fan_li15_path_integral(left,right,gx,gy);
    const bool unchanged=std::memcmp(left_copy,left,sizeof left)==0
        && std::memcmp(right_copy,right,sizeof right)==0;
    std::cout << static_cast<int>(result.status) << ' ' << result.input_side << ' '
              << result.speed_bound << ' ' << unchanged;
    for (Real x : result.integral) std::cout << ' ' << x;
    std::cout << '\n';
  }
}
"""


@pytest.fixture(scope="module")
def path_bridge(tmp_path_factory, native_cxx):
    root = Path(__file__).resolve().parents[4]
    directory = tmp_path_factory.mktemp("fan_li15_header_formula")
    source, executable = directory / "fan_li15_bridge.cpp", directory / "fan_li15_bridge"
    source.write_text(SOURCE)
    header_inputs = [
        root / "include/pops/numerics/moments/fan_li15_interface.hpp",
        root / "include/pops/numerics/moments/fan_li15_path.hpp",
        root / "include/pops/numerics/moments/affine_velocity.hpp",
        root / "include/pops/core/foundation/types.hpp",
    ]
    before_headers = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in header_inputs
    }
    command = [
        native_cxx,
        "-std=c++20",
        "-O2",
        "-fno-fast-math",
        "-ffp-contract=off",
        "-I",
        str(root / "include"),
        str(source),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, capture_output=True, text=True)
    (directory / "compiler.log").write_text(compiled.stdout + compiled.stderr)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    after_headers = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in header_inputs
    }
    assert before_headers == after_headers, "header changed during isolated compilation"
    files = [
        source,
        executable,
        root / "include/pops/numerics/moments/fan_li15_interface.hpp",
        root / "include/pops/numerics/moments/fan_li15_path.hpp",
        root / "include/pops/numerics/moments/affine_velocity.hpp",
        root / "include/pops/core/foundation/types.hpp",
    ]
    (directory / "compile-receipt.json").write_text(
        json.dumps(
            {
                "scope": "isolated header-only Fan-Li formula verification; no PoPS native library",
                "command": command,
                "compiler_version": subprocess.run(
                    [native_cxx, "--version"], capture_output=True, text=True
                ).stdout,
                "sha256": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files
                },
            },
            indent=2,
        )
        + "\n"
    )

    def evaluate(left, right, direction):
        data = "path " + " ".join(repr(float(x)) for x in (*left, *right, *direction)) + "\n"
        completed = subprocess.run(
            [str(executable)], input=data, text=True, capture_output=True, check=True
        )
        row = completed.stdout.split()
        result = {
            "status": int(row[0]),
            "side": int(row[1]),
            "bound": float(row[2]),
            "unchanged": bool(int(row[3])),
            "integral": np.asarray(list(map(float, row[4:]))),
        }
        assert result["unchanged"]
        with (directory / "path-cases.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"input": data.strip(), "output": completed.stdout.strip()}) + "\n"
            )
        return result

    evaluate.executable = executable

    def admissibility(cases):
        data = "".join(
            "admissibility " + " ".join(repr(float(x)) for x in raw) + "\n" for raw in cases
        )
        completed = subprocess.run(
            [str(executable)], input=data, text=True, capture_output=True, check=True
        )
        with (directory / "admissibility-cases.jsonl").open("a") as stream:
            stream.write(json.dumps({"input": data, "output": completed.stdout}) + "\n")
        return tuple(map(int, completed.stdout.split()))

    evaluate.admissibility = admissibility

    def extra(command, left, direction, right=()):
        data = command + " " + " ".join(repr(float(x)) for x in (*left, *right, *direction)) + "\n"
        completed = subprocess.run(
            [str(executable)], input=data, text=True, capture_output=True, check=True
        )
        values = list(map(float, completed.stdout.split()))
        with (directory / "interface-cases.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"input": data.strip(), "output": completed.stdout.strip()}) + "\n"
            )
        if command == "flux":
            return {"status": int(values[0]), "flux": np.asarray(values[1:])}
        return {
            "status": int(values[0]),
            "side": int(values[1]),
            "bound": values[2],
            "flux": np.asarray(values[3:18]),
            "left_ncp": np.asarray(values[18:33]),
            "right_ncp": np.asarray(values[33:48]),
        }

    evaluate.flux = lambda state, direction: extra("flux", state, direction)
    evaluate.interface = lambda left, right, direction: extra("interface", left, direction, right)
    return evaluate


def states():
    left = gaussian_mixture(
        (
            (Q(3, 4), (Q(-1, 3), Q(1, 5)), (Q(3, 4), Q(1, 9), Q(5, 4))),
            (Q(1, 4), (Q(2, 3), Q(-1, 2)), (Q(5, 4), Q(-1, 7), Q(3, 4))),
        )
    )
    right = gaussian_mixture(
        (
            (Q(2, 5), (Q(1, 2), Q(-1, 4)), (Q(4, 5), Q(-1, 8), Q(9, 8))),
            (Q(3, 5), (Q(-1, 4), Q(3, 5)), (Q(6, 5), Q(1, 6), Q(4, 5))),
        )
    )
    return tuple(map(float, left)), tuple(map(float, right))


@pytest.mark.compiler
@pytest.mark.parametrize("density_ratio", [1e-12, 1 - 1e-12, 1.0, 1.5, 1e12])
def test_header_integral_matches_independent_high_precision_quadrature(path_bridge, density_ratio):
    left, right = states()
    right = tuple(density_ratio * x for x in right)
    direction = 0.6, -0.8
    actual = path_bridge(left, right, direction)
    assert actual["status"] == 0
    reference = np.asarray(list(map(float, quadrature_path(left, right, direction))))
    scale = np.max(np.abs(reference))
    normalized_error = np.max(np.abs(actual["integral"] - reference)) / scale
    assert normalized_error < 3e-13
    assert all(actual["integral"][k] == 0 for k in range(15) if k not in TOP)
    print(f"density_ratio={density_ratio:.17g} normalized_integral_error={normalized_error:.17g}")


@pytest.mark.compiler
def test_exact_trace_antisymmetry_direction_linearity_and_binary_density_homogeneity(path_bridge):
    left, right = states()
    for ratio in (1.0, 0.125, 8.0):
        endpoint = tuple(ratio * x for x in right)
        g = 0.375, -0.625
        base = path_bridge(left, endpoint, g)
        assert base["status"] == 0
        reverse = path_bridge(endpoint, left, g)
        normal_reverse = path_bridge(left, endpoint, tuple(-x for x in g))
        np.testing.assert_array_equal(base["integral"], -reverse["integral"])
        np.testing.assert_array_equal(base["integral"], -normal_reverse["integral"])
        assert base["bound"] == reverse["bound"] == normal_reverse["bound"]
        for exponent in (-80, 80):
            scale = 2.0**exponent
            scaled = path_bridge(
                tuple(scale * x for x in left), tuple(scale * x for x in endpoint), g
            )
            assert scaled["status"] == 0
            np.testing.assert_array_equal(scaled["integral"], scale * base["integral"])
            assert scaled["bound"] == base["bound"]


@pytest.mark.compiler
def test_same_state_zero_covector_and_pure_density_path_are_zero(path_bridge):
    left, right = states()
    for first, second, g in (
        (left, left, (1, 0)),
        (left, right, (0, 0)),
        (left, tuple(8 * x for x in left), (1, 2)),
    ):
        result = path_bridge(first, second, g)
        assert result["status"] == 0
        np.testing.assert_array_equal(result["integral"], np.zeros(15))


@pytest.mark.compiler
def test_header_strict_refusals_publish_no_partial_integral(path_bridge):
    left, right = states()
    cases = [
        (0, 0.0, 2),
        (0, -1.0, 2),
        (8, float("nan"), 1),
        (12, float("inf"), 1),
        (2, 0.0, 4),
        (6, 10.0, 4),
    ]
    for slot, value, status in cases:
        invalid = list(left)
        invalid[slot] = value
        for side in (0, 1):
            result = (
                path_bridge(invalid, right, (1, 0))
                if side == 0
                else path_bridge(right, invalid, (1, 0))
            )
            assert result["status"] == status and result["side"] == side
            np.testing.assert_array_equal(result["integral"], np.zeros(15))
    invalid = list(left)
    invalid[0] = 1e-320
    result = path_bridge(invalid, right, (1, 0))
    assert result["status"] == 3
    for direction in ((float("nan"), 0), (0, float("inf"))):
        result = path_bridge(left, right, direction)
        assert result["status"] == 5 and result["side"] == -1
        np.testing.assert_array_equal(result["integral"], np.zeros(15))
    result = path_bridge(left, right, (1e308, 1e308))
    assert result["status"] == 7
    np.testing.assert_array_equal(result["integral"], np.zeros(15))


@pytest.mark.compiler
def test_header_spd_guard_is_not_a_full_moment_cone_claim(path_bridge):
    raw = list(map(float, gaussian_mixture(((Q(1), (Q(0), Q(0)), (Q(1), Q(0), Q(1))),))))
    raw[4] = -1
    result = path_bridge(raw, raw, (1, 0))
    assert result["status"] == 0


@pytest.mark.compiler
def test_gaussian_path_sign_and_noncommuting_amr_average(path_bridge):
    def gaussian(mean):
        return tuple(map(float, gaussian_mixture(((Q(1), (Q(mean), Q(0)), (Q(1), Q(0), Q(1))),))))

    left, fine1, fine2 = gaussian(0), gaussian(1), gaussian(3)
    middle = tuple((a + b) / 2 for a, b in zip(fine1, fine2, strict=True))
    p1, p2, pc = (path_bridge(left, right, (1, 0)) for right in (fine1, fine2, middle))
    assert all(result["status"] == 0 for result in (p1, p2, pc))
    assert p1["integral"][4] == pytest.approx(-1 / 6, abs=2e-14)
    assert p2["integral"][4] == pytest.approx(-81 / 2, abs=3e-12)
    assert pc["integral"][4] == pytest.approx(-31 / 3, abs=3e-12)
    assert pc["integral"][4] - (p1["integral"][4] + p2["integral"][4]) / 2 == pytest.approx(
        10, abs=5e-12
    )


@pytest.mark.compiler
def test_density_integrals_across_series_boundary_and_underflowed_ratio(path_bridge):
    # Scalar quadrature independently isolates the stable density weights;
    # the full 15-component oracle above covers their constitutive use.
    cases = [
        (1.0, ratio) for ratio in (1.0, 1 - 2e-16, 2 / 3 + 1e-12, 2 / 3, 2 / 3 - 1e-12, 0.5, 1e-12)
    ] + [(1e200, 1e-200)]
    request = "".join(f"weights {high!r} {low!r}\n" for high, low in cases)
    rows = subprocess.run(
        [str(path_bridge.executable)], input=request, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    with mp.workdps(65):
        for (high, low), row in zip(cases, rows, strict=True):
            values = list(map(float, row.split()))
            assert values[0] == 1
            scale, weights = values[1], values[2:]
            high_mp, low_mp = mp.mpf(high), mp.mpf(low)
            ell = mp.log(high_mp / low_mp)
            factor = high_mp if ell == 0 else high_mp * ell / mp.expm1(ell)
            panels = max(1, int(mp.ceil(abs(ell) / 12)))
            knots = [mp.mpf(i) / panels for i in range(panels + 1)]
            for k, weight in enumerate(weights):
                reference = factor * mp.quad(
                    lambda t, ell=ell, k=k: (
                        (t if ell == 0 else mp.expm1(ell * t) / mp.expm1(ell)) ** k
                    ),
                    knots,
                )
                relative = abs((mp.mpf(scale) * weight - reference) / reference)
                assert relative < mp.mpf("2e-14"), (high, low, k, relative)
