"""Independent Gaussian moments and high-precision phases test the opt-in native map."""
from __future__ import annotations

from decimal import Decimal as D, ROUND_HALF_EVEN, localcontext
from functools import lru_cache
import math
from pathlib import Path
import subprocess

import numpy as np
import pytest


PQS = tuple((p, q) for q in range(5) for p in range(5 - q))
INDEX = {pair: i for i, pair in enumerate(PQS)}
PI = D("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342")
SOURCE = r'''
#include <pops/numerics/moments/affine_velocity.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
using pops::Real;
using Rotation = pops::moments::AffineVelocityRotation;

int main() {
  std::cout << std::setprecision(17);
  Real omega, half_dt;
  while (std::cin >> omega >> half_dt) {
    Real old[15], endpoint[15], output[15], legacy[15], explicit_default[15];
    for (Real& value : old) if (!(std::cin >> value)) return 1;
    for (Real& value : endpoint) if (!(std::cin >> value)) return 1;
    if (!pops::moments::affine_velocity_push_forward<4>(old, endpoint, omega, half_dt, legacy) ||
        !pops::moments::affine_velocity_push_forward<4, Rotation::cayley>(
            old, endpoint, omega, half_dt, explicit_default) ||
        std::memcmp(legacy, explicit_default, sizeof(legacy)) != 0 ||
        !pops::moments::affine_velocity_push_forward<4, Rotation::exponential>(
            old, endpoint, omega, half_dt, output)) return 2;
    if (output[0] != old[0] || output[1] != endpoint[1] || output[5] != endpoint[5]) return 3;
    Real c, s;
    if (!pops::moments::exponential_rotation(omega, half_dt, c, s)) return 4;
    std::cout << c << ' ' << s;
    for (Real value : output) std::cout << ' ' << value;
    std::cout << '\n';
  }
  Real old[15]{}, endpoint[15]{}, output[15];
  old[0] = endpoint[0] = 1;
  const Real nan = std::numeric_limits<Real>::quiet_NaN();
  const Real inf = std::numeric_limits<Real>::infinity();
  int guards = 0;
  const auto refuses = [&](Real frequency, Real half) {
    std::fill(output, output+15, Real(42));
    const bool valid = pops::moments::affine_velocity_push_forward<4, Rotation::exponential>(
        old, endpoint, frequency, half, output);
    if (valid || !std::all_of(output, output+15, [](Real x) { return x == 42; })) return false;
    ++guards;
    return true;
  };
  endpoint[0] = 2;
  if (!refuses(1, 1)) return 5;
  old[0] = endpoint[0] = 0;
  if (!refuses(1, 1)) return 5;
  old[0] = endpoint[0] = -1;
  if (!refuses(1, 1)) return 5;
  old[0] = endpoint[0] = 1;
  old[7] = nan;
  if (!refuses(1, 1)) return 5;
  old[7] = 0; endpoint[7] = nan;
  if (!refuses(1, 1)) return 5;
  endpoint[7] = 0;
  for (Real frequency : {nan, inf, -inf}) if (!refuses(frequency, 1)) return 5;
  for (Real half : {Real(-1), nan, inf}) if (!refuses(1, half)) return 5;
  if (!refuses(Real(1e200), Real(1e200))) return 5;
  // The overflow-safe default still accepts this limiting Cayley rotation.
  if (!pops::moments::affine_velocity_push_forward<4>(old, endpoint, Real(1e200), Real(1e200), output)) return 6;
  if (!refuses(Real(1e-308), std::numeric_limits<Real>::max())) return 5;
  endpoint[1] = Real(1e100);
  if (!refuses(1, 1)) return 5;  // A finite mean whose fourth raw moment overflows.
  std::cout << "guards=" << guards << '\n';
}
'''


def _rotation(omega, half_dt):
    """Reference the exact binary inputs, independently of FMA or native trig."""
    angle = D.from_float(omega) * (2 * D.from_float(half_dt))
    period = 2 * PI
    angle -= (angle / period).to_integral_value(rounding=ROUND_HALF_EVEN) * period
    sine = term_s = angle
    cosine = term_c = D(1)
    for n in range(1, 160):
        term_s *= -angle * angle / ((2 * n) * (2 * n + 1))
        term_c *= -angle * angle / ((2 * n - 1) * (2 * n))
        sine += term_s
        cosine += term_c
        if max(abs(term_s), abs(term_c)) < D("1e-80"):
            return cosine, sine
    raise AssertionError("independent trigonometric series failed to converge")


def _mean(components):
    return [sum((weight * mean[k] for weight, mean, _ in components), D(0)) for k in range(2)]


def _moments(components, mass):
    answer = [D(0)] * 15
    for weight, mean, cov in components:
        @lru_cache(None)
        def gaussian(p, q):
            if p == q == 0:
                return D(1)
            if p:
                value = mean[0] * gaussian(p - 1, q)
                if p >= 2:
                    value += (p - 1) * cov[0][0] * gaussian(p - 2, q)
                if q:
                    value += q * cov[0][1] * gaussian(p - 1, q - 1)
                return value
            value = mean[1] * gaussian(0, q - 1)
            if q >= 2:
                value += (q - 1) * cov[1][1] * gaussian(0, q - 2)
            return value
        for i, (p, q) in enumerate(PQS):
            answer[i] += mass * weight * gaussian(p, q)
    return answer


def _transform(components, cosine, sine, target):
    r = ((cosine, sine), (-sine, cosine))
    old_mean = _mean(components)
    return [(weight,
             [target[i] + sum((r[i][j] * (mean[j] - old_mean[j]) for j in range(2)), D(0))
              for i in range(2)],
             [[sum((r[i][k] * cov[k][m] * r[j][m] for k in range(2) for m in range(2)), D(0))
               for j in range(2)] for i in range(2)]) for weight, mean, cov in components]


def _cases():
    weights = list(map(D, (".2", ".35", ".45")))
    means = [list(map(D, pair)) for pair in (("-1.2", ".3"), (".7", "1.4"), ("1.1", "-.8"))]
    covariances = [[[D(xx), D(xy)], [D(xy), D(yy)]]
                   for xx, xy, yy in ((".15", ".03", ".06"), (".08", "-.02", ".20"),
                                      (".05", ".012", ".09"))]
    omega = -6283185310000.0
    acceleration = [D("39478417600000") * item for item in (D(".3"), D("-.2"))]
    drift = [acceleration[1] / D.from_float(omega), -acceleration[0] / D.from_float(omega)]
    original = list(zip(weights, means, covariances))
    old_mean = _mean(original)
    prepared = [(w, [u[k] + drift[k] - old_mean[k] for k in range(2)], cov)
                for w, u, cov in original]
    for prepared_mean, components in ((False, original), (True, prepared)):
        u0 = _mean(components)
        for metric in (D(1), D("7.5")):
            mass = D("1.7") * metric
            old = list(map(float, _moments(components, mass)))
            for dt in (0., 1e-16, 1e-14, 1e-12, 1e-10, 1e-6, .0001, .001, .00099999995):
                half = dt / 2
                c, s = _rotation(omega, half)
                z = D.from_float(omega) * D.from_float(half)
                cc, sc = (1 - z * z) / (1 + z * z), 2 * z / (1 + z * z)
                mean = [drift[0] + cc * (u0[0] - drift[0]) + sc * (u0[1] - drift[1]),
                        drift[1] - sc * (u0[0] - drift[0]) + cc * (u0[1] - drift[1])]
                endpoint = old.copy()
                endpoint[1], endpoint[5] = float(mass * mean[0]), float(mass * mean[1])
                expected = _moments(_transform(components, c, s, mean), mass)
                centered = _moments(_transform(components, c, s, [D(0), D(0)]), D(1))
                yield omega, half, old, endpoint, c, s, expected, centered, prepared_mean
    # Zero B and both weak-field signs also retain the supplied translated mean.
    old = list(map(float, _moments(original, D("1.7"))))
    for omega in (0.0, -2.0, 2.0):
        half = .125
        c, s = _rotation(omega, half)
        mean = [old_mean[0] + D(".1"), old_mean[1] - D(".2")]
        endpoint = old.copy()
        endpoint[1], endpoint[5] = float(D("1.7") * mean[0]), float(D("1.7") * mean[1])
        expected = _moments(_transform(original, c, s, mean), D("1.7"))
        centered = _moments(_transform(original, c, s, [D(0), D(0)]), D(1))
        yield omega, half, old, endpoint, c, s, expected, centered, False


@pytest.mark.compiler
def test_exponential_native_map_matches_gaussians_and_stiff_binary_input_phase(tmp_path, native_cxx):
    root = Path(__file__).resolve().parents[4]
    source = tmp_path / "affine_exponential.cpp"
    executable = tmp_path / "affine_exponential"
    source.write_text(SOURCE)
    subprocess.run([native_cxx, "-std=c++20", "-O2", "-I", str(root / "include"),
                    str(source), "-o", str(executable)], check=True, capture_output=True, text=True)
    with localcontext() as context:
        context.prec = 90
        cases = list(_cases())
    inputs = "\n".join(" ".join(map(repr, [omega, half] + old + endpoint))
                       for omega, half, old, endpoint, *_ in cases) + "\n"
    result = subprocess.run([str(executable)], input=inputs, check=True,
                            capture_output=True, text=True)
    rows = result.stdout.splitlines()
    assert rows[-1] == "guards=14"
    assert len(rows) == len(cases) + 1 == 40
    maximum_raw = maximum_phase = maximum_centered = 0.
    for case, row in zip(cases, rows[:-1], strict=True):
        omega, half, old, endpoint, c, s, expected, centered, _ = case
        values = list(map(float, row.split()))
        assert len(values) == 17 and all(map(math.isfinite, values))
        phase_error = max(abs(values[0] - float(c)), abs(values[1] - float(s)))
        maximum_phase = max(maximum_phase, phase_error)
        assert phase_error <= 4e-16
        raw = values[2:]
        assert raw[0] == old[0] and raw[1] == endpoint[1] and raw[5] == endpoint[5]
        error = max(abs(x - float(y)) / max(1., abs(float(y))) for x, y in zip(raw, expected, strict=True))
        maximum_raw = max(maximum_raw, error)
        assert error < 3e-13
        ux, uy = raw[1] / raw[0], raw[5] / raw[0]
        actual_central = [math.fsum(math.comb(p, i) * math.comb(q, j) * (-ux)**(p-i) * (-uy)**(q-j)
                                   * raw[INDEX[i, j]] / raw[0]
                                   for i in range(p+1) for j in range(q+1)) for p, q in PQS]
        covariance = np.array([[actual_central[2], actual_central[6]],
                               [actual_central[6], actual_central[9]]])
        assert np.linalg.eigvalsh(covariance)[0] > 0
        for degree in (2, 3, 4):
            error = math.sqrt(math.fsum(math.comb(degree, p) * (
                actual_central[INDEX[p, degree-p]] - float(centered[INDEX[p, degree-p]]))**2
                for p in range(degree+1)))
            norm = math.sqrt(math.fsum(math.comb(degree, p) * float(centered[INDEX[p, degree-p]])**2
                                      for p in range(degree+1)))
            maximum_centered = max(maximum_centered, error / norm)
            assert error / norm < 2e-12
    print(f"cases={len(cases)} guards=14 default_memcmp=identical "
          f"max_phase={maximum_phase:.17g} max_raw={maximum_raw:.17g} "
          f"max_central={maximum_centered:.17g}")
