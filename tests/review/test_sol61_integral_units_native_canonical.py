"""Compile and exercise the actual native canonical-unit validator, no PoPS JIT."""
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess

import pytest

from pops._ir.quantity import PhysicalDimension
from pops.time._program.integrals import integral_units_bytes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_validator(tmp_path_factory):
    folder = tmp_path_factory.mktemp("native-canonical-units")
    source = folder / "probe.cpp"
    source.write_text(r'''
#include <pops/core/identity/physical_dimension_json.hpp>
#include <iostream>
int main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    try {
      pops::identity::require_canonical_physical_dimension_json(line);
      std::cout << "accept\n";
    } catch (const std::invalid_argument&) { std::cout << "reject\n"; }
  }
}
''')
    exe = folder / "probe"
    subprocess.run(["clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
                    "-I", str(ROOT / "include"), str(source), "-o", str(exe)],
                   check=True, capture_output=True, text=True)

    def check(values):
        result = subprocess.run([str(exe)], input="\n".join(values) + "\n",
                                check=True, capture_output=True, text=True)
        return result.stdout.splitlines()
    return check


def encode(powers):
    return integral_units_bytes(PhysicalDimension(tuple(powers)))


def test_actual_python_images_unicode_and_unbounded_fractions(native_validator):
    names = ["mass", "énergie", "時間", "\ue000", "\U00010000", "😀", "x\"\\\b\f\n\r\t\0\x7f"]
    values = [encode(()), encode((("length", -2),))]
    values += [encode(((name, Fraction(-7, 3)),)) for name in names]
    values += [encode(tuple((name, Fraction(i + 1, 11)) for i, name in enumerate(names)))]
    # Greater than any machine-integer exponent, with a genuine large denominator.
    values += [encode((("huge", Fraction(10**160 + 1, 10**159 + 7)),)),
               encode((("huge", Fraction(-(10**2048 + 1), 10**1024)),))]
    rng = random.Random(409)
    values += [encode((("random", Fraction(rng.randrange(-10**100, 10**100),
                                            rng.randrange(1, 10**100))),)) for _ in range(80)]
    assert native_validator(values) == ["accept"] * len(values)


def test_redigested_malformed_and_noncanonical_images(native_validator):
    valid = encode((("a", Fraction(1, 3)), ("b", Fraction(-2, 5))))
    values = ["not-json", "", "null", "[]", "{}", "true",
              '{"powers":[],"kind":"physical_dimension"}',
              '{"kind":"physical_dimension","powers":[],"extra":1}',
              '{"kind":"physical_dimension","kind":"physical_dimension","powers":[]}',
              '{"kind":"physical_dimension","powers":null}',
              '{"kind":"wrong","powers":[]}',
              valid + " ", " " + valid, "\ufeff" + valid,
              valid.replace(",", ", "), valid.replace('"a"', '""'),
              valid.replace('["b",-2,5]', '["a",-2,5]'),
              valid.replace('["a",1,3],["b",-2,5]', '["b",-2,5],["a",1,3]'),
              valid.replace('["a",1,3]', '["a",1,3,4]'),
              valid.replace('["a",1,3]', '["a",1]'),
              valid.replace('"a"', '1'), valid.replace('"a"', '"\\u0061"'),
              valid.replace('"a"', '"\\/"'), valid.replace('"a"', '"é"'),
              valid.replace('"a"', '"\\u00E9"'),
              valid.replace('"a"', '"\\u000a"'), valid.replace('"a"', '"\\x61"')]
    for bad in ["0", "-0", "+1", "01", "1.0", "1e0", "true", '"1"', "null"]:
        values.append(valid.replace('["a",1,3]', '["a",' + bad + ',3]'))
    for bad in ["0", "-3", "+3", "03", "3.0", "3e0", "false", '"3"']:
        values.append(valid.replace('["a",1,3]', '["a",1,' + bad + ']'))
    values += [valid.replace('["a",1,3]', '["a",2,6]'),
               encode((("a", 1),)).replace(',1,1]', ',10,10]'),
               '{"kind":"physical_dimension","powers":[["x",' + str(10**100) + ',' + str(10**101) + ']]}',
               encode((("\ue000", 1), ("\U00010000", 1))).replace(
                   '["\\ue000",1,1],["\\ud800\\udc00",1,1]',
                   '["\\ud800\\udc00",1,1],["\\ue000",1,1]')]
    # Metadata hashing can authenticate each of these bytes; it cannot type them.
    assert native_validator(values) == ["reject"] * len(values)


def test_capture_both_paths_validate_before_digest_inside_collective_try():
    header = (ROOT / "include/pops/runtime/program/prepared_integral_capture.hpp").read_text()
    require = header.split("static void require_units_", 1)[1].split("static std::string contract_", 1)[0]
    assert require.index("require_canonical_physical_dimension_json(units)") < require.index("sha256_hex")
    for section in [header.split("static PreparedIntegralCapture prepare_", 1)[1].split("double consume_", 1)[0],
                    header.split("double consume_", 1)[1]]:
        assert section.index("try {") < section.index("require_units_(identity, units)")
        assert section.index("require_units_(identity, units)") < section.index("collectively_rethrow_exception")


def test_arbitrary_precision_gcd_rejects_common_factors(native_validator):
    rng = random.Random(183)
    valid, invalid = [], []
    for _ in range(60):
        fraction = Fraction(rng.randrange(1, 10**180), rng.randrange(1, 10**180))
        factor = rng.randrange(2, 10**40)
        valid.append(encode((("rational", fraction),)))
        invalid.append(json.dumps({"kind": "physical_dimension", "powers": [
            ["rational", fraction.numerator * factor, fraction.denominator * factor]]},
            sort_keys=True, separators=(",", ":")))
    assert native_validator(valid + invalid) == ["accept"] * 60 + ["reject"] * 60


def test_python_canonical_image_and_identity_remain_exact():
    import hashlib
    from pops.time import Program

    dimension = PhysicalDimension((("長さ", Fraction(3, 7)), ("mass", -1)))
    units = integral_units_bytes(dimension)
    assert units == json.dumps(dimension.to_data(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    program = Program(name="unit_identity")
    integral = program.integral_state("q", initial=.7, units=dimension)
    assert "/" + hashlib.sha256(units.encode()).hexdigest() + "/q" in integral.identity
