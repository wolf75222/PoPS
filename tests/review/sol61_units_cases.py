"""Independent stdlib vectors against the actual native canonical-unit reader."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess


def encoded(powers):
    return json.dumps({"kind": "physical_dimension", "powers": powers},
                      sort_keys=True, separators=(",", ":"))


def cases():
    result = [("dimensionless", encoded([]), True)]
    for name in ["quote\"", "back\\slash", "slash/", "\x00", "\b\f\n\r\t",
                 "\x1f", "\x7f", "\u00e9", "\ue000", "\U00010000", "\U0010ffff",
                 "\ud800", "\udc00", "\ud800X"]:
        result.append((ascii(name), encoded([[name, -3, 7]]), True))
    result.append(("nonBMP-codepoint-sort", encoded([["\ue000", 1, 1],
                                                   ["\U00010000", 1, 1]]), True))
    rng = random.Random(9247)
    for index in range(40):
        power = Fraction(rng.randrange(-10**100, 10**100) or 1, rng.randrange(1, 10**100))
        result.append((f"fraction-{index}", encoded([["charge", power.numerator, power.denominator]]), True))
    result.append(("giant-coprime", encoded([["giant", 10**2048+1, 10**2048]]), True))
    result.append(("giant-unreduced", encoded([["giant", 3*10**1024, 5*10**1024]]), False))
    invalid = {
        "not-json": "not-json", "reordered": '{"powers":[],"kind":"physical_dimension"}',
        "whitespace": '{"kind":"physical_dimension", "powers":[]}',
        "extra-key": '{"kind":"physical_dimension","powers":[],"x":0}',
        "zero": encoded([["a", 0, 1]]), "unreduced": encoded([["a", 2, 4]]),
        "negative-denominator": encoded([["a", 1, -1]]),
        "leading-zero": '{"kind":"physical_dimension","powers":[["a",01,1]]}',
        "duplicates": encoded([["a", 1, 1], ["a", 2, 1]]),
        "unsorted": encoded([["z", 1, 1], ["a", 1, 1]]),
        "empty-base": encoded([["", 1, 1]]),
        "upper-hex": encoded([["\uefff", 1, 1]]).replace("efff", "EFFF"),
        "ascii-escape": encoded([["a", 1, 1]]).replace('"a"', '"\\u0061"'),
        "slash-escape": encoded([["/", 1, 1]]).replace('"/"', '"\\/"'),
        "raw-DEL": encoded([["\x7f", 1, 1]]).replace("\\u007f", "\x7f"),
        "raw-nonASCII": encoded([["\u00e9", 1, 1]]).replace("\\u00e9", "\u00e9"),
        # These Python names are distinct, but JSON loses their distinction.
        "surrogate-alias": encoded([["\ud800\udc00", 1, 1], ["\U00010000", 1, 1]]),
        "surrogate-order-roundtrip": encoded([["\ud800\udc00", 1, 1], ["\ue000", 1, 1]]),
    }
    result.extend((name, value, False) for name, value in invalid.items())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reader", type=Path)
    args = parser.parse_args()
    vectors = cases()
    run = subprocess.run([str(args.reader)], input="\n".join(row[1] for row in vectors)+"\n",
                         text=True, capture_output=True, check=True)
    outcomes = run.stdout.splitlines()
    assert len(outcomes) == len(vectors)
    for (name, _, expected), outcome in zip(vectors, outcomes, strict=True):
        assert outcome == ("accept" if expected else "refuse"), (name, expected, outcome)
    print(json.dumps({"actual_reader_checks": len(vectors),
                      "accepted": sum(row[2] for row in vectors),
                      "refused": sum(not row[2] for row in vectors)}))
