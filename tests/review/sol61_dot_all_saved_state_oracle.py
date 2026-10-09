"""Offline oracle for authenticated SDK506 states; does not import/run PoPS."""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct

import numpy as np

SOURCE = "f36176fa2a822c67b5ccc114bd21de1bef4be23e"
CASES = (
    ("ro0", "6f99f83b77ab0971be3c97134276ec52dd067e09df5ccd7b8433babc33d8247c", 1., 1.6, -5),
    ("ro1", "9c1fef4ee223dd73a527926772993e9aaf0ba77fab2ee0ec0e708e8ccb582138", .5,
     0.9411764705882353, -10),
)


def ulps(actual, expected):
    assert actual > 0 and expected > 0 and math.isfinite(actual) and math.isfinite(expected)
    return struct.unpack("Q", struct.pack("d", actual))[0] - struct.unpack("Q", struct.pack("d", expected))[0]


def rounded_products(left, right, active=None, coverage=None):
    assert len(left) == len(right)
    if active is None:
        active = [1.] * len(left)
    if coverage is None:
        coverage = [1.] * len(left)
    assert len(active) == len(left) and len(coverage) == len(left)
    products = []
    for a, b, enabled, owner in zip(left, right, active, coverage, strict=True):
        if not math.isfinite(enabled) or not math.isfinite(owner):
            raise OverflowError("nonfinite mask")
        if enabled < .5 or owner < .5:
            continue
        value = float(a) * float(b)
        if not math.isfinite(a) or not math.isfinite(b) or not math.isfinite(value):
            raise OverflowError("nonfinite active input or product")
        products.append(value)
    return products


def exact(products):
    # Reference is the exact real sum of individually rounded binary64
    # products. It does not silently upgrade multiplication to arbitrary precision.
    return sum((Fraction.from_float(value) for value in products), Fraction())


def naive(products):
    result = 0.
    for value in products:  # Python 3.12 sum(float) is already compensated.
        result += value
    return result


def native_legacy_grouping(products):
    return naive(naive(component.flat) for component in products)


def synthetic():
    witnesses = []
    for permutation in itertools.permutations([1e16, 1., -1e16]):
        assert exact(permutation) == 1
        witnesses.append({"products": [value.hex() for value in permutation],
                          "exact": float(exact(permutation)).hex(),
                          "old_naive": naive(permutation).hex()})
    assert rounded_products([math.nan, 2., math.inf], [1., 3., 1.],
                            active=[0., 1., 1.], coverage=[1., 1., 0.]) == [6.]
    refusals = 0
    for args in [([math.nan], [1.], None, None), ([1e308], [1e308], None, None),
                 ([math.inf], [1.], None, None), ([1.], [1.], [math.nan], [0.]),
                 ([1.], [1.], [0.], [math.inf])]:
        try:
            rounded_products(*args)
        except OverflowError:
            refusals += 1
    assert refusals == 5
    limit = float.fromhex("0x1.fffffffffffffp+1023")
    products = rounded_products([limit, limit], [1., 1.])
    assert all(map(math.isfinite, products))
    try:
        float(exact(products))
    except OverflowError:
        pass
    else:
        raise AssertionError("finite products with overflowing exact sum were accepted")
    partitions = [[1e16, 1.], [-1e16]]
    exact_global = float(exact(value for row in partitions for value in row))
    rounded_locals = [float(exact(row)) for row in partitions]
    rounded_rank_sum = naive(rounded_locals)
    assert exact_global == 1. and rounded_rank_sum == 0.
    return {"cancellation_permutations": witnesses, "masked_invalid_values_ignored": True,
            "invalid_input_product_mask_refusals": refusals,
            "finite_products_overflowing_sum_refused": True,
            "cross_rank_limitation": {"products": partitions, "exact": exact_global,
                                     "rounded_locals": rounded_locals,
                                     "ordinary_rank_sum": rounded_rank_sum}}


def run(evidence):
    identity_path = evidence / "identity.json"
    identity = json.loads(identity_path.read_text())
    assert identity["source_commit"] == SOURCE
    assert "headers=506dce78009f" in identity["abi_key"]
    rows = []
    for suffix, digest, dt, target, expected_old_ulps in CASES:
        path = evidence / "pytest-tmp" / ("test_native_scalar_frontier_" + suffix) / "computed_rotation.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with np.load(path, allow_pickle=False) as image:
            assert str(image["abi_key"]) == identity["abi_key"]
            assert list(image["names_rotation"]) == ["x", "y"]
            u = image["state_rotation"].copy()
            assert u.dtype == np.float64 and u.shape == (2, 8, 8)
            assert np.all(np.isfinite(u)) and int(image["ncomp_rotation"]) == 2
            assert int(image["macro_step"]) == 1
            start = float(image["t"])
            receipt = json.loads(str(image["temporal_restart_state"]))["controller_state"]["program_frontier"]
            assert float.fromhex(receipt["requested_duration"]) == dt
            assert float.fromhex(receipt["reached"]) == start
            assert float.fromhex(receipt["duration"]) == start
        rhs0 = np.stack((-u[1], u[0]))
        predictor = u + dt * rhs0
        rhs1 = np.stack((-predictor[1], predictor[0]))
        increment = ((u + (.5 * dt) * rhs0) + (.5 * dt) * rhs1) - u
        contractions = [u * increment, increment * increment]
        assert all(np.all(np.isfinite(value)) for value in contractions)
        old = [native_legacy_grouping(value) for value in contractions]
        oracle = [float(exact(map(float, value.flat))) for value in contractions]
        assert oracle == [math.fsum(map(float, value.flat)) for value in contractions]
        old_endpoint = start + (-2. * old[0] / old[1]) * dt
        oracle_endpoint = start + (-2. * oracle[0] / oracle[1]) * dt
        assert ulps(old_endpoint, target) == expected_old_ulps
        assert abs(ulps(oracle_endpoint, target)) <= 4
        # Exact addition is invariant under mathematical grouping, masks,
        # component permutation and patch splitting. Rounded MPI scalars are not.
        variants = [value[::-1].ravel().tolist() for value in contractions]
        assert oracle == [float(exact(value)) for value in variants]
        rows.append({"path": str(path), "sha256": digest, "shape": list(u.shape),
            "components": ["x", "y"], "start": start.hex(), "requested": dt.hex(),
            "target": target.hex(), "native_receipt": receipt,
            "old_component_patch_grouping": [value.hex() for value in old],
            "exact_sum_rounded_products": [value.hex() for value in oracle],
            "old_endpoint": old_endpoint.hex(), "old_endpoint_ulps": ulps(old_endpoint, target),
            "oracle_endpoint": oracle_endpoint.hex(), "oracle_endpoint_ulps": ulps(oracle_endpoint, target),
            "products": [[float(value).hex() for value in products.flat] for products in contractions]})
    return {"schema_version": 1, "scope": "offline authenticated saved-state oracle; no native run",
            "source_commit": SOURCE, "abi_key": identity["abi_key"],
            "native_sha256_from_receipt": identity["native_sha256"],
            "identity_file_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
            "cases": rows, "synthetic": synthetic()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.evidence.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"saved_cases": len(result["cases"]),
        "old_ulps": [row["old_endpoint_ulps"] for row in result["cases"]],
        "oracle_ulps": [row["oracle_endpoint_ulps"] for row in result["cases"]],
        "cross_rank_limitation": result["synthetic"]["cross_rank_limitation"]}))
