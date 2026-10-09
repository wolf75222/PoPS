"""Independent reader of sealed execution outputs. Never executes/imports the author."""

import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import verify_capture, write_json
from .reference import VARIANTS, check_arrays, check_pairs, require


def check_campaign(root):
    root = Path(root)
    arrays, receipts, reports = {}, {}, {}
    for variant, physics in VARIANTS.items():
        directory = root / variant
        receipts[variant] = verify_capture(directory)
        require(receipts[variant]["variant"] == variant, "wrong execution variant")
        require(
            receipts[variant]["physics"] == {"epsilon": physics.epsilon, "beta": physics.beta},
            "wrong executed parameters",
        )
        with np.load(directory / "actual-arrays.npz", allow_pickle=False) as data:
            arrays[variant] = {name: data[name].copy() for name in data.files}
        with np.load(directory / "actual-bind-inputs.npz", allow_pickle=False) as data:
            require(
                np.array_equal(arrays[variant]["q0"], data["receiver"]),
                "history is not actual bound q0",
            )
            require(
                np.array_equal(arrays[variant]["d_initial"], data["donor"]),
                "donor is not actual bound input",
            )
        reports[variant] = check_arrays(arrays[variant], physics)
    first = receipts["baseline"]["core_before"]
    require(
        all(r["core_before"] == first for r in receipts.values()), "core differs across variants"
    )
    for changed in ("linear", "cubic"):
        predecessor = "baseline" if changed == "linear" else "linear"
        require(
            receipts[changed]["compiler_inputs_digest"]
            != receipts[predecessor]["compiler_inputs_digest"],
            "formula change did not change compiler inputs",
        )

        def binaries(key):
            return {r["sha256"] for r in receipts[key]["loaded_binaries"] if r["role"] != "core"}

        require(
            binaries(changed) != binaries(predecessor), "formula change did not change loaded code"
        )
    return {
        "variants": reports,
        "pairs": check_pairs(arrays),
        "core_unchanged": True,
        "scope": "Independent discrete mathematical acceptance + integrity relative to execution records.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = check_campaign(args.campaign)
    write_json(args.report, result)
    print(json.dumps(result["pairs"], indent=2))


if __name__ == "__main__":
    main()
