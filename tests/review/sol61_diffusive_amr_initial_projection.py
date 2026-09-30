"""Audit actual saved AMR diffusion evidence without importing PoPS or running native code."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def audit(directory):
    directory = Path(directory).resolve()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from tests.python.support.integral_state_receipts import checkpoint_exchange_images

    receipt = json.loads((directory / "accepted-receipt.json").read_text())
    checkpoint_sha256 = hashlib.sha256((directory / "accepted.npz").read_bytes()).hexdigest()
    assert checkpoint_sha256 == receipt["checkpoint_sha256"]
    images, decoded = checkpoint_exchange_images(directory / "accepted.npz")
    assert [hashlib.sha256(image).hexdigest() for image in images] == receipt["ledger_sha256"]
    assert json.dumps(decoded, sort_keys=True) == json.dumps(receipt["ledgers"], sort_keys=True)
    assert receipt["dimension"] == 2 and receipt["time"] == 1e-4 and receipt["macro_step"] == 1
    with np.load(directory / "initial.npz", allow_pickle=False) as archive:
        boxes = archive["patch_boxes"].copy()
    valid = np.zeros((64, 64), dtype=bool)
    coarse_active = np.ones((32, 32), dtype=bool)
    for level, x0, y0, x1, y1 in boxes:
        if level != 1:
            continue
        assert x0 % 2 == y0 % 2 == (x1 + 1) % 2 == (y1 + 1) % 2 == 0
        valid[y0:y1 + 1, x0:x1 + 1] = True
        coarse_active[y0 // 2:(y1 + 1) // 2, x0 // 2:(x1 + 1) // 2] = False
    assert valid[:, -1].all() and coarse_active[:, 0].all() and not coarse_active[:, -1].any()
    levels = {}
    for phase in ("initial", "accepted"):
        with np.load(directory / f"{phase}-state.npz", allow_pickle=False) as archive:
            levels[phase] = (archive["level_0"].reshape(32, 32).copy(),
                             archive["level_1"].reshape(64, 64).copy())
    coarse, fine = levels["initial"]
    affine_coarse = np.broadcast_to(1 + (np.arange(32) + .5) / 32, (32, 32))
    affine_fine = np.broadcast_to(1 + (np.arange(64) + .5) / 64, (64, 64))
    np.testing.assert_array_equal(coarse, affine_coarse)
    restricted = fine.reshape(32, 2, 32, 2).mean(axis=(1, 3))
    np.testing.assert_allclose(restricted[~coarse_active], coarse[~coarse_active], rtol=0, atol=2e-14)
    projection_error = float(np.max(np.abs(fine[valid] - affine_fine[valid])))
    interior = valid.copy()
    interior[:, -2:] = False
    np.testing.assert_allclose(fine[interior], affine_fine[interior], rtol=0, atol=2e-14)
    groups = {}
    for ledger in decoded:
        for row in ledger["records"]:
            if not row["exterior"]:
                continue
            frame = row["context"].removeprefix("pops.exchange.frame.v1/")
            length, suffix = frame.split(":", 1)
            fields = suffix[int(length) + 1:].split("/")
            key = (row["axis"], row["side"], int(fields[1]), int(fields[2]))
            groups.setdefault(key, []).append(row)
    assert set(groups) == {(0, 0, 0, 0), (0, 1, 1, 0), (0, 1, 1, 1)}
    summaries = []
    total = 0.
    right = 0.
    for key, rows in sorted(groups.items()):
        axis, side, level, substep = key
        n, dt = (64, 5e-5) if level else (32, 1e-4)
        assert len(rows) == n
        assert len({(r["operation"], r["occurrence"], r["context"], r["quadrature"])
                    for r in rows}) == n
        assert all(r["measure"] == 1 / n and r["weight"] == dt and
                   r["orientation"] == (1 if side else -1) and r["multiplicity"] == 1
                   for r in rows)
        amount = sum(r["orientation"] * r["measure"] * r["flux"] * r["weight"] for r in rows)
        summaries.append({"axis": axis, "side": side, "level": level, "substep": substep,
                          "count": n, "fluxes": sorted({r["flux"] for r in rows}), "amount": amount})
        total += amount
        if side:
            right += amount
    initial_mass = float(coarse[coarse_active].sum() / 32**2 + fine[valid].sum() / 64**2)
    final_coarse, final_fine = levels["accepted"]
    final_mass = float(final_coarse[coarse_active].sum() / 32**2 + final_fine[valid].sum() / 64**2)
    np.testing.assert_allclose(final_mass - initial_mass, total, rtol=0, atol=2e-14)
    np.testing.assert_allclose(receipt["quantity"] - .7, right, rtol=0, atol=2e-14)
    # Independent local Forward Euler recurrence at the physical right face, from saved input.
    dt, dx, diffusivity = 5e-5, 1 / 64, .1
    last, previous = fine[0, -1], fine[0, -2]
    flux0 = diffusivity * 2 * (2 - last) / dx
    last_after_one = last + dt * diffusivity * (previous - 3 * last + 4) / dx**2
    flux1 = diffusivity * 2 * (2 - last_after_one) / dx
    np.testing.assert_allclose(summaries[1]["fluxes"], [flux0], rtol=0, atol=2e-14)
    np.testing.assert_allclose(summaries[2]["fluxes"], [flux1], rtol=0, atol=2e-14)
    np.testing.assert_allclose(right, dt * (flux0 + flux1), rtol=0, atol=2e-14)
    return {"scope": "saved native evidence, NumPy and exact POPSEX02 decoding; no native execution",
            "artifact": receipt["artifact"], "abi": receipt["platform"]["abi"]["value"],
            "checkpoint_sha256": checkpoint_sha256,
            "ledger_sha256_by_rank": receipt["ledger_sha256"],
            "patch_boxes": boxes.tolist(), "fine_valid_cells": int(valid.sum()),
            "coarse_active_cells": int(coarse_active.sum()), "fine_affine_max_error": projection_error,
            "initial_right_pair": fine[0, -2:].tolist(),
            "expected_right_pair": affine_fine[0, -2:].tolist(), "external_groups": summaries,
            "initial_composite_mass": initial_mass, "final_composite_mass": final_mass,
            "mass_delta": final_mass - initial_mass, "ledger_total": total,
            "right_delta": right, "independent_right_fluxes": [flux0, flux1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-affine", action="store_true")
    args = parser.parse_args()
    result = audit(args.directory)
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.require_affine:
        assert result["fine_affine_max_error"] < 2e-14
