"""Observe T12 native interface receipts against independent scalar formulas.

This module never advances a fine grid. Its fine face integrals come from the
real accepted ledger; they reconstruct the reflux correction to uncovered
coarse cells. The coarse predictor and face-selection oracle use NumPy only.
"""

from fractions import Fraction

import numpy as np


def face_selection(old, *, duration, spacing):
    """Faces are indexed by the cell on their right, with periodic x indexing."""
    values = np.asarray(old, dtype=float)
    assert values.ndim == 2 and np.isfinite(values).all()
    assert 0 < duration / spacing <= 0.25
    left = np.roll(values, 1, axis=1)
    high = 0.5 * (left + values) - 0.5 * duration / spacing * (values - left)
    provisional = values - duration / spacing * (np.roll(high, -1, axis=1) - high)
    troubled = (provisional < 0) | np.roll(provisional < 0, 1, axis=1)
    return high, np.where(troubled, left, high), troubled


def covered_cells(boxes, *, cells):
    nx, ny = cells
    covered = np.zeros((ny, nx), dtype=bool)
    fine_boxes = 0
    for level, lower, upper in boxes:
        if level == 0:
            continue
        assert level == 1
        assert all(lo % 2 == 0 and (hi + 1) % 2 == 0 for lo, hi in zip(lower, upper, strict=True))
        x0, y0 = (int(v) // 2 for v in lower)
        x1, y1 = ((int(v) + 1) // 2 for v in upper)
        assert 0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny
        assert not covered[y0:y1, x0:x1].any()
        covered[y0:y1, x0:x1] = True
        fine_boxes += 1
    assert fine_boxes and covered.any() and (~covered).any()
    return covered


def inspect_selection(old, actual_coarse, boxes, entries, *, cells, duration, selected):
    """Check each x-interface and its actual accepted coarse-cell consequence."""
    nx, ny = cells
    old = np.asarray(old).reshape(ny, nx)
    actual = np.asarray(actual_coarse).reshape(ny, nx)
    high, corrected, troubled = face_selection(old, duration=duration, spacing=1 / nx)
    chosen = corrected if selected else high
    expected = old - duration * nx * (np.roll(chosen, -1, axis=1) - chosen)
    covered = covered_cells(boxes, cells=cells)
    edge = covered != np.roll(covered, 1, axis=1)
    edge_faces = {(int(x), int(y)) for y, x in zip(*np.nonzero(edge), strict=True)}
    assert edge_faces
    coarse, fine = {}, {}
    for row in entries:
        assert row["levels"] == (0, 1) and row["macro"] == 0
        assert row["weight"] == 1 and row["affine"] is None
        assert row["payload"].shape == (1,) and np.isfinite(row["payload"]).all()
        if row["axis"] == 1:
            np.testing.assert_array_equal(row["payload"], np.zeros(1))
            continue
        assert row["axis"] == 0 and row["coarse_face"] in edge_faces
        key = row["coarse_face"]
        if row["level"] == 0:
            assert key not in coarse and row["face"] == key
            assert row["phase"] == 0 and row["evaluation"] == 0
            assert row["begin"] == 0 and row["end"] == 1
            assert row["duration"] == duration and row["area"] == 1 / ny
            coarse[key] = row
        else:
            assert row["level"] == 1
            assert row["phase"] == row["begin"] in (0, Fraction(1, 2))
            assert row["end"] == row["begin"] + Fraction(1, 2)
            assert row["evaluation"] == float(row["begin"]) * duration
            assert row["duration"] == duration / 2 and row["area"] == 1 / (2 * ny)
            assert row["face"][0] == 2 * key[0]
            assert row["face"][1] in (2 * key[1], 2 * key[1] + 1)
            fine.setdefault(key, []).append(row)
    assert set(coarse) == set(fine) == edge_faces
    changes, integrated, largest_reflux = [], {}, 0.0
    for (x, y), coarse_row in coarse.items():
        np.testing.assert_allclose(coarse_row["payload"][0], chosen[y, x], atol=3e-15, rtol=0)
        if troubled[y, x]:
            changes.append(abs(corrected[y, x] - high[y, x]))
        rows = fine[x, y]
        assert len(rows) == 4
        assert {(row["face"][1] % 2, row["begin"]) for row in rows} == {
            (0, Fraction(0)),
            (1, Fraction(0)),
            (0, Fraction(1, 2)),
            (1, Fraction(1, 2)),
        }
        fine_measure = sum(row["duration"] * row["area"] for row in rows)
        assert fine_measure == duration / ny
        integral = sum(row["duration"] * row["area"] * row["payload"][0] for row in rows)
        integrated[x, y] = float(integral)
        coarse_integral = duration * chosen[y, x] / ny
        correction = (integral - coarse_integral) * nx * ny
        largest_reflux = max(largest_reflux, abs(correction))
        # Replace the coarse outgoing measure by the sum of its accepted fine
        # space/time fragments, only on the uncovered incident coarse cell.
        if not covered[y, (x - 1) % nx]:
            expected[y, (x - 1) % nx] -= correction
        if not covered[y, x]:
            expected[y, x] += correction
    assert changes and max(changes) > 1e-4, "no genuinely selected AMR interface face"
    assert largest_reflux > 1e-6, "zero reflux cannot test selected face publication"
    error = float(np.max(abs(actual[~covered] - expected[~covered])))
    np.testing.assert_allclose(actual[~covered], expected[~covered], atol=2e-13, rtol=0)
    return {
        "interface_faces": len(coarse),
        "selected_interface_faces": len(changes),
        "largest_selection": float(max(changes)),
        "largest_reflux": float(largest_reflux),
        "uncovered_state_error": error,
        "fine_integrals": integrated,
        "uncovered": ~covered,
    }
