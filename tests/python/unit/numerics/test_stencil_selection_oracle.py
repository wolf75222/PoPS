"""Discriminating controls for the observer of real T12 interface measures."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from tests.python.support.stencil_selection_oracle import face_selection, inspect_selection


def _observations(selected=True):
    nx, ny, duration = 32, 4, 0.2 / 32
    old = np.full((ny, nx), 0.15)
    old[:, 8:14] = [0.15, 0.06, 0.005, 1.0, 0.65, 0.3]
    boxes = [(0, (0, 0), (31, 3)), (1, (20, 0), (27, 7))]
    high, corrected, _ = face_selection(old, duration=duration, spacing=1 / nx)
    chosen = corrected if selected else high
    # Separate scalar loop for this manufactured observer fixture. This does
    # not stand in for native execution: only the receiver invokes real PoPS.
    actual = old.copy()
    for y in range(ny):
        for x in range(nx):
            actual[y, x] -= duration * nx * (chosen[y, (x + 1) % nx] - chosen[y, x])
    entries = []
    for x in (10, 14):
        for y in range(ny):
            row = dict(
                levels=(0, 1),
                macro=0,
                weight=Fraction(1),
                affine=None,
                axis=0,
                coarse_face=(x, y),
                face=(x, y),
                level=0,
                phase=Fraction(0),
                evaluation=0.0,
                begin=Fraction(0),
                end=Fraction(1),
                duration=duration,
                area=1 / ny,
                payload=np.array([chosen[y, x]]),
            )
            entries.append(row)
            # Four explicitly known face measures: two transverse children
            # times two substeps, with a deliberately nonzero reflux witness.
            for child in range(2):
                for begin in (Fraction(0), Fraction(1, 2)):
                    entries.append(
                        dict(
                            row,
                            face=(2 * x, 2 * y + child),
                            level=1,
                            phase=begin,
                            evaluation=float(begin) * duration,
                            begin=begin,
                            end=begin + Fraction(1, 2),
                            duration=duration / 2,
                            area=1 / (2 * ny),
                            payload=np.array([chosen[y, x] + 0.01]),
                        )
                    )
            actual[y, x - 1 if x == 10 else x] += (-1 if x == 10 else 1) * duration * nx * 0.01
    return old, actual, boxes, entries, dict(cells=(nx, ny), duration=duration, selected=selected)


@pytest.mark.parametrize("selected", (False, True))
def test_observer_checks_nonzero_selection_and_two_substep_reflux(selected):
    old, actual, boxes, entries, options = _observations(selected)
    row = inspect_selection(old, actual, boxes, entries, **options)
    assert row["selected_interface_faces"] == 4
    assert row["largest_selection"] == pytest.approx(0.022, abs=1e-16)
    assert row["largest_reflux"] > 0.001


@pytest.mark.parametrize("missing", ("selection", "reflux", "fine_substep", "interface"))
def test_observer_rejects_missing_native_contracts(missing):
    old, actual, boxes, entries, options = _observations()
    if missing == "selection":
        high, _, _ = face_selection(old, duration=options["duration"], spacing=1 / 32)
        entries = deepcopy(entries)
        for row in entries:
            if row["level"] == 0:
                x, y = row["face"]
                row["payload"][0] = high[y, x]
    elif missing == "reflux":
        actual[:, 9] += 0.002
    elif missing == "fine_substep":
        entries = [row for row in entries if row["level"] == 0 or row["begin"] == 0]
    else:
        boxes = [(0, (0, 0), (31, 3)), (1, (0, 0), (63, 7))]
    with pytest.raises(AssertionError):
        inspect_selection(old, actual, boxes, entries, **options)
