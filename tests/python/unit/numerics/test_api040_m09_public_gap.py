"""Source-only discriminator: M09's random dense G is not a local map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[4]


def test_dense_eight_by_four_incidence_is_not_a_condensed_spatial_gradient(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "python"))
    import pops
    from pops.domain import Rectangle
    from pops.frames import Cartesian2D

    frame = Rectangle("m09_gap", (0., 0.), (1., 1.)).frame(Cartesian2D())
    model = pops.Model("m09_dense_gap", frame=frame)
    state = model.state("U", components=("rho", "mx", "my"))
    rotation = model.operator("rotation", returns=model.local_linear_operator(
        "rotation", on=state, matrix=((0., 0., 0.),
                                      (0., 0., 1.),
                                      (0., -1., 0.))))
    case = pops.Case("m09_gap")
    block = case.block("plasma", model=model)
    program = pops.Program("m09_gap")._bind_operators(model.module)
    current = program.state(block[state]).n
    rng = np.random.default_rng(20260928)
    dense_gradient = rng.normal(size=(8, 4))
    with pytest.raises(ValueError, match="spatial subset must have rank 1, 2, or 3"):
        program.condensed_coeffs(state=current, linear_operator=rotation,
                                 subset=tuple(range(8)), c=0.5, th_dt=0.03)
    with pytest.raises((TypeError, ValueError), match="OperatorHandle|operator handle|typed"):
        program.condensed_coeffs(state=current, linear_operator=rotation,
                                 subset=(1, 2), c=0.5, th_dt=0.03,
                                 gradient_map=dense_gradient)
