"""Actual public AMR declaration/resolve/emission, no built runtime."""

import pytest
import numpy as np
from tests.python.support.m19_amr_consumed_case import build, initial, NAMES
from tests.review.sol61_m19_amr_consumed_offline import (
    potential,
    mapped,
    remap,
    verify_step,
    DT,
)


@pytest.mark.parametrize("reverse", (False, True))
def test_amr_field_public_resolve_and_emit(tmp_path, reverse):
    from pops.codegen.program_slicing import slice_program
    from pops.time._program.detach import detach_compiled_program
    from pops.codegen.program_models import ProgramModelGraph
    from pops.codegen.program_graph_lowering import emit_program_graph

    resolved = build(tmp_path, reverse=reverse)
    assignments = {
        r.subject.local_id: r.layout
        for r in resolved.layout_plan.assignments
        if r.subject_kind == "block"
    }
    sources = []
    for layout in resolved.layout_plan.layouts:
        blocks = tuple(b for b in resolved.blocks if assignments[b.name] == layout.handle)
        p = detach_compiled_program(slice_program(resolved.time, tuple(b.name for b in blocks)))
        sources.append(
            emit_program_graph(
                p.to_graph(),
                lowering_program=p,
                model_graph=ProgramModelGraph.from_resolved_blocks(blocks),
                target="amr_system",
                field_plans={},
            )
        )
    assert len(resolved.layout_plan.mappings) == 2
    assert sum(s.count("stage_field_components(") for s in sources) == 2
    assert any("mapped_field_" in s for s in sources)
    packages = tuple(tmp_path.glob("*/amr_fault.cpp"))
    assert len(packages) == 2 and all(
        "quiet_NaN" in p.read_text() and "apply_physical_support_integral" in p.read_text()
        for p in packages
    )


def test_oracle_discriminates_nonconstant_wrong_field_and_axis():
    values = initial()
    phi = potential(values[NAMES[1]])
    assert np.ptp(phi) > 1e-3 and not np.allclose(phi, values[NAMES[1]][1], rtol=0, atol=1e-5)
    assert np.ptp(mapped(phi, 0, 6)) > 1e-3 and np.max(abs(mapped(phi, 1, 6))) > 1e-4
    np.testing.assert_allclose(remap(phi, 8, 6).mean(), phi.mean(), rtol=0, atol=1e-15)
    after = {
        name: np.stack([remap(c, 6, 1) for c in values[name]]) for name in (NAMES[0], NAMES[2])
    }
    for i, name in enumerate((NAMES[0], NAMES[2])):
        after[name][1, :, 0] += (i + 1) * DT * mapped(phi, i, 6)
    after[NAMES[1]] = np.stack([remap(c, 8, 6) for c in values[NAMES[1]]])
    after[NAMES[1]][1] += 32 * DT
    assert verify_step(values, after)["mapped_span"] > 1e-3
    for corruption in ("constant", "unsolved", "wrong_axis", "stale", "wrong_component", "nan"):
        wrong = {name: value.copy() for name, value in after.items()}
        field = {
            "constant": np.full_like(phi, phi.mean()),
            "unsolved": values[NAMES[1]][1],
            "wrong_axis": phi.T,
            "stale": phi - 0.5,
        }.get(corruption)
        if field is not None:
            wrong[NAMES[0]][1, :, 0] = values[NAMES[0]][1, 0, 0] + DT * mapped(field, 0, 6)
        elif corruption == "nan":
            wrong[NAMES[0]][1, 0, 0] = np.nan
        else:
            wrong[NAMES[0]][0, 0, 0] += 1
        with pytest.raises(ValueError):
            verify_step(values, wrong)


@pytest.mark.parametrize("shape", ((4, 3), (8, 6)))
def test_periodic_stencil_has_independent_manufactured_solution(shape):
    ny, nx = shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    phi = 2 + 0.1 * np.cos(2 * np.pi * x)[None, :] + 0.07 * np.cos(2 * np.pi * y)[:, None]
    lap = nx * nx * (np.roll(phi, -1, axis=1) - 2 * phi + np.roll(phi, 1, axis=1))
    lap += ny * ny * (np.roll(phi, -1, axis=0) - 2 * phi + np.roll(phi, 1, axis=0))
    load = np.stack((np.full(shape, -19.0), phi - 0.125 * lap))
    np.testing.assert_allclose(potential(load), phi, rtol=0, atol=4e-15)
    load[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid saved source"):
        potential(load)
