"""Source admission and synthetic stencil discriminants, never Native evidence."""
import math
import operator

import numpy as np
import pops
import pytest

from tests.python.support.evolved_stage_amr_spatial import (
    ACCEPTANCE, CONTROLS, DENSE_BYTES, DT, FD_STEP, build, composite_flux,
    initial_cell_average, metric_arrays, strip_geometry,
)
from tests.python.support.evolved_stage_mms import DIFFUSION, accumulation


def masks(cells=8):
    covered = np.zeros(cells, dtype=bool)
    covered[cells//4:3*cells//4] = True
    fine = np.repeat(covered, 2)
    return covered, fine


def synthetic_geometry(cells=8):
    covered, fine = masks(cells)
    rows = []
    for level, (valid, active) in enumerate(((np.ones(cells, dtype=bool), ~covered), (fine, fine))):
        n = cells*2**level
        valid = np.tile(valid, (n, 1))
        row = metric_arrays(cells, level, valid)
        row["active"] = np.tile(active, (n, 1))
        row["native_base_shape"] = np.array([cells, cells], dtype=np.int64)
        row["native_patch_boxes"] = np.array([[0, 0, 0, cells-1, cells-1],
            [1, cells//2, 0, 3*cells//2-1, 2*cells-1]], dtype=np.int64)
        rows.append(row)
    return rows


def evaluate_integral(expression, frame, bounds):
    from pops.runtime._analytic_expression_lowering import lower_analytic_components
    ((ops, values),) = lower_analytic_components([expression.to_data()], frame_id=frame.canonical_id)
    inputs = [*bounds, math.prod(bounds[2*a+1]-bounds[2*a] for a in range(2))]
    stack = []
    for op, value in zip(ops, values, strict=True):
        if op == "constant":
            stack.append(value)
        elif op == "input":
            stack.append(inputs[int(value)])
        elif op in ("sin", "cos", "neg"):
            a = stack.pop()
            stack.append(-a if op == "neg" else getattr(math, op)(a))
        else:
            b, a = stack.pop(), stack.pop()
            stack.append({"add":operator.add, "sub":operator.sub, "mul":operator.mul,
                "div":operator.truediv, "pow":operator.pow}[op](a, b))
    assert len(stack) == 1
    return stack[0]


def test_exact_native_initial_integrals_match_independent_quadrature_and_global_amounts():
    case, _ = build()
    initials = tuple(case.initials)
    assert len(initials) == 3 and all(row.value.cell_integrals is not None for row in initials)
    for n in (8, 16):
        expected = initial_cell_average(n)
        for index, initial in enumerate(initials[:2]):
            frame = initial.value.frame
            actual = np.array([evaluate_integral(initial.value.cell_integrals[0], frame,
                [cell/n, (cell+1)/n, 0., 1./n])*n*n for cell in range(n)])
            np.testing.assert_allclose(actual, expected[index], atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(initial_cell_average(1)[:, 0], [.17896125, .3201125], atol=2e-15)


def test_real_public_nonconstant_case_resolve_and_emit_without_compilation():
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.codegen.program_models import ProgramModelGraph
    case, layout = build()
    resolved = pops.resolve(pops.validate(case), layout=layout)
    cpp = emit_cpp_program(resolved.time, model=ProgramModelGraph.from_resolved_blocks(resolved.blocks), target="amr_system")
    assert resolved.time._serialize()["version"] == 16
    assert "AmrFieldRightPreconditioner::kFullResidualBasisLU" in cpp
    assert "PreparedAmrFieldResidual" in cpp
    assert cpp.count("ctx.store_global_field_history(") == 6
    assert cpp.count("evolved_accumulation") >= 2
    assert DENSE_BYTES == 256*1024**2 and DT == .01 and ACCEPTANCE == 3e-8
    from tests.python.support.captured_diffusion_mms import CONTROLS as original_controls, FD_STEP as original_fd
    assert CONTROLS == original_controls and FD_STEP == original_fd


@pytest.mark.parametrize("cells", (8, 16))
def test_signed_composite_flux_nonzero_conservative_and_restriction_noncommuting(cells):
    covered, valid = masks(cells)
    def temperatures(n):
        x = (np.arange(n)+.5)/n
        return np.stack((.15+.02*np.cos(2*np.pi*x), .25+.015*np.sin(2*np.pi*x)))
    coarse, fine = temperatures(cells), temperatures(2*cells)
    divergence, faces, interface, restricted = composite_flux((coarse, fine), covered, valid)
    assert interface.sum() == 2 and DIFFUSION[1, 0] < 0
    # Independent face telescoping over the actual quotient; no assumed SPD.
    amount = divergence[0][:, ~covered].sum(axis=1)/cells+divergence[1][:, valid].sum(axis=1)/(2*cells)
    np.testing.assert_allclose(amount, 0., atol=2e-17)
    np.testing.assert_array_equal(faces[0][:, interface], faces[1][:, 2*np.flatnonzero(interface)])
    gap = accumulation(fine).reshape(2, cells, 2).mean(axis=2)-accumulation(restricted)
    variance = (fine.reshape(2, cells, 2)-restricted[:, :, None])**2
    covariance = ((fine[0].reshape(cells, 2)-restricted[0, :, None])
        *(fine[1].reshape(cells, 2)-restricted[1, :, None])).mean(axis=1)
    expected = np.stack((variance[0].mean(axis=1)+.1*variance[1].mean(axis=1),
        variance[1].mean(axis=1)+.2*covariance))
    np.testing.assert_allclose(gap[:, covered], expected[:, covered], atol=7e-17)
    assert np.max(np.abs(gap[:, covered])) > 1e-10
    assert max(np.max(np.abs(divergence[0][:, ~covered])), np.max(np.abs(divergence[1][:, valid]))) > 1e-6
    # Skipping interface replacement violates closed composite conservation.
    naive_faces = DIFFUSION @ ((restricted-np.roll(restricted, 1, axis=1))*cells)
    naive_divergence = (np.roll(naive_faces, -1, axis=1)-naive_faces)*cells
    naive_amount = naive_divergence[:, ~covered].sum(axis=1)/cells+divergence[1][:, valid].sum(axis=1)/(2*cells)
    assert np.max(np.abs(naive_amount)) > 1e-6


def test_constant_tower_is_only_a_zero_flux_reference_control_not_native_qualification():
    covered, valid = masks()
    d, _, _, _ = composite_flux((np.full((2, 8), .2), np.full((2, 16), .2)), covered, valid)
    np.testing.assert_allclose(d[0][:, ~covered], 0., atol=1e-16)
    np.testing.assert_allclose(d[1][:, valid], 0., atol=1e-16)


@pytest.mark.parametrize("mutation", ("hole_y", "all_covered", "ratio", "active_outside", "volume", "kappa", "edges", "boxes", "shape"))
def test_reference_refuses_foreign_geometry_instead_of_falling_back_to_homogeneous(mutation):
    rows = synthetic_geometry()
    strip_geometry(rows, 8)
    if mutation == "hole_y":
        rows[1]["valid"][1, 4] = False
    elif mutation == "all_covered":
        rows[0]["active"][:] = False
    elif mutation == "ratio":
        rows[1]["valid"][:, 3] = True
    elif mutation == "active_outside":
        rows[1]["active"][:, 0] = True
    elif mutation == "volume":
        rows[0]["cartesian_cell_volume"][0, 0] *= 2
    elif mutation == "kappa":
        rows[0]["declared_no_EB_kappa"][0, 0] = .5
    elif mutation == "edges":
        rows[0]["x_edges"][1] += .01
    elif mutation == "boxes":
        rows[0]["native_patch_boxes"][1, 1] += 1
    else:
        rows[0]["native_base_shape"][0] += 1
    with pytest.raises((ValueError, AssertionError)):
        strip_geometry(rows, 8)
