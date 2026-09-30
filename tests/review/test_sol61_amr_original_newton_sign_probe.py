"""Independent algebra/partition probes; no PoPS/native imports or build."""
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("original_amr_sign_probe", Path(__file__).with_name("sol61_amr_original_newton_sign_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_actual_source_connects_rhs_and_additive_update_to_original_positive_jvp():
    contract = probe.source_contract(ROOT)
    assert contract["jvp_sign"] == contract["update_sign"] == 1
    assert contract["physical_recheck"] == "F"
    assert set(contract["source_sha256"]) == {"amr", "workspace", "uniform", "fixture"}
    assert contract["defect_sign"] in (-1, 1)


@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_real_constant_cubic_equation_wrong_rhs_fails_every_original_armijo_step(permutation):
    report = probe.run_original_constant(1, permutation)
    assert report["status"] == "line_search_failed" and report["iterations"] == 1
    assert report["solution"] == [0., 0., 0.] and len(report["trials"]) == 20
    assert all(row["physical_norm"] > report["initial"] > row["armijo"] for row in report["trials"])
    assert probe.exact_bad_direction()["all_components_strictly_uphill_for_every_positive_step"]


@pytest.mark.parametrize("permutation", [(0, 1, 2), (2, 0, 1)])
def test_correct_defect_convention_solves_original_equations_without_budget_relaxation(permutation):
    report = probe.run_original_constant(-1, permutation)
    assert report["status"] == "converged" and report["iterations"] <= 12
    assert report["physical_residual"] <= 2.e-9 * max(1., report["initial"])
    assert max(abs(a - b) for a, b in zip(report["solution"], report["target"], strict=True)) < 1.e-9
    assert all(row["physical_norm"] <= row["armijo"] for row in report["trials"])


@pytest.mark.parametrize("cells", [16, 32])
@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("replicated", [False, True])
def test_actual_two_level_cell_measures_ownership_and_coverage_do_not_reverse_sign(cells, ranks, replicated):
    vector = (.15, .25, .18)
    partials, squared = probe.hierarchy_squared(vector, cells, ranks, replicated=replicated)
    assert math.isclose(squared, math.fsum(x*x for x in vector), rel_tol=0, abs_tol=5.e-16)
    if replicated and ranks == 2:
        assert partials[1] == 0.
    elif ranks == 2:
        assert math.isclose(partials[0], partials[1], rel_tol=0, abs_tol=5.e-16)


@pytest.mark.parametrize("target", [.15, -.4, .7])
def test_scalar_original_equation_finite_difference_has_positive_jacobian_and_uphill_wrong_direction(target):
    # Exact same central-difference orientation as PreparedAmrFieldResidual.
    q, h, v = 0., 1.e-5, 1.
    def physical(value):
        return value - target
    jvp = (physical(q + h*v) - physical(q - h*v)) / (2*h)
    assert jvp > 0
    wrong_delta = physical(q) / jvp
    step = 1.
    while step >= 1.e-6:
        assert abs(physical(q + step*wrong_delta)) > (1 - 1.e-4*step) * abs(physical(q))
        step *= .5


def test_inventory_source_contains_six_extra_included_declarations():
    import re
    main = (ROOT / "tests/cpp/unit/elliptic/test_composite_general_field.cpp").read_text()
    extra = (ROOT / "tests/cpp/unit/elliptic/amr_original_field_residual.inc").read_text()
    pattern = r"TEST\(CompositeGeneralField,\s*(\w+)\)"
    # The historical diagnosis had six included declarations; subsequent native
    # regressions can add cases without falsifying that historical observation.
    assert len(re.findall(pattern, main)) == 4 and len(re.findall(pattern, extra)) >= 6
    assert '#include "amr_original_field_residual.inc"' in main
    assert "OriginalAmrNonlinearResidualTwoResolutionsAndPermutation" in extra
    assert "OriginalFieldOutcomeStagesAllLevelsAndRevalidatesBeforeAccept" in extra
