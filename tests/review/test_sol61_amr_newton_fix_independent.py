"""Exact c100 source reception plus explicit negative source copies; no native."""
import importlib.util
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("amr_sign_independent_fix", Path(__file__).with_name("sol61_amr_original_newton_sign_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
HEADER = ROOT / "include/pops/runtime/program/prepared_amr_field_residual.hpp"
OLD_NAMES = {
    "ThreeFieldsSignedDiffusionGeneralReactionTwoModesAndReflux",
    "IncompatibleRhsAndIndefiniteDiffusionDoNotPublish", "FacBorrowedOwnedLaneDoesNotClaimOwnership",
    "RefusesMissingAndWrongConstantModes", "OriginalAmrNonlinearResidualTwoResolutionsAndPermutation",
    "OriginalAmrResidualAuthorityAndLocalFailureDoNotPublish", "OriginalApplyOnlyFiniteMatrixDoesNotClaimLinearSpdSolve",
    "OriginalFieldOutcomeStagesAllLevelsAndRevalidatesBeforeAccept", "OriginalSignedDiagonalFaceOperatorPreservesSign",
    "OriginalAmrResidualRefusesNoncanonicalStageFractionsBeforePreparation",
}
NEW_NAME = "OriginalIdentityResidualNewtonDirectionArbitraryComponents"


def inventory(*, main=None, fragment=None, cmake=None):
    main = (ROOT / "tests/cpp/unit/elliptic/test_composite_general_field.cpp").read_text() if main is None else main
    fragment = (ROOT / "tests/cpp/unit/elliptic/amr_original_field_residual.inc").read_text() if fragment is None else fragment
    cmake = (ROOT / "tests/CMakeLists.txt").read_text() if cmake is None else cmake
    target = cmake.split('elseif(name STREQUAL "test_composite_general_field")', 1)[1].split("else()", 1)[0]
    if 'DISCOVERY_SOURCES' not in target or 'cpp/unit/elliptic/amr_original_field_residual.inc' not in target:
        raise ValueError("included declarations absent from target discovery")
    if "SOURCES ${ARG_SOURCES} ${ARG_DISCOVERY_SOURCES}" not in cmake:
        raise ValueError("included declarations not passed to real gtest scanner")
    if '#include "amr_original_field_residual.inc"' not in main:
        raise ValueError("fragment not compiled into actual target")
    names = re.findall(r"TEST\(CompositeGeneralField,\s*(\w+)\)", main + fragment)
    if len(names) != 11 or set(names) != OLD_NAMES | {NEW_NAME}:
        raise ValueError("old ten/new identity native inventory changed")
    return {"CompositeGeneralField." + name for name in names}


def test_fixed_adapter_preserves_physical_equation_and_positive_jvp():
    contract = probe.source_contract(ROOT)
    assert contract["defect_sign"] == -1 and contract["jvp_sign"] == 1 and contract["update_sign"] == 1
    assert contract["physical_recheck"] == "F"
    assert probe.receive(ROOT)["status"] == "sign_convention_compatible"


@pytest.mark.parametrize("label", ["remove_defect_negation", "negate_physical_evaluate", "reverse_jvp",
                                   "negate_jvp", "negate_physical_recheck", "only_first_level"])
def test_negative_source_copies_cannot_claim_fixed_sign_contract(label):
    source = HEADER.read_text()
    if label == "remove_defect_negation":
        altered = source.replace("scale(level, Real(-1));", "scale(level, Real(1));", 1)
    elif label == "negate_physical_evaluate":
        altered = source.replace("add_local(q, captures_, result, evaluation);",
                                 "add_local(q, captures_, result, evaluation);\nfor (auto& level : result) scale(level, Real(-1));", 1)
    elif label == "reverse_jvp":
        altered = source.replace("Real(0.5) / h, plus_[level], -Real(0.5) / h, minus_[level]",
                                 "-Real(0.5) / h, plus_[level], Real(0.5) / h, minus_[level]", 1)
    elif label == "negate_jvp":
        altered = source.replace("++derivatives_;", "for (auto& level : result) scale(level, Real(-1));\n++derivatives_;", 1)
    elif label == "negate_physical_recheck":
        altered = source.replace("evaluate(candidate_, recheck_, 0);",
                                 "evaluate(candidate_, recheck_, 0);\nfor (auto& level : recheck_) scale(level, Real(-1));", 1)
    else:
        altered = source.replace("for (auto& level : result)\n          scale(level, Real(-1));",
                                 "scale(result.front(), Real(-1));", 1)
    assert altered != source
    try:
        contract = probe.source_contract(ROOT, negative_source_copies={"amr": altered})
    except ValueError:
        return
    assert contract["defect_sign"] != -1
    assert probe.run_original_constant(contract["defect_sign"])["status"] == "line_search_failed"


def test_exact_eleven_cases_include_every_historical_native_name():
    assert len(inventory()) == 11


@pytest.mark.parametrize("label", ["remove_old_case", "omit_discovery_argument", "omit_scanned_sources", "drop_cpp_include"])
def test_inventory_negative_source_copies_are_refused(label):
    main = (ROOT / "tests/cpp/unit/elliptic/test_composite_general_field.cpp").read_text()
    fragment = (ROOT / "tests/cpp/unit/elliptic/amr_original_field_residual.inc").read_text()
    cmake = (ROOT / "tests/CMakeLists.txt").read_text()
    if label == "remove_old_case":
        fragment = fragment.replace("TEST(CompositeGeneralField, OriginalFieldOutcomeStagesAllLevelsAndRevalidatesBeforeAccept)",
                                    "TEST(ForeignSuite, OriginalFieldOutcomeStagesAllLevelsAndRevalidatesBeforeAccept)", 1)
    elif label == "omit_discovery_argument":
        cmake = cmake.replace("pops_add_gtest_suite(NAME ${name} DISCOVERY_SOURCES", "pops_add_gtest_suite(NAME ${name}", 1)
    elif label == "omit_scanned_sources":
        cmake = cmake.replace("SOURCES ${ARG_SOURCES} ${ARG_DISCOVERY_SOURCES}", "SOURCES ${ARG_SOURCES}", 1)
    else:
        main = main.replace('#include "amr_original_field_residual.inc"', "", 1)
    with pytest.raises(ValueError):
        inventory(main=main, fragment=fragment, cmake=cmake)


def test_real_cmake_scanner_registers_exact_eleven_from_actual_source_paths(tmp_path):
    expected = inventory()
    main = ROOT / "tests/cpp/unit/elliptic/test_composite_general_field.cpp"
    fragment = main.with_name("amr_original_field_residual.inc")
    (tmp_path / "CMakeLists.txt").write_text('''cmake_minimum_required(VERSION 3.20)
project(IndependentInventory LANGUAGES NONE)
enable_testing()
include(GoogleTest)
add_executable(probe IMPORTED)
set_target_properties(probe PROPERTIES IMPORTED_LOCATION "/never-executed/independent")
gtest_add_tests(TARGET probe SOURCES "%s" "%s" TEST_LIST cases)
file(WRITE "${CMAKE_BINARY_DIR}/cases.txt" "${cases}")
''' % (main, fragment))
    result = subprocess.run(["cmake", "-S", str(tmp_path), "-B", str(tmp_path / "inventory")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert set((tmp_path / "inventory/cases.txt").read_text().split(";")) == expected
