"""Prepared field protocol cores must not interpret any ready backend family."""

from __future__ import annotations

from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[3]
CORE_PROTOCOLS = (
    ROOT / "python/pops/fields/_prepared_field_lowering_registry.py",
    ROOT / "python/pops/fields/_prepared_field_solver_registry.py",
    ROOT / "python/pops/fields/_prepared_field_nullspace_registry.py",
)
GENERIC_CONSUMERS = (
    ROOT / "python/pops/codegen/field_install.py",
    ROOT / "python/pops/codegen/field_install_plan.py",
    ROOT / "python/pops/runtime/_system_unified_install.py",
    ROOT / "python/pops/runtime/_amr_system_install.py",
)
GENERIC_NATIVE_NULLSPACE_PROTOCOLS = (
    ROOT / "include/pops/numerics/elliptic/interface/field_nullspace_provider.hpp",
    ROOT / "include/pops/numerics/elliptic/interface/field_nullspace_builtins.hpp",
    ROOT / "include/pops/numerics/elliptic/interface/field_nullspace_prepare.hpp",
)
CONCRETE_PATTERNS = (
    r"\bgeometric[_-]?mg\b",
    r"\bcomposite[_-]?fac\b",
    r"\bfac(?:_options)?\b",
    r"\bfft\b",
    r"\bconstant\b",
    "connected-constant",
    "operator-topology-derived",
    "mean-value",
    "mean-zero",
)


@pytest.mark.parametrize("path", CORE_PROTOCOLS, ids=lambda path: path.stem)
def test_prepared_field_protocol_core_has_no_concrete_provider_dispatch(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8").casefold()
    assert not {pattern for pattern in CONCRETE_PATTERNS if re.search(pattern, source)}


@pytest.mark.parametrize("path", GENERIC_CONSUMERS, ids=lambda path: path.stem)
def test_field_plan_consumers_have_no_backend_name_dispatch(path: Path) -> None:
    source = path.read_text(encoding="utf-8").casefold()
    backend_patterns = CONCRETE_PATTERNS[:4] + CONCRETE_PATTERNS[6:]
    assert not {pattern for pattern in backend_patterns if re.search(pattern, source)}


@pytest.mark.parametrize(
    "path",
    (
        ROOT / "python/pops/fields/_prepared_field_lowering_registry.py",
        ROOT / "python/pops/codegen/field_install.py",
        ROOT / "python/pops/codegen/field_install_plan.py",
    ),
    ids=lambda path: path.stem,
)
def test_field_lowering_core_has_no_concrete_operator_layout_or_target_dispatch(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8").casefold()
    concrete = (
        r"\blaplacian\b",
        r"\breaction\b",
        r"\bcartesian\b",
        r"\bamr_system\b",
        r"\bfieldoutput\b",
        r"\bgradientoutput\b",
        r"lower_field_method",
    )
    assert not {pattern for pattern in concrete if re.search(pattern, source)}


@pytest.mark.parametrize(
    "path",
    (
        ROOT / "python/pops/runtime/_system_unified_install.py",
        ROOT / "python/pops/runtime/_amr_system_install.py",
    ),
    ids=lambda path: path.stem,
)
def test_runtime_adapters_do_not_interpret_method_owned_install_records(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8").casefold()
    concrete = (
        r"\breaction\b",
        r"\bgradient_sign\b",
        r"\bgradientoutput\b",
        r"\baux_component_index\b",
        r"\bregister_elliptic_field\b",
    )
    assert not {pattern for pattern in concrete if re.search(pattern, source)}


@pytest.mark.parametrize(
    "path", GENERIC_NATIVE_NULLSPACE_PROTOCOLS, ids=lambda path: path.stem
)
def test_native_nullspace_protocol_has_no_cartesian_boundary_record_leak(
    path: Path,
) -> None:
    source = path.read_text(encoding="utf-8")
    cartesian_record_patterns = (
        r"physical_bc\.hpp",
        r"\bBCRec\b",
        r"\bBCType\b",
        r"\b(?:xlo|xhi|ylo|yhi)\b",
        r"std::array\s*<\s*FieldBoundaryNullspaceBehavior\s*,\s*4\s*>",
    )
    assert not {
        pattern for pattern in cartesian_record_patterns if re.search(pattern, source)
    }
