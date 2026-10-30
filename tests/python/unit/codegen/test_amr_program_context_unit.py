"""One instantiation of the AMR Program context per compile identity (B03, F5, CC7)."""
from __future__ import annotations

import pytest

pytest.importorskip("pops")

from pops.codegen import amr_program_context_unit as unit  # noqa: E402

_INSTANTIATION = "template class pops::runtime::program::AmrProgramContext<pops::kNativeDimension>;"


def test_unit_repeats_the_fixed_program_preamble_before_the_instantiation():
    from pops.codegen.program_emit_kernels import _PROGRAM_CPP_TEMPLATE

    source = unit.amr_program_context_unit_source()
    preamble = _PROGRAM_CPP_TEMPLATE.split("{model_helpers}", 1)[0]
    includes = [line for line in preamble.splitlines() if line.startswith("#include")]
    assert includes
    positions = [source.index(line + "\n") for line in includes]
    assert positions == sorted(positions)
    assert source.endswith(unit.AMR_PROGRAM_CONTEXT_INCLUDE + "\n" + _INSTANTIATION + "\n")
    assert source.index(unit.AMR_PROGRAM_CONTEXT_INCLUDE) > positions[-1]
    assert "extern template" not in source
    assert "{" not in source and "}" not in source


def test_generated_include_declares_the_instantiation_extern():
    assert unit.amr_program_context_include() == (
        unit.AMR_PROGRAM_CONTEXT_INCLUDE + "\n" + unit.AMR_PROGRAM_CONTEXT_EXTERN + "\n")
    assert unit.amr_program_context_include("  // provider") == (
        unit.AMR_PROGRAM_CONTEXT_INCLUDE + "  // provider\n" + unit.AMR_PROGRAM_CONTEXT_EXTERN
        + "\n")


def test_unit_identity_follows_the_program_compile_contract(monkeypatch):
    from pops.codegen import toolchain

    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "fixture-native-features")
    base = {
        "compiler": "c++",
        "standard": "c++20",
        "abi_key": "headers|c++|c++20|dim=2",
        "optflags": ["-O3", "-ffp-contract=off"],
        "cflags": ["-DPOPS_HAS_KOKKOS"],
        "toolchain_capture": None,
        "native_dimension": 2,
    }
    semantic, spec = unit.amr_program_context_unit_identity(**base)
    assert unit.amr_program_context_unit_identity(**base) == (semantic, spec)
    for key, value in (
        ("compiler", "g++-14"),
        ("standard", "c++23"),
        ("abi_key", "other headers|c++|c++20|dim=2"),
        ("optflags", ["-O3"]),
        ("cflags", []),
    ):
        changed_semantic, changed_spec = unit.amr_program_context_unit_identity(
            **{**base, key: value})
        assert changed_semantic == semantic, key
        assert changed_spec != spec, key
    other_dimension = unit.amr_program_context_unit_identity(**{**base, "native_dimension": 3})
    assert other_dimension[0] != semantic and other_dimension[1] != spec
