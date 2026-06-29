"""Spec 5 route-inspection + Optimization de-string checks.

The old ``pops.Case.explain_routes()`` assembly has been removed. Route/capability inspection now
comes from the native-backed capability matrix, while the executable path stays
``Module/physics.Model -> compile_problem -> sim.install``.
"""
import importlib
import sys

import pytest

pops = pytest.importorskip("pops")

from pops.codegen import (  # noqa: E402
    ConservativeFusion,
    DebugMath,
    Disabled,
    FastMath,
    GpuRegisterAware,
    Optimization,
    StrictMath,
)


def test_capability_matrix_is_printable_and_native_backed():
    matrix = pops.inspect_capabilities()
    text = str(matrix)
    assert "capability matrix" in text
    assert len(matrix) > 0
    native = [entry for entry in matrix if entry.source == "native"]
    assert native, "native _pops.module_capabilities() rows must be exposed"
    names = {entry.name for entry in native}
    assert {"supports_uniform", "supports_amr", "supports_named_fields"} <= names


def test_capability_matrix_is_json_serialisable_metadata_only():
    import json

    payload = pops.inspect_capabilities().to_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_capability_matrix_covers_spec73_descriptor_families():
    matrix = pops.inspect_capabilities()
    categories = set(matrix.categories())
    assert {
        "backend",
        "optimization",
        "math_mode",
        "fusion_policy",
        "output_policy",
        "checkpoint_policy",
        "output_format",
        "level_policy",
        "refinement_criterion",
        "regrid_policy",
        "patch_layout",
        "nesting_policy",
        "tag_policy",
        "amr_output",
        "closure",
        "realizability",
        "wave_speed",
        "moment_source",
        "solver",
    } <= categories

    backends = {entry.name: entry.to_dict() for entry in matrix.by_category("backend")}
    assert backends["Production"]["capabilities"]["amr"] is True
    assert backends["JIT"]["capabilities"]["amr"] is False

    solvers = {entry.name: entry.to_dict() for entry in matrix.by_category("solver")}
    assert solvers["gmres"]["options"]["max_iter"] == 1
    assert solvers["gmres"]["capabilities"]["supports_amr"] is True

    waves = {entry.name: entry.to_dict() for entry in matrix.by_category("wave_speed")}
    assert waves["ExactSpeeds.bounded"]["capabilities"]["exact_speeds"] is False
    assert waves["ExactSpeeds.roe"]["capabilities"]["roe"] is True


def test_no_case_route_matrix_assembly_survives_publicly():
    assert not hasattr(pops, "Case")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pops.case")


def test_optimization_math_string_is_rejected_with_clear_message():
    with pytest.raises(TypeError) as excinfo:
        Optimization(math="fast")
    message = str(excinfo.value)
    assert "optimization math" in message
    assert "fast" in message
    assert "StrictMath()" in message and "FastMath()" in message
    assert "DebugMath()" in message and "GpuRegisterAware()" in message


def test_optimization_fuse_string_is_rejected():
    with pytest.raises(TypeError) as excinfo:
        Optimization(fuse="conservative")
    assert "optimization fuse" in str(excinfo.value)


def test_optimization_typed_math_still_works():
    for mode in (StrictMath(), FastMath(), DebugMath(), GpuRegisterAware()):
        opt = Optimization(math=mode)
        assert opt.math is mode
        assert opt.options()["math"] == type(mode).__name__

    assert isinstance(Optimization().math, StrictMath)
    assert Optimization().capabilities()["strict_math"] is True
    assert Optimization(math=FastMath()).capabilities()["strict_math"] is False

    opt = Optimization(fuse=ConservativeFusion())
    assert opt.options()["fuse"] == "ConservativeFusion"
    assert Optimization(fuse=Disabled()).options()["fuse"] == "Disabled"


def test_optimization_options_no_longer_crashes_on_a_bad_math():
    opt = Optimization()
    assert opt.options()["math"] == "StrictMath"
    with pytest.raises(TypeError):
        Optimization(math=42)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
