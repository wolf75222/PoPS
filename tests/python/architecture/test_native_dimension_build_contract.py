"""Build-entry contracts for one immutable compile-time spatial specialization."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "scripts" / "build_python.sh"
FINAL_GATE = ROOT / "scripts" / "run_final_gate.py"
PARAVIEW = ROOT / "scripts" / "paraview_python.sh"


def _load_final_gate():
    spec = importlib.util.spec_from_file_location("_native_dimension_final_gate", FINAL_GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_build_entry(*arguments: str, native_dimension: str | None = None):
    environment = os.environ.copy()
    environment.pop("POPS_NATIVE_DIM", None)
    if native_dimension is not None:
        environment["POPS_NATIVE_DIM"] = native_dimension
    return subprocess.run(
        ["bash", str(BUILD), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("arguments", "native_dimension", "message"),
    [
        ((), None, "native dimension is required"),
        ((), "0", "invalid native dimension '0'"),
        (("--dim", "4"), None, "invalid native dimension '4'"),
        (("--dim", "1"), "3", "conflicting native dimensions"),
    ],
)
def test_python_build_fails_closed_before_environment_or_compilation(
    arguments: tuple[str, ...], native_dimension: str | None, message: str
):
    result = _run_build_entry(*arguments, native_dimension=native_dimension)

    assert result.returncode == 2
    assert message in result.stderr
    assert "env 'pops' active" not in result.stdout


def test_python_build_propagates_the_exact_dimension_to_cmake_and_proofs():
    build = BUILD.read_text(encoding="utf-8")

    assert 'export POPS_NATIVE_DIM="$NATIVE_DIM"' in build
    assert '-C cmake.define.POPS_NATIVE_DIM="$POPS_NATIVE_DIM"' in build
    assert '-C build-dir="build/{wheel_tag}-dim${POPS_NATIVE_DIM}"' in build
    assert 'NATIVE_DIM="${CLI_NATIVE_DIM:-$CALLER_NATIVE_DIM}"' in build
    assert '--expect-dim "$POPS_NATIVE_DIM"' in build
    assert 'prove_installed_wheel.py"' in build
    assert '--wheel "${built_wheels[0]}" --expect-dim "$POPS_NATIVE_DIM"' in build


def test_live_and_release_entry_points_require_explicit_dimension_sets():
    final_gate = FINAL_GATE.read_text(encoding="utf-8")
    paraview = PARAVIEW.read_text(encoding="utf-8")

    assert '"--dim", required=True' in final_gate
    assert '"--wheel-dim", action="append", required=True' in final_gate
    assert '"--expect-dim", str(dimension)' in final_gate
    assert "native dimension is required" in paraview
    assert 'export POPS_NATIVE_DIM="$NATIVE_DIM"' in paraview
    assert 'verify_installed_native.py" --expect-dim "$NATIVE_DIM"' in paraview


def test_final_gate_passes_requested_dimension_to_serial_configure_and_ctest():
    source = FINAL_GATE.read_text(encoding="utf-8")
    main = source[source.index("def main(") :]

    assert '"cmake", "--preset", "serial", f"-DPOPS_NATIVE_DIM={args.dim}"' in main
    assert '"cmake", "--build", "--preset", "serial"' in main
    assert '_resolve_ctest_dir(args.ctest_dir, expected_dimension=args.dim)' in main
    assert '"cmake", "--preset", "serial"])' not in main


def test_final_gate_rejects_ctest_tree_with_a_different_native_dimension(tmp_path):
    gate = _load_final_gate()
    ctest_dir = tmp_path / "build"
    ctest_dir.mkdir()
    (ctest_dir / "CTestTestfile.cmake").write_text("# generated\n", encoding="utf-8")
    (ctest_dir / "CMakeCache.txt").write_text(
        "POPS_BUILD_TESTS:BOOL=ON\nPOPS_NATIVE_DIM:STRING=2\n", encoding="utf-8"
    )

    with pytest.raises(gate.FinalGateError, match=r"POPS_NATIVE_DIM=2, but --dim=3"):
        gate._resolve_ctest_dir(ctest_dir, expected_dimension=3)


def test_every_checked_in_preset_inherits_an_explicit_dimension():
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    configure = {preset["name"]: preset for preset in presets["configurePresets"]}

    assert configure["base"]["cacheVariables"]["POPS_NATIVE_DIM"] == "2"

    def selected_dimension(name: str, seen: frozenset[str] = frozenset()) -> str | None:
        assert name not in seen, f"cyclic preset inheritance through {name}"
        preset = configure[name]
        direct = preset.get("cacheVariables", {}).get("POPS_NATIVE_DIM")
        if direct is not None:
            return str(direct)
        parents = preset.get("inherits", ())
        if isinstance(parents, str):
            parents = (parents,)
        inherited = {selected_dimension(parent, seen | {name}) for parent in parents}
        inherited.discard(None)
        assert len(inherited) <= 1, f"ambiguous native dimension inherited by {name}"
        return next(iter(inherited), None)

    for name in configure:
        assert selected_dimension(name) == "2", f"preset {name} has no exact specialization"
