"""Build-entry contracts for one immutable compile-time spatial specialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "scripts" / "build_python.sh"
FINAL_GATE = ROOT / "scripts" / "run_final_gate.py"
PARAVIEW = ROOT / "scripts" / "paraview_python.sh"


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
