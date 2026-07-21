from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "ci_python_module_objects", ROOT / "scripts" / "ci_python_module_objects.py"
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _target(path: str, owner: str) -> str:
    return f"src/CMakeFiles/{owner}.dir/{path}: CXX_COMPILER__{owner}_Release"


def test_runtime_object_prewarm_lanes_are_an_exact_disjoint_cover():
    inventory = "\n".join(
        (
            _target("runtime/program/step_transaction.cpp.o", "pops_runtime_core_objects"),
            _target("runtime/system/system.cpp.o", "pops_runtime_system"),
            _target("runtime/output/hdf5_collective.cpp.o", "pops_runtime_output"),
            _target("runtime/amr/amr_system.cpp.o", "pops_runtime_amr"),
            _target(
                "generated_seams/amr/block/base/amr_block_exb.cpp.o",
                "pops_runtime_amr",
            ),
            _target(
                "generated_seams/amr/compiled/base/amr_compiled_exb.cpp.o",
                "pops_runtime_amr",
            ),
            "python/CMakeFiles/_pops.dir/bindings.cpp.o: CXX_COMPILER___pops_Release",
        )
    )
    lanes = module.partition_runtime_objects(inventory)
    assert lanes == {
        "system": [
            "src/CMakeFiles/pops_runtime_core_objects.dir/runtime/program/step_transaction.cpp.o",
            "src/CMakeFiles/pops_runtime_output.dir/runtime/output/hdf5_collective.cpp.o",
            "src/CMakeFiles/pops_runtime_system.dir/runtime/system/system.cpp.o",
        ],
        "amr-block": [
            "src/CMakeFiles/pops_runtime_amr.dir/generated_seams/amr/block/base/amr_block_exb.cpp.o",
            "src/CMakeFiles/pops_runtime_amr.dir/runtime/amr/amr_system.cpp.o",
        ],
        "amr-compiled": [
            "src/CMakeFiles/pops_runtime_amr.dir/generated_seams/amr/compiled/base/amr_compiled_exb.cpp.o",
        ],
    }


def test_runtime_object_prewarm_rejects_an_empty_lane():
    inventory = "\n".join(
        (
            _target("runtime/system/system.cpp.o", "pops_runtime_system"),
            _target("runtime/amr/amr_system.cpp.o", "pops_runtime_amr"),
        )
    )
    with pytest.raises(SystemExit, match="empty Python module prewarm lanes"):
        module.partition_runtime_objects(inventory)
