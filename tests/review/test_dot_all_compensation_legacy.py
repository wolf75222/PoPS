"""Exact source regression against the SDK506 received source; no native import."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BASE = "f36176fa2a822c67b5ccc114bd21de1bef4be23e"
UNIFORM = "include/pops/runtime/program/program_context.hpp"
AMR = "include/pops/runtime/program/amr_program_context_spatial_operations.inc"
MESH = "include/pops/mesh/storage/mf_arith.hpp"
ITERATION = "include/pops/mesh/execution/for_each.hpp"


def body(text, signature):
    start = text.index(signature)
    opening = text.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


@pytest.mark.parametrize("path,signature", [
    (path, f"Real {method}(int program_block,")
    for path in (UNIFORM, AMR)
    for method in (
        "sum_component", "abs_sum_component", "max_component", "min_component",
        "norm2", "norm_inf", "dot",
    )
] + [
    (AMR, "template <class Visitor>\nvoid for_each_owner_active_level_"),
    (AMR, "template <class Visitor>\nvoid for_each_owner_finest_active_level_"),
    (MESH, "template <int Dim>\nstruct FiniteOwnedDotKernel"),
    (MESH, "template <int Dim, class MemorySpace>\nReal dot_all_local("),
    (ITERATION, "Real for_each_cell_reduce_sum(const ExecutionSpace&"),
    (ITERATION, "Real for_each_cell_reduce_sum(const Box<Dim>&"),
])
def test_legacy_reductions_visitors_and_product_kernel_byte_identical(path, signature):
    old = subprocess.run(["git", "show", f"{BASE}:{path}"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert body((ROOT / path).read_text(), signature) == body(old, signature)
