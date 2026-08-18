"""IF-08 planner cases: TR-01 at dim 1, GE-03 as the Dim2 refuse example.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

TR01_ID = "TR-01"
MATCHING_DIM = 1
TR01_CASE = {"id": TR01_ID, "native_dimensions": [MATCHING_DIM]}

DIM2_ID = "GE-03"
DIM2_REQUIRED = 2
DIM2_CASE = {"id": DIM2_ID, "native_dimensions": [DIM2_REQUIRED]}


def tr01_case() -> dict:
    """Return the planner record for TR-01 (native_dimensions = [1])."""
    return {
        "id": TR01_CASE["id"],
        "native_dimensions": list(TR01_CASE["native_dimensions"]),
    }


def dim2_case() -> dict:
    """Return the planner record for the Dim2 case (GE-03, native_dimensions = [2])."""
    return {
        "id": DIM2_CASE["id"],
        "native_dimensions": list(DIM2_CASE["native_dimensions"]),
    }
