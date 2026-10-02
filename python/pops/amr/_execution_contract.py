"""Shared exact execution-v2/v3 protocol admission, before runtime mutations."""
from copy import deepcopy


def validate_execution_data(data, *, dimension=None):
    if dimension is not None and (type(dimension) is not int or dimension not in (1, 2, 3)):
        raise ValueError("AMR execution requires exact spatial dimension 1, 2, or 3")
    keys = {"schema_version", "authority_type", "mode", "relations"}
    if type(data) is not dict or type(data.get("schema_version")) is not int:
        raise TypeError("AMR execution must be an exact versioned dict")
    version = data["schema_version"]
    if version not in (2, 3) or set(data) != (keys | {"accepted_halo"} if version == 3 else keys):
        raise ValueError("unsupported exact AMR execution v2/v3 contract")
    if data["authority_type"] != "amr_execution" or data["mode"] not in ("synchronous", "subcycled"):
        raise ValueError("AMR execution authority/mode is not canonical")
    if type(data["relations"]) is not list:
        raise TypeError("AMR execution relations must be a list")
    if data["mode"] == "synchronous" and data["relations"]:
        raise ValueError("synchronous AMR execution accepts no authored relations")
    for index, row in enumerate(data["relations"]):
        if type(row) is not dict or set(row) != {
            "parent_level", "child_level", "temporal_ratio", "remainder_policy"
        }:
            raise ValueError("AMR execution temporal relation has incomplete keys")
        ratio = row["temporal_ratio"]
        if (type(row["parent_level"]) is not int or type(row["child_level"]) is not int
            or row["parent_level"] != index or row["child_level"] != index + 1
            or type(ratio) is not dict or set(ratio) != {"numerator", "denominator"}
            or type(ratio["numerator"]) is not int or type(ratio["denominator"]) is not int
            or ratio["denominator"] <= 0 or ratio["numerator"] < ratio["denominator"]
            or row["remainder_policy"] not in ("integral_only", "explicit_final_substep")):
            raise ValueError("AMR execution temporal relation is not canonical")
        if ratio["numerator"] % ratio["denominator"] and row["remainder_policy"] == "integral_only":
            raise ValueError("non-integral AMR temporal relation requires an explicit remainder")
    if version == 3:
        halo = data["accepted_halo"]
        if (type(halo) is not dict or set(halo) != {
            "schema_version", "effect", "point_authority", "cells", "components"
        } or type(halo["schema_version"]) is not int or halo["schema_version"] != 1
            or halo["effect"] != "prepare_accepted_halo"
            or halo["point_authority"] != "candidate_accepted_clock"
            or halo["components"] != "all_state_components"):
            raise ValueError("unsupported exact accepted halo preparation authority")
        cells = halo["cells"]
        widths = (cells,) if type(cells) is int else cells
        if (type(widths) is not tuple or not widths
            or any(type(value) is not int or value <= 0 for value in widths)
            or (dimension is not None and type(cells) is not int and len(widths) != dimension)
            or (dimension is not None and any(value > 2147483647 for value in widths))):
            raise ValueError("accepted halo preparation requires positive exact ranked cells")
    return deepcopy(data)


def runtime_execution_data(execution, *, dimension=None):
    protocol = getattr(execution, "runtime_execution_data", None)
    if not callable(protocol):
        raise TypeError("adaptive execution authority must implement runtime_execution_data()")
    # Snapshot the first result before the second call: a reused mutable dict cannot hide drift.
    first = validate_execution_data(protocol(), dimension=dimension)
    second = validate_execution_data(protocol(), dimension=dimension)
    if first != second:
        raise TypeError("AMR runtime_execution_data() must return one deterministic v2/v3 dict")
    identity = getattr(execution, "to_data", None)
    if callable(identity) and first != validate_execution_data(identity(), dimension=dimension):
        raise ValueError("AMR execution runtime projection differs from its declared authority")
    return first
