"""Explicit pytest node inventories for independently schedulable cases.

Selection still begins with manifest-owned files. A partitioned file is collected in
full on each worker, checked against its exact inventory, then filtered to the
assigned nodeids. An added, removed, duplicated or renamed test fails before run.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "tests/python/test_partitions.json"
ENVIRONMENT = "POPS_CI_TEST_UNITS"


def owner(unit: str) -> str:
    return unit.split("::", 1)[0]


def load_contract(path: Path = CONTRACT) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text())
    if type(raw) is not dict or set(raw) != {"schema", "files"} or raw["schema"] != 1:
        raise ValueError("unsupported Python test partition contract")
    rows = raw["files"]
    if type(rows) is not dict:
        raise ValueError("Python test partitions require an explicit file table")
    result = {}
    for file, row in rows.items():
        if (type(file) is not str or not file.startswith("tests/python/")
                or not file.endswith(".py") or "::" in file or ".." in Path(file).parts
                or type(row) is not dict or set(row) != {"basis", "nodes"}
                or type(row["basis"]) is not str or not row["basis"]
                or type(row["nodes"]) is not list or not row["nodes"]):
            raise ValueError("invalid Python test partition file")
        nodes = {}
        for entry in row["nodes"]:
            if type(entry) is not dict or set(entry) != {"nodeid", "seconds"}:
                raise ValueError("invalid Python test partition entry")
            node, seconds = entry["nodeid"], entry["seconds"]
            if (type(node) is not str or owner(node) != file or "::" not in node
                    or node in nodes or type(seconds) not in (int, float)
                    or not math.isfinite(seconds) or seconds <= 0):
                raise ValueError("invalid or duplicate Python test partition node")
            nodes[node] = float(seconds)
        result[file] = nodes
    return result


def expand(files, contract=None) -> list[str]:
    contract = load_contract() if contract is None else contract
    if len(files) != len(set(files)) or any("::" in file for file in files):
        raise ValueError("partition expansion requires unique manifest-owned files")
    return [node for file in files for node in contract.get(file, {file: 0})]


def weights(durations, contract=None):
    contract = load_contract() if contract is None else contract
    result = dict(durations)
    for nodes in contract.values():
        result.update(nodes)
    return result


def collected_selection(nodeids, selected, contract):
    """Validate complete collection before selecting any case, without test execution."""
    if len(nodeids) != len(set(nodeids)) or len(selected) != len(set(selected)):
        raise ValueError("duplicate Python test collection or selection")
    by_file = {}
    for node in nodeids:
        by_file.setdefault(owner(node), set()).add(node)
    requested = {}
    for unit in selected:
        requested.setdefault(owner(unit), []).append(unit)
    if set(by_file) != set(requested):
        raise ValueError("collected Python files differ from exact assigned owners")
    keep = set()
    for file, units in requested.items():
        if file in contract:
            if by_file[file] != set(contract[file]):
                raise ValueError("collected Python nodes differ from inventory: " + file)
            if file in units or not set(units) <= by_file[file]:
                raise ValueError("partition selection must use exact known nodeids: " + file)
            keep.update(units)
        elif units != [file]:
            raise ValueError("unpartitioned Python file selected by unknown nodeid: " + file)
        else:
            keep.update(by_file[file])
    return keep


def pytest_collection_modifyitems(config, items):
    raw = os.environ.get(ENVIRONMENT)
    if raw is None:
        return
    selected = json.loads(raw)
    if type(selected) is not list or any(type(unit) is not str for unit in selected):
        raise ValueError("explicit Python test units must be a string list")
    keep = collected_selection([item.nodeid for item in items], selected, load_contract())
    deselected = [item for item in items if item.nodeid not in keep]
    items[:] = [item for item in items if item.nodeid in keep]
    config.hook.pytest_deselected(items=deselected)
