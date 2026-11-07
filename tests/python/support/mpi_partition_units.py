"""Run one declared partition of an exactly inventoried pytest selection under MPI.

The MPI Python allocation splits a costly selection by measured case cost, without changing a
watchdog. Every worker still collects all cases its selection owns, refuses any added, removed,
renamed or duplicated case against the declared inventory, and only then runs its assigned
units. The partitions of one selection are disjoint and their union is the inventory, so no case
can leave the MPI lane silently. Loaded with ``-p`` only by the MPI launchers, which always set
the environment variable below.

PYTEST_DONT_REWRITE: ``mpi_pytest_entrypoint.run`` imports this module before pytest loads it as
a plugin, and the module holds no assertion to rewrite.
"""
from __future__ import annotations

import json
import os
import re

ENVIRONMENT = "POPS_CI_MPI_TEST_UNITS"
_PARTITION = re.compile(r"[a-z][a-z0-9_]*")


def validate(spec: object) -> dict:
    """Return the exact partition spec, or raise before any test is collected or run."""
    if type(spec) is not dict or set(spec) != {"partition", "inventory", "units"}:
        raise ValueError("MPI partition units require exactly a partition, inventory and units")
    partition, inventory, units = spec["partition"], spec["inventory"], spec["units"]
    if type(partition) is not str or not _PARTITION.fullmatch(partition):
        raise ValueError("MPI partition units require an exact partition identity")
    if (type(inventory) is not list or type(units) is not list or not units
            or any(type(node) is not str or "::" not in node or node != node.strip()
                   for node in inventory + units)):
        raise ValueError("MPI partition units require nonempty pytest node lists")
    if len(set(inventory)) != len(inventory) or len(set(units)) != len(units):
        raise ValueError("MPI partition units contain a duplicate pytest node")
    if not set(units) <= set(inventory):
        raise ValueError("MPI partition units select a node outside their declared inventory")
    return {"partition": partition, "inventory": list(inventory), "units": list(units)}


def encode(partition: str, inventory, units) -> str:
    """Serialize one validated spec for the launcher environment."""
    spec = validate({"partition": partition, "inventory": list(inventory), "units": list(units)})
    return json.dumps(spec, separators=(",", ":"))


def from_environment(environment=None) -> dict | None:
    raw = (os.environ if environment is None else environment).get(ENVIRONMENT)
    return None if raw is None else validate(json.loads(raw))


def selected(collected, spec: dict) -> set[str]:
    """Prove the complete collection equals the inventory, then return the assigned units."""
    collected = list(collected)
    if len(collected) != len(set(collected)):
        raise ValueError("MPI partition collection duplicated a pytest node")
    expected = set(spec["inventory"])
    if set(collected) != expected:
        raise ValueError(
            "MPI partition %s collection differs from its declared inventory: missing=%s extra=%s"
            % (spec["partition"], sorted(expected - set(collected)),
               sorted(set(collected) - expected)))
    return set(spec["units"])


def pytest_collection_modifyitems(config, items):
    spec = from_environment()
    if spec is None:
        raise ValueError("MPI partition plugin loaded without its declared units")
    keep = selected([item.nodeid for item in items], spec)
    deselected = [item for item in items if item.nodeid not in keep]
    items[:] = [item for item in items if item.nodeid in keep]
    config.hook.pytest_deselected(items=deselected)


def function_of(node: str) -> str:
    return node.split("::", 1)[1].split("[", 1)[0]


def parameter_partitions(partitions: dict[str, tuple[str, ...]]) -> dict:
    """Exact parameter partitions of whole test functions: id -> (selectors, cases, units).

    The selectors are the functions that own the units, so each child collects every parameter
    case of those functions; the encoded units carry the union of all partitions as inventory.
    """
    inventory = [node for units in partitions.values() for node in units]
    selectors = tuple(sorted({node.split("[", 1)[0] for node in inventory}))
    if len(set(inventory)) != len(inventory) or not all(partitions.values()):
        raise ValueError("MPI parameter partitions must be nonempty and disjoint")
    result = {}
    for identity, units in partitions.items():
        cases: dict[str, int] = {}
        for node in units:
            cases[function_of(node)] = cases.get(function_of(node), 0) + 1
        result[identity] = (selectors, cases, encode(identity, inventory, units))
    return result
