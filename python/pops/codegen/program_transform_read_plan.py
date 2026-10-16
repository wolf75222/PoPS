"""Exact free-input schemas for a generated pointwise transform's read journal.

This descriptor describes only the generated equations and declared input types.
It is not a permission to discharge a representation: the runtime must also bind
the actual journal, authenticated spatial support and exact source incarnation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from pops._ir.expr import Var
from pops._ir.values import RuntimeParamRef
from pops._ir.visitors import _children, _dag_key_data
from pops._structural_data import canonical_json
from pops.model.hash_data import canonical_hash_data


@dataclass(frozen=True, slots=True)
class TransformReadPlan:
    parameter_indices: tuple[int, ...]
    provider_count: int
    descriptor: str
    authorities: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.parameter_indices) + self.provider_count

    @property
    def components(self) -> int:
        return 1 + 2 * self.count

    @property
    def identity(self) -> str:
        return hashlib.sha256(self.descriptor.encode("utf-8")).hexdigest()


def transform_read_plan(impl: Any, roots: Any, binding: Any) -> TransformReadPlan:
    """Bind the expanded roots to their exact used native inputs, without defaults.

    Slots are obtained from the same model and ProviderPack as the equation
    kernel. Unused model parameters, unrelated providers, qids and evaluation
    ordinals do not enter the descriptor. Native parameter slot numbers only
    select reads; parameter declaration identities provide the logical schema.
    """
    nodes = impl.assign_runtime_indices()
    native_indices = {node.name: index for index, node in enumerate(nodes)}
    parameters: dict[str, RuntimeParamRef] = {}
    providers: set[str] = set()
    pending, seen = list(roots), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, RuntimeParamRef):
            prior = parameters.setdefault(node.name, node)
            if prior.dtype != node.dtype:
                raise ValueError("transform input has conflicting parameter types")
        elif isinstance(node, Var):
            if node.kind == "aux":
                providers.add(node.name)
            elif node.kind != "cons":
                raise ValueError("transform read plan requires expanded, owned input leaves")
        pending.extend(_children(node))
    names = tuple(sorted(parameters, key=native_indices.__getitem__))
    slots = binding["slots"]
    if set(slots) != providers or sorted(slots.values()) != list(range(len(slots))):
        raise ValueError("transform read plan differs from its exact provider kernel plan")
    rows = []
    pack = impl._auxiliary_provider_pack
    for name in sorted(slots, key=slots.__getitem__):
        keys = [key for key in pack
                if key.space_kind in {"aux", "field"} and key.component == name]
        if len(keys) != 1:
            raise ValueError("transform input has no unique authenticated provider")
        key = keys[0]
        rows.append({"key": key.to_data(), "contract": pack.contract(key).to_data()})
    descriptor = {
        "protocol": "pops.transform.read-plan.v1",
        "expressions": _dag_key_data(roots),
        "state_components": tuple(impl.cons_names),
        "parameters": tuple({"name": name, "dtype": parameters[name].dtype} for name in names),
        "providers": rows,
    }
    authority_rows = (
        *({"kind": "parameter", "name": name, "dtype": parameters[name].dtype} for name in names),
        *({"kind": "provider", **row} for row in rows),
    )
    return TransformReadPlan(
        tuple(native_indices[name] for name in names), len(rows),
        canonical_json(canonical_hash_data(descriptor), allow_nan=False),
        tuple(canonical_json(canonical_hash_data(row), allow_nan=False) for row in authority_rows),
    )
