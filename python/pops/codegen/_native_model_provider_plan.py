"""One aggregate native input pack projected from the resolved component authority."""
from __future__ import annotations

import re
from typing import Any

from pops.model.provider_pack import ComponentKey, ProviderPack

from .component_provider_packs import consumer_provider_plan

NATIVE_MODEL_PROVIDER_CONTRACT = 1


def native_model_provider_plan(model: Any) -> tuple:
    """Union only roles evaluated by the generated aggregate model, never outputs."""
    complete = getattr(model, "_component_provider_pack", None)
    if type(complete) is not ProviderPack:
        raise TypeError("native model inputs require the resolved complete ProviderPack")
    roles = [model._component_flux_consumer_plan]
    operators = model._component_operator_consumer_plans
    if model._source is not None:
        roles.append(operators.get("source_default", ()))
    if model._proj is not None:
        roles.append(operators.get("projection", ()))
    keys = [ComponentKey(**row["key"]) for role in roles for row in role]
    return consumer_provider_plan(complete.select(keys))


def provider_slot_projection(role: Any, native: Any) -> dict[int, int]:
    """Authenticate each role row before projecting its local slot into the union."""
    by_key = {ComponentKey(**row["key"]): row for row in native}
    result = {}
    for row in role:
        target = by_key.get(ComponentKey(**row["key"]))
        if target is None or row["contract"] != target["contract"] \
                or row["provider"] != target["provider"]:
            raise ValueError("native model inputs do not authenticate an operation provider")
        result[row["consumer_slot"]] = target["consumer_slot"]
    return result


def project_provider_locals(lines: list[str], role: Any, native: Any) -> list[str]:
    """Rewrite compiler-owned provider-read tokens using exact qualified slot evidence."""
    if native is None:
        return lines
    slots = provider_slot_projection(role, native)

    def project(match: re.Match) -> str:
        return "pops::provider_value<%d>(a)" % slots[int(match.group(1))]

    return [re.sub(r"pops::provider_value<(\d+)>\(a\)", project, line) for line in lines]
