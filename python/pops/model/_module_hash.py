"""Canonical content identity for an operator-first :class:`Module`."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .hash_data import body_identity, canonical_hash_data


def module_content_hash(module: Any) -> str:
    """Hash every semantic declaration and public alias while preserving registry order."""
    payload = {
        "schema": "spec2-module",
        "name": module.name,
        "state_spaces": [
            module._state_spaces[name].to_data() for name in sorted(module._state_spaces)
        ],
        "field_spaces": [
            module._field_spaces[name].to_data() for name in sorted(module._field_spaces)
        ],
        "parameters": [
            declaration.artifact_data()
            for _, declaration in sorted(module._param_registry.items())
        ],
        "aux": [module._aux[name].to_data() for name in sorted(module._aux)],
        "eigenvalues": None if module._eigenvalues is None else {
            direction: [canonical_hash_data(value)
                        for value in module._eigenvalues[direction]]
            for direction in ("x", "y")
        },
        "wave_speed_provider": module._wave_speed_provider,
        # Registry order is semantic: it determines stable OperatorId values.
        "operators": [{
            "name": operator.name,
            "kind": operator.kind,
            "signature": operator.signature.to_data(),
            "capabilities": operator.capabilities,
            "requirements": operator.requirements,
            "lowering": operator.lowering,
            "body": body_identity(operator.body),
        } for operator in module._registry],
        # Public aliases are authenticated declaration identities, not presentation-only labels.
        # Two Modules exposing different author-facing handles must therefore never collapse to
        # the same model-definition owner even when they lower to the same native operator route.
        "operator_aliases": module._registry.aliases(),
    }
    canonical = json.dumps(
        canonical_hash_data(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["module_content_hash"]
