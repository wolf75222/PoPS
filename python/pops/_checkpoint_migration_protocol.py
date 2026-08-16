"""Shared, dependency-neutral Uniform-v2 checkpoint migration provenance protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


# Offline migrations carry one reviewed JSON provenance member.  Its spelling is deliberately
# capped independently of the live checkpoint controls: it is not an authority for unbounded
# allocation merely because a producer has already authenticated its ordinary payload arrays.
_CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS = 1 << 14
UNIFORM_V2_MIGRATION_PROTOCOL = "pops.uniform-checkpoint-v2-offline-migration.v1"


def validate_checkpoint_migration_provenance(payload: Mapping[str, Any]) -> Any | None:
    """Validate optional Uniform-v2 provenance and return its exact mapping identity.

    Native checkpoints deliberately have no migration member.  When one is present it is an
    authenticated, consumed provenance record, never descriptive archive decoration.
    """
    if "checkpoint_migration" not in payload:
        return None

    import numpy as np

    from pops._manifest_protocol import strict_json_loads
    from pops.identity import make_identity

    value = np.asarray(payload["checkpoint_migration"])
    if value.shape != () or value.dtype.kind != "U":
        raise TypeError("checkpoint migration provenance must be an exact Unicode scalar")
    text = value.item()
    if not isinstance(text, str) or len(text) > _CHECKPOINT_MIGRATION_PROVENANCE_MAX_CHARACTERS:
        raise ValueError("checkpoint migration provenance exceeds its fixed character capacity")
    provenance = strict_json_loads(text, where="checkpoint migration provenance")
    required = {"protocol", "mapping_identity", "mapping"}
    if type(provenance) is not dict or set(provenance) != required:
        raise ValueError(
            "checkpoint migration provenance must contain exactly protocol, mapping_identity, mapping"
        )
    if provenance["protocol"] != UNIFORM_V2_MIGRATION_PROTOCOL:
        raise ValueError("checkpoint migration provenance has an unknown protocol")
    if not isinstance(provenance["mapping_identity"], str) or not provenance["mapping_identity"]:
        raise TypeError("checkpoint migration provenance mapping_identity must be non-empty text")
    if type(provenance["mapping"]) is not dict:
        raise TypeError("checkpoint migration provenance mapping must be an exact object")
    identity = make_identity("uniform-v2-migration-map", provenance["mapping"])
    if provenance["mapping_identity"] != identity.token:
        raise ValueError("checkpoint migration provenance mapping_identity does not match mapping")
    return identity


def _checkpoint_migration_provenance_characters(payload: Mapping[str, Any]) -> int:
    """Validate the optional fixed-envelope migration provenance member."""
    if validate_checkpoint_migration_provenance(payload) is None:
        return 0
    return len(str(payload["checkpoint_migration"]))
