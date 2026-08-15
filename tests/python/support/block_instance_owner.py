"""Official Case-block instance owners for typed plan fixtures."""

from __future__ import annotations

import hashlib
from typing import Any

from pops.codegen._plans import canonical_block_instance_owner
from pops.model.ownership import OwnerKind, OwnerPath


def testing_model_owner(name: str) -> OwnerPath:
    owner = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
    owner._bind_definition_fingerprint(
        "test-model:sha256:%s" % hashlib.sha256(name.encode("utf-8")).hexdigest()
    )
    return owner


def make_testing_block_instance_owner(case: str, block: str, model_name: str) -> Any:
    return canonical_block_instance_owner(
        case=case, block=block, model_owner=testing_model_owner(model_name)
    )
