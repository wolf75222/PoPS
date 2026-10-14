"""Full-family requests reject every remaining corpus obligation before construction."""
import importlib

import pytest

from examples.api_v02.common import MissingCapabilityError
from examples.api_v02.coverage import OBLIGATIONS


# Enumerate the normative 28 IDs independently of the implementation mapping.
@pytest.mark.parametrize("identifier", tuple(f"M{number:02d}" for number in range(1, 29)))
def test_complete_model_request_rejects_unqualified_family(identifier):
    module = importlib.import_module("examples.api_v02." + (
        "models_M01_M14" if int(identifier[1:]) <= 14 else "models_M15_M28"))
    with pytest.raises(MissingCapabilityError) as failure:
        module.build_model(identifier, require_complete=True)
    assert failure.value.model == identifier
    assert failure.value.required == OBLIGATIONS[identifier]
    assert failure.value.required
