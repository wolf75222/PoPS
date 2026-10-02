"""Authoring capability versus detached canonical content: real registry resolver."""
import pytest
import pops
from pops.model.handles import Handle
from tests.review.test_m17_parameter_phase_identity import public_case


def test_foreign_live_clone_is_refused_by_actual_path_resolver():
    case, layout, model, parameter, block = public_case()
    other, other_layout, foreign, foreign_parameter, _ = public_case()
    pops.resolve(pops.validate(other), layout=other_layout)
    registry = case._block_registry
    with pytest.raises(ValueError):
        registry.qualify(foreign_parameter, block=block)
    from pops._ir.expr_references import resolve_reference_value
    foreign_ref = foreign.value(foreign_parameter)
    with pytest.raises(ValueError):
        resolve_reference_value(foreign_ref,
            lambda handle: registry.canonicalize(handle, block=block), {})


def test_canonical_clone_and_serialized_roundtrip_are_content_authenticated():
    case, layout, _, parameter, block = public_case()
    other, other_layout, _, foreign_parameter, _ = public_case()
    pops.resolve(pops.validate(other), layout=other_layout)
    pops.resolve(pops.validate(case), layout=layout)
    registry = case._block_registry
    canonical = foreign_parameter._resolved()
    restored = Handle.from_canonical_identity(canonical.canonical_identity())
    assert registry.qualify(canonical, block=block).declaration_ref == parameter
    assert registry.qualify(restored, block=block).declaration_ref == parameter
