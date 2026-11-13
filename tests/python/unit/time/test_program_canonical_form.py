"""One canonical form per frozen authoring Program (N11, F5): bit for bit, and its witness."""
import json
from types import MappingProxyType

import pytest

import pops
from examples.api_v02.finite_structural_depth import build_finite_structural_depth
from examples.api_v02.profile_contracts import SELECTED_PROFILES, named_profiles
from examples.api_v02.profile_dispatch import build_profile
from pops.time._program import canonical_form
from pops.time._program.serialization import ir_identity_reuse

GALLERY = [(identifier, choice.name) for identifier in SELECTED_PROFILES
           for choice in named_profiles(identifier)]
# These projections hold a tuple (history_field_metadata_budget), so they are not closed builtin
# JSON: a decoded copy would hold a list, and the Program keeps no form.
NOT_CLOSED_JSON = {
    ("M06", "one-phase-stefan"), ("T01", "stage-constrained-ssprk2"),
    ("T12", "uniform-spatial-face"), ("T12", "field-after-replacement"),
}


def _texts(program):
    return tuple(json.dumps(program._serialize(include_provenance=variant), sort_keys=False)
                 for variant in (True, False)) + (program._ir_hash(),)


def _snapshot_texts(snapshot):
    return (snapshot._canonical_json, snapshot._artifact_canonical_json,
            snapshot._semantic_canonical_json, snapshot.hash, snapshot.artifact_hash,
            snapshot.semantic_identity.token)


def _resolved(spec, validated):
    try:
        plan = pops.resolve(validated, layout=spec.layout)
    except Exception as error:  # a profile whose resolve needs native code or a capability
        return ("refused", type(error).__name__, str(error))
    return (plan.plan_identity.token, *_snapshot_texts(plan.snapshot))


@pytest.fixture
def form_switch(monkeypatch):
    def switch(enabled):
        monkeypatch.setattr(canonical_form, "_ENABLED", enabled)
    return switch


def _validated(identifier, profile):
    spec = build_profile(identifier, profile=profile)
    validated = pops.validate(spec.case)
    return spec, validated, spec.case._time


@pytest.mark.parametrize(("identifier", "profile"), GALLERY)
def test_gallery_form_is_bit_identical_and_kept(identifier, profile, form_switch):
    spec, validated, program = _validated(identifier, profile)
    form_switch(False)
    expected = _texts(program), _resolved(spec, validated)
    assert not canonical_form.has_form(program)
    form_switch(True)
    first = _texts(program)
    # Every gallery Program reaches only immutable objects and sealed registries: a new mutable
    # leaf class fails here instead of silently disabling the form.
    if (identifier, profile) in NOT_CLOSED_JSON:
        assert type(program.history_field_metadata_budget) is tuple
        assert not canonical_form.has_form(program)
    else:
        assert canonical_form.has_form(program)
    assert (first, _resolved(spec, validated)) == expected
    assert _texts(program) == expected[0]


def test_structural_depth_form_is_bit_identical(form_switch):
    spec = build_finite_structural_depth(depth=8)
    validated = pops.validate(spec.case)
    program = spec.case._time
    form_switch(False)
    expected = _texts(program), _resolved(spec, validated)
    form_switch(True)
    assert (_texts(program), _resolved(spec, validated)) == expected
    assert canonical_form.has_form(program)


def _frozen_with_form():
    spec, _validated_case, program = _validated("M01", "periodic-advection")
    reference = _texts(program)
    assert canonical_form.has_form(program)
    return spec, program, reference


def _projected(program):
    return canonical_form.projected_ir_hash(program)


def test_reads_decode_a_fresh_dict_in_insertion_order():
    _spec, program, reference = _frozen_with_form()
    data = program._serialize(include_provenance=False)
    assert list(data) == list(program._canonical_projection(include_provenance=False))
    data["nodes"].clear()
    data["name"] = "mutated by a caller"
    assert _texts(program) == reference


def test_rebinding_a_program_attribute_drops_the_form():
    _spec, program, reference = _frozen_with_form()
    object.__setattr__(program, "name", "renamed out of contract")
    assert program._ir_hash() == _projected(program) != reference[2]


def test_rebinding_a_node_attribute_drops_the_form():
    _spec, program, reference = _frozen_with_form()
    node = next(value for value in program._values if value.name)
    object.__setattr__(node, "name", node.name + "-edited")
    assert program._ir_hash() == _projected(program) != reference[2]


def test_rebinding_inside_a_leaf_object_drops_the_form():
    _spec, program, reference = _frozen_with_form()
    node = program._values[-1]
    record = node.provenance
    span = record.primary
    object.__setattr__(span, "line", span.line + 1000)
    with_provenance = json.dumps(program._serialize(include_provenance=True), sort_keys=False)
    assert with_provenance != reference[0]
    assert with_provenance == json.dumps(
        program._canonical_projection(include_provenance=True), sort_keys=False)


def test_a_mutable_object_in_the_ir_keeps_no_form():
    spec = build_profile("M01", profile="periodic-advection")
    pops.validate(spec.case)
    program = spec.case._time
    node = program._values[-1]
    object.__setattr__(node, "attrs", MappingProxyType({**node.attrs, "extension": [1, 2]}))
    program._ir_hash()
    assert not canonical_form.has_form(program)


def test_a_mutable_program_keeps_no_form():
    spec = build_profile("M01", profile="periodic-advection")
    program = spec.case._time
    assert program._frozen is False
    program._ir_hash()
    assert not canonical_form.has_form(program)


def test_a_frozen_program_of_an_open_case_keeps_no_form():
    spec = build_profile("M01", profile="periodic-advection")
    program = spec.case._time
    program.freeze()
    program._ir_hash()
    # Its Case registries are still open: nothing is kept until pops.validate seals them.
    assert not canonical_form.has_form(program)
    pops.validate(spec.case)
    program._ir_hash()
    assert canonical_form.has_form(program)


def test_a_changed_definition_gives_the_new_identity(monkeypatch):
    _spec, program, reference = _frozen_with_form()
    form = canonical_form._FORMS[program]
    assert form.authorities
    authority, fingerprint = form.authorities[0]
    changed = "test:sha256:" + "0" * 64
    assert fingerprint != changed
    monkeypatch.setattr(type(authority), "fingerprint", lambda self: changed)
    assert not canonical_form._witness_holds(program, form)


def test_a_reuse_scope_trusts_a_checked_form_until_it_ends():
    _spec, program, reference = _frozen_with_form()
    with pytest.raises(RuntimeError, match="changed during pops.compile"):
        with ir_identity_reuse("pops.compile"):
            assert program._ir_hash() == reference[2]
            object.__setattr__(program, "name", "renamed out of contract")
            assert program._ir_hash() == reference[2]
    assert program._ir_hash() != reference[2]


def test_the_compile_scope_projects_a_kept_form_for_real(monkeypatch):
    _spec, program, reference = _frozen_with_form()
    projected = []
    original = canonical_form.projected_ir_hash

    def counted(value):
        projected.append(value)
        return original(value)

    monkeypatch.setattr(canonical_form, "projected_ir_hash", counted)
    for operation in ("pops.resolve", "pops.compile"):
        with ir_identity_reuse(operation):
            assert program._ir_hash() == reference[2]
    assert projected == [program]


def test_only_closed_finite_json_and_refusing_classes_are_kept():
    assert canonical_form._closed_json({"a": [1, 2.5, None, True, "x"]})
    assert not canonical_form._closed_json({"a": float("nan")})
    assert not canonical_form._closed_json({"a": (1,)})
    assert not canonical_form._closed_json({1: "a"})

    class Claims:
        __pops_ir_immutable__ = True

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)

    claim = Claims()
    assert not canonical_form._immutable_class(Claims, claim)
    assert not hasattr(claim, canonical_form._PROBE)

    class Refuses:
        __slots__ = ("value",)

        def __setattr__(self, name, value):
            raise AttributeError("immutable")

    assert canonical_form._immutable_class(Refuses, Refuses())
    import numpy as np
    from pops.time import CopyCurrent

    for mutable in ([], {}, set(), bytearray(), np.zeros(2), lambda: None):
        assert not canonical_form._immutable_class(type(mutable), mutable)
    assert canonical_form._immutable_class(CopyCurrent, CopyCurrent())
