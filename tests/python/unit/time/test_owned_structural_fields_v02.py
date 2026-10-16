"""Owned control-flow storage preserves values through suspension and active arms."""
from pops.codegen.program_structural_control import make_slots, copy_structure, _OwnedFieldSlot
from pops.codegen.program_emit_hierarchy_control import _heap_slots

FIELD = {"vtype": "scalar_field"}


def test_every_field_slot_owns_storage_without_reading_source_support():
    for pointers in (False, True):
        lines = []
        slot = make_slots(FIELD, "foreign_level", 7, "carry", lines, pointers=pointers)
        assert isinstance(slot, _OwnedFieldSlot)
        assert slot == "(*carry_7_s0_owner)"
        assert "shared_ptr" in "\n".join(lines)
        assert "foreign_level" not in "\n".join(lines)
        assert "scratch_state" not in "\n".join(lines)
        # [=] captures the holder because the Field expression refers to it,
        # rather than capturing a raw pointer and leaving its holder behind.
        assert slot.owner_token in slot


def test_product_copy_prepares_all_fields_and_scalars_before_any_publication():
    sig = {"vtype": "product", "members": (("a", FIELD), ("b", FIELD), ("n", {"vtype": "int64"}))}
    declarations = []
    slots = make_slots(sig, None, 2, "out", declarations)
    lines = []
    copy_structure(sig, slots, {"a": "first", "b": "second", "n": "old_n"}, lines, references=True)
    source = "\n".join(lines)
    vote = source.index('"structured control owned copy"')
    assert source.index("ctx.own_structural_field(first)") < vote
    assert source.index("ctx.own_structural_field(second)") < vote
    for slot in (slots["a"], slots["b"]):
        assert source.index(slot.owner_token + " = std::move") > vote
    assert source.index(slots["n"] + " = std::move") > vote
    assert "is_nothrow_move_assignable_v" in source
    assert "= &first" not in source and "= &second" not in source


def test_inactive_variant_field_is_never_read_and_tag_is_published_last():
    sig = {"vtype": "variant", "members": (("empty", {"vtype": "int64"}), ("field", FIELD))}
    declarations = []
    slots = make_slots(sig, None, 2, "variant", declarations)
    lines = []
    copy_structure(sig, slots, {"tag": "input_tag", "payloads": {"empty": "n", "field": "unreachable_field"}}, lines, references=False)
    source = "\n".join(lines)
    assert "if (structured_pending_0_active_1) structured_pending_2 = ctx.own_structural_field(unreachable_field);" in source
    assert source.index("invalid native Variant discriminant") < source.index("ctx.own_structural_field")
    assert source.index(slots["tag"] + " = structured_pending_0_tag") > source.index('"structured control owned copy"')


def test_suspended_heap_field_keeps_its_owner_in_the_captured_frame():
    lines = []
    _, frame, slot = _heap_slots(FIELD, 9, "hierarchy", lines, carry=True, extra_members=("int marker=0;",))
    assert isinstance(slot, _OwnedFieldSlot)
    assert slot.owner_token == frame + "->s0_owner"
    assert "shared_ptr<pops::MultiFab" in "\n".join(lines)
    assert "int marker=0;" in lines
    copy_structure(FIELD, slot, "source", lines, references=False)
    assert "ctx.own_structural_field(source)" in "\n".join(lines)
    assert "scratch_state" not in "\n".join(lines)


def test_plain_borrowed_output_is_not_a_valid_structural_destination():
    import pytest
    with pytest.raises(TypeError, match="owning slot"):
        copy_structure(FIELD, "(*borrowed)", "source", [], references=True)
