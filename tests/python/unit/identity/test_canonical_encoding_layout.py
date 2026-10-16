"""Repeated canonical layouts preserve live values and observable subclass behavior."""
from __future__ import annotations

import pytest

from pops.identity import canonical_bytes


class GeneralMap(dict):
    """Exercise the uncached public dictionary-subclass semantics."""


def test_repeated_layout_never_reuses_values_or_hides_cycles():
    left = {"b": 1, "aa": 2}
    right = {"b": 3, "aa": 4}
    assert canonical_bytes([left, right]).hex() == "82a261620162616102a261620362616104"
    right["aa"] = right
    with pytest.raises(ValueError, match=r"reference cycle at \$\[1\]\.aa"):
        canonical_bytes([left, right])


def test_layout_order_and_capacity_do_not_restrict_canonical_values():
    rows = [{f"field-{index}": index, "same": -index} for index in range(4200)]
    assert canonical_bytes(rows) == canonical_bytes([GeneralMap(row) for row in rows])
    row = {"é": 0, "aa": 1, "b": 2}
    assert canonical_bytes([row, dict(reversed(tuple(row.items())))]) == (
        b"\x82" + canonical_bytes(GeneralMap(row)) * 2)


def test_dictionary_and_key_subclass_hooks_execute_for_every_occurrence():
    class Redirected(dict):
        def items(self):
            return (("actual", 9),)

    class Text(str):
        calls = 0

        def encode(self, *args, **kwargs):
            self.calls += 1
            return super().encode(*args, **kwargs)

    key = Text("same")
    assert canonical_bytes([Redirected(ignored=1)] * 2) == canonical_bytes([{"actual": 9}] * 2)
    assert canonical_bytes([{key: 1}, {key: 2}]) == canonical_bytes([{"same": 1}, {"same": 2}])
    assert key.calls == 2


def test_small_integer_subclass_comparisons_are_not_elided():
    events = []

    class Integer(int):
        def __lt__(self, other):
            events.append("lt")
            return super().__lt__(other)

        def __gt__(self, other):
            events.append("gt")
            return super().__gt__(other)

        def __ge__(self, other):
            events.append("ge")
            return super().__ge__(other)

    assert canonical_bytes(Integer(7)) == b"\x07"
    assert events == ["lt", "gt", "ge", "lt"]
    assert canonical_bytes(list(range(-24, 24))) == canonical_bytes([Integer(x) for x in range(-24, 24)])


def test_layout_cache_observes_mutation_from_a_value_hook():
    shared = {"value": None}

    class MutatingText(str):
        def encode(self, *args, **kwargs):
            shared["value"] = "after"
            return super().encode(*args, **kwargs)

    shared["value"] = MutatingText("before")
    assert canonical_bytes([shared, shared]) == canonical_bytes([{"value": "before"}, {"value": "after"}])


@pytest.mark.parametrize(("value", "error", "message"), [
    ({"key": {"bad": 1.0}}, TypeError, r"float at \$\.key\.bad"),
    ({"key": {"\ud800": 0}}, ValueError, r"string at \$\.key\{key\}"),
    ({"key": {1: 0}}, TypeError, r"map key at \$\.key must be a string"),
    ({"key": {"bad": 1 << 63}}, OverflowError, r"integer at \$\.key\.bad"),
])
def test_layout_errors_preserve_path_and_type(value, error, message):
    with pytest.raises(error, match=message):
        canonical_bytes(value)


def test_opaque_hooks_remain_refused():
    class Opaque:
        def to_data(self):
            raise AssertionError("The codec must never invoke extension hooks")

    with pytest.raises(TypeError, match=r"opaque Opaque at \$\.same"):
        canonical_bytes({"same": Opaque()})
