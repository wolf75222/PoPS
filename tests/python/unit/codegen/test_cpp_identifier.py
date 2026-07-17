"""C++ identifier hygiene for author-facing model and emitter names."""

import re

import pytest

from pops.codegen.cpp_writer import _cpp_identifier


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("two-fluid", "two_fluid"),
        ("2fluid", "pops_2fluid"),
        ("class", "pops_class"),
        ("__x", "pops_x"),
        ("_Upper", "pops_Upper"),
        ("a__b", "a_b"),
        ("a--b", "a_b"),
    ),
)
def test_cpp_identifier_is_ascii_valid_and_non_reserved(raw: str, expected: str) -> None:
    identifier = _cpp_identifier(raw)

    assert identifier == expected
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier)
    assert not identifier.startswith("_")
    assert "__" not in identifier
