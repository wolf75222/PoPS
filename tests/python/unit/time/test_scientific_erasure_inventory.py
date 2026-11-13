"""G03 lot E, session 1 (ROM-21, ``scientific_type_erasure_contract``): the inventory of erasures.

A ``ScientificType`` is erased to its finite type at three points only: the lowering of a validated
Program into regions, the ABI of ``bind``, and the outputs (G03 note, section 5). Every other
operation that the ticket names keeps the scientific type or refuses. This module is the inventory:
one row per path, the guard that holds it, and the tests that prove it. The rows whose proof lived
elsewhere are checked to still exist, so the inventory cannot point at a test that is gone; the
paths that had no direct test get one here (clone, the call of a closed body, the property of every
row). The concrete erasure that lot E closes, the surface states of b6, is tested in
``tests/python/unit/surface/test_published_representation.py``.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from pops.graph_functions import graph
from pops.representations import ScientificValue
from pops.scientific_graph import ANNOTATION, _key
from pops.time import Program
from pops.time.variant_types import Float64, Product
from pops._ir.quantity import PhysicalDimension
from tests.python.unit.time.test_scientific_graph_v14 import L, T, typ

AREA = PhysicalDimension((("length", 2),))

ROOT = pathlib.Path(__file__).resolve().parents[4]

#: path of the ticket -> (guard in the code, tests that prove « keeps or refuses »)
INVENTORY = {
    "projection .value": (
        "python/pops/scientific_graph.py:bind_scientific_value (the declaration lives on the "
        "canonical SSA, not on the ScientificValue wrapper)",
        ("tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_scientific_type_survives_value_extraction_and_canonical_ssa_alias",
         "tests/python/unit/time/test_scientific_erasure_v15.py::"
         "test_root_validation_rejects_erased_derived_annotations",
         "tests/python/unit/time/test_scientific_erasure_inventory.py::"
         "test_projection_keeps_the_declaration_or_refuses")),
    "clone": (
        "python/pops/time/_program/rebuild.py:clone_attrs (attributes copied through "
        "remap_metadata), the shared rebuild engine of _rebuild and of the passes",
        ("tests/python/unit/time/test_scientific_erasure_inventory.py::"
         "test_clone_keeps_every_declaration",
         "tests/python/unit/time/test_scientific_formals_v15.py::"
         "test_nested_legacy_boundaries_survive_reload_and_whole_program_detach")),
    "call of a closed body": (
        "python/pops/graph_functions.py:GraphFunction.__call__ -> "
        "scientific_graph.validate_scientific_graph_call",
        ("tests/python/unit/time/test_scientific_erasure_inventory.py::"
         "test_a_closed_body_keeps_its_dimensions_or_refuses",
         "tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_plain_graph_call_checks_unused_scientific_operations_atomically",
         "tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_applied_contract_rejects_modified_numerical_body")),
    "detachment of a stage value": (
        "python/pops/scientific_graph.py:scientific_packet_schema and "
        "validate_scientific_continuation (read and freeze of a continuation); "
        "python/pops/time/_program/detach.py:prepare_scientific_contracts (compiled detach)",
        ("tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_continuation_schema_preserves_scientific_meaning_across_read_and_freeze",
         "tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_bundle_reconstruction_cannot_relabel_scientific_history",
         "tests/python/unit/time/test_scientific_formals_v15.py::"
         "test_iterate_context_cannot_disappear_on_unannotated_output")),
    "passage between Field and Array": (
        "python/pops/representations/__init__.py:ScientificType (an array type requires its "
        "exact dof family) and the analyzer of scientific_graph.py on map, at and fold",
        ("tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_array_diffusion_checks_coordinate_and_diffusivity_units_on_actual_graph",
         "tests/python/unit/time/test_scientific_graph_v14.py::"
         "test_continuation_scientific_schema_covers_nested_arrays_and_rejects_tampered_units",
         "tests/python/unit/time/test_scientific_spatial_v14.py::"
         "test_branch_cannot_import_array_from_a_sibling_region")),
    "elaboration of a surface quantity into a b6 state": (
        "python/pops/surface/_elaborate.py:elaborated_representation, checked by "
        "python/pops/surface/_run.py:require_published_under at bind (closed by lot E)",
        ("tests/python/unit/surface/test_published_representation.py::"
         "test_a_published_array_is_refused_on_another_support_of_equal_shape",
         "tests/python/unit/surface/test_published_representation.py::"
         "test_another_representation_on_one_support_is_refused")),
}

#: the three erasure points of the contract, where the finite type is all that remains
ERASURE_POINTS = (
    "the lowering of a validated Program into regions",
    "the ABI of bind, whose arrays carry a schema (Q3)",
    "the outputs, whose metadata stay in the manifest",
)


def _annotations(program):
    return sorted((value.id, _key(value.attrs[ANNOTATION]))
                  for value in program._values if ANNOTATION in value.attrs)


@pytest.mark.parametrize("path", sorted(INVENTORY))
def test_each_row_names_tests_that_exist(path):
    guard, tests = INVENTORY[path]
    assert guard and tests
    for test in tests:
        module, name = test.split("::")
        source = ROOT / module
        assert source.is_file(), test
        names = {node.name for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef)}
        assert name in names, test


def test_the_contract_has_three_erasure_points():
    assert len(ERASURE_POINTS) == 3


def test_projection_keeps_the_declaration_or_refuses():
    p = Program("erasure_projection")
    value = ScientificValue(p.float64(3), typ(L))
    raw = value.value
    assert ScientificValue(raw, typ(L)).value is raw            # the same declaration: kept
    with pytest.raises(TypeError, match="reannotation"):
        ScientificValue(raw, typ(T))                             # another one: refused
    assert _key(p._canonical_value(raw).attrs[ANNOTATION]) == _key(typ(L).to_data())


@pytest.mark.parametrize("clone", ("rebuild", "common subexpressions"))
def test_clone_keeps_every_declaration(clone):
    p = Program("erasure_clone")
    length = ScientificValue(p.float64(3), typ(L))
    ScientificValue(p.float64(2), typ(T))
    ScientificValue(length.value * length.value, typ(AREA))
    copy = (p._rebuild(lambda _value: True) if clone == "rebuild"
            else p.eliminate_common_subexpressions())
    assert _annotations(copy) == _annotations(p) and len(_annotations(p)) == 3
    with pytest.raises(TypeError, match="reannotation"):
        ScientificValue(copy._values[length.value.id], typ(T))


def test_a_closed_body_keeps_its_dimensions_or_refuses():
    square = graph(inputs=Product(x=Float64), output=Float64)(lambda g, args: args["x"] * args["x"])
    identity = graph(inputs=Product(x=Float64), output=Float64)(lambda g, args: args["x"])
    p = Program("erasure_call")
    length = ScientificValue(p.float64(3), typ(L))
    ScientificValue(identity(p, x=length.value), typ(L))
    ScientificValue(square(p, x=length.value), typ(AREA))
    with pytest.raises(TypeError, match="dimensions"):
        ScientificValue(square(p, x=length.value), typ(L))
