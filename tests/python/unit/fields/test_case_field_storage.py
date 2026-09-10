"""Independent Case field unknowns preserve exact ownership and existing provider authority."""
from __future__ import annotations

import pytest

import pops
from pops.fields import FieldDiscretization, FieldProblem, SharedMeanGauge
from pops.fields.methods import CellCenteredSecondOrder
from pops.math import laplacian
from pops.model import Handle, OwnerPath, OwnerKind
from pops.model.ownership import MissingOwnershipError
from pops.solvers import CartesianCG


def _registration(case, *, name="joint", unknowns=None):
    if unknowns is None:
        unknowns = tuple(Handle("phi", kind="field",
                                owner=OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, owner))
                         for owner in ("first", "second"))
    problem = FieldProblem(name, unknowns=unknowns,
                           equations=tuple(-laplacian(item) == 0 for item in unknowns),
                           gauge=SharedMeanGauge(unknowns))
    method = FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(),
                                 solver=CartesianCG())
    return case.field(problem, method), problem


def test_unknowns_have_case_field_storage_without_any_species_block():
    case = pops.Case("field assembly")
    field, problem = _registration(case)
    first, second = (field[unknown] for unknown in problem.unknowns)
    assert first.local_id == second.local_id == "phi"
    assert first != second
    assert first.block_ref is second.block_ref is None
    assert first.owner_path.nodes[0].kind is OwnerKind.CASE
    assert case.resolve(problem.unknowns[0]) == case.resolve(first)
    assert case.resolve(problem.unknowns[1]) == case.resolve(second)
    resolved = case._field_registry.resolved_registration(field)
    assert resolved.operator.unknowns == (case.resolve(first), case.resolve(second))
    assert all(unknown.is_resolved for unknown in resolved.operator.unknowns)
    assert len(case.blocks()) == 0


def test_registered_unknowns_refuse_foreign_case_and_species_qualification():
    first_case, second_case = pops.Case("one"), pops.Case("two")
    field, problem = _registration(first_case)
    foreign, foreign_problem = _registration(second_case)
    with pytest.raises(MissingOwnershipError):
        first_case.resolve(foreign[foreign_problem.unknowns[0]])
    with pytest.raises(MissingOwnershipError):
        field[foreign_problem.unknowns[0]]
    with pytest.raises(TypeError, match="do not accept block"):
        first_case.resolve(problem.unknowns[0], block=object())
    with pytest.raises(ValueError, match="already belongs"):
        _registration(first_case, name="conflicting", unknowns=problem.unknowns)
    assert tuple(first_case.fields()) == ("joint",)


def test_canonical_storage_identity_roundtrips_and_detached_handle_cannot_rebind():
    case = pops.Case("roundtrip")
    field, problem = _registration(case)
    canonical = case.resolve(field[problem.unknowns[0]])
    restored = Handle.from_canonical_identity(canonical.canonical_identity())
    assert case.resolve(restored) == canonical
    detached = case.resolve(field)
    with pytest.raises(MissingOwnershipError, match="detached"):
        detached[problem.unknowns[0]]
    with pytest.raises(MissingOwnershipError, match="detached"):
        detached.default_program_solver()


def test_field_storage_registration_freezes_with_its_case_registry():
    case = pops.Case("sealed")
    field, problem = _registration(case)
    original_identity = case.resolve(field[problem.unknowns[0]]).canonical_identity()
    case._field_registry.freeze()
    assert case.resolve(problem.unknowns[0]).canonical_identity() == original_identity
    with pytest.raises(RuntimeError, match="frozen"):
        _registration(case, name="later")
