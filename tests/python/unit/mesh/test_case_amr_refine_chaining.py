"""Spec 5 sec.5.14 / sec.8.6: case.amr.refine records authenticated criteria.

``case.amr`` is a thin authoring shim, NOT a separate AMR engine. It resolves every model-local
criterion through the case's block registry, records the resolved criterion in the sole constraint
registry and returns the case so the call chains. The user-owned layout stays unchanged; compile
materialises a detached layout from those constraints.

Pure Python; needs only ``import pops`` (nothing computes on a grid).
"""
import pytest

pops = pytest.importorskip("pops")

from pops.mesh.amr import (  # noqa: E402
    PatchLayout, ProperNesting, Refine, RegridEvery, TagUnion)
from pops.mesh.cartesian import CartesianMesh  # noqa: E402
from pops.mesh.layouts import AMR  # noqa: E402
from pops.model import (  # noqa: E402
    AmbiguousReferenceError, DeclarationIndex, Handle, MissingOwnershipError,
    OperatorHandle, OwnerKind, OwnerPath)
from pops.ir.ops import dx, dy, sqrt  # noqa: E402
from pops.ir import ValueExpr  # noqa: E402
from pops.problem.handles import (  # noqa: E402
    FieldHandle as ProblemFieldHandle, OperatorHandle as ProblemOperatorHandle)


class _FakeModel:
    """A minimal model advertising its declared subjects (mirrors HyperbolicModel's surface)."""

    def __init__(self):
        self.name = "amr-model"
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)
        self.rho = Handle("rho", kind="state", owner=self.owner_path)

    def declaration_index(self):
        return DeclarationIndex(owner=self.owner_path, handles=(self.rho,))


def _case():
    model = _FakeModel()
    case = pops.Problem(layout=AMR(base=CartesianMesh(n=32)))
    case.block("ne", model)
    return case, model


def _gradient_norm(handle):
    value = ValueExpr(handle)
    return sqrt(dx(value) ** 2 + dy(value) ** 2)


def _gradient_norm_leaf(expression):
    return expression.a.a.a.field.handle


def test_refine_records_a_resolved_criterion_without_mutating_the_layout():
    case, model = _case()
    criterion = Refine.on(model.rho).above(0.1)
    assert case.amr.refine(criterion) is case
    stored = case._constraints.refinement["refine"]
    assert stored is not criterion
    assert stored.subject.is_resolved
    assert stored.subject.is_instance
    assert case.layout.refine is None


def test_refine_chains_regrid_nesting_and_patches_in_one_call():
    case, model = _case()
    regrid = RegridEvery(4)
    nesting = ProperNesting(buffer=2)
    patches = PatchLayout(distribute_coarse=True, coarse_max_grid=16)
    result = case.amr.refine(Refine.on(model.rho).above(0.1),
                             regrid=regrid, nesting=nesting, patches=patches)
    assert result is case
    slots = case._constraints.refinement
    assert slots["regrid"] is regrid
    assert slots["nesting"] is nesting
    assert slots["patches"] is patches


def test_refine_slots_chain_across_separate_calls():
    # Each slot can also be set on its own call; later calls only touch the slots they pass.
    case, _ = _case()
    case.amr.refine(regrid=RegridEvery(2))
    case.amr.refine(nesting=ProperNesting(buffer=1))
    case.amr.refine(patches=PatchLayout(coarse_max_grid=8))
    slots = case._constraints.refinement
    assert slots["regrid"].steps == 2
    assert slots["nesting"].buffer == 1
    assert slots["patches"].coarse_max_grid == 8
    # No criterion was ever passed: refine stays unset.
    assert "refine" not in slots


def test_refine_tag_union_is_resolved_in_the_constraint_registry():
    case, model = _case()
    case.amr.refine(TagUnion(Refine.on(model.rho).above(0.1),
                             Refine.on(model.rho).below(-0.1)),
                    regrid=RegridEvery(3))
    slots = case._constraints.refinement
    union = slots["refine"]
    assert isinstance(union, TagUnion)
    assert len(union.criteria) == 2
    assert all(criterion.subject.is_resolved for criterion in union.criteria)
    assert slots["regrid"].steps == 3


def test_refine_rejects_an_unregistered_handle_before_writing_the_registry():
    case, model = _case()
    ghost = Handle("definitely_not_a_role", kind="state", owner=model.owner_path)
    with pytest.raises(MissingOwnershipError):
        case.amr.refine(Refine.on(ghost).above(0.05))
    # The rejected criterion never lands in the authoritative registry (fail before mutate).
    assert "refine" not in case._constraints.refinement


def test_refine_rejects_a_string_subject_before_resolution():
    with pytest.raises(TypeError, match="names and strings"):
        Refine.on("rho")


def test_expression_indicator_is_resolved_recursively_without_mutating_the_source_graph():
    case, model = _case()
    indicator = _gradient_norm(model.rho)
    case.amr.refine(Refine.on(indicator).above(0.2))

    stored = case._constraints.refinement["refine"].subject
    assert stored is not indicator
    assert _gradient_norm_leaf(stored).is_resolved
    assert _gradient_norm_leaf(stored).is_instance
    assert not _gradient_norm_leaf(indicator).is_resolved


def test_reused_model_requires_explicit_block_handle_and_lists_candidate_owners():
    model = _FakeModel()
    case = pops.Problem(name="multi", layout=AMR(base=CartesianMesh(n=32)))
    left = case.block("left", model)
    case.block("right", model)

    with pytest.raises(AmbiguousReferenceError) as exc:
        case.amr.refine(Refine.on(_gradient_norm(model.rho)).above(0.1))
    message = str(exc.value)
    assert "block:left" in message and "block:right" in message
    assert "block[declaration]" in message
    assert case._constraints.refinement == {}

    case.amr.refine(Refine.on(_gradient_norm(left[model.rho])).above(0.1))
    subject = _gradient_norm_leaf(case._constraints.refinement["refine"].subject)
    assert subject.is_resolved and subject.is_instance
    assert subject.block_ref.local_id == "left"


def test_refine_rejects_block_and_operator_control_handles_as_non_values():
    case, model = _case()
    block = case.blocks()["ne"]
    operator = OperatorHandle("rhs", kind="local_rate", owner=model.owner_path)

    for reference in (block, operator):
        with pytest.raises(TypeError, match="value-readable"):
            Refine.on(reference)
        with pytest.raises(TypeError, match="readable value Handle"):
            ValueExpr(reference)

    problem_operator = ProblemOperatorHandle("coupling", owner=case.owner_path)
    with pytest.raises(TypeError, match="value-readable"):
        Refine.on(problem_operator)
    with pytest.raises(TypeError, match="readable value Handle"):
        ValueExpr(problem_operator)

    field = ProblemFieldHandle("phi", owner=case.owner_path)
    assert field.expression_readable is True
    assert ValueExpr(field).handle is field


def test_direct_layout_rejects_canonical_foreign_and_nested_ghost_handles():
    from pops.codegen.orchestration import _resolve_layout

    _, model = _case()
    case = pops.Problem(name="layout-auth")
    case.block("ne", model)
    ghost = Handle("ghost", kind="state", owner=model.owner_path.canonical())
    foreign = Handle("rho", kind="state", owner=OwnerPath.model("foreign-model"))

    direct = AMR(base=CartesianMesh(n=32), refine=Refine.on(foreign).above(0.1))
    with pytest.raises(MissingOwnershipError):
        _resolve_layout(case, direct)

    nested = AMR(
        base=CartesianMesh(n=32),
        refine=Refine.on(_gradient_norm(ghost)).above(0.1),
    )
    with pytest.raises(MissingOwnershipError):
        _resolve_layout(case, nested)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
