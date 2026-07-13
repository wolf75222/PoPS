"""Typed hierarchy-provider identity, validation, and open authoring contract."""

from __future__ import annotations

from fractions import Fraction

import pytest

from pops.identity.semantic import semantic_identity_of
from pops.ir.literals import scalar_data
from pops.linalg import LinearProblem
from pops.solvers import CG, CompositeTensorFAC, Hierarchy, HierarchySolveProvider
from pops.time import FailRun, Program


def _hierarchy_program(provider, *, name="hierarchy-provider"):
    program = Program(name)
    operator = program.matrix_free_operator("tensor", scope=Hierarchy(), provider=provider)
    program.set_apply(operator, lambda _program, _out, value: value)
    rhs = program.scalar_field("rhs")
    program.solve(
        LinearProblem(operator, rhs, scope=Hierarchy()),
        solver=CG(max_iter=11, rel_tol=1.0e-6),
        name="phi",
    ).consume(action=FailRun())
    return program, operator


@pytest.mark.parametrize(
    "option",
    [
        {"fine_sweeps": True},
        {"fine_sweeps": 1.5},
        {"fine_sweeps": 0},
        {"coarse_cycles": False},
        {"coarse_cycles": "4"},
        {"coarse_cycles": -1},
        {"coarse_rel_tol": True},
        {"coarse_rel_tol": 0},
        {"coarse_rel_tol": 1},
        {"coarse_rel_tol": float("nan")},
        {"coarse_rel_tol": float("inf")},
        {"verbose": 0},
        {"verbose": "true"},
    ],
)
def test_provider_options_are_strict(option):
    with pytest.raises((TypeError, ValueError)):
        CompositeTensorFAC(**option)


def test_omitted_options_preserve_native_default_authority_and_exact_identity():
    default = CompositeTensorFAC()
    assert default.fine_sweeps is None
    assert default.coarse_rel_tol is None
    assert default.coarse_cycles is None
    assert default.verbose is None
    assert default.canonical_identity()["options"] == {
        "fine_sweeps": None,
        "coarse_rel_tol": None,
        "coarse_cycles": None,
        "verbose": None,
    }

    configured = CompositeTensorFAC(
        fine_sweeps=7,
        coarse_rel_tol=Fraction(1, 8),
        coarse_cycles=9,
        verbose=False,
    )
    identity = configured.canonical_identity()
    assert identity["provider_id"] == "composite_tensor_fac"
    assert identity["capabilities"] == ["amr_hierarchy", "tensor_elliptic"]
    assert identity["options"] == {
        "fine_sweeps": 7,
        "coarse_rel_tol": scalar_data(Fraction(1, 8)),
        "coarse_cycles": 9,
        "verbose": False,
    }
    assert configured.identity != default.identity


def test_provider_identity_roundtrips_and_participates_in_every_program_identity():
    provider = CompositeTensorFAC(
        fine_sweeps=7, coarse_rel_tol=2.0e-7, coarse_cycles=9, verbose=True
    )
    left, operator = _hierarchy_program(provider)
    right, _ = _hierarchy_program(
        CompositeTensorFAC(fine_sweeps=7, coarse_rel_tol=2.0e-7, coarse_cycles=9, verbose=True)
    )
    changed, _ = _hierarchy_program(
        CompositeTensorFAC(fine_sweeps=8, coarse_rel_tol=2.0e-7, coarse_cycles=9, verbose=True)
    )

    solve = next(value for value in left._values if value.op == "solve_linear")
    expected = provider.canonical_identity()
    for frozen_identity in (
        operator.attrs["hierarchy_provider_identity"],
        solve.attrs["hierarchy_provider_identity"],
    ):
        assert frozen_identity["provider_id"] == expected["provider_id"]
        assert tuple(frozen_identity["capabilities"]) == tuple(expected["capabilities"])
        assert dict(frozen_identity["options"]) == expected["options"]
    serialized_solve = next(
        node
        for node in left._serialize(include_provenance=False)["nodes"]
        if node["op"] == "solve_linear"
    )
    assert serialized_solve["attrs"]["hierarchy_provider_identity"] == expected

    graph = left.to_graph()
    graph_hash = graph.graph_hash
    graph_solve = next(node for node in graph.nodes if node.kind == "solve")
    graph_identity = graph_solve.attrs.to_data()["attrs"]["hierarchy_provider_identity"]
    assert graph_identity["provider_id"] == "composite_tensor_fac"
    assert set(graph_identity["options"]) == {
        "fine_sweeps",
        "coarse_rel_tol",
        "coarse_cycles",
        "verbose",
    }
    assert left.to_graph().graph_hash == graph_hash

    assert left._ir_hash() == right._ir_hash()
    assert semantic_identity_of(program=left) == semantic_identity_of(program=right)
    assert graph_hash == right.to_graph().graph_hash
    assert left._ir_hash() != changed._ir_hash()
    assert semantic_identity_of(program=left) != semantic_identity_of(program=changed)
    assert graph_hash != changed.to_graph().graph_hash


class _ExternalHierarchyProvider:
    provider_id = "external_composite"
    capabilities = frozenset({"amr_hierarchy"})
    __pops_ir_immutable__ = True

    def __init__(self):
        self.options = {"external_knob": ["original"]}

    def canonical_identity(self):
        return {
            "schema_version": 1,
            "provider_id": self.provider_id,
            "capabilities": sorted(self.capabilities),
            "options": self.options,
        }


def test_structural_provider_is_accepted_for_authoring_then_refused_by_backend_lowering():
    provider = _ExternalHierarchyProvider()
    assert isinstance(provider, HierarchySolveProvider)
    program, operator = _hierarchy_program(provider, name="external-provider")
    assert operator.attrs["hierarchy_provider"] == "external_composite"

    # The Program keeps detached canonical data even if a dishonest third-party implementation
    # mutates a container after declaring the immutable extension marker.
    provider.options["external_knob"].append("mutated")
    assert tuple(operator.attrs["hierarchy_provider_identity"]["options"]["external_knob"]) == (
        "original",
    )

    from pops.codegen.program_emit_control import _emit_amr_hierarchy_bodies

    with pytest.raises(NotImplementedError, match="provider 'external_composite'.*not lowerable"):
        _emit_amr_hierarchy_bodies(program)
