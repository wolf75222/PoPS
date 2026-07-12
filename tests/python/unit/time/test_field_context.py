#!/usr/bin/env python3
"""ADC-588: the typed FieldContext a Program field solve carries.

``P.solve_fields(...)`` now returns a ProgramValue tagged with a real
:class:`pops.time.field_context.FieldContext` (previously a docstring-only concept). This pins:
  - the default solve is tagged with the ``phi`` problem and the historical phi/grad outputs;
  - a named field solve is tagged with that field's problem and its single output;
  - the context identifies every exact (block, stage-source) pair and rejects a cross-context read;
  - coupled field solves retain all block sources, never an arbitrary first-block projection;
  - every field-reading Program op enforces the token, while a local solve accepts a RHS whose
    provenance was explicitly propagated from the same token.

Pure Python (no _pops numerics beyond importing the package); no compilation.
"""
from typed_program_support import (
    fresh_state_refs,
    state_refs,
    typed_field,
    typed_state,
)

import sys

import pytest

from pops.time.field_context import DEFAULT_FIELD_PROBLEM, FieldContext, FieldReadProvenance
from pops.time.program import Program
from pops.time.points import TimePoint


def test_field_context_matches_and_rejects_triple():
    layout_outputs = ("phi", "grad_x", "grad_y")
    plasma, _ = fresh_state_refs("plasma")
    other, _ = fresh_state_refs("other")
    ctx = FieldContext("phi", ((plasma, 7),), layout_outputs)
    assert ctx.matches("phi", plasma, 7)
    assert ctx.matches(None, plasma, 7)  # None problem matches any (default case)
    assert not ctx.matches("phi", plasma, 8)  # stage mismatch
    assert not ctx.matches("phi", other, 7)  # block mismatch
    assert not ctx.matches("psi", plasma, 7)  # problem mismatch
    assert hash(ctx)
    with pytest.raises((AttributeError, TypeError)):
        ctx.stage_sources = ((plasma, 8),)


def test_require_read_raises_structured_error():
    plasma, _ = fresh_state_refs("plasma")
    other, _ = fresh_state_refs("other_block")
    ctx = FieldContext("phi", ((plasma, 7),))
    with pytest.raises(ValueError) as exc:
        ctx.require_read("phi", other, 7)
    msg = str(exc.value)
    assert "incompatible field context" in msg
    assert "plasma" in msg and "other_block" in msg


def test_output_lookup_names_known_outputs():
    plasma, _ = fresh_state_refs("plasma")
    ctx = FieldContext("phi", ((plasma, 0),), ("phi", "grad_x", "grad_y"))
    assert ctx.output("grad_y") == "grad_y"
    with pytest.raises(KeyError) as exc:
        ctx.output("E")
    # KeyError repr keeps the message; it must name the missing handle and the known set.
    assert "E" in str(exc.value) and "phi" in str(exc.value)


def test_default_field_problem_sentinel():
    plasma, _ = fresh_state_refs("plasma")
    ctx = FieldContext(None, ((plasma, 0),))
    assert ctx.field_problem == DEFAULT_FIELD_PROBLEM == "phi"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field_problem": object(), "stage_sources": (("plasma", 0),)},
        {"field_problem": "phi", "stage_sources": ((object(), 0),)},
        {"field_problem": "phi", "stage_sources": (("plasma", []),)},
        {"field_problem": "phi", "stage_sources": (("plasma", True),)},
        {"field_problem": "phi", "stage_sources": (("plasma", 0),), "outputs": [object()]},
    ],
)
def test_field_context_rejects_mutable_or_unstable_identity_leaves(kwargs):
    with pytest.raises(TypeError):
        FieldContext(**kwargs)


def test_field_context_detaches_container_inputs_and_remains_hashable():
    plasma, _ = fresh_state_refs("plasma")
    other, _ = fresh_state_refs("other")
    sources = [[plasma, 0]]
    outputs = ["phi"]
    context = FieldContext("phi", sources, outputs)
    sources[0][0] = other
    outputs.append("grad_x")

    assert context.stage_sources == ((plasma, 0),)
    assert context.outputs == ("phi",)
    assert hash(context)


def test_solve_fields_tags_default_context():
    P = Program("p")
    U = typed_state(P, "plasma")
    block, _ = state_refs(P, "plasma")
    f = P.solve_fields(state=U)
    assert f.vtype == "fields"
    ctx = f.field_context
    assert ctx.field_problem == "phi"
    assert ctx.stage_sources == ((block, U.id),)
    assert ctx.outputs == ("phi", "grad_x", "grad_y")
    # The default op keeps empty physics attrs; provenance has its own canonical IR field.
    assert f.attrs == {}


def test_solve_fields_named_field_context():
    P = Program("p")
    U = typed_state(P, "plasma")
    psi = typed_field(P, "psi")
    g = P.solve_fields(name="psi_solve", state=U, field=psi)
    ctx = g.field_context
    assert ctx.field_problem is psi
    assert ctx.outputs == ("psi",)
    # The named field records its route in the IR (non-empty attrs) as before ADC-588.
    assert g.attrs == {"field": psi}


def test_solves_from_different_states_have_distinct_stage_sources():
    P = Program("p")
    Ua = typed_state(P, "blockA")
    Ub = typed_state(P, "blockB")
    block_a, _ = state_refs(P, "blockA")
    block_b, _ = state_refs(P, "blockB")
    fa = P.solve_fields(state=Ua)
    fb = P.solve_fields(state=Ub)
    # Each call carries its OWN context object; solves from different stage states are distinct
    # (different block AND different stage source), so one cannot be read as the other.
    assert fa.field_context is not fb.field_context
    assert fa.field_context.stage_sources == ((block_a, Ua.id),)
    assert fb.field_context.stage_sources == ((block_b, Ub.id),)
    assert not fa.field_context.matches("phi", block_b, Ub.id)


def test_coupled_context_tracks_every_block_source_and_rejects_stale_stage():
    P = Program("coupled")
    Ua = typed_state(P, "a")
    Ub = typed_state(P, "b")
    block_a, _ = state_refs(P, "a")
    block_b, _ = state_refs(P, "b")
    fields = P.solve_fields_from_blocks((Ua, Ub))

    assert fields.block is None
    assert fields.field_context.stage_sources == ((block_a, Ua.id), (block_b, Ub.id))
    assert P._rhs_legacy(state=Ua, fields=fields, sources=[]).block is block_a
    assert P._rhs_legacy(state=Ub, fields=fields, sources=[]).block is block_b

    Ua_stage = P.linear_combine("a_stage", Ua)
    with pytest.raises(ValueError, match="incompatible field context"):
        P._rhs_legacy(state=Ua_stage, fields=fields, sources=[])


def test_field_consumers_reject_stale_state_but_local_solve_accepts_derived_rhs():
    P = Program("derived")
    U0 = typed_state(P, "plasma")
    fields0 = P.solve_fields(U0)
    R0 = P._rhs_legacy(state=U0, fields=fields0, sources=[])
    q = P.linear_combine("q", U0 + P.dt * R0, at=TimePoint(P.clock, 1))
    linear = P._linear_source("relax")

    # q carries the exact context through the RHS graph, so the implicit local solve is valid.
    solved = P.solve_local_linear(
        "solved", operator=P.I - P.dt * linear, rhs=q, fields=fields0)
    assert q.field_context == fields0.field_context
    assert solved.field_context == fields0.field_context

    # Physics evaluation is stricter: q is a new stage State and must first get its own field solve.
    with pytest.raises(ValueError, match="incompatible field context"):
        P._rhs_legacy(state=q, fields=fields0, sources=[])
    with pytest.raises(ValueError, match="incompatible field context"):
        P._source("reaction", state=q, fields=fields0)
    with pytest.raises(ValueError, match="incompatible field context"):
        P._apply(linear, state=q, fields=fields0)

    plain_stage = P.linear_combine("plain_stage", U0)
    with pytest.raises(ValueError, match="incompatible field context"):
        P.solve_local_linear(
            "stale", operator=P.I - P.dt * linear, rhs=plain_stage, fields=fields0)


def test_multistage_provenance_is_explicit_and_operator_context_cannot_be_substituted():
    P = Program("multistage")
    U0 = typed_state(P, "plasma")
    fields0 = P.solve_fields(U0)
    R0 = P._rhs_legacy(state=U0, fields=fields0, sources=[])
    U1 = P.linear_combine("U1", U0 + P.dt * R0, at=TimePoint(P.clock, 1))
    fields1 = P.solve_fields(U1)
    R1 = P._rhs_legacy(state=U1, fields=fields1, sources=[])
    q = P.linear_combine(
        "q", U0 + P.dt * R0 + P.dt * R1, at=TimePoint(P.clock, step=1))

    assert isinstance(q.field_context, FieldReadProvenance)
    assert q.field_context.contexts == (fields0.field_context, fields1.field_context)

    # Typed P.call(L, fields1) carries this validation witness. The runtime semantics still live on
    # solve_local_linear's explicit fields input, but substituting fields0 now fails at authoring.
    linear = P._replace_value(
        P._linear_source("relax"), field_context=fields1.field_context)
    solved = P.solve_local_linear(
        "solved", operator=P.I - P.dt * linear, rhs=q, fields=fields1)
    assert isinstance(solved.field_context, FieldReadProvenance)
    with pytest.raises(ValueError, match="operator was authored for field provenance"):
        P.solve_local_linear(
            "wrong_fields", operator=P.I - P.dt * linear, rhs=q, fields=fields0)


def test_rk4_final_combine_retains_all_stage_contexts_without_rejection():
    from pops.lib import time as libtime

    block, state = fresh_state_refs("plasma")
    program = libtime.rk4(block, state)
    final = next(iter(program._commits.values()))
    assert isinstance(final.field_context, FieldReadProvenance)
    assert len(final.field_context.contexts) == 4
    serialized = program._serialize()
    final_data = next(node for node in serialized["nodes"] if node["id"] == final.id)
    reads = final_data["field_context"]["reads"]
    assert len(reads) == 4
    assert program._serialize() == serialized
    inspected = next(node for node in program.ir_nodes() if node["name"] == final.name)
    assert inspected["field_context"] == final_data["field_context"]

    rebuilt = program.eliminate_dead_nodes()
    rebuilt_final = next(iter(rebuilt._commits.values()))
    source_ids = {source for context in rebuilt_final.field_context.contexts
                  for _, source in context.stage_sources}
    assert source_ids <= {value.id for value in rebuilt._values}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
