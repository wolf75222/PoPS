#!/usr/bin/env python3
"""ADC-588: the typed FieldContext a Program field solve carries.

``P.solve_fields(...)`` now returns a Value tagged with a real
:class:`pops.time.field_context.FieldContext` (previously a docstring-only concept). This pins:
  - the default solve is tagged with the ``phi`` problem and the historical phi/grad outputs;
  - a named field solve is tagged with that field's problem and its single output;
  - the context identifies the (problem, block, stage-source) triple and rejects a cross-context
    read with a structured error;
  - the default ``solve_fields`` IR stays byte-identical (empty ``attrs``) so the compiled-Program
    cache key is unchanged (parity: the context is build-time metadata, never serialized).

Pure Python (no _pops numerics beyond importing the package); no compilation.
"""
import sys

import pytest

from pops.time.field_context import DEFAULT_FIELD_PROBLEM, FieldContext
from pops.time.program import Program


def test_field_context_matches_and_rejects_triple():
    layout_outputs = ("phi", "grad_x", "grad_y")
    ctx = FieldContext("phi", "plasma", 7, layout_outputs)
    assert ctx.matches("phi", "plasma", 7)
    assert ctx.matches(None, "plasma", 7)  # None problem matches any (default case)
    assert not ctx.matches("phi", "plasma", 8)  # stage mismatch
    assert not ctx.matches("phi", "other", 7)  # block mismatch
    assert not ctx.matches("psi", "plasma", 7)  # problem mismatch


def test_require_read_raises_structured_error():
    ctx = FieldContext("phi", "plasma", 7)
    with pytest.raises(ValueError) as exc:
        ctx.require_read("phi", "other_block", 7)
    msg = str(exc.value)
    assert "incompatible field context" in msg
    assert "plasma" in msg and "other_block" in msg


def test_output_lookup_names_known_outputs():
    ctx = FieldContext("phi", "plasma", 0, ("phi", "grad_x", "grad_y"))
    assert ctx.output("grad_y") == "grad_y"
    with pytest.raises(KeyError) as exc:
        ctx.output("E")
    # KeyError repr keeps the message; it must name the missing handle and the known set.
    assert "E" in str(exc.value) and "phi" in str(exc.value)


def test_default_field_problem_sentinel():
    ctx = FieldContext(None, "plasma", 0)
    assert ctx.field_problem == DEFAULT_FIELD_PROBLEM == "phi"


def test_solve_fields_tags_default_context():
    P = Program("p")
    U = P.state("plasma")
    f = P.solve_fields(state=U)
    assert f.vtype == "fields"
    ctx = f.field_context
    assert ctx.field_problem == "phi"
    assert ctx.block == "plasma"
    assert ctx.stage_source == U.id
    assert ctx.outputs == ("phi", "grad_x", "grad_y")
    # PARITY: the default op keeps empty attrs -> byte-identical .so cache key.
    assert f.attrs == {}


def test_solve_fields_named_field_context():
    P = Program("p")
    U = P.state("plasma")
    g = P.solve_fields(name="psi_solve", state=U, field="psi")
    ctx = g.field_context
    assert ctx.field_problem == "psi"
    assert ctx.outputs == ("psi",)
    # The named field records its route in the IR (non-empty attrs) as before ADC-588.
    assert g.attrs == {"field": "psi"}


def test_solves_from_different_states_have_distinct_stage_sources():
    P = Program("p")
    Ua = P.state("blockA")
    Ub = P.state("blockB")
    fa = P.solve_fields(state=Ua)
    fb = P.solve_fields(state=Ub)
    # Each call carries its OWN context object; solves from different stage states are distinct
    # (different block AND different stage source), so one cannot be read as the other.
    assert fa.field_context is not fb.field_context
    assert fa.field_context.stage_source == Ua.id
    assert fb.field_context.stage_source == Ub.id
    assert not fa.field_context.matches("phi", "blockB", Ub.id)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
