"""Resolve-time guards for atomic shared-interface RHS groups."""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pops.codegen._interface_validation import validate_shared_interface_program
from pops.codegen.program_emit_control import _emit_contiguous_rhs_group
from pops.codegen.program_codegen import emit_cpp_program
from pops.numerics.terms import Flux
from pops.time import Program, TimePoint
from typed_program_support import typed_state


@dataclass(frozen=True)
class _Layout:
    qualified_id: str


@dataclass(frozen=True)
class _Side:
    boundary: str
    layout: _Layout


class _Interface:
    qualified_id = "case::shared-interface"

    def __init__(self, layout: _Layout) -> None:
        self.left = _Side("left.xhi", layout)
        self.right = _Side("right.xlo", layout)

    def canonical_identity(self) -> dict[str, object]:
        return {
            "qualified_id": self.qualified_id,
            "left": self.left.boundary,
            "right": self.right.boundary,
            "layout": self.left.layout.qualified_id,
        }


def _resolved_context() -> tuple[tuple[object, ...], object]:
    layout = _Layout("case::layout")
    interface = _Interface(layout)

    def block(name: str, boundary: str) -> object:
        plan = SimpleNamespace(
            productions=(SimpleNamespace(
                region=SimpleNamespace(boundary=boundary)),),
            interfaces=(interface,),
        )
        return SimpleNamespace(
            name=name,
            numerics=SimpleNamespace(boundaries=(plan,)),
        )

    blocks = (
        block("left", interface.left.boundary),
        block("right", interface.right.boundary),
    )
    layout_plan = SimpleNamespace(assignments=tuple(
        SimpleNamespace(
            subject_kind="block",
            subject=SimpleNamespace(local_id=name),
            layout=layout,
        )
        for name in ("left", "right")
    ))
    return blocks, layout_plan


def _validate(program: Program) -> None:
    blocks, layout_plan = _resolved_context()
    validate_shared_interface_program(
        blocks, layout_plan, program, target="system")


def test_shared_interface_rejects_default_flux_rhs_nested_in_branch() -> None:
    program = Program("nested_shared_branch")
    left = typed_state(program, "left")
    typed_state(program, "right")
    condition = program.norm2(left) > 0
    program.branch(
        condition,
        lambda branch: branch.rhs(
            "left_true_rate", state=left, terms=[Flux()]),
        lambda branch: branch.rhs(
            "left_false_rate", state=left, terms=[Flux()]),
    )

    with pytest.raises(
            NotImplementedError,
            match=r"nested under control flow branch\.true_block.*top-level coherence round"):
        _validate(program)


def test_shared_interface_rejects_default_flux_rhs_nested_in_loop() -> None:
    program = Program("nested_shared_loop")
    left = typed_state(program, "left")
    typed_state(program, "right")

    def body(loop: Program, state: object) -> object:
        rate = loop.rhs("left_loop_rate", state=state, terms=[Flux()])
        return loop.value(
            "left_loop_state", state + loop.dt * rate,
            at=TimePoint(loop.clock, step=1),
        )

    program.range(left, 2, body)

    with pytest.raises(
            NotImplementedError,
            match=r"nested under control flow range\.body_block.*top-level coherence round"):
        _validate(program)


def test_group_codegen_keeps_atomic_and_per_rate_identities_distinct() -> None:
    program = Program("group_identity")
    left_block = object()
    right_block = object()
    left_state = SimpleNamespace(id=3)
    right_state = SimpleNamespace(id=4)
    point = TimePoint(program.clock, Fraction(1, 2))
    left_rate = SimpleNamespace(
        id=11, name="left_rate", point=point, block=left_block,
        inputs=(left_state,), attrs={"sources": None})
    right_rate = SimpleNamespace(
        id=12, name="right_rate", point=point, block=right_block,
        inputs=(right_state,), attrs={"sources": ()})
    variables = {3: "u3", 4: "u4"}
    lines: list[str] = []

    _emit_contiguous_rhs_group(
        [left_rate, right_rate], {left_block: 0, right_block: 1},
        variables, lines, group_identity=29)

    assert lines[-1] == (
        "ctx.rhs_group(29, {{0, &u3, &r11, 11, 0}, "
        "{1, &u4, &r12, 12, 1}});")


def test_validation_and_codegen_share_noncontiguous_stage_coherence_plan() -> None:
    """A pure SSA node between sibling rates must not expose an accepted-state fallback."""
    program = Program("noncontiguous_boundary_stage_dependencies")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    left_rate = program.rhs("left_rate", state=left.n, terms=[Flux()])
    program.norm2(left.n)  # deliberately separates the two same-StagePoint residual nodes
    right_rate = program.rhs("right_rate", state=right.n, terms=[Flux()])
    left_next = program.value(
        "left_next", left.n + program.dt * left_rate, at=left.next.point)
    right_next = program.value(
        "right_next", right.n + program.dt * right_rate, at=right.next.point)
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)

    # Resolve-time validation and both native targets consume the same planner. This is the load-
    # bearing integration point that prevents a validator/emitter semantic split.
    _validate(program)
    for target in ("system", "amr_system"):
        source = emit_cpp_program(program, target=target)
        assert source.count("ctx.rhs_group(") == 1
        assert "ctx.neg_div_flux_default_into(" not in source
        assert "ctx.rhs_into(" not in source
        group = next(line for line in source.splitlines() if "ctx.rhs_group(" in line)
        requests = re.findall(r"\{(\d+), &u\d+, &r\d+, \d+, 1\}", group)
        assert requests == ["0", "1"]


def test_same_stage_repeated_blocks_form_two_deterministic_rhs_rounds() -> None:
    """A0,B0,A1,B1 must lower as two atomic groups, never four one-sided evaluations."""
    program = Program("two_shared_interface_rounds")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    rates = (
        program.rhs("left_rate_0", state=left.n, terms=[Flux()]),
        program.rhs("right_rate_0", state=right.n, terms=[Flux()]),
        program.rhs("left_rate_1", state=left.n, terms=[Flux()]),
        program.rhs("right_rate_1", state=right.n, terms=[Flux()]),
    )
    left_next = program.value(
        "left_next", left.n + program.dt * (rates[0] + rates[2]), at=left.next.point)
    right_next = program.value(
        "right_next", right.n + program.dt * (rates[1] + rates[3]), at=right.next.point)
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)

    _validate(program)
    for target in ("system", "amr_system"):
        source = emit_cpp_program(program, target=target)
        groups = [line for line in source.splitlines() if "ctx.rhs_group(" in line]
        assert len(groups) == 2
        assert [
            re.findall(r"\{(\d+), &u\d+, &r\d+, \d+, 1\}", group)
            for group in groups
        ] == [["0", "1"], ["0", "1"]]


def test_shared_interface_rejects_incomplete_second_rhs_round() -> None:
    program = Program("incomplete_shared_interface_round")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    program.rhs("left_rate_0", state=left.n, terms=[Flux()])
    program.rhs("right_rate_0", state=right.n, terms=[Flux()])
    program.rhs("left_rate_1", state=left.n, terms=[Flux()])

    with pytest.raises(ValueError, match=r"right.*coherence round 1"):
        _validate(program)


def test_rhs_coherence_rejects_interleaved_rounds() -> None:
    program = Program("interleaved_shared_interface_rounds")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    program.rhs("left_rate_0", state=left.n, terms=[Flux()])
    program.rhs("left_rate_1", state=left.n, terms=[Flux()])
    program.rhs("right_rate_0", state=right.n, terms=[Flux()])
    program.rhs("right_rate_1", state=right.n, terms=[Flux()])

    with pytest.raises(ValueError, match="rounds.*interleaved"):
        _validate(program)


def test_amr_codegen_refuses_to_move_stage_group_across_side_effect() -> None:
    """Atomic sibling evaluation must never reorder an authored diagnostic side effect."""
    program = Program("boundary_stage_ordering_barrier")
    left = typed_state(program, "left", state_name="U")
    right = typed_state(program, "right", state_name="U")
    left_rate = program.rhs("left_rate", state=left.n, terms=[Flux()])
    diagnostic = program.norm2(left.n)
    program.record_scalar("before_right_rate", diagnostic)
    right_rate = program.rhs("right_rate", state=right.n, terms=[Flux()])
    left_next = program.value(
        "left_next", left.n + program.dt * left_rate, at=left.next.point)
    right_next = program.value(
        "right_next", right.n + program.dt * right_rate, at=right.next.point)
    program.commit(left.next, left_next)
    program.commit(right.next, right_next)

    with pytest.raises(ValueError, match="ordering barrier.*record_scalar"):
        _validate(program)
