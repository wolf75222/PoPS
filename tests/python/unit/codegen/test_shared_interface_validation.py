"""Resolve-time guards for atomic shared-interface RHS groups."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from pops.codegen._interface_validation import validate_shared_interface_program
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
            match=r"nested under control flow branch\.true_block.*top-level contiguous RHS group"):
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
            match=r"nested under control flow range\.body_block.*top-level contiguous RHS group"):
        _validate(program)
