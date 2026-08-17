"""Authoring-only proofs for hold-then-catch-up and the implicit-source mask."""

from __future__ import annotations

from pops.time import (
    HoldCatchupBlock,
    adaptive_strides,
    hold_catchup_program,
    step_adaptive_program,
)
from pops.time._program.operations import _keep_components
from pops.model.spaces import StateSpace


def test_adaptive_strides_oracle() -> None:
    assert adaptive_strides({"fast": 4.0, "slow": 1.0}) == {"fast": 1, "slow": 4}
    assert adaptive_strides({"a": 5.0, "b": 2.5, "c": 1.0}) == {"a": 1, "b": 2, "c": 5}


def test_keep_components_resolves_names_and_complement() -> None:
    space = StateSpace("U", components=("rho", "q"), roles={"rho": "density", "q": "scalar"})
    assert _keep_components(
        space, implicit_vars=("q",), implicit_roles=(), complement=False
    ) == (1,)
    assert _keep_components(
        space, implicit_vars=("q",), implicit_roles=(), complement=True
    ) == (0,)
    assert _keep_components(
        space, implicit_vars=(), implicit_roles=(), complement=False
    ) == (0, 1)


def test_hold_catchup_program_uses_cadence_and_subcycle() -> None:
    from pops.time import Program
    from typed_program_support import state_refs

    holder = Program("hold-catchup-unit")
    fast_block, fast_decl = state_refs(holder, "fast")
    slow_block, slow_decl = state_refs(holder, "slow")
    authored = hold_catchup_program(
        [
            HoldCatchupBlock(fast_block[fast_decl], stride=1, linear_rate=-0.5, name="fast"),
            HoldCatchupBlock(slow_block[slow_decl], stride=4, linear_rate=-0.25, name="slow"),
        ]
    )
    assert authored.cadence_contract().stride == 4
    ops = [node["op"] for node in authored.ir_nodes()]
    assert "subcycle" in ops
    assert "synchronize" in ops


def test_step_adaptive_program_uses_oracle_strides() -> None:
    from pops.time import Program
    from typed_program_support import state_refs

    holder = Program("step-adaptive-unit")
    fast_block, fast_decl = state_refs(holder, "fast")
    slow_block, slow_decl = state_refs(holder, "slow")
    authored = step_adaptive_program(
        {"fast": fast_block[fast_decl], "slow": slow_block[slow_decl]},
        {"fast": 4.0, "slow": 1.0},
        linear_rates={"fast": -0.5, "slow": -0.25},
    )
    assert authored.cadence_contract().stride == 4


def test_hold_catchup_imex_mask_authors_implicit_source() -> None:
    from pops.model.spaces import StateSpace
    from pops.time import Program
    from typed_program_support import state_refs

    space = StateSpace("U", components=("rho", "q"), roles={"rho": "density", "q": "scalar"})
    holder = Program("imex-mask-unit")
    block, decl = state_refs(holder, "fluid", space=space)
    authored = hold_catchup_program(
        [
            HoldCatchupBlock(
                block[decl],
                stride=1,
                kind="imex",
                implicit_vars=("q",),
                name="fluid",
            )
        ]
    )
    ops = [node["op"] for node in authored.ir_nodes()]
    assert ops.count("implicit_source") == 2
    keeps = [
        node["attrs"]["keep_components"]
        for node in authored.ir_nodes()
        if node["op"] == "implicit_source"
    ]
    assert [0] in keeps and [1] in keeps
    from pops.codegen.program_lowerability import check_schedules_lowerable

    check_schedules_lowerable(authored, target="amr_system")
