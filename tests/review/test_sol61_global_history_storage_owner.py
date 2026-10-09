"""Source-only independent storage-owner witnesses; no native execution/JIT.

The physical coupled field is global and independent of the chosen history
storage block. These baseline probes do not qualify the future owner_block API.
"""
from __future__ import annotations

import pytest

import pops
from pops.codegen.program_emit_amr import _emit_checkpoint_shape_metadata
from pops.domain import Rectangle
from pops.fields import FieldBoundary, FieldDiscretization, FieldProblem, bcs
from pops.fields.methods import CellCenteredSecondOrder
from pops.frames import Cartesian2D
from pops.math import ddt, div, laplacian
from pops.model import Handle, OwnerPath
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.solvers import CG
from pops.time import FailRun


def authored():
    frame = Rectangle("storage-domain", lower=(0, 0), upper=(1, 1)).frame(Cartesian2D())
    case = pops.Case("storage-case")
    program = pops.Program("global-coupled-field")
    blocks, states, inputs = [], [], {}
    for name in ("thermal", "matter", "spectator"):
        model = pops.Model("physical-"+name, frame=frame)
        state = model.state("U", components=("load", "independent-component"))
        flux = model.flux("stationary", frame=frame, state=state,
            components={axis: tuple(0*x for x in state) for axis in frame.axes},
            waves={axis: tuple(0*x for x in state) for axis in frame.axes})
        rate = model.rate("balance", equation=ddt(state) == -div(flux))
        numerics = DiscretizationPlan()
        numerics.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
        block = case.block(name, model)
        case.numerics(numerics, block=block)
        current = program.state(block[state])
        blocks.append(block)
        states.append(state)
        inputs[block[state]] = current.n
    unknowns = tuple(Handle(name, kind="field", owner=OwnerPath.model("global-equations"))
                     for name in ("T", "z", "w"))
    from pops._ir.elliptic import Reaction
    equations = []
    for i, (q, state) in enumerate(zip(unknowns, states, strict=True)):
        lhs = -laplacian(q)+Reaction(q, 2)
        for j, other in enumerate(unknowns):
            if i != j:
                lhs -= Reaction(other, .25)
        equations.append(lhs == state[0])
    problem = FieldProblem("original-global", unknowns=unknowns, equations=tuple(equations),
        boundaries=tuple(FieldBoundary(q, bcs.BoundaryCondition(bcs.AllPhysicalBoundaries(), bcs.Periodic()))
                         for q in unknowns))
    field = case.field(problem, FieldDiscretization(method=CellCenteredSecondOrder(), boundaries=(), solver=CG(max_iter=200)))
    point = program.stage("same-physical-point", c=0)
    consumed = program.solve(field, values=inputs, at=point).consume(action=FailRun())
    observation = field.observe(consumed)
    return case, program, blocks, states, problem, observation, field, point


@pytest.mark.parametrize("component", (0, 1, 2))
def test_global_observation_has_no_invented_physical_owner_or_state(component):
    _case, program, blocks, _states, problem, observed, field, point = authored()
    value = observed[field[problem.unknowns[component]]]
    before = program._ir_hash()
    assert value.block is value.state_ref is None
    assert value.point == point
    assert blocks[0] != blocks[1] != blocks[2]
    program.store_history("global%d" % component, value, depth=1)
    assert program._histories_ncomp["global%d" % component] == 1
    assert "global%d" % component not in program._history_blocks
    assert program._ir_hash() != before  # history is a real resource, not a field surrogate.
    with pytest.raises(ValueError, match="requires explicit block owner provenance"):
        _emit_checkpoint_shape_metadata(program)
    assert value.block is value.state_ref is None and value.point == point


def test_same_canonical_names_do_not_issue_foreign_storage_authority():
    case, _program, blocks, states, _problem, _observed, _field, _point = authored()
    foreign, _other_program, other_blocks, other_states, *_rest = authored()
    assert case is not foreign and blocks[0] is not other_blocks[0]
    # Live registry ownership is authenticated independently of presentation names.
    with pytest.raises((ValueError, TypeError)):
        case.resolve(other_blocks[0][other_states[0]])
    assert case.resolve(blocks[0][states[0]]) is not None


def test_detached_equal_port_cannot_mint_a_live_state_qualification():
    _case, _program, blocks, states, *_rest = authored()
    block = blocks[0]
    from pops.problem.handles import BlockHandle
    detached = BlockHandle(block.local_id, owner=block.owner_path, model_owner=block.model_owner_path)
    assert detached is not block and detached == block
    with pytest.raises((ValueError, TypeError)):
        detached[states[0]]


def test_mutable_observation_alias_does_not_issue_a_new_consumed_field():
    _case, _program, _blocks, _states, problem, observed, field, _point = authored()
    value = observed[field[problem.unknowns[0]]]
    from pops.fields._observation_contract import validate_field_observation
    from types import SimpleNamespace
    forged = SimpleNamespace(op=value.op, vtype=value.vtype, prog=value.prog,
                             inputs=value.inputs, attrs=dict(value.attrs))
    forged.attrs["component"] = 1
    with pytest.raises(ValueError, match="declared solved unknown"):
        validate_field_observation(forged)
