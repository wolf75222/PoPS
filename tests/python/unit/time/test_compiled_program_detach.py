"""ADC-655: a compiled Program retains no live authoring graph."""
from __future__ import annotations

import gc
import weakref
from types import MappingProxyType

import pytest

from pops.fields import FieldProblem
from pops.model import (
    DeclarationIndex,
    Handle,
    Operator,
    OperatorRegistry,
    OwnerKind,
    OwnerPath,
    RateSpace,
    Signature,
    StateSpace,
)
from pops.problem import Case
from pops.problem.handles import BlockHandle, FieldHandle
from pops.time import Program, StagePoint, TimePoint
from pops.time.program_detach import detach_compiled_program
from pops.time.values import ProgramValue


class _Model:
    def __init__(self) -> None:
        self.name = "transport"
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)
        self.state = Handle("u", kind="state", owner=self.owner_path)
        self._registry = OperatorRegistry(owner=self.owner_path)
        space = StateSpace("U", ("u",))
        operator = self._registry.register(Operator(
            "decay",
            "local_source",
            Signature((space,), RateSpace(space)),
            lowering={"source": "default"},
        ))
        self.rate = next(
            handle
            for handle in self._registry.declaration_index().records()
            if handle.local_id == operator.name
        )

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=(self.state,))

    def operator_registry(self) -> OperatorRegistry:
        return self._registry


def _authored_program():
    model = _Model()
    problem = Case(name="case")
    block = problem.block("fluid", model)
    field = problem.field(FieldProblem(name="psi"))
    program = Program("compiled-step")._bind_operators(model)
    state = program.state(block, model.state)
    program.keep_history(state, depth=1)
    previous = state.prev.value
    solved = program.solve_fields("solve-psi", state.n, field=field)
    rate = model.rate(state.n)
    predictor_point = StagePoint(
        "predictor", {"main": TimePoint(state.clock, 1)})
    stage = state.stage("predictor", point=predictor_point)
    candidate = program.value(
        "candidate", state.n + program.dt * rate, at=stage.point)
    looped = program.range(
        candidate,
        2,
        lambda builder, value: builder.value("range-body", value),
    )
    program.value(stage, looped)
    final = program.value("final", stage.value, at=state.next.point)
    program.commit(state.next, final)
    return {
        "model": model,
        "problem": problem,
        "block": block,
        "field": field,
        "program": program,
        "state": state,
        "previous": previous,
        "solved": solved,
        "rate": rate,
    }


def _walk_values(program):
    seen = set()

    def visit(value):
        if value.id in seen:
            return
        seen.add(value.id)
        yield value
        for key in ("cond_block", "body_block", "apply_block", "residual_block"):
            for nested in value.attrs.get(key) or ():
                yield from visit(nested)

    for value in program._values:
        yield from visit(value)


def test_detach_reowns_values_canonicalizes_handles_and_preserves_ir_identity():
    authored = _authored_program()
    source = authored["program"]
    serialized = source._serialize(include_provenance=False)
    ir_hash = source._ir_hash()

    detached = detach_compiled_program(source)

    assert detached is not source
    assert detached._compiled_detached is True
    assert detached._frozen is True
    assert detached.owner_path.is_canonical
    assert detached._case_owner_path.is_canonical
    assert detached._serialize(include_provenance=False) == serialized
    assert detached._ir_hash() == ir_hash
    assert detached._operator_registries == {}
    assert detached._default_state_spaces == {}
    assert detached._default_field_spaces == {}
    assert isinstance(detached._values, tuple)
    assert isinstance(detached._commits, MappingProxyType)

    values = tuple(_walk_values(detached))
    assert values
    assert all(isinstance(value, ProgramValue) and value.prog is detached for value in values)
    source_value_ids = {id(value) for value in source._issued_values.values()}
    assert all(id(value) not in source_value_ids for value in values)
    blocks = {id(value.block): value.block for value in values if value.block is not None}
    assert len(blocks) == 1
    [block] = blocks.values()
    assert isinstance(block, BlockHandle)
    assert block is not authored["block"]
    assert block.is_resolved and block._instance_registry is None
    for value in values:
        if value.state_ref is not None:
            assert value.state_ref.is_resolved
            assert value.state_ref.block_ref is value.block
            assert value.state_ref.declaration_ref.is_resolved

    solved = next(value for value in values if value.op == "solve_fields")
    detached_field = solved.attrs["field"]
    assert isinstance(detached_field, FieldHandle)
    assert detached_field is not authored["field"]
    assert detached_field.is_resolved and detached_field._field_registry is None
    assert solved.field_context.field_problem is detached_field
    assert solved.field_context.stage_sources[0][0] is block

    rate = next(value for value in values if "operator_handle" in value.attrs)
    operator = rate.attrs["operator_handle"]
    assert operator is not authored["model"].rate
    assert operator.is_resolved

    [time_state] = detached._time_states.values()
    assert time_state._program is detached
    assert time_state.block is block
    assert time_state.state.block_ref is block
    assert all(handle._program is detached for handle in detached._time_stage_handles.values())
    assert all(handle._program is detached for handle in detached._time_history_handles.values())
    assert all(value.prog is detached for value in detached._time_stage_values.values())
    assert all(value.prog is detached for value in detached._time_history_values.values())


def test_source_mutations_and_stale_containers_cannot_change_detached_program():
    authored = _authored_program()
    source = authored["program"]
    stale_values = source._values
    detached = detach_compiled_program(source)
    before = detached._serialize()
    before_hash = detached._ir_hash()

    stale_values.clear()
    authored["model"]._registry.register_alias("decay-alias", "decay")
    authored["problem"].field(FieldProblem(name="late-field"))

    assert detached._serialize() == before
    assert detached._ir_hash() == before_hash
    with pytest.raises(RuntimeError, match="frozen"):
        detached.capture_source_locations()
    state_ref = next(iter(detached._commits))
    with pytest.raises(TypeError):
        detached._commits[state_ref] = next(iter(detached._commits.values()))


def test_detached_program_does_not_keep_source_program_or_registries_alive():
    def build():
        authored = _authored_program()
        refs = {
            "program": weakref.ref(authored["program"]),
            "model": weakref.ref(authored["model"]),
            "problem": weakref.ref(authored["problem"]),
            "operator_registry": weakref.ref(authored["model"]._registry),
            "block_registry": weakref.ref(authored["problem"]._block_registry),
            "field_registry": weakref.ref(authored["problem"]._field_registry),
        }
        return detach_compiled_program(authored["program"]), refs

    detached, refs = build()
    gc.collect()

    assert detached._ir_hash()
    assert {name: ref() for name, ref in refs.items()} == {
        name: None for name in refs
    }


def test_detachment_hook_stays_internal_and_rejects_non_program_values():
    import pops.time as public_time

    assert not hasattr(public_time, "detach_compiled_program")
    assert "detach_compiled_program" not in getattr(public_time, "__all__", ())
    with pytest.raises(TypeError, match="pops.time.Program"):
        detach_compiled_program(object())


def test_compiled_problem_constructor_uses_the_detached_program_boundary():
    from pops.codegen.loader import CompiledProblem

    authored = _authored_program()
    source = authored["program"]
    compiled = CompiledProblem(
        "<not-loaded>", source, authored["model"], "abi", "c++", "c++20",
        generated_cpp="// exact compiler-owned source\n",
    )

    assert compiled.program is not source
    assert compiled.program._compiled_detached is True
    assert compiled.program._ir_hash() == source._ir_hash()
    assert compiled.program._operator_registries == {}


def test_ordinary_rebuild_keeps_authoring_registries_and_ir_identity():
    authored = _authored_program()
    source = authored["program"]

    rebuilt = source._rebuild(lambda _value: True)

    assert rebuilt._serialize(include_provenance=False) == source._serialize(
        include_provenance=False)
    assert rebuilt._ir_hash() == source._ir_hash()
    assert rebuilt._operator_registries == source._operator_registries
    assert any(
        registry is authored["model"]._registry
        for registry in rebuilt._operator_registries.values()
    )
