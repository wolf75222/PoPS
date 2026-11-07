"""ADC-655: a compiled Program retains no live authoring graph."""
from __future__ import annotations

import gc
import weakref
from types import MappingProxyType

import pytest

from pops.descriptors import Descriptor
from pops.fields import FieldDiscretization, FieldOperator
from pops.math import ValueExpr
from pops.math import laplacian
from pops.model import (
    DeclarationIndex,
    FieldSpace,
    Handle,
    Operator,
    OperatorRegistry,
    OwnerKind,
    OwnerPath,
    RateSpace,
    Signature,
    StateHandle,
    StateSpace,
)
from pops.problem import Case
from pops.problem.handles import BlockHandle, FieldHandle
from pops.time import FailRun, Program, StagePoint, TimePoint
from pops.time._program.detach import detach_compiled_program
from pops.time.values import ProgramValue


class _Model:
    def __init__(self) -> None:
        self.name = "transport"
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)
        self._registry = OperatorRegistry(owner=self.owner_path)
        space = StateSpace("U", ("u",))
        self.state = StateHandle("U", owner=self.owner_path, space=space)
        self.field_space = FieldSpace("psi", ("psi",))
        self.field = Handle("psi", kind="field", owner=self.owner_path)
        operator = self._registry.register(Operator(
            "decay",
            "local_source",
            Signature((space,), RateSpace(space)),
            lowering={"source": "default"},
        ))
        provider_operator = self._registry.register(Operator(
            "psi_provider",
            "field_operator",
            Signature((space,), self.field_space),
            lowering={"field_provider": {"key": "psi_provider"}},
        ))
        self.rate = next(
            handle
            for handle in self._registry.declaration_index().records()
            if handle.local_id == operator.name
        )
        self.field_provider = next(
            handle
            for handle in self._registry.declaration_index().records()
            if handle.local_id == provider_operator.name
        )

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=(self.state, self.field))

    def field_spaces(self):
        return {self.field_space.name: self.field_space}

    def operator_registry(self) -> OperatorRegistry:
        return self._registry

    def primitive_recipes(self):
        """This metadata-only fixture declares no primitive expression recipes."""
        return MappingProxyType({})


class _FieldMethod(Descriptor):
    category = "field_method"

    def to_data(self):
        return {"type": "unit-second-order"}


class _FieldSolver(Descriptor):
    category = "elliptic_solver"

    def to_data(self):
        return {"type": "unit-krylov"}


def _field(case: Case, model: _Model, name: str):
    unknown = case.qualify(model.field)
    operator = FieldOperator(
        name,
        unknown=unknown,
        equation=-laplacian(ValueExpr(unknown)) == ValueExpr(case.qualify(model.state)),
        providers=model.field_provider,
    )
    return case.field(
        operator,
        FieldDiscretization(
            method=_FieldMethod(), boundaries=(), solver=_FieldSolver()),
    )


def _authored_program():
    model = _Model()
    problem = Case(name="case")
    block = problem.block("fluid", model)
    field = _field(problem, model, "psi")
    program = Program("compiled-step")
    state = program.state(block[model.state])
    program.keep_history(state, depth=1)
    previous = state.prev.value
    solved = field(state.n, name="solve-psi").consume(action=FailRun())
    rate = model.rate(state.n)
    predictor_point = StagePoint(
        "predictor", {"main": TimePoint(state.clock, 1)})
    stage = state.stage("predictor", point=predictor_point)
    candidate = program.value(
        "candidate", state.n + program.dt * rate, at=stage.point)
    looped = program.range(
        candidate,
        2,
        lambda builder, value: builder.value("range-body", 1 * value),
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
    assert detached._state_spaces
    assert all(state_ref.space == space
               for state_ref, space in detached._state_spaces.items())
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
    assert solved.field_context.field is detached_field
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
    _field(authored["problem"], authored["model"], "late-field")

    assert detached._serialize() == before
    assert detached._ir_hash() == before_hash
    with pytest.raises(RuntimeError, match="frozen"):
        detached.capture_source_locations()
    state_ref = next(iter(detached._commits))
    with pytest.raises(TypeError):
        detached._commits[state_ref] = next(iter(detached._commits.values()))


def _counted_projections(monkeypatch):
    """Record each Program whose IR identity projection (``include_provenance=False``) runs."""
    projections = []
    original = Program._serialize

    def counted(self, *args, **kwargs):
        if kwargs.get("include_provenance", True) is False:
            projections.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Program, "_serialize", counted)
    return projections


def test_compile_identity_reuse_projects_once_and_reauthenticates_on_success(monkeypatch):
    from pops.time._program.serialization import ir_identity_reuse

    authored = _authored_program()
    source = authored["program"].freeze()
    unfrozen = _authored_program()["program"]
    expected = source._ir_hash()
    projections = _counted_projections(monkeypatch)
    with ir_identity_reuse():
        assert [source._ir_hash() for _ in range(4)] == [expected] * 4
        with ir_identity_reuse():
            assert source._ir_hash() == expected
        detached = detach_compiled_program(source)
        assert detached.to_graph() is detached.to_graph()
        unfrozen._ir_hash()
        unfrozen._ir_hash()
    # One captured identity and one fresh exit authentication per frozen Program; a mutable
    # Program is never reused.
    assert projections == [source, detached, unfrozen, unfrozen, source, detached]
    projections.clear()
    assert [source._ir_hash(), source._ir_hash()] == [expected, expected]
    assert projections == [source, source]

    projections.clear()
    with pytest.raises(ValueError, match="lowering failed"):
        with ir_identity_reuse():
            source._ir_hash()
            raise ValueError("lowering failed")
    assert projections == [source]


@pytest.mark.parametrize("change", ["definition", "out_of_contract_edit"])
def test_compile_identity_reuse_fails_when_a_reused_identity_changed(change):
    from pops.time._program.serialization import ir_identity_reuse

    authored = _authored_program()
    source = authored["program"].freeze()
    detached = detach_compiled_program(source)
    expected = detached._ir_hash()
    changed = source if change == "definition" else detached
    with pytest.raises(RuntimeError, match="changed during pops.compile"):
        with ir_identity_reuse():
            assert changed._ir_hash() == expected
            if change == "definition":
                authored["model"]._registry.register_alias("decay-alias", "decay")
            else:
                object.__setattr__(detached, "name", "altered")
            assert changed._ir_hash() == expected
    assert changed._ir_hash() != expected


def test_compile_identity_reuse_leaves_no_graph_binding_after_the_scope():
    from pops.codegen.program_graph_lowering import emit_program_graph
    from pops.time._program.serialization import ir_identity_reuse

    program = Program("cached")
    program.int64(3)
    detached = detach_compiled_program(program)
    with ir_identity_reuse():
        detached._ir_hash()
        # A transient out-of-contract edit, undone before the scope ends: the exit passes.
        object.__setattr__(detached, "name", "altered")
        altered = detached.to_graph()
        object.__setattr__(detached, "name", "cached")
    # The graph converted under the first read's identity does not outlive the scope.
    assert detached.to_graph() is not altered
    with pytest.raises(ValueError, match="does not match"):
        emit_program_graph(altered, lowering_program=detached)


def test_compile_identity_reuse_is_bound_to_the_thread_that_opens_it(monkeypatch):
    import contextvars
    import threading

    from pops.time._program.serialization import ir_identity_reuse

    source = _authored_program()["program"].freeze()
    expected = source._ir_hash()
    projections = _counted_projections(monkeypatch)
    with ir_identity_reuse():
        assert source._ir_hash() == expected
        context = contextvars.copy_context()
        reads = []
        worker = threading.Thread(target=lambda: context.run(
            lambda: reads.extend(source._ir_hash() for _ in range(2))))
        worker.start()
        worker.join()
        assert reads == [expected, expected]
    # The opening thread's first read and exit authentication, plus a fresh projection for each
    # read of the copied context on the other thread.
    assert projections == [source] * 4


def test_public_compile_reads_identities_in_one_reuse_scope(monkeypatch):
    import sys
    from types import ModuleType

    import pops
    import pops._native_selector as selector
    import pops.codegen._phases as phases
    import pops.codegen._plans as plans
    import pops.codegen.region_facts as region_facts
    from pops import _api

    source = _authored_program()["program"].freeze()
    expected = source._ir_hash()
    projections = _counted_projections(monkeypatch)
    result = object()

    class Resolved:
        resolved_dimension = 2

        @staticmethod
        def verify() -> None:
            assert source._ir_hash() == expected

    def compile_phase(plan):
        assert [source._ir_hash(), source._ir_hash()] == [expected, expected]
        return result

    bootstrap = ModuleType("pops._bootstrap")
    monkeypatch.setattr(plans, "ResolvedSimulationPlan", Resolved)
    # pops.compile also checks the expected region classes of the verified plan (#710, IR21);
    # this stand-in plan has no regions.
    monkeypatch.setattr(region_facts, "check_expected_classes", lambda plan: ())
    monkeypatch.setattr(selector, "select_native_dimension", lambda dimension: None)
    monkeypatch.setitem(sys.modules, "pops._bootstrap", bootstrap)
    monkeypatch.setattr(pops, "_bootstrap", bootstrap, raising=False)
    monkeypatch.setattr(phases, "compile", compile_phase)

    assert _api.compile(Resolved()) is result
    # The plan check and the compile phase read in one scope: the first read projects, the
    # later reads reuse it, and the successful exit authenticates it once.
    assert projections == [source, source]


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
