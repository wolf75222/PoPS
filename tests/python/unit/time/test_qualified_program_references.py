"""ADC-653: the time IR retains qualified semantic handles until lowering."""
from __future__ import annotations

import pytest

from pops.descriptors import Descriptor
from pops.fields import FieldDiscretization, FieldOperator
from pops.ir import ValueExpr
from pops.math import laplacian
from pops.model import (
    DeclarationIndex,
    DoubleOwnershipError,
    FieldSpace,
    Handle,
    Operator,
    OperatorRegistry,
    Module,
    MissingOwnershipError,
    OwnerKind,
    OwnerPath,
    RateSpace,
    Signature,
    StateHandle,
    StateSpace,
)
from pops.problem import Case
from pops.problem.handles import BlockHandle
from pops.time import FailRun, Program, StagePoint, TimePoint
from pops.lib.time import ForwardEuler, SSPRK2
from pops.numerics.terms import DefaultSource, SourceTerm


class _Model:
    def __init__(self, name: str = "transport", *, with_rate: bool = False,
                 components: tuple[str, ...] = ("u",)) -> None:
        self.name = name
        self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, name)
        self.space = StateSpace("u", components)
        self.u = StateHandle("u", owner=self.owner_path, space=self.space)
        self.field_space = FieldSpace("psi", ("psi",))
        self.field = Handle("psi", kind="field", owner=self.owner_path)
        self._registry = OperatorRegistry(owner=self.owner_path)
        provider = self._registry.register(Operator(
            "psi_provider",
            "field_operator",
            Signature((self.space,), self.field_space),
            lowering={"field_provider": {"key": "psi_provider"}},
        ))
        self.field_provider = next(
            handle for handle in self._registry.declaration_index().records()
            if handle.local_id == provider.name)
        self.rate = None
        if with_rate:
            operator = self._registry.register(Operator(
                "decay",
                "local_source",
                Signature((self.space,), RateSpace(self.space)),
                lowering={"source": "default"},
            ))
            self.rate = next(
                handle for handle in self._registry.declaration_index().records()
                if handle.local_id == operator.name)

    def declaration_index(self) -> DeclarationIndex:
        return DeclarationIndex(owner=self.owner_path, handles=(self.u, self.field))

    def field_spaces(self):
        return {self.field_space.name: self.field_space}

    def operator_registry(self) -> OperatorRegistry:
        return self._registry


def _declarations(*, with_rate: bool = False):
    model = _Model(with_rate=with_rate)
    case = Case(name="case")
    block = case.block("fluid", model)
    return model, block


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
    return case.field(
        FieldOperator(
            name,
            unknown=unknown,
            equation=-laplacian(ValueExpr(unknown)) == ValueExpr(case.qualify(model.u)),
            providers=model.field_provider,
        ),
        FieldDiscretization(
            method=_FieldMethod(), boundaries=(), solver=_FieldSolver()),
    )


def _forward_copy():
    model, block = _declarations()
    program = Program("step")
    state = program.state(block[model.u])
    next_value = program.value("u_next", state.n, at=state.next.point)
    program.commit(state.next, next_value)
    return model, block, program, state, next_value


def test_state_qualifies_once_and_every_value_retains_semantic_provenance():
    model, block, program, state, next_value = _forward_copy()

    assert isinstance(block, BlockHandle)
    assert state.state is block[model.u]
    assert state.n.block is block
    assert state.n.state_ref is state.state
    assert next_value.block is block
    assert next_value.state_ref is state.state
    assert program.commits() == {state.state: next_value}


def test_state_refuses_strings_wrong_kinds_and_redundant_qualification():
    model, block = _declarations()
    program = Program("strict")

    with pytest.raises(TypeError, match="StateHandle"):
        program.state("fluid")
    with pytest.raises(MissingOwnershipError, match=r"block-qualified state|block\[U\]"):
        program.state(model.u)
    with pytest.raises(DoubleOwnershipError, match="already block-qualified"):
        block[block[model.u]]
    with pytest.raises(TypeError, match="StateHandle"):
        program.state(Handle("phi", kind="field", owner=model.owner_path))


def test_two_instances_of_one_model_remain_distinct_in_time_ir():
    model = _Model()
    case = Case(name="case")
    left = case.block("left", model)
    right = case.block("right", model)
    program = Program("two-blocks")

    left_state = program.state(left[model.u])
    right_state = program.state(right[model.u])

    assert left_state.state != right_state.state
    assert left_state.n.block is left
    assert right_state.n.block is right
    assert left_state.n.state_ref != right_state.n.state_ref

    from pops.time.program_serialization import _json_ready
    encoded = _json_ready({right_state.state: "right", left_state.state: "left"})
    entries = encoded["mapping_entries"]
    assert [entry[0]["handle"]["block_ref"]["local_id"] for entry in entries] == [
        "left", "right",
    ]

    program.commit(left_state.next, program.value(
        "left_next", left_state.n, at=left_state.next.point))
    program.commit(right_state.next, program.value(
        "right_next", right_state.n, at=right_state.next.point))
    from pops.codegen.program_codegen import _check_lowerable
    _check_lowerable(program, model=model)


@pytest.mark.parametrize("case_names", [("same", "same"), ("first", "second")])
def test_one_program_refuses_blocks_from_distinct_case_authorities(case_names):
    model = _Model()
    first_case = Case(name=case_names[0])
    second_case = Case(name=case_names[1])
    first = first_case.block("fluid", model)
    second = second_case.block("fluid", model)
    program = Program("one-case")

    first_state = program.state(first[model.u])
    assert first_state.block is first
    with pytest.raises(ValueError, match="one Program cannot combine blocks from"):
        program.state(second[model.u])

    assert tuple(program._time_states.values()) == (first_state,)
    assert program._case_owner_path == first_case.owner_path


def test_program_serialization_is_canonical_and_contains_no_authoring_capability():
    _, _, first, _, _ = _forward_copy()
    _Model("unrelated-owner-allocation")
    _, _, second, _, _ = _forward_copy()

    first_data = first._serialize()
    second_data = second._serialize()
    assert first_data == second_data
    assert first._ir_hash() == second._ir_hash()
    assert all("#authoring=" not in state.local_id for state in first._time_states.values())
    assert first_data["version"] == 4
    assert first_data["nodes"][0]["block"]["kind"] == "block"
    assert first_data["nodes"][0]["state"]["kind"] == "state"
    assert first_data["commits"][0]["state"] == first_data["nodes"][0]["state"]
    assert "#authoring=" not in repr(first_data)


def test_history_tables_and_serialization_retain_the_qualified_state():
    model, block = _declarations()
    program = Program("history")
    state = program.state(block[model.u])
    program.keep_history(state, depth=2)
    previous = state.prev(2).value
    program.commit(state.next, program.value(
        "next", state.n + previous, at=state.next.point))

    assert program._history_state_refs["fluid.u"] is state.state
    assert previous.state_ref is state.state
    data = program._serialize()
    assert data["histories"] == [{
        "name": "fluid.u",
        "lag": 2,
        "ncomp": None,
        "state": data["nodes"][0]["state"],
    }]


def test_public_call_keeps_the_operator_handle_separate_from_its_lowering_name():
    model, block = _declarations(with_rate=True)
    program = Program("operator")
    state = program.state(block[model.u])

    rate = model.rate(state.n)

    assert rate.attrs["operator_handle"] is model.rate
    assert rate.attrs["sources"] == ("default",)
    serialized = next(node for node in program._serialize()["nodes"] if node["id"] == rate.id)
    identity = serialized["attrs"]["operator_handle"]["handle"]
    assert identity["local_id"] == "decay"
    assert identity["registered_operator_name"] == "decay"
    assert identity["owner_path"]["nodes"] == [
        {"kind": "model_definition", "name": "transport"},
    ]


def test_callable_operator_handle_refuses_free_names():
    model, block = _declarations(with_rate=True)
    program = Program("board")
    state = program.state(block[model.u])

    value = model.rate(state.n)
    assert value.attrs["operator_handle"] is model.rate
    assert not hasattr(program, "call")
    assert not hasattr(program, "op")
    assert not hasattr(program, "fields")


def test_named_field_solve_requires_authenticated_field_from_the_state_case():
    model = _Model()
    case = Case(name="case")
    block = case.block("fluid", model)
    field = _field(case, model, "psi")
    foreign_case = Case(name="case")
    foreign_case.block("fluid", model)
    foreign = _field(foreign_case, model, "psi")
    program = Program("field")
    state = program.state(block[model.u])

    solved = field(state.n, name="psi_solve").consume(action=FailRun())
    solve_token = next(value for value in program._values if value.op == "solve_fields")
    assert solve_token.attrs["field"] is field
    assert solved.field_context.field is field
    serialized = next(
        node for node in program._serialize()["nodes"] if node["id"] == solve_token.id)
    assert serialized["attrs"]["field"]["handle"]["owner_path"]["nodes"] == [
        {"kind": "case", "name": "case"},
    ]
    assert not hasattr(program, "solve_fields")
    with pytest.raises(ValueError, match="different Cases"):
        foreign(state.n, name="foreign").consume(action=FailRun())


def test_case_field_retains_its_exact_typed_provider_selector():
    module = Module("field-operator")
    state_space = module.state_space("U", ("u",))
    field_space = module.field_space("psi_fields", ("psi",))
    operator = module.operator(
        "psi", kind="field_operator",
        signature=Signature((state_space,), field_space), expr="field-solve")
    case = Case(name="case")
    block = case.block("fluid", module)
    unknown = block[module.field_handle(field_space)]
    field = case.field(
        FieldOperator(
            "psi",
            unknown=unknown,
            equation=-laplacian(ValueExpr(unknown)) == ValueExpr(
                block[module.state_handle(state_space)]),
            providers=operator,
        ),
        FieldDiscretization(
            method=_FieldMethod(), boundaries=(), solver=_FieldSolver()),
    )
    program = Program("field-operator")
    state = program.state(block[module.state_handle(state_space)])

    solved = field(state.n, name="psi_solve").consume(action=FailRun())
    solve_token = next(value for value in program._values if value.op == "solve_fields")
    assert solve_token.attrs["field"] is field
    assert solved.field_context.field is field
    contribution, = tuple(case._fields.get("psi").operator.providers)
    assert contribution.provider is operator

    with pytest.raises(TypeError, match="not Program solve authorities"):
        operator(state.n, name="provider-is-not-a-solve")


def test_rhs_wrappers_and_ready_presets_keep_typed_source_ownership():
    model, block = _declarations(with_rate=True)
    program = Program("typed-rhs")
    state = program.state(block[model.u])

    wrapped = program.rhs(state=state.n, terms=[SourceTerm(model.rate)])
    assert wrapped.attrs["sources"] == ("default",)
    assert wrapped.attrs["source_handles"] == (model.rate,)
    default = program.rhs(state=state.n, terms=[DefaultSource()])
    assert default.attrs["sources"] == ("default",)
    assert "source_handles" not in default.attrs
    with pytest.raises(TypeError, match="typed OperatorHandle"):
        SourceTerm(model.rate.name)

    preset = ForwardEuler(block[model.u], rate=model.rate)
    rate = next(value for value in preset._values if "operator_handle" in value.attrs)
    assert rate.attrs["operator_handle"] is model.rate
    with pytest.raises(TypeError, match="OperatorHandle"):
        ForwardEuler(block[model.u], rate="decay")


def test_homonymous_operators_from_two_models_resolve_by_owner_and_block_provenance():
    first = _Model("first", with_rate=True, components=("u",))
    second = _Model("second", with_rate=True, components=("v", "w"))
    case = Case(name="coupled")
    first_block = case.block("first_block", first)
    second_block = case.block("second_block", second)
    program = Program("multi-owner")
    first_state = program.state(first_block[first.u])
    second_state = program.state(second_block[second.u])
    assert len(first_state.space.components) == 1
    assert len(second_state.space.components) == 2

    first_rate = first.rate(first_state.n)
    second_rate = second.rate(second_state.n)

    assert first_rate.attrs["operator_handle"] is first.rate
    assert second_rate.attrs["operator_handle"] is second.rate
    assert first_rate.attrs["operator_handle"] != second_rate.attrs["operator_handle"]
    with pytest.raises(ValueError, match="none of its block-qualified arguments"):
        first.rate(second_state.n)
    from pops.codegen.program_codegen import _check_lowerable
    with pytest.raises(NotImplementedError, match="ProgramModelGraph"):
        _check_lowerable(program, model=first)


def test_ready_presets_take_typed_references_and_match_the_explicit_program():
    model, block = _declarations(with_rate=True)
    manual = Program("ForwardEuler")
    state = manual.state(block[model.u])
    point = StagePoint(
        "forward_euler_stage_0", {"main": TimePoint(manual.clock, 0)})
    rate = manual.value("forward_euler_k_0", model.rate(state.n), at=point)
    manual.commit(state.next, manual.value(
        "forward_euler_step", state.n + manual.dt * rate, at=state.next.point))

    preset = ForwardEuler(block[model.u], rate=model.rate)
    assert preset._serialize(include_provenance=False) == manual._serialize(
        include_provenance=False)

    factory = SSPRK2(block[model.u], rate=model.rate)
    assert isinstance(factory, Program)
    assert factory.validate() is True
    assert all(value.block is block for value in factory._values if value.block is not None)
    with pytest.raises(TypeError, match="block-qualified"):
        ForwardEuler(model.u, rate=model.rate)


def test_commit_report_contains_full_case_block_model_and_state_identity():
    _, _, program, state, _ = _forward_copy()

    commit = program.inspect().to_dict()["commits"][0]
    assert commit["local_id"] == state.state.local_id
    assert commit["block_ref"]["local_id"] == state.block.local_id
    assert commit["owner_path"]["nodes"] == [
        {"kind": "case", "name": "case"},
        {"kind": "block", "name": "fluid"},
        {"kind": "model_definition", "name": "transport"},
    ]


def test_codegen_block_index_helpers_never_default_unknown_blocks_to_zero():
    model, block = _declarations()
    foreign_case = Case(name="foreign")
    foreign = foreign_case.block("fluid", model)
    from pops.codegen.program_emit_ops import _required_block_index
    from pops.codegen.program_emit_params import _required_param_block_index

    assert _required_block_index({block: 0}, block, "test") == 0
    with pytest.raises(ValueError, match="not declared"):
        _required_block_index({block: 0}, foreign, "test")
    with pytest.raises(ValueError, match="routing is unavailable"):
        _required_block_index(None, block, "test")
    marker = type("Node", (), {"name": "rhs"})()
    with pytest.raises(ValueError, match="outside Program._block_indices"):
        _required_param_block_index({block: 0}, foreign, marker)


def test_raw_module_lowering_preserves_the_module_owner_authority():
    module = Module("raw-module")
    module.state_space("U", ("u",))

    from pops.codegen.module_lowering import lower_and_validate
    emit_model, source_module = lower_and_validate(module)

    assert source_module is module
    assert emit_model._m.owner_path == module.owner_path
    assert emit_model.operator_registry().owner_path == module.owner_path
