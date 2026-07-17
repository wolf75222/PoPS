"""Runtime-parameter routing uses each Program node's exact model owner."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.codegen.program_emit_params import emit_program_params, program_param_entries
from pops.codegen.program_models import ProgramModelGraph
from pops._ir.values import RuntimeParamRef
from pops.model import OwnerPath
from pops.model.handles import ParamHandle
from pops.problem.handles import BlockHandle


class _Impl:
    def __init__(self, owner, name="alpha", default=1.0, *, parameter_owner=None):
        self.owner_path = owner
        handle = ParamHandle(
            name, owner=parameter_owner or owner, param_kind="runtime")
        self.parameter = RuntimeParamRef(name, default, handle=handle)
        self._source_terms = {"term": [self.parameter]}
        self._linear_sources = {}
        self._flux_terms = {}

    def has_runtime_params(self):
        return True

    def assign_runtime_indices(self):
        return [self.parameter]


class _SparseReadImpl(_Impl):
    def __init__(self, owner):
        super().__init__(owner, "alpha", 1.0)
        self.unused = self.parameter
        self.parameter = RuntimeParamRef(
            "omega", 3.0,
            handle=ParamHandle("omega", owner=owner, param_kind="runtime"),
        )
        self._source_terms = {"term": [self.parameter]}

    def assign_runtime_indices(self):
        return [self.unused, self.parameter]


class _Program:
    def __init__(self, blocks, nodes):
        self._blocks = tuple(blocks)
        self._values = list(nodes)

    def _block_indices(self):
        return {block: index for index, block in enumerate(self._blocks)}


def _block(name, model_owner):
    return BlockHandle(name, owner=OwnerPath.case("case"), model_owner=model_owner)


def _node(name, block):
    return SimpleNamespace(
        name=name, block=block, op="source", attrs={"source": "term"})


def _graph(routes):
    models = {owner: impl for _block_name, owner, impl in routes}
    return ProgramModelGraph(
        models_by_owner=models,
        source_modules_by_owner={owner: {"module": owner.name} for owner in models},
        owners_by_block={block_name: owner for block_name, owner, _impl in routes},
        authorities_by_owner={owner: owner for owner in models},
    )


def test_homonymous_params_are_qualified_and_indexed_per_block_owner():
    ions_owner = OwnerPath.model("ions")
    electrons_owner = OwnerPath.model("electrons")
    ions = _Impl(ions_owner, "alpha", 1.0)
    electrons = _Impl(electrons_owner, "alpha", 2.0)
    ion_block = _block("ions", ions_owner)
    electron_block = _block("electrons", electrons_owner)
    program = _Program(
        (ion_block, electron_block),
        (_node("ion_source", ion_block), _node("electron_source", electron_block)),
    )
    models = _graph((
        ("ions", ions_owner, ions),
        ("electrons", electrons_owner, electrons),
    ))

    assert program_param_entries(program, models) == [
        (0, "alpha", 0, 1.0),
        (1, "alpha", 0, 2.0),
    ]
    source = emit_program_params(program, models)
    assert "pops_program_param_count() { return 2; }" in source
    assert "static const int v[] = {0, 1};" in source
    assert source.count('return "alpha";') == 2


def test_graph_collection_never_uses_the_first_models_parameter_table():
    first_owner = OwnerPath.model("first")
    second_owner = OwnerPath.model("second")
    first = _Impl(first_owner, "first_only", 1.0)
    second = _Impl(second_owner, "second_only", 3.0)
    first_block = _block("first", first_owner)
    second_block = _block("second", second_owner)
    program = _Program(
        (first_block, second_block),
        (_node("second_source", second_block),),
    )
    models = _graph((
        ("first", first_owner, first),
        ("second", second_owner, second),
    ))

    assert program_param_entries(program, models) == [
        (1, "second_only", 0, 3.0),
    ]


def test_sparse_parameter_read_materialises_the_complete_stable_abi_vector():
    owner = OwnerPath.model("sparse")
    impl = _SparseReadImpl(owner)
    block = _block("fluid", owner)
    program = _Program((block,), (_node("source", block),))
    models = _graph((("fluid", owner, impl),))

    assert program_param_entries(program, models) == [
        (0, "alpha", 0, 1.0),
        (0, "omega", 1, 3.0),
    ]


def test_graph_rejects_a_parameter_read_owned_by_another_model():
    declared_owner = OwnerPath.model("declared")
    foreign_owner = OwnerPath.model("foreign")
    impl = _Impl(
        declared_owner, "alpha", 1.0, parameter_owner=foreign_owner)
    block = _block("fluid", declared_owner)
    program = _Program((block,), (_node("bad_source", block),))
    models = _graph((("fluid", declared_owner, impl),))

    with pytest.raises(ValueError, match=r"block 'fluid'.*model owner.*foreign.*declared"):
        program_param_entries(program, models)


def test_unqualified_runtime_param_is_refused_on_graph_route():
    owner = OwnerPath.model("model")
    impl = _Impl(owner)
    impl.parameter = RuntimeParamRef("alpha", 1.0)
    impl._source_terms = {"term": [impl.parameter]}
    block = _block("fluid", owner)
    program = _Program((block,), (_node("source", block),))
    models = _graph((("fluid", owner, impl),))

    with pytest.raises(ValueError, match="no owner-qualified ParamHandle"):
        program_param_entries(program, models)
