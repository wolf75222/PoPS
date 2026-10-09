"""Authored aliases are authenticated before immutable public bind evidence."""
import pytest

from pops import Case
from pops.codegen._plans import BindInputs, _canonical_bind_params
from pops.model import Module
from pops.model.bind_schema import BindSchema
from pops.params import RuntimeParam, ConstParam


def declarations():
    module = Module("same_model")
    speed = module.param(RuntimeParam("speed", default=1.))
    order = module.param(ConstParam("order", value=2))
    case = Case("same_case")
    block = case.block("fluid", module)
    alias, const_alias = block[speed], block[order]
    schema = BindSchema.from_problem(case)
    return schema, alias, const_alias, speed


def test_live_and_canonical_aliases_produce_the_same_detached_bind_identity():
    schema, live, _, _ = declarations()
    canonical = schema.slot(live).handle
    assert not live.is_resolved and canonical.is_resolved
    authored = BindInputs(params=_canonical_bind_params(schema, {live: 2.}))
    restored = BindInputs(params=_canonical_bind_params(schema, {canonical: 2.}))
    assert tuple(authored.params) == (canonical,)
    assert authored.inputs_identity == restored.inputs_identity
    authored.verify()
    resolved = schema.resolve_bind(authored.params, compile_values=schema.resolve_compile())
    assert resolved[canonical] == 2.


def test_duplicate_aliases_and_foreign_authoring_handles_do_not_collapse_by_name():
    schema, live, _, declaration = declarations()
    canonical = schema.slot(live).handle
    with pytest.raises(ValueError, match="multiple bind entries"):
        _canonical_bind_params(schema, {live: 2., canonical: 3.})
    _, foreign, _, _ = declarations()
    with pytest.raises(KeyError, match="not present"):
        _canonical_bind_params(schema, {foreign: 2.})
    with pytest.raises(ValueError, match="not block-qualified"):
        _canonical_bind_params(schema, {declaration: 2.})
    with pytest.raises(TypeError, match="ParamHandle"):
        _canonical_bind_params(schema, {"speed": 2.})


def test_compile_time_parameters_remain_unsettable_at_bind():
    schema, _, const_alias, _ = declarations()
    with pytest.raises(TypeError, match="only RuntimeParam"):
        _canonical_bind_params(schema, {const_alias: 3})
