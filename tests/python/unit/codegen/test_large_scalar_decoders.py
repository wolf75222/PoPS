"""Canonical exact values remain readable beyond Python's decimal digit limit."""
import importlib
import json
import sys
from fractions import Fraction

import pytest
from pops.identity.scalar import ScalarLiteral, parse_canonical_integer


def _large_fraction():
    return Fraction(10**5100+17,10**5100+31)


@pytest.mark.parametrize("value",(0,7,-12,10**5100+1,-10**5100-1),
                         ids=("zero","positive","negative","large","large_negative"))
def test_exact_integer_roundtrip_does_not_modify_global_digit_policy(value):
    limit=sys.get_int_max_str_digits()
    data=ScalarLiteral.from_value(value).to_data()
    assert parse_canonical_integer(data["value"])==value
    assert ScalarLiteral.from_value(parse_canonical_integer(data["value"])).to_data()==data
    assert sys.get_int_max_str_digits()==limit


@pytest.mark.parametrize("raw",("","-","+1","01","-0","1_0"," 1","1 ","١","1.0",True,12))
def test_integer_parser_rejects_noncanonical_spellings(raw):
    with pytest.raises((TypeError,ValueError)):
        parse_canonical_integer(raw)


@pytest.mark.parametrize("module,name",(
    ("pops.params.constraints","_literal_from_data"),
    ("pops.codegen._compiled_parameter","_decode_scalar"),
    ("pops.time._methods.properties","_literal"),
))
@pytest.mark.parametrize("kind",("integer","rational"))
def test_large_literals_roundtrip_through_consuming_decoders(module,name,kind):
    value=10**5100+19 if kind=="integer" else _large_fraction()
    data=ScalarLiteral.from_value(value).to_data()
    restored=getattr(importlib.import_module(module),name)(json.loads(json.dumps(data)))
    assert restored==value


def test_large_field_and_bind_literals_preserve_canonical_identity():
    from pops.fields._program_expression import decode_field_literal
    from pops.model._bind_schema_data import literal_value
    from pops.identity import canonical_bytes
    for value in (10**5100+19,_large_fraction()):
        original=ScalarLiteral.from_value(value).to_data()
        field=decode_field_literal(original)
        assert canonical_bytes(field.to_data())==canonical_bytes(original)
        assert field.to_python()==value
        assert literal_value(original,where="large default")==value


def test_large_integer_metadata_decoders_keep_exact_values():
    from pops.codegen.program_emit_ops import _canonical_metadata_int
    from pops.time._graph.base import _canonical_int
    from pops.time._program.spatial_solve import spatial_scalar
    value=10**5100+19
    data={"scalar":ScalarLiteral.from_value(value).to_data()}
    assert _canonical_metadata_int(data,where="large scalar")==value
    assert _canonical_int(data)==value
    assert spatial_scalar(data)==value


def test_large_rational_reaches_existing_numerical_provider_codegen():
    from pops.numerics.riemann.providers import _delta_from_token,_delta_token
    from pops.solvers.krylov._prepared_method_registry import _richardson_options,_emit_richardson
    value=_large_fraction()
    assert _delta_from_token(_delta_token(value))==value
    data=ScalarLiteral.from_value(value).to_data()
    assert _richardson_options({"relaxation":data})=={"relaxation":data}
    assert "pops::Real(1.0)" in _emit_richardson(None,{"relaxation":data})


def test_large_rational_data_dag_detach_and_cpp_share_one_exact_identity():
    from pops._ir import Const
    from pops._ir.visitors import _dag_key_data,validate_dag_key_data
    from pops.codegen.program_codegen import emit_cpp_program
    from pops.time import Program
    from pops.time._program.detach import detach_compiled_program
    from tests.python.support.typed_program import program_states,synthetic_module
    limit=sys.get_int_max_str_digits()
    data=ScalarLiteral.from_value(_large_fraction()).to_data()
    restored=Fraction(parse_canonical_integer(data["numerator"]),
                      parse_canonical_integer(data["denominator"]))
    graph=_dag_key_data((Const(restored),))
    validate_dag_key_data(json.loads(json.dumps(graph)),root_count=1)
    program=Program("large_rational")
    _,states=program_states(program,synthetic_module("large_rational_model",components=("u",)),("owner",))
    q=states["owner"]
    value=program.value("authored_equation",(restored*q.n[0],),at=q.next.point)
    program.commit(q.next,value)
    original=program._ir_hash()
    clone=detach_compiled_program(program)
    assert clone._ir_hash()==original
    assert clone._serialize(include_provenance=False)==program._serialize(include_provenance=False)
    source=emit_cpp_program(clone)
    assert "pops::Real(1.0)" in source
    assert original==program._ir_hash()
    assert sys.get_int_max_str_digits()==limit
