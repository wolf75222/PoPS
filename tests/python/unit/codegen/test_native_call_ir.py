"""Authenticated native function contracts and one local typed joint evaluation."""
from dataclasses import replace

import pytest

from pops._ir.application import ApplicationContext
from pops._ir.expr import Var
from pops._ir.visitors import _key
from pops.codegen.cpp_writer import _cse_emit
from pops.model import Module, Signature
from pops.model.bundles import ProductSpace
from pops.native_components import PreparedNativeComponent
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain


def _function(tmp_path, **options):
    (tmp_path / "law.hpp").write_text("#pragma once\n")
    module = Module("native_example")
    left = module.state_space("left", ("momentum", "thermal"))
    right = module.state_space("right", ("momentum", "thermal", "mass"))
    component = PreparedNativeComponent.header_only("example.law", include_root=tmp_path,
                                                    entry_headers=("law.hpp",))
    function = NativeFunction(component, "example::Law::evaluate",
        Signature((left, right), ProductSpace({"left": left, "right": right})), **options)
    return module, left, right, function


def test_native_call_is_immutable_opaque_and_typed(tmp_path):
    module, left, right, function = _function(tmp_path)
    call = function(module.state_symbols(left), module.state_symbols(right))
    assert len(call.left) == 2 and len(call.right) == 3
    assert call.left[0].call is call.right[2].call
    with pytest.raises(AttributeError, match="immutable"):
        call.inputs = ()
    with pytest.raises(TypeError, match="no Python callback"):
        call.eval({})
    with pytest.raises(ValueError, match="input shape"):
        function((1,), (1, 2, 3))
    with pytest.raises(ValueError, match="support/representation"):
        function(module.state_symbols(right)[:2], module.state_symbols(right))


def test_native_call_context_and_header_bytes_authenticate_identity(tmp_path):
    _module, _left, _right, function = _function(tmp_path)
    first = function((1, 2), (3, 4, 5), context=ApplicationContext(stage=1, iterate=0, sampling="cell"))
    assert _key(first) == _key(function((1, 2), (3, 4, 5), context=first.context))
    for context in (ApplicationContext(stage=2, iterate=0, sampling="cell"),
                    ApplicationContext(stage=1, iterate=1, sampling="cell"),
                    ApplicationContext(stage=1, iterate=0, sampling="face")):
        assert _key(first) != _key(function((1, 2), (3, 4, 5), context=context))
    (tmp_path / "law.hpp").write_text("#pragma once\n// changed\n")
    with pytest.raises(ValueError, match="changed after registration"):
        function.component.stage_verified(tmp_path / "stage")


def test_read_footprint_is_conservative_or_exact_including_domain(tmp_path):
    _module, _left, _right, function = _function(tmp_path)
    left = tuple(Var(name, "cons") for name in ("a", "b"))
    right = tuple(Var(name, "cons") for name in ("c", "d", "e"))
    assert function(left, right).left[0].deps() == {"a", "b", "c", "d", "e"}
    function = replace(function, reads=((0, 0),), domains=(NativeInputDomain(1, 2, lower=0),))
    assert function(left, right).left[0].deps() == {"a", "e"}
    with pytest.raises(ValueError, match="footprint"):
        replace(function, reads=((3, 0),))


def test_projection_cse_emits_one_static_call_without_erasing_repeated_use(tmp_path):
    _module, _left, _right, function = _function(tmp_path,
        domains=(NativeInputDomain(0, 0, lower=0, lower_open=True),))
    call = function((Var("a", "cons"), 2), (3, 4, 5))
    lines, outputs, statuses = _cse_emit(
        [call.left[0] + call.left[0], call.right[2]], "pops::Real", "", return_native_statuses=True)
    source = "\n".join(lines)
    assert source.count("example::Law::evaluate(") == 1
    assert "a > 0" in source and source.index("if (") < source.index("example::Law::evaluate(")
    assert len(statuses) == 1
    assert " + " in outputs[0]
    assert any(".read(4)" in value for value in outputs + lines)


def test_derivative_and_execution_routes_are_explicit(tmp_path):
    _module, _left, _right, function = _function(tmp_path,
        derivatives=(NativeDerivative("exact", "example::Law::jacobian"),
                     NativeDerivative("approximate", "example::Law::approximate")))
    assert function.require_derivative("exact")["target"].endswith("jacobian")
    assert function.require_derivative("approximate")["target"].endswith("approximate")
    assert function.require_derivative("finite_difference")["target"] is None
    assert function.require_derivative("exact")["component"] == function.component.authority()
    with pytest.raises(ValueError, match="no declared unavailable"):
        function.require_derivative("unavailable")
    with pytest.raises(ValueError, match="no compiled device"):
        function.require_execution("device")
    with pytest.raises(ValueError, match="source interface"):
        replace(function, interface="foreign-abi")


def test_autodiff_uses_only_declared_exact_native_jacobian(tmp_path):
    from pops._ir.lowering import diff
    _module, _left, _right, function = _function(tmp_path,
        derivatives=(NativeDerivative("exact", "example::Law::jacobian"),))
    a = Var("a", "cons")
    call = function((a, 2), (3, 4, 5))
    derivative = diff(call.left[0], "a")
    lines, values = _cse_emit([derivative], "pops::Real", "")
    assert "example::Law::jacobian(" in "\n".join(lines)
    assert len(values) == 1
    with pytest.raises(ValueError, match="no declared exact"):
        diff(replace(function, derivatives=())((a, 2), (3, 4, 5)).left[0], "a")
    with pytest.raises(ValueError, match="lagged"):
        diff(replace(function, effects=("lagged",))((a, 2), (3, 4, 5)).left[0], "a")


def test_unsupported_native_side_effects_refuse_and_diagnostic_occurrences_stay_distinct(tmp_path):
    for effect in ("mutation", "io", "write_state", "external_publication"):
        with pytest.raises(ValueError, match="transactional admission"):
            _function(tmp_path, effects=("fallible", effect))
    _module, _left, _right, function = _function(tmp_path, effects=("fallible", "diagnostic_counter"))
    with pytest.raises(ValueError, match="explicit logical occurrence"):
        function((1, 2), (3, 4, 5))
    first = function((1, 2), (3, 4, 5), occurrence="first")
    second = function((1, 2), (3, 4, 5), occurrence="second")
    declarations, _outputs = _cse_emit((first.left[0] + first.left[0], second.right[0]), "double", "")
    assert sum("example::Law::evaluate(" in line for line in declarations) == 2
    assert _key(first) != _key(second)
