"""A joint imported constitutive law at one consumer-selected endpoint."""
from pops._ir.expr import Var
from pops._ir.lowering import diff
from pops.codegen.native_constitutive import emit_native_constitutive
from pops.model import FieldSpace, Module, Signature
from pops.model.bundles import ProductSpace
from pops.native_calls import NativeDerivative, NativeFunction, NativeInputDomain
from pops.native_components import PreparedNativeComponent


def constitutive_function(directory):
    directory.mkdir()
    (directory / "constitutive.hpp").write_text(r'''#pragma once
#include <pops/core/model/native_call.hpp>
#include <atomic>
namespace constitutive {
static inline std::atomic<long> calls{0}, jacobian_calls{0};
inline pops::NativeCallResult<3> evaluate(double q) {
  ++calls;
  pops::NativeCallResult<3> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values = {q+q*q, 1+q*q, 2+q*q};
  return result;
}
inline pops::NativeCallResult<3> jacobian(double q) {
  ++jacobian_calls;
  pops::NativeCallResult<3> result;
  result.status = pops::EvaluationStatus::kOk;
  result.values = {1+2*q, 2*q, 2*q};
  return result;
}
}
''')
    component = PreparedNativeComponent.header_only("constitutive.joint", include_root=directory,
        entry_headers=("constitutive.hpp",))
    module = Module("constitutive")
    state = module.state_space("scalar", ("q",))
    output = ProductSpace({"transform": FieldSpace("W", components=("value",)),
                           "diffusivity": FieldSpace("A", components=("x", "y"))})
    function = NativeFunction(component, "constitutive::evaluate", Signature((state,), output),
        derivatives=(NativeDerivative("exact", "constitutive::jacobian"),),
        domains=(NativeInputDomain(0, 0, lower=0, lower_open=True),),
        effects=("fallible", "diagnostic_counter"))
    return function


def test_endpoint_joint_W_and_diagonal_A_preserve_exact_derivative_and_status(tmp_path):
    function = constitutive_function(tmp_path / "external")
    q = Var("q", "cons")
    call = function((q,), occurrence="constitutive-endpoint")
    expressions = (*call.transform, *call.diffusivity, diff(call.transform[0], q))
    emitted = emit_native_constitutive(expressions)
    text = "\n".join(emitted.lines)
    assert text.count("constitutive::evaluate(") == 1
    assert text.count("constitutive::jacobian(") == 1
    assert len(emitted.values) == 4
    assert len(emitted.functions) == 2
    assert "constitutive_status_" == emitted.status
    assert "constitutive_reason_" == emitted.reason
    assert "selected_ > constitutive_status_" in text
    assert ".reason_code > constitutive_reason_" in text
    assert all(".read(" in value for value in emitted.values)
