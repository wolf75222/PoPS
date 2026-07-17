import pytest

from pops import model as model_pkg
from pops.codegen.lowering_coverage import LoweringRejection
from pops.codegen.module_lowering import _module_to_model, lower_and_validate
from pops._ir.expr import Const


def _lowerable_module():
    module = model_pkg.Module("coverage")
    state = module.state_space("U", ("rho",))
    module.operator(
        name="source", signature=(state,) >> model_pkg.Rate(state),
        kind="local_source", expr=[Const(0.0)])
    return module


def test_module_lowering_attaches_total_coverage_graph():
    lowered = _module_to_model(_lowerable_module())
    report = lowered.lowering_coverage_report
    rows = {row.source: row for row in report.rows}
    assert rows["state_space:U"].disposition == "lowered"
    assert rows["derived:primitive:rho"].disposition == "derived"
    assert rows["operator:source"].targets == ("dsl:source_term",)
    assert rows["operator_metadata:source"].disposition == "documentary"
    assert report.target_to_sources["dsl:source_term"] == ("operator:source",)


def test_unsupported_operator_kind_is_a_structured_partial_rejection():
    module = model_pkg.Module("unsupported")
    state = module.state_space("U", ("rho",))
    matrix = model_pkg.MatrixFreeOperator(state, state)
    module.operator(
        name="implicit", signature=(state,) >> matrix,
        kind="matrix_free_operator", expr=Const(0.0))
    with pytest.raises(LoweringRejection) as caught:
        lower_and_validate(module)
    rejection = caught.value
    assert rejection.source == "operator:implicit"
    assert rejection.gate == "operator_kind_not_lowerable"
    row = next(row for row in rejection.coverage_report.rows
               if row.source == "operator:implicit")
    assert row.disposition == "rejected" and row.targets == ()
    assert "state_space:U" in rejection.coverage_report.source_to_targets


def test_bodyless_operator_rejection_also_carries_partial_report():
    module = model_pkg.Module("bodyless")
    state = module.state_space("U", ("rho",))
    module.operator(
        name="source", signature=(state,) >> model_pkg.Rate(state),
        kind="local_source", expr=lambda: None)
    with pytest.raises(LoweringRejection, match="no IR body") as caught:
        _module_to_model(module)
    assert caught.value.gate == "expression_body_required"
