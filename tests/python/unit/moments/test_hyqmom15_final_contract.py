"""ADC-694 canonical HyQMOM15 model, closure and Case acceptance contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from pops.lib.models.moments import HyQMOM15
from pops.moments import LocalClosure, closure, moment_names
from pops.physics import Model


def test_hyqmom15_is_a_real_model_with_exact_generic_handles() -> None:
    definition = HyQMOM15.vlasov_lorentz(exact_speeds=False)
    assert type(definition.model) is Model
    assert tuple(definition.state.components) == tuple(moment_names(4))
    assert len(definition.state.components) == 15
    assert definition.model.rate_contract(definition.explicit_rate)["flux"] == definition.flux
    assert definition.implicit_source.kind == "local_linear_operator"
    assert definition.realizable_set.options() == {"order": 4}
    assert set(definition.model.module.operator_registry().names()) >= {
        definition.explicit_rate.registered_operator_name,
        definition.electric_source.reg_name,
        definition.implicit_source.registered_operator_name,
    }


def test_local_closure_is_model_agnostic_and_order_checked() -> None:
    @closure(4)
    def zero_fifth_order(_standardized):
        return {"S%d%d" % (p, 5 - p): 0.0 for p in range(6)}

    assert isinstance(zero_fifth_order, LocalClosure)
    definition = HyQMOM15.vlasov_lorentz(
        closure=zero_fifth_order, exact_speeds=False)
    assert definition.closure.contract_data() == {
        "kind": "local_moment_closure", "order": 4, "name": "zero_fifth_order"}

    @closure(2)
    def wrong_order(_standardized):
        return {"S30": 0.0, "S21": 0.0, "S12": 0.0, "S03": 0.0}

    with pytest.raises(ValueError, match="declares order 2"):
        HyQMOM15.vlasov_lorentz(
            closure=wrong_order, exact_speeds=False)


def test_final_example_resolves_and_declares_complete_rollback_surface() -> None:
    root = Path(__file__).resolve().parents[4]
    path = root / "EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"
    spec = importlib.util.spec_from_file_location("hyqmom15_final_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    case, physics = module.build_case()

    import pops
    from pops.mesh import CartesianMesh
    from pops.mesh.layouts import Uniform

    resolved = pops.resolve(
        pops.validate(case), layout=Uniform(CartesianMesh(n=8, periodic=True)))
    assert len(resolved.blocks) == 1
    assert len(physics.components) == 15
    transaction = case._time.transaction_plan()
    assert transaction.guards == ("moments.realizable(order=4)",)
    assert set(transaction.staged_effects) == {
        "state", "fields", "flux_ledgers", "histories", "schedules", "consumers"}
    assert len(case._consumers.inspect()["nodes"]) == 2
    emitted = case._time.emit_cpp_program(model=physics.model.dsl)
    assert "pops::detail::mat_inverse<15>(" in emitted
    assert "pops::Real M_[15][15];" in emitted
