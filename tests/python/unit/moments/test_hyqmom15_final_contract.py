"""ADC-694 canonical HyQMOM15 model, closure and final authoring contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from pops.lib.models.moments import HyQMOM15
from pops.moments import LocalClosure, closure, moment_names
from pops.physics import Model


ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples/final/EXEMPLE_SPEC_FINALE_15_MOMENTS_HYQMOM.py"


def _load_example():
    spec = importlib.util.spec_from_file_location("pops_final_hyqmom15", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hyqmom15_is_a_real_model_with_exact_generic_handles() -> None:
    model = HyQMOM15.vlasov_lorentz(exact_speeds=False)
    assert type(model) is Model
    state = model.states["U"]
    flux = model.fluxes["transport"]
    explicit_rate = model.operators["transport"]
    implicit_source = model.operators["magnetic_rotation"]
    electric_source = model.sources["electric"]
    assert tuple(state.components) == tuple(moment_names(4))
    assert len(state.components) == 15
    assert model.rate_contract(explicit_rate) == {
        "state": state, "flux": flux, "sources": (electric_source,)}
    assert implicit_source.kind == "local_linear_operator"
    assert set(model.module.operator_registry().names()) >= {
        explicit_rate.registered_operator_name,
        electric_source.reg_name,
        implicit_source.registered_operator_name,
    }


def test_local_closure_is_model_agnostic_and_order_checked() -> None:
    @closure(4)
    def zero_fifth_order(_standardized):
        return {"S%d%d" % (p, 5 - p): 0.0 for p in range(6)}

    assert isinstance(zero_fifth_order, LocalClosure)
    model = HyQMOM15.vlasov_lorentz(
        closure=zero_fifth_order, exact_speeds=False)
    assert type(model) is Model
    assert tuple(model.states["U"].components) == tuple(moment_names(4))

    @closure(2)
    def wrong_order(_standardized):
        return {"S30": 0.0, "S21": 0.0, "S12": 0.0, "S03": 0.0}

    with pytest.raises(ValueError, match="declares order 2"):
        HyQMOM15.vlasov_lorentz(
            closure=wrong_order, exact_speeds=False)


def test_final_authoring_derives_field_storage_and_complete_generic_program() -> None:
    target = _load_example().build_authoring()

    assert type(target.model) is Model
    assert target.components == tuple(moment_names(4))
    assert target.model.field_spaces()[target.field.local_id].components == (
        "phi", "grad_x", "grad_y")
    assert target.field_provider == target.model.operators["fields"]
    assert target.program.transaction_plan() is not None
    assert len(target.case._consumers.inspect()["nodes"]) == 3
    local_map = target.implicit_operator.signature.output
    assert len(local_map.domain.components) == 15
    assert local_map.domain == local_map.range


def test_final_example_uses_only_the_root_lifecycle_and_public_layout_home() -> None:
    source = EXAMPLE.read_text(encoding="utf-8")
    assert "from pops.layouts import Uniform" in source
    assert "pops.mesh.layouts" not in source
    assert "BindInputs" not in source
    assert "simulation.run(" not in source
    assert "from pops.runtime import System" not in source
    assert "from pops.runtime import AmrSystem" not in source
    for call in (
        "pops.validate(", "pops.resolve(", "pops.compile(", "pops.bind(", "pops.run(",
    ):
        assert call in source
