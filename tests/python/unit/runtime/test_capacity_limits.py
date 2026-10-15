"""Runtime parameter cardinality is dynamic; carrier bytes are bounded at bind.

Coupled-source legacy capacities remain covered separately. Native report tests
require the installed SDK of this change; source-only receipts select authoring
and manifest tests explicitly rather than pretending to qualify an older SDK.
"""
import pytest

import pops  # noqa: F401 -- ensures the package import path is set up like the sibling suites
from pops.physics._facade import Model
from pops.params import RuntimeParam


def _model_with_runtime_params(n):
    """An isothermal-like model whose source reads @p n distinct runtime params (p_0..p_{n-1}), so the
    model declares exactly n runtime params. The formula is physically trivial (a sum of scaled densities)
    -- the point is the COUNT, which drives assign_runtime_indices."""
    m = Model("cap")
    rho, mx, my = m.conservative_vars("rho", "rho_u", "rho_v")
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    # A source term that reads every runtime param, forcing each to get a stable index.
    acc = rho * 0.0
    for k in range(n):
        handle = m.param(RuntimeParam("p_%d" % k, default=float(k) + 1.0))
        acc = acc + m.value(handle) * rho
    m.primitive_vars(rho=rho, u=u, v=v)
    m.conservative_from([rho, rho * u, rho * v])
    m.flux(x=[mx, mx * u, my * u], y=[my, mx * v, my * v])
    m.source([acc, rho * 0.0, rho * 0.0])
    return m


def _impl(model):
    """The authoring HyperbolicModel implementation behind a facade Model (where assign_runtime_indices
    lives). The facade stores it on ``_m`` (pops.physics._facade.Model)."""
    for attr in ("_m", "_impl", "impl", "_model"):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "assign_runtime_indices"):
            return obj
    if hasattr(model, "assign_runtime_indices"):
        return model
    raise AssertionError("could not find the authoring model behind the facade")


def _limit():
    from pops.physics.aux import max_runtime_params
    return max_runtime_params()


def test_runtime_param_storage_has_no_fixed_model_count_limit():
    assert _limit() is None


@pytest.mark.parametrize("count", [33, 65, 129])
def test_finite_runtime_parameter_families_are_accepted(count):
    nodes = _impl(_model_with_runtime_params(count)).assign_runtime_indices()
    assert len(nodes) == count
    assert [node.to_cpp() for node in nodes] == [f"params.get({index})" for index in range(count)]


def test_python_capacity_contract_matches_native_owned_carrier():
    from pops import _pops

    assert _pops.__max_runtime_params__ is None
    first = _pops.runtime_parameter_storage_bytes(65)
    assert first > 2 * 65 * 8
    assert _pops.runtime_parameter_storage_bytes(129) - first == 2 * (129 - 65) * 8
    assert _pops.runtime_parameter_storage_bytes(0) == 0


def test_module_manifest_surfaces_runtime_param_utilization():
    from pops.model.manifest import module_manifest_of
    model = _model_with_runtime_params(3)
    manifest = module_manifest_of(model)
    assert manifest is not None
    util = manifest.params_utilization
    assert util["limit"] == _limit()
    # count = the runtime params surfaced (>= 0); the limit and status are always present.
    assert util["count"] >= 0
    assert util["status"] == "dynamic"
    # It round-trips into the JSON dict view.
    assert manifest.to_dict()["params_utilization"]["limit"] == _limit()


def test_params_utilization_helper_computes_count_limit_status():
    from pops.model import Module
    from pops.params import ConstParam

    limit = _limit()
    module = Module("utilization")
    module.parameters(
        RuntimeParam("a", default=1.0),
        RuntimeParam("b", default=2.0),
        ConstParam("c", 3.0),
    )
    util = module.manifest().params_utilization
    assert util == {"count": 2, "limit": limit, "status": "dynamic"}
    large = Module("large-family")
    large.parameters(*(
        RuntimeParam("p%d" % k, default=float(k)) for k in range(65)
    ))
    assert large.manifest().params_utilization == {"count": 65, "limit": None, "status": "dynamic"}


def test_coupled_source_overflow_errors_are_exact():
    from pops.physics import multispecies as ms
    # Too many source terms (> kCsMaxTerms): the compile() validation names the count and the bound.
    src = ms.CoupledSource("cs")
    ne = src.block("e").role("density")
    # Build kCsMaxTerms + 1 output terms on distinct (block, role) pairs.
    over = ms._CS_MAX_TERMS + 1
    for k in range(over):
        src.add(block="b%d" % k, role="density", expr=ne)
    with pytest.raises(ValueError) as exc:
        src.compile()
    msg = str(exc.value)
    assert "too many source terms" in msg
    assert str(ms._CS_MAX_TERMS) in msg


def test_compiled_coupled_source_utilization_surfaces_bounds():
    from pops.physics import multispecies as ms
    src = ms.CoupledSource("cs")
    ne = src.block("e").role("density")
    ni = src.block("i").role("density")
    src.add(block="e", role="density", expr=ne * ni)
    compiled = src.compile()
    util = compiled.utilization()
    assert util["registers"]["limit"] == ms._CS_MAX_REG
    assert util["terms"]["limit"] == ms._CS_MAX_TERMS
    assert util["program"]["limit"] == ms._CS_MAX_PROG
    assert util["terms"]["count"] == 1


if __name__ == "__main__":  # allow running as a script (some CI shards import-run tests)
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
