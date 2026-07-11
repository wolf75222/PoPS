"""Fixed-capacity limits are SURFACED and overflow is refused EARLY (ADC-610).

The runtime library carries several FIXED, device-copyable capacities that were previously hidden C++
constants: kMaxRuntimeParams (RuntimeParams::values[]), and the coupled-source kCsMax* bounds. This suite
proves the three ADC-610 guarantees for the RUNTIME-PARAM bound (the poster child) plus the coupled-source
bounds:

  1. OVERFLOW REJECTED EARLY, at codegen, with a USER-FACING error that NAMES the limit, the count and the
     offending params -- before any .so is emitted or any device array is read out of bounds. Pure Python
     (the guard fires in assign_runtime_indices), so no compiled _pops is required.
  2. The kMaxRuntimeParams bound is MIRRORED from the single C++ source: when _pops is importable, the
     Python literal (physics.aux) agrees with _pops.__max_runtime_params__ -- they cannot silently drift.
  3. The capacity is SURFACED in the reports/manifest: the ModuleManifest params_utilization row and the
     runtime-param report row carry {count, limit}.
  4. The coupled-source kCsMax* overflow still errors with the exact messages, and CompiledCoupledSource
     exposes a utilization() view of the bounds.

pops is never faked: the real Model / manifest / CoupledSource are used; _pops-dependent checks are
hasattr / import gated so the suite runs on a tree without a freshly built module.
"""
import pytest

import pops  # noqa: F401 -- ensures the package import path is set up like the sibling suites
from pops.physics.facade import Model
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
    lives). The facade stores it on ``_m`` (pops.physics.facade.Model)."""
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


def test_runtime_param_limit_is_a_positive_int():
    assert isinstance(_limit(), int) and _limit() >= 1


def test_thirty_three_runtime_params_raise_at_codegen_naming_the_limit():
    limit = _limit()
    model = _model_with_runtime_params(limit + 1)  # one over the fixed bound
    impl = _impl(model)
    with pytest.raises(ValueError) as exc:
        impl.assign_runtime_indices()
    msg = str(exc.value)
    # The error NAMES the limit, the count, and the model, and points at the C++ header.
    assert "kMaxRuntimeParams" in msg
    assert str(limit) in msg
    assert str(limit + 1) in msg
    assert "runtime_params.hpp" in msg
    # It NAMES the offending params (at least one of the declared names).
    assert "'p_0'" in msg


def test_exactly_the_limit_is_accepted():
    model = _model_with_runtime_params(_limit())
    nodes = _impl(model).assign_runtime_indices()
    assert len(nodes) == _limit()


def test_python_limit_matches_cpp_constant_when_module_present():
    _pops = pytest.importorskip("pops._pops")
    if not hasattr(_pops, "__max_runtime_params__"):
        pytest.skip("built _pops predates the __max_runtime_params__ export (ADC-610)")
    assert _limit() == int(_pops.__max_runtime_params__)


def test_module_manifest_surfaces_runtime_param_utilization():
    from pops.model.manifest import module_manifest_of
    model = _model_with_runtime_params(3)
    manifest = module_manifest_of(model)
    if manifest is None:
        pytest.skip("this facade Model exposes no backing Module (manifest honestly absent)")
    util = manifest.params_utilization
    assert util["limit"] == _limit()
    # count = the runtime params surfaced (>= 0); the limit and status are always present.
    assert util["count"] >= 0
    assert util["status"] in ("ok", "at_limit", "exceeded")
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
    assert util == {"count": 2, "limit": limit, "status": "ok"}
    at_limit = Module("at-limit")
    at_limit.parameters(*(
        RuntimeParam("p%d" % k, default=float(k)) for k in range(limit)
    ))
    assert at_limit.manifest().params_utilization["status"] == "at_limit"


def test_coupled_source_overflow_errors_are_exact():
    ms = pytest.importorskip("pops.physics.multispecies")
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
    ms = pytest.importorskip("pops.physics.multispecies")
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
