"""Source algebra and discriminating oracles; these tests make no native claim."""
import numpy as np
import pops

from examples.api_v02.models_M01_M14 import build_m07, build_m10
from examples.api_v02.models_drift_wetdry_oracles import (
    bernoulli_quadrature, drift_diffusion_rhs, periodic_potential, ssprk2, wet_dry_rhs,
)
from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph


def test_wet_dry_oracle_rejects_noop_and_each_source_counterexample():
    spec = build_m07(nonstationary=True)
    initial = next(iter(spec.initial_values.values()))
    expected = ssprk2(initial, lambda q: wet_dry_rhs(q, 1/24), dt=spec.dt)
    assert spec.invariants({"fluid": expected})["passed"]
    assert not spec.invariants({"fluid": initial})["passed"]
    for mode in ("none", "omit_left", "omit_right", "swap"):
        wrong = ssprk2(initial, lambda q, mode=mode: wet_dry_rhs(q, 1/24, source_mode=mode), dt=spec.dt)
        assert not spec.invariants({"fluid": wrong})["passed"], mode


def test_hydrostatic_oracle_balances_both_sources_at_equilibrium():
    spec = build_m07()
    initial = next(iter(spec.initial_values.values()))
    assert spec.invariants({"fluid": initial})["passed"]
    np.testing.assert_allclose(spec.dt*wet_dry_rhs(initial, 1/24), 0, rtol=0,
                               atol=8*np.finfo(float).eps)


def test_exponential_integral_oracle_and_live_field_controls():
    # Its independent integral also obeys B(-z)-B(z)=z and B(0)=1.
    z = np.array([-4., -.3, -1.e-6, 0., 1.e-6, .3, 4.])
    np.testing.assert_allclose(bernoulli_quadrature(-z)-bernoulli_quadrature(z), z,
                               rtol=2.e-14, atol=4.e-16)
    spec = build_m10()
    initial = next(iter(spec.initial_values.values()))
    lengths = (2*np.pi, 2*np.pi)
    expected = ssprk2(initial, lambda q: drift_diffusion_rhs(q, lengths), dt=spec.dt)
    stale = periodic_potential(initial[0], lengths)
    for option in ({"mobility": 0.}, {"mobility": -.1}, {"potential": stale}):
        wrong = ssprk2(initial, lambda q, option=option: drift_diffusion_rhs(q, lengths, **option), dt=spec.dt)
        assert np.max(abs(expected-wrong)) > 1.e-9, option
    assert np.max(abs(expected-initial)) > 1.e-5
    assert abs(np.mean(expected-initial)) < 3.e-15


def test_fitted_face_is_lowered_from_python_without_special_native_flux():
    spec = build_m10()
    plan = pops.resolve(pops.validate(spec.case), layout=spec.layout)
    cpp = emit_cpp_program(plan.time, target="system",
        model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert "apply_fitted" not in cpp
    assert "scharfetter_gummel_bernoulli" not in cpp
    assert "expm1(" in cpp
    assert "PreparedStencilInput" in cpp


def test_legacy_receivers_name_their_actual_builders():
    import ast
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/"integration/runtime/test_api_v02_methods_T01_T12.py"
    tree = ast.parse(path.read_text())
    required = {"T03_legacy_extended_Strang": "build_t03",
                "T10_legacy_homogeneous_Heun": "build_t10",
                "T11_legacy_two_species": "build_t11"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_compile"]
    for label, builder in required.items():
        call = next(c for c in calls if c.args and isinstance(c.args[0], ast.Constant)
                    and c.args[0].value == label)
        value = next(k.value for k in call.keywords if k.arg == "builder")
        assert isinstance(value, ast.Name) and value.id == builder
