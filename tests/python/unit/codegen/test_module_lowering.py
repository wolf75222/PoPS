#!/usr/bin/env python3
"""ADC-557 acceptance: ONE validation + lowering, facade error remap, Module as canonical IR.

``compile_problem`` lowers and validates the model via the SINGLE ``lower_and_validate`` entry (the
divergent standalone ``model.check()`` compile step is gone), returning the emit model plus the
operator-first :class:`pops.model.Module` -- the canonical compile-IR authority carried as the
lowered-module trace. A lowering / validation error is remapped onto the user's facade handles.

These checks stay pure Python (no compiler / no ``.so``); they pin:

  1  a raw Module with a bodyless codegen operator raises the SAME error through
     ``lower_and_validate`` as through ``_module_to_model`` (one validation path);
  2  a facade Model resolves to its operator-first Module (``source_module``) with NO manual
     ``lower()`` and carries a ``module_hash``;
  3  a facade dependency error is remapped, citing the model name / states / operators;
  4  raw program emission fails closed until the model passed through the canonical Module route.

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
from pops.codegen.program_codegen import emit_cpp_program
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pops.numerics.terms import DefaultSource, Flux

from typed_program_support import typed_state
from tests.python.support.requirements import repo_include

pytest.importorskip("pops")
from pops import model as model_pkg  # noqa: E402
from pops import time as adctime  # noqa: E402
from pops._ir.expr import Const  # noqa: E402
from pops.physics._facade import Model  # noqa: E402
from pops.codegen.module_lowering import (  # noqa: E402
    _lower_native_role, _module_to_model, _typed_lowering_roles,
    lower_and_validate, remap_lowering_error)
from pops.frames import X_AXIS, Z_AXIS  # noqa: E402
from pops.physics import Axial, Density, Momentum, RoleKey, Scalar  # noqa: E402
from pops.physics.roles import native_role_token  # noqa: E402
from pops.physics._coupled_abi import role_canonical  # noqa: E402


def test_module_role_lowering_preserves_typed_boundary_semantics():
    assert native_role_token(_lower_native_role(Density())) == "density"
    assert native_role_token(_lower_native_role(Momentum(axis=X_AXIS))) == "momentum:0"
    assert native_role_token(_lower_native_role(Axial(axis=X_AXIS))) == "axial:0"
    assert native_role_token(_lower_native_role(Axial(axis=Z_AXIS))) == "axial:2"
    assert native_role_token(_lower_native_role(Scalar())) == "scalar"
    assert native_role_token(_lower_native_role("momentum:1")) == "momentum:1"
    assert native_role_token(_lower_native_role("q1")) == "q1"
    with pytest.raises(ValueError, match="malformed reserved physical role"):
        _lower_native_role("momentum:01")
    assert native_role_token(_lower_native_role("momentum_x")) == "momentum_x"
    assert native_role_token(_lower_native_role("Custom")) == "Custom"
    assert role_canonical("axial:2") == "axial:2"


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_module_exact_role_tokens_reenter_private_authoring_as_typed_values(dimension):
    components = ("density", *("p%d" % axis for axis in range(dimension)))
    state = SimpleNamespace(
        components=components,
        roles={
            "density": "density",
            **{"p%d" % axis: "momentum:%d" % axis for axis in range(dimension)},
        },
    )

    roles = _typed_lowering_roles(state)
    assert roles is not None
    assert tuple(native_role_token(role) for role in roles) == (
        "density",
        *("momentum:%d" % axis for axis in range(dimension)),
    )


def test_module_custom_role_tokens_reenter_authoring_without_label_loss():
    state = SimpleNamespace(
        components=("first", "second"),
        roles={"first": "q1", "second": "q2"},
    )

    lowered = _typed_lowering_roles(state)

    assert lowered is not None
    assert all(isinstance(role, RoleKey) for role in lowered)
    assert tuple(native_role_token(role) for role in lowered) == ("q1", "q2")

    module = model_pkg.Module("custom_role_module")
    module.state_space(
        "U", ("first", "second"), roles={"first": "q1", "second": "q2"})
    emitted = _module_to_model(module)
    assert tuple(native_role_token(role) for role in emitted._m.cons_roles) == ("q1", "q2")


def test_module_pascal_role_remains_an_exact_custom_label_not_a_physical_alias():
    state = SimpleNamespace(components=("q",), roles={"q": "MomentumX"})
    roles = _typed_lowering_roles(state)
    assert roles is not None
    assert native_role_token(roles[0]) == "MomentumX"


def _facade_model(name="ep"):
    m = Model(name)
    rho, mx, my = m.conservative_vars("rho", "mx", "my")
    gx = m.aux("grad_x")
    gy = m.aux("grad_y")
    m.flux(x=[mx, mx * mx / rho, mx * my / rho], y=[my, mx * my / rho, my * my / rho])
    m.source_term("electric", [Const(0.0), -rho * gx, -rho * gy])
    m.elliptic_rhs(rho - 1.0)
    return m


def _fe_program(model, name="p"):
    P = adctime.Program(name)
    U = typed_state(P, "ep", model=model)
    electric = model.module.operator_handle("electric")
    R = P.rhs(state=U, terms=[Flux(), electric])
    endpoint = typed_state(P, "ep", state_name="U", model=model).next
    P.commit(endpoint, P.value("U1", U + P.dt * R, at=endpoint.point))
    return P


# --- 1: ONE validation -- a bodyless Module operator raises the SAME error both ways ------------

def _bodyless_module():
    mod = model_pkg.Module("bodyless")
    u = mod.state_space("U", ("rho",))
    # A grid_operator (flux) with a CALLABLE (non-IR) body is not compilable -> the single validation
    # rejects it with the "no IR body" error.
    mod.operator(name="flux", signature=(u,) >> model_pkg.Rate(u), kind="grid_operator",
                 expr=lambda: None)
    return mod


def test_one_validation_bodyless_operator_same_error():
    direct = None
    via_lower = None
    try:
        _module_to_model(_bodyless_module())
    except ValueError as exc:
        direct = str(exc)
    try:
        lower_and_validate(_bodyless_module(), facade=None)
    except ValueError as exc:
        via_lower = str(exc)
    assert direct is not None, "_module_to_model rejects a bodyless codegen operator"
    assert via_lower is not None, "lower_and_validate rejects it too (one validation path)"
    assert direct == via_lower, "the SAME error text is raised via both entries (no divergence)"


# --- 2: a facade Model resolves to its operator-first Module with no manual lower() -------------

def test_facade_model_carries_operator_first_module():
    m = _facade_model()
    emit_model, source_module = lower_and_validate(m, facade=m)
    assert isinstance(source_module, model_pkg.Module), \
        "lower_and_validate returns the operator-first Module as the canonical IR authority"
    assert source_module.module_hash(), "the Module carries a stable hash (drift detection)"
    # The emit model is the facade Model itself (consumed as-is; byte-identical emit).
    assert emit_model is m


# --- 3: a facade dependency error is remapped to the user's handles ----------------------------

def test_remap_cites_facade_handles():
    m = _facade_model("mymodel")
    try:
        remap_lowering_error(ValueError("undefined variables ['ghost']"), m)
    except ValueError as exc:
        msg = str(exc)
        assert "mymodel" in msg, "the remap names the model the user wrote"
        assert "states" in msg and "operators" in msg, "the remap lists the declared handles"
        assert "undefined variables" in msg, "the original cause is preserved"
    else:
        pytest.fail("remap_lowering_error must re-raise")


def test_remap_without_facade_reraises_unchanged():
    original = ValueError("raw internal error")
    with pytest.raises(ValueError) as exc:
        remap_lowering_error(original, None)
    assert exc.value is original, "no facade -> the original error is re-raised verbatim"


# --- 4: compilation consumes the canonical lowered Module authority -----------------------------

def test_emit_requires_lowered_module_provider_authority():
    direct_model = _facade_model()
    with pytest.raises(ValueError, match="exact auxiliary ProviderPack"):
        emit_cpp_program(_fe_program(direct_model, "cmp"),
                         model=direct_model, target="system")

    candidate = _facade_model()
    emit_model, _ = lower_and_validate(candidate, facade=None)
    first = emit_cpp_program(_fe_program(emit_model, "cmp"), model=emit_model, target="system")
    second = emit_cpp_program(_fe_program(emit_model, "cmp"), model=emit_model, target="system")
    assert first == second, "the canonical Module route is stable and deterministic"


def test_lowered_program_installs_a_local_provider_plan_before_its_context():
    model, _ = lower_and_validate(_facade_model(), facade=None)
    source = emit_cpp_program(_fe_program(model, "provider-plan"), model=model, target="system")

    assert "install_auxiliary_consumer_plan(ConsumerPlan{" in source
    assert "ctx.template provider_values_view<2>(" in source
    assert source.index("install_auxiliary_consumer_plan(ConsumerPlan{") < source.index(
        "make_program_execution_provider"
    )
    assert "ctx.aux()" not in source


def test_one_typed_named_flux_supplies_the_native_base_flux_without_losing_its_name():
    module = model_pkg.Module("named_flux_route")
    state = module.state_space("fluid", ("rho",))
    (rho,) = module.state_symbols(state)
    flux = module.operator(
        name="transport",
        signature=(state,) >> model_pkg.Rate(state),
        kind="grid_operator",
        expr={"x": (rho,), "y": (rho,)},
    )
    module.eigenvalues(x=(Const(1.0),), y=(Const(1.0),))
    module.rate_operator(
        "advance", state_space=module.state_handle(state), flux=True, fluxes=(flux,))

    lowered = _module_to_model(module)

    assert lowered._m._flux, "the native HyperbolicModel base-flux concept is satisfied"
    assert tuple(lowered._m._flux_terms) == ("transport",)
    assert lowered._m._rate_operators["advance"]["fluxes"] == ["transport"]


def test_module_lowering_rejects_conflicting_native_default_fluxes_for_one_state():
    module = model_pkg.Module("conflicting_default_fluxes")
    state = module.state_space("fluid", ("rho",))
    (rho,) = module.state_symbols(state)
    signature = (state,) >> model_pkg.Rate(state)
    first = module.operator(
        "first",
        signature=signature,
        kind="grid_operator",
        expr={"x": (rho,), "y": (rho,)},
    )
    second = module.operator(
        "second",
        signature=signature,
        kind="grid_operator",
        expr={"x": (2.0 * rho,), "y": (2.0 * rho,)},
    )
    module.rate_operator(
        "first_rate", state_space=state, fluxes=(first,), default_flux=first)
    module.rate_operator(
        "second_rate", state_space=state, fluxes=(second,), default_flux=second)

    with pytest.raises(ValueError, match="conflicting native-default flux operators"):
        _module_to_model(module)


# --- 5: the handle carries the module_hash for drift detection + the lowered-module trace -------

def test_handle_carries_module_hash_and_trace():
    from pops.codegen.loader import CompiledProblem
    from pops.model.manifest import module_manifest_of
    m = _facade_model("gas")
    _, source_module = lower_and_validate(m, facade=m)
    manifest = module_manifest_of(source_module)
    program = _fe_program(m, "trace")
    handle = CompiledProblem("/tmp/none.so", program, m._m, "SIG|c++|c++23", "c++", "c++23",
                             module_manifest=manifest, module_hash=source_module.module_hash())
    assert handle.module_hash() == source_module.module_hash(), \
        "the handle carries the compile-time Module hash (bind drift detection)"
    # The low-level handle retains the immutable operator-first trace even before a real artifact
    # is loaded; full inspect() is covered below through the actual compile_problem chain.
    assert handle.module_manifest.to_dict()["name"] == "gas"


# --- 6: the REAL compile_problem chain threads the trace + hash onto the handle -----------------
# Only the toolchain seams are stubbed (no compiler / Kokkos in the unit lane); everything from
# lower_and_validate through manifest/hash capture to the CompiledProblem kwargs is the REAL code.
# This pins the exact chain the compiler-gated integration test (test_compile_module_trace.py)
# exercises on CI, for BOTH model shapes: the facade Model (trace present) and the native brick
# ModelSpec (trace honestly absent) -- the CI-only failure mode a stubbed handle test cannot see.

def _stub_toolchain(monkeypatch, tmp_path):
    import pops.codegen._compile_drivers as cd
    import pops.codegen.toolchain as toolchain
    import pops.native_components as native_components
    import pops._native_selector as native_selector

    def fake_run_compile(cmd, what):
        del what
        output = Path(cmd[cmd.index("-o") + 1])
        dependency_file = Path(cmd[cmd.index("-MF") + 1])
        generated = Path(next(item for item in cmd if item.endswith("problem.cpp")))

        def dep_escape(path):
            return str(path).replace("\\", "\\\\").replace(" ", "\\ ").replace("$", "$$")

        output.write_bytes(b"FAKE-SO")
        dependency_file.write_text(
            "%s: %s\n" % (dep_escape(output), dep_escape(generated)),
            encoding="utf-8",
        )

    monkeypatch.setattr(cd, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: "c++23")
    monkeypatch.setattr(cd, "pops_header_signature", lambda inc: "TESTSIG")
    monkeypatch.setattr(cd, "_run_compile", fake_run_compile)
    # This is a pure codegen trace test.  Compilation still receives one explicit
    # native rank, but no extension is loaded from the test worktree.
    monkeypatch.setattr(native_selector, "select_native_dimension", lambda dimension: dimension)
    monkeypatch.setattr(toolchain, "loader_native_dimension", lambda: 2)
    monkeypatch.setattr(toolchain, "_native_feature_key", lambda: "test-native-features")
    monkeypatch.setattr(native_components, "verify_prepared_native_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    return cd


def test_compile_problem_chain_threads_trace_for_facade_model(monkeypatch, tmp_path):
    cd = _stub_toolchain(monkeypatch, tmp_path)
    model = _facade_model("ep")
    compiled = cd.compile_problem(time=_fe_program(model), model=model,
                                  include=repo_include(), native_dimension=2)
    assert compiled.module_manifest is not None, \
        "the REAL compile chain attaches the operator-first Module manifest"
    assert compiled.module_hash(), "the REAL compile chain attaches the module_hash"
    ops = [op.get("name") for op in compiled.module_manifest.to_dict()["operators"]]
    assert "flux_default" in ops, "the trace lists the facade's operators: %s" % ops


def _fe_program_default(model, name="spec"):
    """An FE program on the DEFAULT source only (ctx.rhs_into: no model kernels emitted), the same
    minimal lowering the sibling integration tests compile with a native brick ModelSpec."""
    P = adctime.Program(name)
    U = typed_state(P, "ep", model=model)
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    endpoint = typed_state(P, "ep", state_name="U", model=model).next
    P.commit(endpoint, P.value("U1", U + P.dt * R, at=endpoint.point))
    return P


def test_compile_problem_chain_refuses_a_moduleless_model_duck(monkeypatch, tmp_path):
    # Semantic identity requires an authenticated Module authority or one of the explicitly
    # supported physical models. A shape-compatible duck must fail instead of producing an
    # unauthenticated artifact with an "honestly absent" identity.
    class _ModulelessModel:
        name = "spec"
        cons_names = ["rho", "mx", "my"]

        def __init__(self):
            from pops.model import OwnerKind, OwnerPath, StateHandle, StateSpace

            self.owner_path = OwnerPath.fresh(OwnerKind.MODEL_DEFINITION, self.name)
            self._state = StateHandle(
                "U", owner=self.owner_path,
                space=StateSpace("U", tuple(self.cons_names)))

        def declaration_index(self):
            from pops.model import DeclarationIndex

            return DeclarationIndex(owner=self.owner_path, handles=(self._state,))

    cd = _stub_toolchain(monkeypatch, tmp_path)
    model = _ModulelessModel()
    with pytest.raises(
        TypeError,
        match="OperatorRegistry|Module authority|supported model|semantic model identity",
    ):
        cd.compile_problem(time=_fe_program_default(model), model=model,
                           include=repo_include(), native_dimension=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
