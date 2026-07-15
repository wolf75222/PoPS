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
     ``to_module()`` / ``lower()`` and carries a ``module_hash``;
  3  a facade dependency error is remapped, citing the model name / states / operators;
  4  the emit model of a facade Model is BYTE-IDENTICAL through ``lower_and_validate`` vs direct.

Guarded with ``pytest.importorskip("pops")``; the ``__main__`` block runs pytest.
"""
from pops.codegen.program_codegen import emit_cpp_program
import sys

import pytest
from pops.numerics.terms import DefaultSource, Flux

from typed_program_support import typed_state

pytest.importorskip("pops")
from pops import model as model_pkg  # noqa: E402
from pops import time as adctime  # noqa: E402
from pops._ir.expr import Const  # noqa: E402
from pops.physics._facade import Model  # noqa: E402
from pops.codegen.module_lowering import (  # noqa: E402
    _module_to_model, lower_and_validate, remap_lowering_error)


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


# --- 2: a facade Model resolves to its operator-first Module with no manual to_module -----------

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


# --- 4: byte-identical emit through lower_and_validate vs direct --------------------------------

def test_emit_is_byte_identical_through_lower_and_validate():
    direct_model = _facade_model()
    direct = emit_cpp_program(_fe_program(direct_model, "cmp"),
        model=direct_model, target="system")
    candidate = _facade_model()
    emit_model, _ = lower_and_validate(candidate, facade=None)
    via = emit_cpp_program(_fe_program(emit_model, "cmp"), model=emit_model, target="system")
    assert direct == via, "routing a facade Model through lower_and_validate is byte-identical"


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

    def fake_run_compile(cmd, what):
        with open(cmd[cmd.index("-o") + 1], "wb") as handle:
            handle.write(b"FAKE-SO")

    monkeypatch.setattr(cd, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: "c++23")
    monkeypatch.setattr(cd, "pops_header_signature", lambda inc: "TESTSIG")
    monkeypatch.setattr(cd, "_run_compile", fake_run_compile)
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    return cd


def _final_trace_artifact(compiled):
    """Wrap the real low-level compile result in the exact final artifact/plan records."""

    from pops.codegen._compiled_artifact import CompiledBlockArtifact, CompiledSimulationArtifact
    from pops.codegen._plans import ResolvedBlock, ResolvedSimulationPlan
    from pops.identity import make_identity
    from pops.model.bind_schema import BindSchema
    from pops.problem._snapshot import AuthoringSnapshot
    from tests.python.support.layout_plan import resolved_layout_contract

    class _CompiledTraceModel:
        so_path = "/nonexistent/trace-model.so"
        backend = "production"
        target = "system"
        abi_key = compiled.abi_key
        cxx = compiled.cxx
        std = compiled.std
        model_hash = "trace-model"
        gamma = None
        caps = {"cpu": True, "mpi": False, "amr": False, "gpu": False}
        artifact_identity = make_identity("artifact", {"component": "trace-model"})

        @staticmethod
        def __pops_artifact_model_metadata__():
            return {
                "schema_version": 2,
                "state_spaces": ("U",),
                "cons_names": ("rho", "mx", "my"),
                "n_vars": 3,
                "params": {},
                "aux_names": (),
                "n_aux": 0,
                "capabilities": {"mpi": False},
                "wave_speed_provider": None,
            }

    layout = {"kind": "uniform"}
    layout_plan, coverage = resolved_layout_contract(
        layout, target="system", block_names=("ep",))
    schema = BindSchema()
    plan = ResolvedSimulationPlan(
        snapshot=AuthoringSnapshot({"case": "module-trace"}),
        target="system",
        backend="production",
        layout=layout,
        layout_plan=layout_plan,
        layout_targets={layout_plan.layouts[0].handle.qualified_id: "system"},
        time=compiled.program,
        blocks=(ResolvedBlock(
            "ep", {"model": "trace-model"}, {"ghost_depth": 2}, "production",
            ("U",), ("test::ep::state::U",)),),
        bind_schema=schema,
        compile_values=schema.resolve_compile(),
        field_plans={},
        libraries=(),
        requirements={},
        capabilities={"cpu": True},
        lowering_coverage=coverage,
    )
    block = CompiledBlockArtifact(
        "ep", _CompiledTraceModel(), plan.blocks[0].spatial, ("U",))
    return CompiledSimulationArtifact(plan=plan, program=compiled, blocks=(block,))


def test_compile_problem_chain_threads_trace_for_facade_model(monkeypatch, tmp_path):
    cd = _stub_toolchain(monkeypatch, tmp_path)
    model = _facade_model("ep")
    compiled = cd.compile_problem(time=_fe_program(model), model=model,
                                  include=str(tmp_path))
    assert compiled.module_manifest is not None, \
        "the REAL compile chain attaches the operator-first Module manifest"
    assert compiled.module_hash(), "the REAL compile chain attaches the module_hash"
    report = _final_trace_artifact(compiled).inspect().to_dict()
    ops = [op.get("name") for op in report["module_manifest"]["operators"]]
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
                           include=str(tmp_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
