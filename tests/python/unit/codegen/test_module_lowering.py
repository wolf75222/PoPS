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
import sys

import pytest

from typed_program_support import typed_state

pytest.importorskip("pops")
from pops import model as model_pkg  # noqa: E402
from pops import time as adctime  # noqa: E402
from pops.ir.expr import Const  # noqa: E402
from pops.physics.facade import Model  # noqa: E402
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
    f = P.solve_fields(U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["electric"])
    endpoint = typed_state(P, "ep", state_name="U", model=model).next
    P.commit(endpoint, P.linear_combine("U1", U + P.dt * R, at=endpoint.point))
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
    direct = _fe_program(direct_model, "cmp").emit_cpp_program(
        model=direct_model, target="system")
    candidate = _facade_model()
    emit_model, _ = lower_and_validate(candidate, facade=None)
    via = _fe_program(emit_model, "cmp").emit_cpp_program(model=emit_model, target="system")
    assert direct == via, "routing a facade Model through lower_and_validate is byte-identical"


# --- 5: the handle carries the module_hash for drift detection + the lowered-module trace -------

def test_handle_carries_module_hash_and_trace():
    from pops.codegen.loader import CompiledProblem
    from pops.model.manifest import module_manifest_of
    m = _facade_model("gas")
    _, source_module = lower_and_validate(m, facade=m)
    manifest = module_manifest_of(source_module)
    handle = CompiledProblem("/tmp/none.so", None, m, "SIG|c++|c++23", "c++", "c++23",
                             module_manifest=manifest, module_hash=source_module.module_hash())
    assert handle.module_hash() == source_module.module_hash(), \
        "the handle carries the compile-time Module hash (bind drift detection)"
    # The lowered-module trace is present in inspect() (the operator-first Module manifest).
    report = handle.inspect().to_dict()
    assert report["module_manifest"] is not None, "inspect() carries the lowered-module trace"
    assert report["module_manifest"]["name"] == "gas"


# --- 6: the REAL compile_problem chain threads the trace + hash onto the handle -----------------
# Only the toolchain seams are stubbed (no compiler / Kokkos in the unit lane); everything from
# lower_and_validate through manifest/hash capture to the CompiledProblem kwargs is the REAL code.
# This pins the exact chain the compiler-gated integration test (test_compile_module_trace.py)
# exercises on CI, for BOTH model shapes: the facade Model (trace present) and the native brick
# ModelSpec (trace honestly absent) -- the CI-only failure mode a stubbed handle test cannot see.

def _stub_toolchain(monkeypatch, tmp_path):
    import pops.codegen.compile_drivers as cd

    def fake_run_compile(cmd, what):
        with open(cmd[cmd.index("-o") + 1], "wb") as handle:
            handle.write(b"FAKE-SO")

    monkeypatch.setattr(cd, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: "c++23")
    monkeypatch.setattr(cd, "pops_header_signature", lambda inc: "TESTSIG")
    monkeypatch.setattr(cd, "_run_compile", fake_run_compile)
    monkeypatch.setenv("POPS_CACHE_DIR", str(tmp_path))
    return cd


def test_compile_problem_chain_threads_trace_for_facade_model(monkeypatch, tmp_path):
    cd = _stub_toolchain(monkeypatch, tmp_path)
    model = _facade_model("ep")
    compiled = cd.compile_problem(time=_fe_program(model), model=model,
                                  include=str(tmp_path))
    assert compiled.module_manifest is not None, \
        "the REAL compile chain attaches the operator-first Module manifest"
    assert compiled.module_hash(), "the REAL compile chain attaches the module_hash"
    report = compiled.inspect().to_dict()
    ops = [op.get("name") for op in report["module_manifest"]["operators"]]
    assert "flux_default" in ops, "the trace lists the facade's operators: %s" % ops


def _fe_program_default(model, name="spec"):
    """An FE program on the DEFAULT source only (ctx.rhs_into: no model kernels emitted), the same
    minimal lowering the sibling integration tests compile with a native brick ModelSpec."""
    P = adctime.Program(name)
    U = typed_state(P, "ep", model=model)
    f = P.solve_fields(U)
    R = P._rhs_legacy(state=U, fields=f, flux=True, sources=["default"])
    endpoint = typed_state(P, "ep", state_name="U", model=model).next
    P.commit(endpoint, P.linear_combine("U1", U + P.dt * R, at=endpoint.point))
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
    with pytest.raises(TypeError, match="Module authority|supported model|semantic model identity"):
        cd.compile_problem(time=_fe_program_default(model), model=model,
                           include=str(tmp_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
