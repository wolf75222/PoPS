#!/usr/bin/env python3
"""Spec 5 sec.12.4: codegen ``POPS_*`` environment completeness and inspectability.

The contract is additive and honest:

  * each implemented variable supplies a DEFAULT -- an explicit Python argument to
    ``compile_problem`` always wins (asserted below against a conflicting env);
  * coercion of implemented controls is lenient (an unrecognised value falls back to the safe
    default, never raises);
  * ``POPS_AUTOTUNE`` is rejected because no autotuning engine exists; no inert control is accepted.

These checks are PURE: they exercise the resolver, the recording on the handle and the inspect
surface WITHOUT a real Kokkos compile. The one end-to-end ``compile_problem`` check MOCKS the
compiler invocation (``_run_compile`` / the Kokkos build flags) so the wiring -- not the heavy
Kokkos-gated build -- is tested; it never fakes the engine's numerics.

Pytest + __main__ guard (CI runs ``python3 <file>``).
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

from pops.codegen.env import CodegenEnv, resolve_log_level
from pops.codegen.loader import CompiledModel, CompiledProblem
from pops.numerics.terms import DefaultSource, Flux

from tests.python.unit.runtime._typed_program import (
    typed_compiled_artifact,
    typed_program_state,
)


INCLUDE = str(Path(__file__).resolve().parents[4] / "include")


def _program_fixture(name="env_demo"):
    """A real in-memory Program (a state, a Forward-Euler commit) -- no compile."""
    P, module, _, _, _, temporal = typed_program_state(name, block_name="plasma")
    dt = P.dt
    U = temporal.n
    R = P.rhs(state=U, terms=[Flux(), DefaultSource()])
    P.commit(temporal.next, P.value("U1", U + dt * R, at=temporal.next.point))
    return P, module


def _program(name="env_demo"):
    return _program_fixture(name)[0]


def _compiled_model(*, abi_key="SIG|c++|c++23", cxx="c++", std="c++23"):
    """Exact metadata carrier for synthetic handles; no compiler is involved."""
    return CompiledModel(
        so_path="<synthetic>", backend="production", cons_names=["u"],
        cons_roles=["Scalar"], prim_names=["u"], n_vars=1, gamma=None, n_aux=0,
        params={}, caps={"cpu": True}, abi_key=abi_key, model_hash="env-model",
        cxx=cxx, std=std,
    )


def _handle(env, program=None):
    """A final artifact carrying a synthetic executable and resolved CodegenEnv."""
    P = program if program is not None else _program()
    model = _compiled_model()
    component = CompiledProblem(
        "/tmp/pops-cache/problem.so",
        P,
        model,
        "SIG|c++|c++23",
        "c++",
        "c++23",
        codegen_env=env,
    )
    return typed_compiled_artifact(component, model)


def _write_fake_compile_outputs(cmd, payload=b"FAKE-SO"):
    """Publish the two outputs guaranteed by a successful compiler invocation."""
    output = Path(cmd[cmd.index("-o") + 1])
    dependency_file = Path(cmd[cmd.index("-MF") + 1])
    generated = Path(next(item for item in cmd if item.endswith("problem.cpp")))

    def dep_escape(path):
        return str(path).replace("\\", "\\\\").replace(" ", "\\ ").replace("$", "$$")

    output.write_bytes(payload)
    dependency_file.write_text(
        "%s: %s\n" % (dep_escape(output), dep_escape(generated)),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Each wired var reads from the env (lenient coercion).
# ---------------------------------------------------------------------------

def test_log_level_codegen_specific_wins_and_is_lenient():
    # POPS_CODEGEN_LOG (specific) wins over POPS_LOG (broad); aliases map; a bad value -> quiet.
    assert resolve_log_level({}) == "quiet"
    assert resolve_log_level({"POPS_LOG": "info"}) == "info"
    assert resolve_log_level({"POPS_LOG": "verbose"}) == "debug"
    assert resolve_log_level({"POPS_LOG": "debug", "POPS_CODEGEN_LOG": "info"}) == "info"
    assert resolve_log_level({"POPS_CODEGEN_LOG": "garbage"}) == "quiet"  # lenient, not raised


@pytest.mark.parametrize("value", ["", "off", "basic", "aggressive", "nonsense"])
def test_autotune_is_rejected_until_an_engine_exists(value):
    with pytest.raises(NotImplementedError, match="no autotuning engine"):
        CodegenEnv.from_env(env={"POPS_AUTOTUNE": value})

    clean = CodegenEnv.from_env(env={})
    assert "autotune" not in clean.to_dict()
    assert not hasattr(clean, "autotune")


def test_codegen_dir_keep_dump_read_from_env():
    e = CodegenEnv.from_env(env={"POPS_CODEGEN_DIR": "/cg", "POPS_KEEP_GENERATED": "1",
                                 "POPS_DUMP_IR": "yes", "POPS_DUMP_CPP": "true",
                                 "POPS_CACHE_DIR": "/cache", "POPS_PROFILE": "advanced"})
    assert e.codegen_dir == "/cg"
    assert e.keep_generated is True
    assert e.dump_ir is True and e.dump_cpp is True
    assert e.cache_dir == "/cache"
    assert e.profile == "advanced"


# ---------------------------------------------------------------------------
# Explicit argument overrides the env (additive contract).
# ---------------------------------------------------------------------------

def test_explicit_codegen_dir_overrides_env():
    assert CodegenEnv.from_env(codegen_dir="/explicit",
                               env={"POPS_CODEGEN_DIR": "/env"}).codegen_dir == "/explicit"
    # No explicit -> the env supplies the default.
    assert CodegenEnv.from_env(env={"POPS_CODEGEN_DIR": "/env"}).codegen_dir == "/env"


def test_explicit_keep_generated_overrides_env():
    # debug=True forces keep regardless of the env (explicit-arg-wins).
    assert CodegenEnv.from_env(keep_generated=True, env={}).keep_generated is True
    # The env still supplies the default when the explicit flag is falsey.
    assert CodegenEnv.from_env(keep_generated=False,
                               env={"POPS_KEEP_GENERATED": "1"}).keep_generated is True


# ---------------------------------------------------------------------------
# Inspectability (criterion #47): the active env state is surfaced in inspect().
# ---------------------------------------------------------------------------

def test_env_state_surfaced_in_inspect():
    e = CodegenEnv.from_env(env={"POPS_CODEGEN_LOG": "info", "POPS_CODEGEN_DIR": "/cg"})
    rep = _handle(e).inspect()
    d = rep.to_dict()
    assert d["env"]["log_level"] == "info"
    assert d["env"]["codegen_dir"] == "/cg"
    assert "autotune" not in d["env"]
    assert "autotune" not in str(rep)


def test_inspect_without_env_is_empty_not_faked():
    # A handle built outside compile_problem carries no env -> {} (documented absence, not a default).
    model = _compiled_model()
    component = CompiledProblem(
        "/tmp/x/problem.so", _program(), model, model.abi_key, "c++", "c++23"
    )
    bare = typed_compiled_artifact(component, model)
    assert bare.codegen_env is None
    assert bare.inspect().env == {}


# ---------------------------------------------------------------------------
# End-to-end compile_problem wiring (mocked compiler -- no real Kokkos build).
# ---------------------------------------------------------------------------

def test_compile_problem_records_env_and_honors_dirs(monkeypatch):
    """compile_problem resolves + records the env, redirects to POPS_CODEGEN_DIR, keeps + dumps.

    The Kokkos-gated compiler invocation is MOCKED (we do not build a real .so): we patch the build
    flags + the compile runner so the body runs to completion and writes a placeholder .so. The
    POINT is the env wiring (record on the handle, codegen-dir redirect, keep-generated, dump-on-
    compile), not the compile itself.
    """
    from pops.codegen import _compile_drivers as cd

    def _fake_build_flags(cxx=None):
        return ("c++", [], [])

    def _fake_run_compile(cmd, where):
        del where
        # The compile command's "-o <so_path>" output is the artifact; create a placeholder so the
        # cache-hit path on a second call is exercised too. The compiler-observed dependency
        # contract is part of the same success seam, so publish its depfile as well.
        _write_fake_compile_outputs(cmd, b"// mock .so placeholder\n")

    monkeypatch.setattr(cd, "pops_loader_build_flags", _fake_build_flags)
    monkeypatch.setattr(cd, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: std or "c++23")
    monkeypatch.setattr(cd, "_run_compile", _fake_run_compile)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("POPS_CODEGEN_DIR", tmp)
        monkeypatch.setenv("POPS_KEEP_GENERATED", "1")
        monkeypatch.setenv("POPS_DUMP_IR", "1")
        monkeypatch.setenv("POPS_DUMP_CPP", "1")
        program, module = _program_fixture("wired")
        compiled = cd.compile_problem(
            model=module, time=program, force=True, include=INCLUDE)

        # The env snapshot is recorded on the handle and surfaced in inspect().
        assert compiled.codegen_env is not None
        artifact = typed_compiled_artifact(
            compiled,
            _compiled_model(abi_key=compiled.abi_key, cxx=compiled.cxx, std=compiled.std),
        )
        assert "autotune" not in artifact.inspect().env
        # The .so landed in POPS_CODEGEN_DIR.
        assert os.path.dirname(compiled.so_path) == tmp
        # POPS_KEEP_GENERATED kept the source next to the .so.
        assert compiled.generated_sources and os.path.exists(compiled.generated_sources[0])
        # POPS_DUMP_IR / POPS_DUMP_CPP wrote dumps into the codegen dir.
        produced = set(os.listdir(tmp))
        assert "wired.ir.json" in produced, produced
        assert "wired.cpp" in produced, produced

        # A second call hits the cache (the placeholder .so exists) and STILL records the env.
        again = cd.compile_problem(
            model=module, time=program, force=False, include=INCLUDE)
        assert again.codegen_env is not None
        assert again.so_path == compiled.so_path


def test_explicit_debug_keeps_generated_over_env(monkeypatch):
    """compile_problem(debug=True) forces keep-generated even with POPS_KEEP_GENERATED unset."""
    from pops.codegen import _compile_drivers as cd

    monkeypatch.setattr(cd, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(cd, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: std or "c++23")
    monkeypatch.setattr(
        cd,
        "_run_compile",
        lambda cmd, where: _write_fake_compile_outputs(cmd, b"// mock\n"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("POPS_CODEGEN_DIR", tmp)
        monkeypatch.delenv("POPS_KEEP_GENERATED", raising=False)
        program, module = _program_fixture("dbg")
        compiled = cd.compile_problem(
            model=module, time=program, force=True, debug=True, include=INCLUDE)
        assert compiled.codegen_env.keep_generated is True
        assert compiled.generated_sources and os.path.exists(compiled.generated_sources[0])


def test_compile_problem_rejects_external_optflag_inputs_before_compiler_invocation(
    monkeypatch, tmp_path,
):
    from pops.codegen import _compile_drivers as cd

    monkeypatch.setenv("POPS_CODEGEN_DIR", str(tmp_path))
    monkeypatch.setenv("POPS_DSL_OPTFLAGS", "-O3 @/tmp/untrusted-native-input.rsp")
    monkeypatch.setattr(cd, "pops_loader_build_flags", lambda cxx=None: ("c++", [], []))
    monkeypatch.setattr(cd, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: std or "c++23")
    monkeypatch.setattr(
        cd,
        "_run_compile",
        lambda *_: pytest.fail("unsafe POPS_DSL_OPTFLAGS reached the compiler"),
    )
    program, module = _program_fixture("unsafe-optflags")

    with pytest.raises(ValueError, match="closed path-free.*allowlist"):
        cd.compile_problem(
            model=module, time=program, force=True, include=INCLUDE
        )


def test_compile_problem_rejects_autotune_before_compiler_discovery(monkeypatch):
    """The public compile path fails closed before it can invoke any compiler/toolchain probe."""
    from pops.codegen import _compile_drivers as cd

    monkeypatch.setenv("POPS_AUTOTUNE", "off")
    monkeypatch.setattr(
        cd,
        "pops_loader_build_flags",
        lambda cxx=None: pytest.fail("compiler discovery must not run for POPS_AUTOTUNE"),
    )
    program, module = _program_fixture("unsupported_autotune")
    with pytest.raises(NotImplementedError, match="remove POPS_AUTOTUNE"):
        cd.compile_problem(model=module, time=program, force=True, include=INCLUDE)


# ---------------------------------------------------------------------------
# Coverage guard: every sec.12.4 POPS_* the doc lists is read by the resolver.
# ---------------------------------------------------------------------------

def test_every_documented_var_is_read():
    names = ["POPS_LOG", "POPS_CODEGEN_LOG", "POPS_CODEGEN_DIR", "POPS_KEEP_GENERATED",
             "POPS_DUMP_IR", "POPS_DUMP_CPP", "POPS_CACHE_DIR", "POPS_PROFILE"]
    # Set every var to a non-default and assert the resolved snapshot reflects each one.
    env = {"POPS_LOG": "debug", "POPS_CODEGEN_LOG": "info", "POPS_CODEGEN_DIR": "/cg",
           "POPS_KEEP_GENERATED": "1", "POPS_DUMP_IR": "1", "POPS_DUMP_CPP": "1",
           "POPS_CACHE_DIR": "/cache", "POPS_PROFILE": "advanced"}
    e = CodegenEnv.from_env(env=env)
    d = e.to_dict()
    # Each documented name has a corresponding resolved, non-default field.
    assert d["log_level"] == "info"            # codegen-specific wins over POPS_LOG=debug
    assert d["codegen_dir"] == "/cg"
    assert d["keep_generated"] is True
    assert d["dump_ir"] is True and d["dump_cpp"] is True
    assert d["cache_dir"] == "/cache"
    assert d["profile"] == "advanced"
    # POPS_LOG is read (it is the fallback when POPS_CODEGEN_LOG is absent).
    assert resolve_log_level({"POPS_LOG": "debug"}) == "debug"
    assert set(names)  # the list above is the sec.12.4 surface this test pins


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
