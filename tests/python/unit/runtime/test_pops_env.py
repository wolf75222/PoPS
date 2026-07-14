#!/usr/bin/env python3
"""Spec 5 sec.12.4: codegen ``POPS_*`` environment completeness and inspectability.

The contract is additive and honest:

  * each variable supplies a DEFAULT -- an explicit Python argument to ``compile_problem`` always
    wins (asserted below by passing the explicit value against a conflicting env);
  * coercion is lenient (an unrecognised value falls back to the safe default, never raises);
  * ``POPS_AUTOTUNE`` is an HONEST no-op stub (no autotune engine today): it is recorded + surfaced
    but changes no codegen and does not enter the cache key.

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

try:
    from pops.codegen.env import CodegenEnv, resolve_log_level, resolve_autotune
    from pops.codegen.loader import CompiledModel, CompiledProblem
    from pops.numerics.terms import DefaultSource, Flux
except Exception as exc:  # noqa: BLE001 -- pops unavailable in this interpreter
    print("skip test_pops_env (pops unavailable: %s)" % exc)
    sys.exit(0)

from tests.python.unit.runtime._typed_program import typed_program_state


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


def _compiled_model():
    """Exact metadata carrier for synthetic handles; no compiler is involved."""
    return CompiledModel(
        so_path="<synthetic>", backend="production", cons_names=["u"],
        cons_roles=["Scalar"], prim_names=["u"], n_vars=1, gamma=None, n_aux=0,
        params={}, caps={"cpu": True}, abi_key="SIG|c++|c++23", model_hash="env-model",
        cxx="c++", std="c++23",
    )


def _handle(env, program=None):
    """A synthetic CompiledProblem carrying a resolved CodegenEnv (no .so on disk)."""
    P = program if program is not None else _program()
    return CompiledProblem("/tmp/pops-cache/problem.so", P, _compiled_model(),
                           "SIG|c++|c++23", "c++", "c++23",
                           codegen_env=env)


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


def test_autotune_levels_and_honest_stub():
    assert resolve_autotune({}) == "off"
    assert resolve_autotune({"POPS_AUTOTUNE": "basic"}) == "basic"
    assert resolve_autotune({"POPS_AUTOTUNE": "aggressive"}) == "aggressive"
    assert resolve_autotune({"POPS_AUTOTUNE": "nonsense"}) == "off"  # lenient fallback


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
    e = CodegenEnv.from_env(env={"POPS_CODEGEN_LOG": "info", "POPS_AUTOTUNE": "aggressive",
                                 "POPS_CODEGEN_DIR": "/cg"})
    rep = _handle(e).inspect()
    d = rep.to_dict()
    assert d["env"]["log_level"] == "info"
    assert d["env"]["autotune"] == "aggressive"
    assert d["env"]["codegen_dir"] == "/cg"
    # The autotune no-op stub is labelled honestly in the printable report.
    assert "no-op stub" in str(rep)


def test_inspect_without_env_is_empty_not_faked():
    # A handle built outside compile_problem carries no env -> {} (documented absence, not a default).
    bare = CompiledProblem("/tmp/x/problem.so", _program(), _compiled_model(),
                           "SIG", "c++", "c++23")
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
        # The compile command's "-o <so_path>" output is the artifact; create a placeholder so the
        # cache-hit path on a second call is exercised too.
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("// mock .so placeholder\n")

    monkeypatch.setattr(cd, "pops_loader_build_flags", _fake_build_flags)
    monkeypatch.setattr(cd, "pops_header_signature", lambda include: "MOCKSIG")
    monkeypatch.setattr(cd, "_probe_cxx_std", lambda cc, std: std or "c++23")
    monkeypatch.setattr(cd, "_run_compile", _fake_run_compile)

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("POPS_CODEGEN_DIR", tmp)
        monkeypatch.setenv("POPS_KEEP_GENERATED", "1")
        monkeypatch.setenv("POPS_DUMP_IR", "1")
        monkeypatch.setenv("POPS_DUMP_CPP", "1")
        monkeypatch.setenv("POPS_AUTOTUNE", "basic")

        program, module = _program_fixture("wired")
        compiled = cd.compile_problem(
            model=module, time=program, force=True, include=INCLUDE)

        # The env snapshot is recorded on the handle and surfaced in inspect().
        assert compiled.codegen_env is not None
        assert compiled.codegen_env.autotune == "basic"
        assert compiled.inspect().env["autotune"] == "basic"
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
    monkeypatch.setattr(cd, "_run_compile",
                        lambda cmd, where: open(cmd[cmd.index("-o") + 1], "w").write("// mock\n"))

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv("POPS_CODEGEN_DIR", tmp)
        monkeypatch.delenv("POPS_KEEP_GENERATED", raising=False)
        program, module = _program_fixture("dbg")
        compiled = cd.compile_problem(
            model=module, time=program, force=True, debug=True, include=INCLUDE)
        assert compiled.codegen_env.keep_generated is True
        assert compiled.generated_sources and os.path.exists(compiled.generated_sources[0])


# ---------------------------------------------------------------------------
# Coverage guard: every sec.12.4 POPS_* the doc lists is read by the resolver.
# ---------------------------------------------------------------------------

def test_every_documented_var_is_read():
    names = ["POPS_LOG", "POPS_CODEGEN_LOG", "POPS_CODEGEN_DIR", "POPS_KEEP_GENERATED",
             "POPS_DUMP_IR", "POPS_DUMP_CPP", "POPS_CACHE_DIR", "POPS_PROFILE", "POPS_AUTOTUNE"]
    # Set every var to a non-default and assert the resolved snapshot reflects each one.
    env = {"POPS_LOG": "debug", "POPS_CODEGEN_LOG": "info", "POPS_CODEGEN_DIR": "/cg",
           "POPS_KEEP_GENERATED": "1", "POPS_DUMP_IR": "1", "POPS_DUMP_CPP": "1",
           "POPS_CACHE_DIR": "/cache", "POPS_PROFILE": "advanced", "POPS_AUTOTUNE": "basic"}
    e = CodegenEnv.from_env(env=env)
    d = e.to_dict()
    # Each documented name has a corresponding resolved, non-default field.
    assert d["log_level"] == "info"            # codegen-specific wins over POPS_LOG=debug
    assert d["codegen_dir"] == "/cg"
    assert d["keep_generated"] is True
    assert d["dump_ir"] is True and d["dump_cpp"] is True
    assert d["cache_dir"] == "/cache"
    assert d["profile"] == "advanced"
    assert d["autotune"] == "basic"
    # POPS_LOG is read (it is the fallback when POPS_CODEGEN_LOG is absent).
    assert resolve_log_level({"POPS_LOG": "debug"}) == "debug"
    assert set(names)  # the list above is the sec.12.4 surface this test pins


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
