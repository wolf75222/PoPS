"""Actual Host headers and public emission; no installed Native or MPI/GPU claims.

POPS_FIELD_COUNTER_HOST_PREFIX explicitly selects a readonly Kokkos SDK for the
small compile/run probe. Without it the portable Source emission tests still run.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from pops.codegen.program_codegen import emit_cpp_program
from pops.codegen.program_models import ProgramModelGraph
from tests.python.unit.fields.test_field_program_reuse import _emit_variant


def _effects(code):
    preparation = re.search(
        r"std::shared_ptr<std::array<pops::Real, 2>> field_counters_0;.*?"
        r'"Program field diagnostic preparation"\);', code, re.S).group()
    solve = re.search(r"\+\+\(\*field_counters_0\)\[0\];\s*"
                      r"ctx.record_scalar\([^\n]+\);\s*ctx.record_scalar\([^\n]+\);", code).group()
    reuse = re.search(r"\+\+\(\*field_counters_0\)\[1\];\s*"
                      r"ctx.record_scalar\([^\n]+\);\s*ctx.record_scalar\([^\n]+\);", code).group()
    keys = re.findall(r"ctx.record_scalar\((\"[^\"]+\")", solve)
    return preparation, solve, reuse, keys


def test_public_mapped_two_stage_field_effect_is_prepared_before_first_transport(tmp_path):
    from tests.python.support.m19_vlasov_poisson_case import build
    plan = build(tmp_path / "physical-providers")
    code = emit_cpp_program(plan.time, model_graph=ProgramModelGraph.from_resolved_blocks(plan.blocks))
    assert code.count("ctx.suspend_map(") == 8
    assert code.count("++(*field_counters_0)[0];") == 2
    assert code.count("std::make_shared<std::array<pops::Real, 2>>") == 1
    preparation = code.index('"Program field diagnostic preparation"')
    assert code.index("ctx.begin_step(dt)") < preparation < code.index("ctx.suspend_map(")
    assert preparation < code.index("ctx.solve_prepared_linear(")
    assert "field_solve_count_0" not in code and "field_reuse_count_0" not in code
    kernels = re.findall(r"KOKKOS_LAMBDA.*?\}\);", code, re.S)
    assert all("field_counters" not in kernel for kernel in kernels)
    prefix = os.environ.get("POPS_FIELD_COUNTER_HOST_PREFIX")
    if prefix:
        root = Path(__file__).resolve().parents[2]
        source = tmp_path / "public-mapped-field.cpp"
        source.write_text(code)
        argv = [shutil.which(os.environ.get("CXX", "clang++")), "-std=c++20",
                "-DPOPS_NATIVE_DIM=2", "-DPOPS_HAS_KOKKOS=1",
                "-DPOPS_RUNTIME_SHARED_EXCEPTION_ABI=1", "-I" + str(root / "include"),
                "-I" + str(Path(prefix) / "include"), "-fsyntax-only", str(source)]
        argv[1:1] = (["-Xpreprocessor", "-fopenmp"]
                     if os.uname().sysname == "Darwin" else ["-fopenmp"])
        compile_result = subprocess.run(argv, text=True, capture_output=True)
        (tmp_path / "syntax.stdout").write_text(compile_result.stdout)
        (tmp_path / "syntax.stderr").write_text(compile_result.stderr)
        (tmp_path / "syntax-receipt.json").write_text(json.dumps(
            {"argv": argv, "exit": compile_result.returncode,
             "installed_Native_MPI_GPU_qualification": False}, indent=2))
        assert compile_result.returncode == 0, compile_result.stderr


def test_distinct_public_coefficients_observe_emitted_effects_across_actual_temporal_continuations(tmp_path):
    prefix = os.environ.get("POPS_FIELD_COUNTER_HOST_PREFIX")
    if not prefix:
        pytest.skip("explicit readonly Host Kokkos prefix is required")
    compiler = shutil.which(os.environ.get("CXX", "clang++"))
    assert compiler is not None
    root = Path(__file__).resolve().parents[2]
    code = _emit_variant("same")  # two-state variable-coefficient field, no VP names/law
    preparation, solve, reuse, keys = _effects(code)
    template = Path(__file__).with_name("field_diagnostic_continuation_host.cpp.in").read_text()
    for token, text in {"@PREPARATION@": preparation, "@SOLVE@": solve,
                        "@REUSE@": reuse, "@SOLVE_KEY@": keys[0],
                        "@REUSE_KEY@": keys[1]}.items():
        template = template.replace(token, text)
    source = tmp_path / "actual-header-host.cpp"
    source.write_text(template)
    binary = tmp_path / "actual-header-host"
    argv = [compiler, "-std=c++20", "-O0", "-DPOPS_NATIVE_DIM=2", "-DPOPS_HAS_KOKKOS=1",
            "-I" + str(root / "include"), "-I" + str(Path(prefix) / "include"), str(source),
            "-L" + str(Path(prefix) / "lib"), "-Wl,-rpath," + str(Path(prefix) / "lib"),
            "-lkokkoscore", "-o", str(binary)]
    if os.uname().sysname == "Darwin":
        argv[1:1] = ["-Xpreprocessor", "-fopenmp"]
        argv.insert(-2, "-lomp")
    else:
        argv.insert(1, "-fopenmp")
    environment = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false")
    environment.pop("PYTHONPATH", None)
    build = subprocess.run(argv, capture_output=True, text=True, env=environment)
    (tmp_path / "compile.stdout").write_text(build.stdout)
    (tmp_path / "compile.stderr").write_text(build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(binary)], capture_output=True, text=True, env=environment)
    (tmp_path / "run.stdout").write_text(run.stdout)
    (tmp_path / "run.stderr").write_text(run.stderr)
    (tmp_path / "receipt.json").write_text(json.dumps({"compile_argv": argv,
        "compile_exit": build.returncode, "run_exit": run.returncode,
        "Host_actual_header_probe": True, "installed_Native_MPI_GPU_qualification": False}, indent=2))
    assert run.returncode == 0, run.stdout + run.stderr
