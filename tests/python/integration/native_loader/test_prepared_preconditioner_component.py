"""Real compile/install/run proof for an external prepared-preconditioner native component."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pops._native_selector import select_native_dimension, selected_native_dimension
from tests.python.support.requirements import (
    default_cxx,
    missing_compiler_requirement,
    repo_include,
    require_native_or_skip,
)


def _require_dim2_native_selection() -> int:
    dimension = selected_native_dimension()
    if dimension is not None:
        if dimension != 2:
            raise RuntimeError(
                "prepared preconditioner harness is Cartesian2D and cannot use selected Dim=%d"
                % dimension
            )
        return dimension
    configured_dimension = os.environ.get("POPS_NATIVE_DIM")
    if configured_dimension is None:
        raise RuntimeError(
            "prepared preconditioner harness requires selected Dim=2 or explicit POPS_NATIVE_DIM=2"
        )
    if configured_dimension != "2":
        raise RuntimeError(
            "prepared preconditioner harness is Cartesian2D and requires canonical POPS_NATIVE_DIM=2"
        )
    dimension = 2
    select_native_dimension(dimension)
    return dimension


def _require_native() -> str:
    missing = missing_compiler_requirement(repo_include())
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    cxx = default_cxx()
    if cxx is None:
        require_native_or_skip("compilateur C++ absent", optional_skip=pytest.skip)
        raise AssertionError("require_native_or_skip must not return")
    from pops.codegen.toolchain import pops_loader_build_flags

    try:
        _require_dim2_native_selection()
        compiler, _compile_flags, _link_flags = pops_loader_build_flags(cxx)
    except (RuntimeError, ValueError) as exc:
        require_native_or_skip(
            str(exc),
            optional_skip=pytest.skip,
        )
        raise AssertionError("require_native_or_skip must not return") from exc
    try:
        import pops.runtime._engine_descriptors  # noqa: F401
        import pops.runtime._system  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "prepared component runtime bindings unavailable: %s" % exc,
            optional_skip=pytest.skip,
        )
        raise AssertionError("require_native_or_skip must not return") from exc
    return compiler


def test_preconditioner_dim2_selection_refuses_missing(monkeypatch):
    monkeypatch.setattr(
        sys.modules[__name__],
        "selected_native_dimension",
        lambda: None,
    )
    monkeypatch.delenv("POPS_NATIVE_DIM", raising=False)
    with pytest.raises(RuntimeError, match="requires selected Dim=2"):
        _require_dim2_native_selection()


@pytest.mark.parametrize("raw_dimension", ("invalid", "02", "+2", " 2"))
def test_preconditioner_dim2_selection_refuses_noncanonical_text(monkeypatch, raw_dimension):
    monkeypatch.setattr(
        sys.modules[__name__],
        "selected_native_dimension",
        lambda: None,
    )
    monkeypatch.setenv("POPS_NATIVE_DIM", raw_dimension)
    with pytest.raises(RuntimeError, match="canonical POPS_NATIVE_DIM=2"):
        _require_dim2_native_selection()


def test_preconditioner_dim2_selection_refuses_conflicting_selected_dimension(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "selected_native_dimension",
        lambda: 1,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "select_native_dimension",
        lambda dimension: calls.append(dimension),
    )
    monkeypatch.setenv("POPS_NATIVE_DIM", "2")
    with pytest.raises(RuntimeError, match="selected Dim=1"):
        _require_dim2_native_selection()
    assert calls == []


def test_preconditioner_dim2_selection_reuses_selected_dimension(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sys.modules[__name__],
        "selected_native_dimension",
        lambda: 2,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "select_native_dimension",
        lambda dimension: calls.append(dimension),
    )
    assert _require_dim2_native_selection() == 2
    assert calls == []


def _passive_model(name: str):
    from pops.physics._facade import Model

    model = Model(name)
    (rho,) = model.conservative_vars("rho")
    model.primitive_vars(rho)
    model.conservative_from([rho])
    model.flux(x=[0.0 * rho], y=[0.0 * rho])
    model.eigenvalues(x=[0.0 * rho], y=[0.0 * rho])
    return model


def _program(model, descriptor):
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.solvers import GMRES
    from pops.time import FailRun, Program
    from tests.python.support.typed_program import program_states

    program = Program("external-header-preconditioner")
    _, states = program_states(program, model, ("blk",))
    temporal = states["blk"]
    operator = program.matrix_free_operator("helmholtz")

    def apply(scope, _out, value):
        laplacian = scope.scalar_field("laplacian")
        scope.laplacian(laplacian, value)
        return value - 0.1 * laplacian

    program.set_apply(operator, apply)
    rhs = program.value("rhs", temporal.n, at=temporal.next.point)
    solution = program.solve(
        LinearProblem(
            operator,
            rhs,
            at=temporal.next.point,
            properties=LinearOperatorProperties.general(),
            nullspace=None,
        ),
        solver=GMRES(
            max_iter=100,
            rel_tol=1.0e-10,
            restart=20,
            preconditioner=descriptor,
        ),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    return program


def _public_program(state, descriptor):
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.solvers import GMRES
    from pops.time import FailRun, FixedDt, Program

    program = Program("public-external-header-preconditioner")
    temporal = program.state(state)
    operator = program.matrix_free_operator("helmholtz")

    def apply(_scope, _out, value):
        return value

    program.set_apply(operator, apply)
    solution = program.solve(  # noqa: F841 - consumes the native solve before the unchanged commit
        LinearProblem(
            operator,
            temporal.n,
            at=temporal.next.point,
            properties=LinearOperatorProperties.general(),
            nullspace=None,
        ),
        solver=GMRES(
            max_iter=100,
            rel_tol=1.0e-10,
            restart=20,
            preconditioner=descriptor,
        ),
    ).consume(action=FailRun())
    program.commit(
        temporal.next,
        program.value("unchanged", temporal.n, at=temporal.next.point),
    )
    program.step_strategy(FixedDt(0.01))
    return program


def _write_component(root: Path) -> None:
    header = root / "vendor" / "prepared_identity.hpp"
    header.parent.mkdir(parents=True)
    header.write_text(
        """#pragma once
#include <pops/numerics/elliptic/linear/prepared_affine_problem.hpp>
#include <pops/runtime/export.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace vendor {
namespace {
// Keep the complete instrumented provider local to each generated Program DSO. ELF may coalesce
// externally linked inline variables, functions and class members across the low-level and public
// artifacts, turning two independent lifecycle proofs into one misleading cumulative counter set.
std::atomic<std::uint64_t> factory_calls{0};
std::atomic<std::uint64_t> session_state_constructions{0};
std::atomic<std::uint64_t> prepare_calls{0};
std::atomic<std::uint64_t> apply_calls{0};
std::atomic<std::uint64_t> apply_before_prepare_calls{0};

struct PreparedIdentitySessionState {
  PreparedIdentitySessionState() {
    session_state_constructions.fetch_add(1, std::memory_order_relaxed);
  }

  std::atomic<bool> prepared{false};
};

pops::PreparedLinearPreconditionerSessionFactory<POPS_NATIVE_DIM> prepared_identity_session_factory() {
  return [](const pops::ExecutionLane&) {
    factory_calls.fetch_add(1, std::memory_order_relaxed);
    auto state = std::make_shared<PreparedIdentitySessionState>();
    return pops::PreparedLinearPreconditionerSessionCallbacks<POPS_NATIVE_DIM>{
        [state] {
          prepare_calls.fetch_add(1, std::memory_order_relaxed);
          state->prepared.store(true, std::memory_order_release);
        },
        [state](pops::MultiFab<POPS_NATIVE_DIM>& out, const pops::MultiFab<POPS_NATIVE_DIM>& in) {
          if (!state->prepared.load(std::memory_order_acquire))
            apply_before_prepare_calls.fetch_add(1, std::memory_order_relaxed);
          apply_calls.fetch_add(1, std::memory_order_relaxed);
          pops::PureFieldAlgebra::copy(out, in);
        },
        [] { return std::size_t{0}; }};
  };
}
}  // namespace
}  // namespace vendor

extern "C" POPS_EXPORT std::uint64_t pops_test_preconditioner_factory_calls() noexcept {
  return vendor::factory_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_preconditioner_session_state_constructions() noexcept {
  return vendor::session_state_constructions.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_preconditioner_prepare_calls() noexcept {
  return vendor::prepare_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_preconditioner_apply_calls() noexcept {
  return vendor::apply_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_preconditioner_apply_before_prepare_calls() noexcept {
  return vendor::apply_before_prepare_calls.load(std::memory_order_relaxed);
}
""",
        encoding="utf-8",
    )


def _native_counters(so_path: str) -> tuple[int, int, int, int, int]:
    library = ctypes.CDLL(so_path)
    library.pops_test_preconditioner_factory_calls.restype = ctypes.c_uint64
    library.pops_test_preconditioner_session_state_constructions.restype = ctypes.c_uint64
    library.pops_test_preconditioner_prepare_calls.restype = ctypes.c_uint64
    library.pops_test_preconditioner_apply_calls.restype = ctypes.c_uint64
    library.pops_test_preconditioner_apply_before_prepare_calls.restype = ctypes.c_uint64
    return (
        int(library.pops_test_preconditioner_factory_calls()),
        int(library.pops_test_preconditioner_session_state_constructions()),
        int(library.pops_test_preconditioner_prepare_calls()),
        int(library.pops_test_preconditioner_apply_calls()),
        int(library.pops_test_preconditioner_apply_before_prepare_calls()),
    )


def _assert_native_session_lifecycle(counters: tuple[int, int, int, int, int]) -> None:
    factory_calls, state_constructions, prepare_calls, apply_calls, unprepared_applies = counters
    assert factory_calls == 2, "the prepared problem and bound workspace own exactly two sessions"
    assert state_constructions == 2, "each lifecycle session must own one fresh state"
    assert prepare_calls == 2, "each lifecycle session is prepared exactly once"
    assert apply_calls >= 2, "both prepared sessions must execute their native callback"
    assert unprepared_applies == 0, "the runtime must never apply an unprepared session"


def test_external_header_only_provider_compiles_links_installs_and_runs(
    tmp_path, isolated_native_cache,
):
    cxx = _require_native()
    from pops.codegen._compile_drivers import compile_problem
    from pops.solvers import preconditioners

    include_root = tmp_path / "component-include"
    _write_component(include_root)

    def emit(_node, _prelude, _prototype, _vector_distribution, _provider):
        return preconditioners.NativeEmission("vendor::prepared_identity_session_factory()")

    provider = preconditioners.register(preconditioners.Provider(
        provider_id="pops.test.prepared-header-identity",
        interface_version=1,
        options_schema="pops.test.prepared-header-identity.options@1",
        scheme="e2e_external_header_identity",
        descriptor_name="e2e_external_header_identity",
        display_name="ExternalHeaderIdentity()",
        native_id="vendor::prepared_identity",
        validator_id="pops.test.prepared-header-identity.validate@1",
        planner_id="pops.test.prepared-header-identity.plan@1",
        emitter_id="pops.test.prepared-header-identity@1",
        preconditioned=True,
        prepared_buffers=2,
        use_policy=preconditioners.UsePolicy(
            "pops.test.prepared-header-identity.use", 1,
            {"methods": ("gmres",), "components": "any", "nullspaces": "any"},
            lambda _use, _where: None,
        ),
        options=(),
        emitter=emit,
        native_component=preconditioners.HeaderOnlyComponent(
            "pops.test.prepared-header-identity",
            include_root=include_root,
            entry_headers=("vendor/prepared_identity.hpp",),
        ),
    ))
    descriptor = preconditioners.Prepared(provider)
    model = _passive_model("external_preconditioner_program_model")
    program = _program(model, descriptor)

    compiled = compile_problem(
        so_path=str(tmp_path / "program.so"),
        model=model,
        time=program,
        include=repo_include(),
        cxx=cxx,
    )
    assert Path(compiled.so_path).is_file()
    generated_path = compiled.dump_cpp(tmp_path / "generated.cpp")
    generated = Path(generated_path).read_text(encoding="utf-8")
    assert "#include <vendor/prepared_identity.hpp>" in generated
    assert "vendor::prepared_identity_session_factory()" in generated
    compiled_solve = next(
        value for value in compiled.program._values if value.op == "solve_linear"
    )
    assert (
        compiled_solve.attrs["preconditioner_provider"]["native_component"]
        ["manifest_sha256"]
        == provider.native_component.manifest_sha256
    )

    axis = (np.arange(8) + 0.5) / 8
    x, y = np.meshgrid(axis, axis, indexing="ij")
    initial = 1.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)

    import pops
    from tests.python.integration._final_field_program import (
        resolve_periodic_field_program,
        scalar_advection_model,
    )

    public_model = scalar_advection_model("public_external_preconditioner_model")
    resolved = resolve_periodic_field_program(
        public_model,
        lambda state, _rate, _field: _public_program(state, descriptor),
        name="public-external-preconditioner",
        block_name="blk",
        target="system",
        n=8,
        cxx=cxx,
        include=repo_include(),
    )
    public_compiled = pops.compile(resolved)
    public_runtime = pops.bind(
        public_compiled,
        initial_state={"blk": np.stack([initial])},
    )
    public_report = pops.run(public_runtime, t_end=0.01, max_steps=1)
    public_result = np.asarray(public_runtime.state_global("blk"), dtype=np.float64)[0]
    assert public_report.accepted_steps == 1
    assert np.isfinite(public_result).all()
    np.testing.assert_allclose(public_result, initial)
    public_counters = _native_counters(public_compiled.so_path)
    _assert_native_session_lifecycle(public_counters)
