"""Real compile/install/run proof for an external prepared-preconditioner native component."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
    require_native_or_skip,
)


def _require_native() -> None:
    missing = missing_native_compile_requirement(repo_include(), default_cxx())
    if missing:
        require_native_or_skip(missing, optional_skip=pytest.skip)
    try:
        import pops.runtime._engine_descriptors  # noqa: F401
        import pops.runtime._system  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        require_native_or_skip(
            "prepared component runtime bindings unavailable: %s" % exc,
            optional_skip=pytest.skip,
        )


def _passive_model(name: str):
    from pops.physics._facade import Model

    model = Model(name)
    (rho,) = model.conservative_vars("rho")
    velocity = model.primitive("u", 0.0 * rho)
    model.primitive_vars(rho=rho, u=velocity)
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

    def apply(scope, _out, value):
        laplacian = scope.scalar_field("laplacian")
        scope.laplacian(laplacian, value)
        return value - 0.1 * laplacian

    program.set_apply(operator, apply)
    solution = program.solve(
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
    program.commit(temporal.next, solution)
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
inline std::atomic<std::uint64_t> factory_calls{0};
inline std::atomic<std::uint64_t> session_state_constructions{0};
inline std::atomic<std::uint64_t> prepare_calls{0};
inline std::atomic<std::uint64_t> apply_calls{0};
inline std::atomic<std::uint64_t> apply_before_prepare_calls{0};

struct PreparedIdentitySessionState {
  PreparedIdentitySessionState() {
    session_state_constructions.fetch_add(1, std::memory_order_relaxed);
  }

  std::atomic<bool> prepared{false};
};

inline pops::PreparedLinearPreconditionerSessionFactory prepared_identity_session_factory() {
  return [](const pops::ExecutionLane&) {
    factory_calls.fetch_add(1, std::memory_order_relaxed);
    auto state = std::make_shared<PreparedIdentitySessionState>();
    return pops::PreparedLinearPreconditionerSessionCallbacks{
        [state] {
          prepare_calls.fetch_add(1, std::memory_order_relaxed);
          state->prepared.store(true, std::memory_order_release);
        },
        [state](pops::MultiFab& out, const pops::MultiFab& in) {
          if (!state->prepared.load(std::memory_order_acquire))
            apply_before_prepare_calls.fetch_add(1, std::memory_order_relaxed);
          apply_calls.fetch_add(1, std::memory_order_relaxed);
          pops::PureFieldAlgebra::copy(out, in);
        },
        [] { return std::size_t{0}; }};
  };
}
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
    _require_native()
    from pops.codegen._compile_drivers import compile_problem
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._engine_descriptors import Explicit, Spatial
    from pops.runtime._system import System
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
        cxx=default_cxx(),
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

    compiled_model = _passive_model("external_preconditioner_block_model").compile(
        backend="production", include=repo_include(), cxx=default_cxx()
    )
    simulation = System(n=8, L=1.0, periodic=True)
    simulation.add_equation(
        "blk",
        compiled_model,
        spatial=Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=Explicit(method="euler"),
    )
    axis = (np.arange(8) + 0.5) / 8
    x, y = np.meshgrid(axis, axis, indexing="ij")
    initial = 1.0 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    simulation.set_state("blk", np.stack([initial]))
    simulation.install_program(compiled.so_path)
    simulation.step(0.01)
    result = np.asarray(simulation.get_state("blk"))[0]
    assert np.isfinite(result).all()
    assert float(np.max(np.abs(result - initial))) > 1.0e-8
    low_counters = _native_counters(compiled.so_path)
    _assert_native_session_lifecycle(low_counters)

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
        cxx=default_cxx(),
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
    assert float(np.max(np.abs(public_result - initial))) > 1.0e-8
    public_counters = _native_counters(public_compiled.so_path)
    _assert_native_session_lifecycle(public_counters)
