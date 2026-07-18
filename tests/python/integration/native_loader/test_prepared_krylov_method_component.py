"""Real Python -> codegen -> C++ -> solve proof for an external Krylov method."""
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
            "prepared Krylov runtime bindings unavailable: %s" % exc,
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


def _write_component(root: Path) -> None:
    header = root / "vendor" / "prepared_one_step_krylov.hpp"
    header.parent.mkdir(parents=True)
    header.write_text(
        r"""#pragma once
#include <pops/numerics/elliptic/linear/generic_krylov.hpp>
#include <pops/runtime/export.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace vendor::one_step_krylov {
namespace {
// Keep the complete instrumented provider local to each generated Program DSO. ELF may coalesce
// externally linked inline variables, functions and class members across the low-level and public
// artifacts, turning two independent lifecycle proofs into one misleading cumulative counter set.
std::atomic<std::uint64_t> workspace_calls{0};
std::atomic<std::uint64_t> solve_calls{0};

inline const double* physical_step(const pops::PreparedProviderOptions& options) noexcept {
  if (options.schema_identity != "vendor.one-step-krylov.options@1" ||
      options.values.size() != 1)
    return nullptr;
  const auto& item = *options.values.begin();
  return item.first == "physical_step" ? std::get_if<double>(&item.second) : nullptr;
}

class Provider final : public pops::PreparedKrylovMethodProvider {
 public:
  std::string_view identity() const noexcept override {
    return "vendor.one-step-krylov";
  }
  std::uint64_t interface_version() const noexcept override { return 1; }
  std::string_view collective_contract() const noexcept override {
    return "vendor.one-step-krylov@1";
  }

  pops::KrylovMethodValidation validate_controls(
      const pops::KrylovMethodControls& controls,
      const pops::PreparedProviderOptions& options) const noexcept override {
    if (controls.max_iterations < 1 ||
        !std::isfinite(static_cast<double>(controls.rel_tol)) ||
        !std::isfinite(static_cast<double>(controls.abs_tol)) ||
        controls.rel_tol < pops::Real(0) || controls.rel_tol >= pops::Real(1) ||
        controls.abs_tol < pops::Real(0) ||
        (controls.rel_tol == pops::Real(0) && controls.abs_tol == pops::Real(0)))
      return pops::KrylovMethodValidation::reject(1, "invalid universal controls");
    const double* step = physical_step(options);
    if (step == nullptr || !std::isfinite(*step) || *step <= 0.0)
      return pops::KrylovMethodValidation::reject(2, "invalid physical_step");
    return pops::KrylovMethodValidation::accept();
  }

  pops::KrylovMethodValidation validate_problem(
      const pops::KrylovMethodProblemFacts& facts,
      const pops::PreparedProviderOptions&) const noexcept override {
    if (!facts.properties.valid() || facts.footprint.components < 1 ||
        facts.footprint.input_ghosts < 0 || facts.robust_payload_width == 0)
      return pops::KrylovMethodValidation::reject(3, "invalid vector-space facts");
    if (facts.has_preconditioner || facts.footprint.preconditioned)
      return pops::KrylovMethodValidation::reject(4, "one-step method is unpreconditioned");
    return pops::KrylovMethodValidation::accept();
  }

  pops::KrylovWorkspaceRequirements workspace_requirements(
      const pops::KrylovWorkspaceRequest& request,
      const pops::PreparedProviderOptions&) const override {
    if (request.footprint.preconditioned)
      throw std::invalid_argument("one-step method is unpreconditioned");
    workspace_calls.fetch_add(1, std::memory_order_relaxed);
    return {.field_count = 2,
            .real_count = 1,
            .state_word_count = 1,
            .initial_residual_field = 1};
  }

  pops::SolveReport solve(
      pops::PreparedKrylovSolveContext& context,
      const pops::PreparedProviderOptions& options) const override {
    const auto call = solve_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    context.state_word(0) = call;
    context.real_value(0) = static_cast<pops::Real>(*physical_step(options));
    context.add_physical_direction(
        context.iterate(), context.real_value(0), context.initial_residual());
    const pops::Real residual = context.true_residual_norm(context.initial_residual());
    return context.report(
        residual, 1,
        residual <= context.physical_threshold() ? pops::SolveStatus::kSolved
                                                 : pops::SolveStatus::kIterationLimit);
  }
};

inline pops::PreparedKrylovMethod method(double step) {
  static const auto provider = std::make_shared<const Provider>();
  return pops::PreparedKrylovMethod(
      provider,
      pops::PreparedProviderOptions{
          "vendor.one-step-krylov.options@1", {{"physical_step", step}}});
}

}  // namespace
}  // namespace vendor::one_step_krylov

extern "C" POPS_EXPORT std::uint64_t pops_test_krylov_workspace_calls() noexcept {
  return vendor::one_step_krylov::workspace_calls.load(std::memory_order_relaxed);
}

extern "C" POPS_EXPORT std::uint64_t pops_test_krylov_solve_calls() noexcept {
  return vendor::one_step_krylov::solve_calls.load(std::memory_order_relaxed);
}
""",
        encoding="utf-8",
    )


def _provider(include_root: Path):
    from pops.native_components import PreparedNativeComponent
    from pops.solvers import krylov

    def prepare_options(options):
        if set(options) != {"physical_step"}:
            raise ValueError("one-step method requires exactly physical_step")
        raw_step = options["physical_step"]
        if isinstance(raw_step, str):
            try:
                step = float.fromhex(raw_step)
            except ValueError as exc:
                raise ValueError("physical_step is not a canonical hexadecimal scalar") from exc
            if step.hex() != raw_step:
                raise ValueError("physical_step is not a canonical hexadecimal scalar")
        else:
            step = float(raw_step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("physical_step must be finite and positive")
        return {"physical_step": step.hex()}

    def validate(use, where):
        if use.preconditioned:
            raise ValueError("%s one-step method is unpreconditioned" % where)

    return krylov.register_prepared_krylov_method_provider(
        krylov.PreparedKrylovMethodProvider(
            provider_id="vendor.one-step-krylov",
            interface_version=1,
            options_schema="vendor.one-step-krylov.options@1",
            emitter_id="vendor.one-step-krylov@1",
            capabilities={
                "contract_version": 2,
                "preconditioning_placement": "none",
            },
            native_component=PreparedNativeComponent.header_only(
                "vendor.one-step-krylov",
                include_root=include_root,
                entry_headers=("vendor/prepared_one_step_krylov.hpp",),
            ),
            option_preparer=prepare_options,
            validator=validate,
            emitter=lambda _node, options: (
                "vendor::one_step_krylov::method(static_cast<double>(%s))"
                % options["physical_step"]
            ),
        )
    )


def _program(model, provider):
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.solvers import krylov
    from pops.time import FailRun, Program
    from tests.python.support.typed_program import program_states

    program = Program("external-header-krylov-method")
    _, states = program_states(program, model, ("blk",))
    temporal = states["blk"]
    operator = program.matrix_free_operator("double-identity")
    program.set_apply(operator, lambda _scope, _out, value: 2.0 * value)
    rhs = program.value("rhs", temporal.n, at=temporal.next.point)
    solution = program.solve(
        LinearProblem(
            operator,
            rhs,
            at=temporal.next.point,
            properties=LinearOperatorProperties.general(),
            nullspace=None,
        ),
        solver=krylov.Prepared(
            provider,
            max_iter=1,
            rel_tol=1.0e-12,
            method_options={"physical_step": 0.5},
            name="ExternalOneStep",
        ),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    return program


def _public_program(state, provider):
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.solvers import krylov
    from pops.time import FailRun, FixedDt, Program

    program = Program("public-external-header-krylov-method")
    temporal = program.state(state)
    operator = program.matrix_free_operator("double-identity")
    program.set_apply(operator, lambda _scope, _out, value: 2.0 * value)
    solution = program.solve(
        LinearProblem(
            operator,
            temporal.n,
            at=temporal.next.point,
            properties=LinearOperatorProperties.general(),
            nullspace=None,
        ),
        solver=krylov.Prepared(
            provider,
            max_iter=1,
            rel_tol=1.0e-12,
            method_options={"physical_step": 0.5},
            name="PublicExternalOneStep",
        ),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    program.step_strategy(FixedDt(0.01))
    return program


def _native_counters(so_path: str) -> tuple[int, int]:
    library = ctypes.CDLL(so_path)
    library.pops_test_krylov_workspace_calls.restype = ctypes.c_uint64
    library.pops_test_krylov_solve_calls.restype = ctypes.c_uint64
    return (
        int(library.pops_test_krylov_workspace_calls()),
        int(library.pops_test_krylov_solve_calls()),
    )


def test_external_krylov_provider_compiles_and_executes_its_native_recurrence(
    tmp_path, isolated_native_cache,
):
    _require_native()
    from pops.codegen._compile_drivers import compile_problem
    from pops.numerics.reconstruction import FirstOrder
    from pops.numerics.riemann import Rusanov
    from pops.runtime._engine_descriptors import Explicit, Spatial
    from pops.runtime._system import System

    include_root = tmp_path / "component-include"
    _write_component(include_root)
    provider = _provider(include_root)
    model = _passive_model("external_krylov_program_model")
    compiled = compile_problem(
        so_path=str(tmp_path / "program.so"),
        model=model,
        time=_program(model, provider),
        include=repo_include(),
        cxx=default_cxx(),
    )
    assert Path(compiled.so_path).is_file()
    generated_path = compiled.dump_cpp(tmp_path / "generated.cpp")
    generated = Path(generated_path).read_text(encoding="utf-8")
    assert "#include <vendor/prepared_one_step_krylov.hpp>" in generated
    assert "vendor::one_step_krylov::method" in generated
    assert "KrylovWorkspaceRequirements" not in generated
    assert ".initial_residual_field" not in generated

    compiled_model = _passive_model("external_krylov_block_model").compile(
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
    np.testing.assert_allclose(result, 0.5 * initial, rtol=0.0, atol=1.0e-13)
    low_workspace_calls, low_solve_calls = _native_counters(compiled.so_path)
    assert low_workspace_calls == 1
    assert low_solve_calls == 1

    # The exact same provider must survive the final public lifecycle rather than only the private
    # compile/install seam used above.  Keep both proofs: the first exposes generated C++, while this
    # one guards Case -> resolve -> compile -> bind -> run.
    import pops
    from tests.python.integration._final_field_program import (
        resolve_periodic_field_program,
        scalar_advection_model,
    )

    public_model = scalar_advection_model("public_external_krylov_model")
    resolved = resolve_periodic_field_program(
        public_model,
        lambda state, _rate, _field: _public_program(state, provider),
        name="public-external-krylov",
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
    np.testing.assert_allclose(public_result, 0.5 * initial, rtol=0.0, atol=1.0e-13)

    public_workspace_calls, public_solve_calls = _native_counters(public_compiled.so_path)
    assert public_workspace_calls == 1
    assert public_solve_calls == 1
