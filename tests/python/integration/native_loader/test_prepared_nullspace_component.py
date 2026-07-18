"""Real compile/install/run proof for an external prepared-nullspace native component."""
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
            "prepared nullspace runtime bindings unavailable: %s" % exc,
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
    header = root / "vendor" / "prepared_nullspace.hpp"
    header.parent.mkdir(parents=True)
    header.write_text(
        """#pragma once
#include <pops/numerics/elliptic/interface/field_nullspace.hpp>
#include <pops/runtime/export.hpp>

#include <atomic>
#include <cstdint>

namespace vendor {
inline std::atomic<std::uint64_t> plan_calls{0};

inline pops::FieldNullspacePlan periodic_constant_plan() {
  plan_calls.fetch_add(1, std::memory_order_relaxed);
  return pops::constant_mean_zero_nullspace(
      "vendor-periodic-constant", "external prepared nullspace provider");
}
}  // namespace vendor

extern "C" POPS_EXPORT std::uint64_t pops_test_nullspace_plan_calls() noexcept {
  return vendor::plan_calls.load(std::memory_order_relaxed);
}
""",
        encoding="utf-8",
    )


def _provider(include_root: Path):
    from pops.identity.scalar import scalar_cpp, scalar_literal
    from pops.fields import MeanValueGauge, nullspace

    def author(options, gauge, _properties, where):
        if options != {"recipe_version": 1}:
            raise ValueError("%s requires recipe_version=1" % where)
        if type(gauge) is not MeanValueGauge:
            raise TypeError("%s requires MeanValueGauge" % where)
        return nullspace.Contracts(
            {"basis_recipe": "periodic-constant", "recipe_version": 1},
            {"constraint": "mean-value", "value": scalar_literal(gauge.value)},
        )

    def validate(use, where):
        if use.components is not None and use.components != 1:
            raise ValueError("%s is scalar-only" % where)
        if not use.operator_properties["symmetric"]:
            raise ValueError("%s requires a symmetric operator certificate" % where)
        if dict(use.contracts.nullspace) != {
            "basis_recipe": "periodic-constant",
            "recipe_version": 1,
        }:
            raise ValueError("%s carries a stale basis recipe" % where)

    def emit(_node, _prelude, contracts, _identity, _provider):
        return nullspace.NativeEmission(
            "[&]() { auto plan = vendor::periodic_constant_plan(); "
            "plan.gauges.front().value = static_cast<pops::Real>(%s); "
            "return plan; }()" % scalar_cpp(contracts.gauge["value"])
        )

    return nullspace.register(nullspace.Provider(
        provider_id="pops.test.external-periodic-nullspace",
        emitter_id="pops.test.external-periodic-nullspace@1",
        singular=True,
        use_policy=nullspace.UsePolicy(
            "pops.test.external-periodic-nullspace.use",
            1,
            {"components": 1, "operator": "symmetric", "gauge": "mean-value"},
            validate,
        ),
        author=author,
        emitter=emit,
        native_component=nullspace.HeaderOnlyComponent(
            "pops.test.external-periodic-nullspace",
            include_root=include_root,
            entry_headers=("vendor/prepared_nullspace.hpp",),
        ),
    ))


def _program(model, provider):
    from pops.fields import MeanValueGauge, nullspace
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.solvers import CG
    from pops.time import FailRun, Program
    from tests.python.support.typed_program import program_states

    program = Program("external-header-nullspace")
    _, states = program_states(program, model, ("blk",))
    temporal = states["blk"]
    operator = program.matrix_free_operator("negative_laplacian")

    def apply(scope, _out, value):
        laplacian = scope.scalar_field("laplacian")
        scope.laplacian(laplacian, value)
        return -1.0 * laplacian

    program.set_apply(operator, apply)
    rhs = program.value("rhs", temporal.n, at=temporal.next.point)
    solution = program.solve(
        LinearProblem(
            operator,
            rhs,
            at=temporal.next.point,
            properties=(
                LinearOperatorProperties
                .symmetric_positive_definite_on_nullspace_complement()
            ),
            nullspace=nullspace.Prepared(provider, recipe_version=1),
            gauge=MeanValueGauge(0),
        ),
        solver=CG(max_iter=200, rel_tol=1.0e-10),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    return program


def _native_plan_calls(so_path: str) -> int:
    library = ctypes.CDLL(so_path)
    library.pops_test_nullspace_plan_calls.restype = ctypes.c_uint64
    return int(library.pops_test_nullspace_plan_calls())


def _public_program(state, provider):
    from pops.fields import MeanValueGauge, nullspace
    from pops.linalg import LinearOperatorProperties, LinearProblem
    from pops.solvers import CG
    from pops.time import FailRun, FixedDt, Program

    program = Program("public-external-header-nullspace")
    temporal = program.state(state)
    operator = program.matrix_free_operator("negative_laplacian")

    def apply(scope, _out, value):
        laplacian = scope.scalar_field("laplacian")
        scope.laplacian(laplacian, value)
        return -1.0 * laplacian

    program.set_apply(operator, apply)
    solution = program.solve(
        LinearProblem(
            operator,
            temporal.n,
            at=temporal.next.point,
            properties=(
                LinearOperatorProperties
                .symmetric_positive_definite_on_nullspace_complement()
            ),
            nullspace=nullspace.Prepared(provider, recipe_version=1),
            gauge=MeanValueGauge(0),
        ),
        solver=CG(max_iter=200, rel_tol=1.0e-10),
    ).consume(action=FailRun())
    program.commit(temporal.next, solution)
    program.step_strategy(FixedDt(0.01))
    return program


def test_external_nullspace_provider_compiles_links_installs_and_runs(
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
    model = _passive_model("external_nullspace_program_model")
    program = _program(model, provider)

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
    assert "#include <vendor/prepared_nullspace.hpp>" in generated
    assert "vendor::periodic_constant_plan()" in generated
    compiled_solve = next(
        value for value in compiled.program._values if value.op == "solve_linear"
    )
    assert (
        compiled_solve.attrs["nullspace_provider"]["native_component"]
        ["manifest_sha256"]
        == provider.native_component.manifest_sha256
    )

    compiled_model = _passive_model("external_nullspace_block_model").compile(
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
    initial = np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    simulation.set_state("blk", np.stack([initial]))
    simulation.install_program(compiled.so_path)
    simulation.step(0.01)
    result = np.asarray(simulation.get_state("blk"))[0]
    assert np.isfinite(result).all()
    assert abs(float(np.mean(result))) < 1.0e-12
    assert float(np.max(np.abs(result - initial))) > 1.0e-4
    low_plan_calls = _native_plan_calls(compiled.so_path)
    assert low_plan_calls >= 1

    # The same external header/provider must survive the final public lifecycle.  The low-level
    # proof above remains valuable for the exact generated seam; this second half prevents a private
    # compile/install API from masking a broken Case -> resolve -> compile -> bind -> run path.
    import pops
    from tests.python.integration._final_field_program import (
        resolve_periodic_field_program,
        scalar_advection_model,
    )

    public_model = scalar_advection_model("public_external_nullspace_model")
    resolved = resolve_periodic_field_program(
        public_model,
        lambda state, _rate, _field: _public_program(state, provider),
        name="public-external-nullspace",
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
    assert abs(float(np.mean(public_result))) < 1.0e-12
    assert float(np.max(np.abs(public_result - initial))) > 1.0e-4
    public_plan_calls = _native_plan_calls(public_compiled.so_path)
    assert public_plan_calls >= 1
    assert low_plan_calls + public_plan_calls >= 2
