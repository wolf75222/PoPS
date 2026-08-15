"""pops.codegen.program_emit_amr -- the AMR install-entry emitter (epic ADC-511 / ADC-508, Spec 6).

Split out of :mod:`pops.codegen.program_codegen` so that module stays under the Spec-4 500-line
budget. ``_emit_amr_install`` is the only public name; ``program_codegen`` re-imports it and calls
it from ``emit_cpp_program`` when ``target='amr_system'``.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any


def _flux_fraction(value: Any) -> Fraction:
    """Canonical exact coefficient used by the native reflux ledger."""
    value = value.to_python() if hasattr(value, "to_python") else value
    try:
        return Fraction.from_float(value) if isinstance(value, float) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise TypeError(
            "AMR FluxExpression coefficient is not an exact rational literal"
        ) from error


def _flux_polynomial(value: Any) -> dict[int, Fraction]:
    polynomial = {}
    for power, coefficient in value.items():
        if isinstance(power, bool) or not isinstance(power, int) or power < 0:
            raise TypeError("AMR FluxExpression coefficient has an invalid dt power")
        ratio = _flux_fraction(coefficient)
        if ratio:
            polynomial[power] = polynomial.get(power, Fraction(0)) + ratio
    return {power: coefficient for power, coefficient in polynomial.items() if coefficient}


def _flux_polynomial_product(
    left: dict[int, Fraction], right: dict[int, Fraction]
) -> dict[int, Fraction]:
    result = {}
    for left_power, left_coefficient in left.items():
        for right_power, right_coefficient in right.items():
            power = left_power + right_power
            result[power] = (
                result.get(power, Fraction(0)) + left_coefficient * right_coefficient
            )
    return {power: coefficient for power, coefficient in result.items() if coefficient}


def _flux_expression_linear_combine(
    terms: Any,
) -> tuple[dict[Any, dict[int, Fraction]], int]:
    result: dict[Any, dict[int, Fraction]] = {}
    maximum_terms = 0
    for expression, coefficient in terms:
        for basis, polynomial in expression.items():
            scaled = _flux_polynomial_product(polynomial, coefficient)
            destination = result.setdefault(basis, {})
            for power, factor in scaled.items():
                destination[power] = destination.get(power, Fraction(0)) + factor
            result[basis] = {
                power: factor for power, factor in destination.items() if factor
            }
            if not result[basis]:
                del result[basis]
        maximum_terms = max(
            maximum_terms, *(len(polynomial) for polynomial in result.values()), 0
        )
    return result, maximum_terms


def _flux_expression_budgets(program: Any) -> tuple[tuple[int, int], ...]:
    """Derive exact finite per-block FluxExpression bounds from the frozen Program IR.

    Every flux-producing RHS evaluation creates one distinct native basis. Fixed ``range`` and
    ``subcycle`` regions are interpreted the authored number of times, and lazy branches are
    explored independently. Coefficients use the same rational polynomial multiplication,
    addition, and exact cancellation as ``AmrProgramContext``. An unbounded ``while`` may remain in
    a source/local-solve region, but is refused when it can create or carry a flux expression.
    """
    blocks = program._block_indices()
    ordered_blocks = sorted(blocks, key=blocks.get)

    def contains_flux(values: Any, block: Any) -> bool:
        for value in values:
            if value.op == "rhs" and value.block == block and value.attrs.get("flux", True):
                return True
            for key in (
                "cond_block", "body_block", "apply_block", "residual_block",
                "true_block", "false_block",
            ):
                nested = value.attrs.get(key)
                if isinstance(nested, (list, tuple)) and contains_flux(nested, block):
                    return True
        return False

    alias_input = {
        "acceptance_guard": 0,
        "fill_boundary": 0,
        "project": 0,
        "solve_fields": 0,
        "store_history": 0,
        "synchronize": 0,
    }

    def analyze(block: Any) -> tuple[int, int]:
        # (value expressions, total flux RHS evaluations, largest coefficient term count,
        #  next path-local basis identity). Expressions are immutable-by-convention, so a shallow
        # environment clone is sufficient when a branch forks.
        paths: list[tuple[dict[int, Any], int, int, int]] = [({}, 0, 0, 0)]

        def note(expression: Any, maximum: int) -> int:
            return max(maximum, *(len(polynomial) for polynomial in expression.values()), 0)

        def execute(values: Any, active: Any) -> Any:
            for value in values:
                next_paths = []
                for environment, basis_count, maximum_terms, next_basis in active:
                    if value.op == "branch":
                        for arm_key, result_key in (
                            ("true_block", "true_result"),
                            ("false_block", "false_result"),
                        ):
                            fork = execute(
                                value.attrs[arm_key],
                                [(dict(environment), basis_count, maximum_terms, next_basis)],
                            )
                            for arm_env, arm_count, arm_maximum, arm_next in fork:
                                arm_result = arm_env.get(value.attrs[result_key].id, {})
                                arm_env[value.id] = arm_result if value.block == block else {}
                                next_paths.append(
                                    (arm_env, arm_count, note(arm_result, arm_maximum), arm_next)
                                )
                        continue

                    if value.op in ("range", "subcycle"):
                        loop_paths = [(dict(environment), basis_count, maximum_terms, next_basis)]
                        loop_input = value.inputs[0]
                        for _ in range(int(value.attrs["count"])):
                            seeded = []
                            for loop_env, loop_count, loop_maximum, loop_next in loop_paths:
                                loop_env = dict(loop_env)
                                if value.block == block:
                                    loop_env[loop_input.id] = loop_env.get(loop_input.id, {})
                                seeded.append(
                                    (loop_env, loop_count, loop_maximum, loop_next)
                                )
                            loop_paths = execute(value.attrs["body_block"], seeded)
                            for loop_env, _, _, _ in loop_paths:
                                if value.block == block:
                                    loop_env[loop_input.id] = loop_env.get(
                                        value.attrs["body"].id, {}
                                    )
                        for loop_env, loop_count, loop_maximum, loop_next in loop_paths:
                            expression = (
                                loop_env.get(loop_input.id, {}) if value.block == block else {}
                            )
                            loop_env[value.id] = expression
                            next_paths.append(
                                (loop_env, loop_count, note(expression, loop_maximum), loop_next)
                            )
                        continue

                    if value.op == "while":
                        expression = environment.get(value.inputs[0].id, {})
                        if contains_flux(value.attrs["cond_block"], block) or contains_flux(
                            value.attrs["body_block"], block
                        ) or (value.block == block and expression):
                            raise ValueError(
                                "AMR FluxExpression budget is unbounded: while node %r can create "
                                "or carry a conservative flux basis; use a finite range"
                                % value.name
                            )
                        # With no new basis and no target-block loop-carried expression, every
                        # iteration sees the same captured flux expressions. Interpret each region
                        # once to retain its exact finite coefficient maximum; its local results do
                        # not escape the while region or feed the other block's loop carrier.
                        probes = execute(
                            value.attrs["cond_block"],
                            [(dict(environment), basis_count, maximum_terms, next_basis)],
                        )
                        probes = execute(value.attrs["body_block"], probes)
                        for probe_env, probe_count, probe_maximum, probe_next in probes:
                            probe_env[value.id] = {}
                            next_paths.append(
                                (probe_env, probe_count, probe_maximum, probe_next)
                            )
                        continue

                    expression = {}
                    if value.block == block:
                        if value.op == "rhs" and value.attrs.get("flux", True):
                            expression = {(value.id, next_basis): {0: Fraction(1)}}
                            next_basis += 1
                            basis_count += 1
                        elif value.op == "linear_combine":
                            expression, combine_maximum = _flux_expression_linear_combine(
                                (
                                    environment.get(source.id, {}),
                                    _flux_polynomial(coefficient),
                                )
                                for source, coefficient in zip(
                                    value.inputs, value.attrs["coeffs"], strict=True
                                )
                            )
                            maximum_terms = max(maximum_terms, combine_maximum)
                        elif value.op in alias_input and len(value.inputs) > alias_input[value.op]:
                            expression = environment.get(
                                value.inputs[alias_input[value.op]].id, {}
                            )
                    environment[value.id] = expression
                    next_paths.append(
                        (environment, basis_count, note(expression, maximum_terms), next_basis)
                    )
                active = next_paths
            return active

        paths = execute(program._values, paths)
        return (
            max((basis_count for _, basis_count, _, _ in paths), default=0),
            max((maximum_terms for _, _, maximum_terms, _ in paths), default=0),
        )

    return tuple(analyze(block) for block in ordered_blocks)


def _emit_flux_expression_budget(program: Any) -> str:
    budgets = _flux_expression_budgets(program)
    rhs_bounds = ", ".join("UINT64_C(%d)" % rhs for rhs, _ in budgets)
    coefficient_bounds = ", ".join(
        "UINT64_C(%d)" % coefficient for _, coefficient in budgets
    )
    count = len(budgets)

    def lookup(name: str, values: str) -> str:
        return (
            f'extern "C" std::uint64_t {name}(int program_block) {{\n'
            f"  static constexpr std::array<std::uint64_t, {count}> values{{{{{values}}}}};\n"
            f"  if (program_block < 0 || program_block >= {count}) return UINT64_C(0);\n"
            "  return values[static_cast<std::size_t>(program_block)];\n"
            "}\n"
        )

    has_flux = any(rhs > 0 for rhs, _ in budgets)
    return (
        "// Frozen-IR FluxExpression budgets in exact pops_program_block_name order.\n"
        f'extern "C" bool pops_program_has_flux_expression() {{ return '
        f'{str(has_flux).lower()}; }}\n'
        f'extern "C" int pops_program_flux_expression_budget_count() {{ return {count}; }}\n'
        + lookup("pops_program_flux_rhs_basis_bound", rhs_bounds)
        + lookup("pops_program_flux_coefficient_term_bound", coefficient_bounds)
    )


def _require_bounded_cell_local_program(program: Any, target: Any,
                                        hierarchy_bodies: Any) -> Any:
    """Validate the exact Program shape consumed by the first local-time provider.

    The native provider performs one transport-only forward-Euler update itself.  Accepting a
    broader IR and then skipping its generated body would be a second, divergent temporal
    authority, so every unsupported node is refused before source emission.
    """
    contract = program.cell_local_time_contract()
    if contract is None:
        return None
    if target != "amr_system":
        raise ValueError("Program.cell_local_time requires target='amr_system'")
    if not program.cadence_contract().is_default:
        raise ValueError(
            "Program.cell_local_time currently requires the default Program cadence")
    if hierarchy_bodies is not None:
        raise ValueError(
            "Program.cell_local_time does not support hierarchy-scoped field solves")
    if getattr(program, "_dt_bound", None) is not None:
        raise ValueError("Program.cell_local_time does not support a Program dt-bound body")
    if getattr(program, "_histories", None):
        raise ValueError("Program.cell_local_time does not support history operators")

    values = tuple(program._values)
    if len(values) != 3 or tuple(value.op for value in values) != (
            "state", "rhs", "linear_combine"):
        raise ValueError(
            "Program.cell_local_time currently requires exactly one transport-only "
            "ForwardEuler state/rhs/commit chain")
    state, rhs, result = values
    if tuple(rhs.inputs) != (state,) or rhs.attrs.get("flux") is not True or \
            rhs.attrs.get("fluxes") is not None or tuple(rhs.attrs.get("sources", ())) != ():
        raise ValueError(
            "Program.cell_local_time currently requires one default-flux RHS without sources "
            "or fields")
    if tuple(result.inputs) != (state, rhs):
        raise ValueError(
            "Program.cell_local_time ForwardEuler result must consume its accepted state and RHS")
    coefficients = tuple(result.attrs.get("coeffs", ()))
    if len(coefficients) != 2 or dict(coefficients[0]) != {0: 1} or \
            dict(coefficients[1]) != {1: 1}:
        raise ValueError(
            "Program.cell_local_time requires the exact update U_next = U + dt * rhs(U)")
    commits = tuple(program._commits.items())
    if len(commits) != 1 or commits[0][1] is not result or commits[0][0] != state.state_ref:
        raise ValueError(
            "Program.cell_local_time requires one exact commit to the advanced state")
    if len(program._block_indices()) != 1:
        raise ValueError("Program.cell_local_time currently requires exactly one Program block")
    return contract


def _emit_amr_install(program: Any, target: Any, prelude: Any, body: Any,
                      hierarchy_bodies: Any = None, provider_plan_install: str = "") -> str:
    """C++ source of the AMR install entry the .so exports (epic ADC-511 / ADC-508, Spec 6).

    ``target='system'`` emits NOTHING (a System-only .so carries only ``pops_install_program``).
    ``target='amr_system'`` emits ``pops_install_program_amr``, the entry ``AmrSystem::install_program``
    resolves (it dlopens the .so, validates the ABI key + section-24 requirements, binds the blocks by
    name, seeds the runtime params, then calls this). It asks the exact-ranked facade-to-provider
    factory for the ``AmrSystem<Dim>`` execution provider. It then installs the recursively subcycled per-level
    macro-step driver: the IDENTICAL lowered ``{body}`` -- the
    one ``pops_install_program`` runs on ``System`` -- wrapped in an explicit level-clock scheduler.
    Its install-time prelude is materialized once per native level, not once per hierarchy: each
    closure therefore owns fields/workspaces with the exact level layout. A topology-epoch or
    process-local materialization-generation change rematerializes the bundles before the next
    advance. The body references only the variable ``ctx``
    (never the provider type), so codegen has no concrete Uniform/AMR context dispatch.

    Shape: one macro-step recursively advances each child on its declared parent/child clock relation,
    with exact stage abscissae and mandatory temporal interpolation from parent old/new snapshots, then
    synchronizes finest-first by conservative reflux followed by average-down. Authored single-state
    field nodes use the exact point/provider-qualified solve at each active level. The context exposes
    only an explicitly level-0-only default-field route for legacy/manual drivers, so coarse auxiliary
    injection can never masquerade as a requested fine-level solve. The C/F interface is now conservative to round-off: the
    per-level effective flux is captured through the Program's own linear combination and routed
    through the native ``route_reflux`` at level sync (ADC-639), so mass/momentum/energy are conserved
    across the interface on a genuinely multilevel run; a coarse-only / flat Program stays
    bit-identical."""
    cell_local_time = _require_bounded_cell_local_program(
        program, target, hierarchy_bodies)
    if target != "amr_system":
        return ""
    flux_expression_budget = _emit_flux_expression_budget(program)
    if cell_local_time is not None:
        clock_identity = json.dumps(program.clock.qualified_id)
        return (
            flux_expression_budget
            + '\n#include <pops/runtime/program/amr_program_context.hpp>\n'
            'extern "C" void pops_install_program_amr('
            'pops::AmrSystem<pops::kNativeDimension>* sys) {\n'
            + provider_plan_install + ("\n" if provider_plan_install else "") +
            '  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);\n'
            '  auto& ctx = *ctx_owner;\n'
            f'  ctx.configure_primary_clock({clock_identity});\n'
            '  ctx.prepare_same_level_cell_temporal_execution('
            f'{clock_identity}, {cell_local_time.tick_denominator}, '
            f'{cell_local_time.rung});\n'
            '  ctx.install([ctx_owner](double dt) {\n'
            '    ctx_owner->advance_same_level_cell_temporal(dt);\n'
            '  }, ctx_owner);\n'
            '}\n'
        )

    def walk(values: Any) -> Any:
        for value in values:
            yield value
            for key in (
                "cond_block",
                "body_block",
                "apply_block",
                "residual_block",
                "true_block",
                "false_block",
            ):
                nested = value.attrs.get(key)
                if isinstance(nested, (list, tuple)):
                    yield from walk(nested)

    transform_guard = ""
    transform_refresh_guard = ""
    if any(value.op == "local_transform" for value in walk(program._values)):
        transform_guard = (
            '  auto _require_local_transform_level_contract = [ctx_owner]() {\n'
            '    auto& ctx = *ctx_owner;\n'
            '    if (ctx.program_resource_topology().levels > 1)\n'
            '      throw std::runtime_error("local_transform on multi-level AMR requires a typed '
            'post-synchronization Program phase; refusing pre-reflux execution");\n'
            "  };\n"
            "  _require_local_transform_level_contract();\n"
        )
        transform_refresh_guard = "    _require_local_transform_level_contract();\n"
    if hierarchy_bodies is None:
        phase_fields = "    std::function<void(double)> step;\n"
        phase_initializers = (
            '      [=](double dt) {\n'
            '        auto& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + body + '\n'
            '      }\n')
        installed_driver = (
            "    auto _advance_level = [&](double level_dt) {\n"
            "      _refresh_level_programs();\n"
            "      _level_programs->at(static_cast<std::size_t>(ctx.level())).step(level_dt);\n"
            "    };\n"
            "    ctx.advance_hierarchy(dt, _advance_level);\n"
        )
    else:
        gather, solve, publish = hierarchy_bodies
        phase_fields = (
            "    std::function<void(double)> step;\n"
            "    std::function<void(double)> gather;\n"
            "    std::function<void(double)> solve;\n"
            "    std::function<void(double)> publish;\n"
        )
        phase_initializers = (
            '      [=](double dt) {\n'
            '        auto& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + body + '\n'
            '      },\n'
            '      [=](double dt) {\n'
            '        auto& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + gather + '\n'
            '      },\n'
            '      [=](double dt) {\n'
            '        auto& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + solve + '\n'
            '      },\n'
            '      [=](double dt) {\n'
            '        auto& ctx = *ctx_owner;\n'
            '        (void)dt;\n' + publish + '\n'
            '      }\n')
        installed_driver = (
            '    auto _advance_hierarchy = [&](double hierarchy_dt) {\n'
            '      _refresh_level_programs();\n'
            '      const int _nlev = ctx.program_resource_topology().levels;\n'
            '      if (ctx.uses_prepared_krylov_fallback()) {\n'
            '        for (int _k = 0; _k < _nlev; ++_k) {\n'
            '          ctx.with_program_resource_level(_k, [&]() {\n'
            '            _level_programs->at(static_cast<std::size_t>(_k)).step(hierarchy_dt);\n'
            '          });\n'
            '        }\n'
            '      } else {\n'
            '        // Gather every level before the unique hierarchy-scoped solve.\n'
            '        for (int _k = 0; _k < _nlev; ++_k) {\n'
            '          ctx.with_program_resource_level(_k, [&]() {\n'
            '            _level_programs->at(static_cast<std::size_t>(_k)).gather(hierarchy_dt);\n'
            '          });\n'
            '        }\n'
            '        ctx.with_program_resource_level(0, [&]() {\n'
            '          _level_programs->front().solve(hierarchy_dt);\n'
            '        });\n'
            '        // The composite solution is complete before any level reconstructs or commits.\n'
            '        for (int _k = 0; _k < _nlev; ++_k) {\n'
            '          ctx.with_program_resource_level(_k, [&]() {\n'
            '            _level_programs->at(static_cast<std::size_t>(_k)).publish(hierarchy_dt);\n'
            '          });\n'
            '        }\n'
            '      }\n'
            '    };\n'
            '    ctx.advance_synchronized_hierarchy(dt, _advance_hierarchy);\n')

    # Every generated prelude allocation is layout-bound. Materialize one complete closure bundle
    # per level before the first advance, and rebuild the set exactly once after a topology epoch or
    # process-local materialization-generation change. This is deliberately generic: scalar scratch,
    # condensed coefficients, matrix-free
    # apply captures, prepared problems and Krylov workspaces all follow the same lifetime protocol.
    level_resources = (
        '  struct _PopsAmrLevelProgram {\n' + phase_fields + '  };\n'
        '  auto _make_level_program = [ctx_owner]() {\n'
        '    auto& ctx = *ctx_owner;\n'
        + prelude + '\n'
        '    return _PopsAmrLevelProgram{\n' + phase_initializers + '    };\n'
        '  };\n'
        '  auto _level_programs = std::make_shared<std::vector<_PopsAmrLevelProgram>>();\n'
        '  auto _level_program_epoch = std::make_shared<std::uint64_t>(\n'
        '      std::numeric_limits<std::uint64_t>::max());\n'
        '  auto _level_program_generation = std::make_shared<std::uint64_t>(\n'
        '      std::numeric_limits<std::uint64_t>::max());\n'
        '  auto _refresh_level_programs = [=]() {\n'
        '    auto& ctx = *ctx_owner;\n'
        '    const auto topology = ctx.program_resource_topology();\n'
        '    const std::uint64_t epoch = topology.epoch;\n'
        '    const std::uint64_t generation = topology.generation;\n'
        '    const int levels = topology.levels;\n'
        + transform_refresh_guard +
        '    if (*_level_program_epoch == epoch &&\n'
        '        *_level_program_generation == generation &&\n'
        '        _level_programs->size() == static_cast<std::size_t>(levels))\n'
        '      return;\n'
        '    _level_programs->clear();\n'
        '    _level_programs->reserve(static_cast<std::size_t>(levels));\n'
        '    ctx.for_each_program_resource_level([&](int) {\n'
        '      _level_programs->emplace_back(_make_level_program());\n'
        '    });\n'
        '    *_level_program_epoch = epoch;\n'
        '    *_level_program_generation = generation;\n'
        '  };\n'
        '  _refresh_level_programs();\n')
    return (
        flux_expression_budget
        + '\n#include <pops/runtime/program/amr_program_context.hpp>  // registers the AmrSystem provider\n'
        '// AMR install entry (epic ADC-511 / ADC-508, Spec 6): the target=\'amr_system\' counterpart\n'
        '// of pops_install_program. AmrSystem::install_program resolves + calls it after binding the\n'
        '// blocks by name and seeding the runtime params. The shared factory selects the provider and\n'
        '// the wrapper installs only the parent/child clock driver: the SAME\n'
        '// lowered body is recursively subcycled, temporally interpolated and conservatively synced.\n'
        'extern "C" void pops_install_program_amr('
        'pops::AmrSystem<pops::kNativeDimension>* sys) {\n'
        + provider_plan_install + ("\n" if provider_plan_install else "") +
        '  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);\n'
        '  auto& ctx = *ctx_owner;\n'
        + transform_guard + level_resources +
        '\n  ctx.install([=](double dt) {\n'
        '    auto& ctx = *ctx_owner;\n'
        '    _refresh_level_programs();\n'
        + installed_driver +
        '  }, ctx_owner, _refresh_level_programs);\n'
        '}\n')
