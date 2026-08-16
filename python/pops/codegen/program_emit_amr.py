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
            result[power] = result.get(power, Fraction(0)) + left_coefficient * right_coefficient
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
            result[basis] = {power: factor for power, factor in destination.items() if factor}
            if not result[basis]:
                del result[basis]
        maximum_terms = max(maximum_terms, *(len(polynomial) for polynomial in result.values()), 0)
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
                "cond_block",
                "body_block",
                "apply_block",
                "residual_block",
                "true_block",
                "false_block",
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
                                seeded.append((loop_env, loop_count, loop_maximum, loop_next))
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
                        if (
                            contains_flux(value.attrs["cond_block"], block)
                            or contains_flux(value.attrs["body_block"], block)
                            or (value.block == block and expression)
                        ):
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
                            next_paths.append((probe_env, probe_count, probe_maximum, probe_next))
                        continue

                    expression = {}
                    if value.block == block:
                        if value.op == "rhs" and value.attrs.get("flux", True):
                            expression = {(value.id, next_basis): {0: Fraction(1)}}
                            next_basis += 1
                            basis_count += 1
                        elif value.op == "history":
                            # A history read is a second live FluxExpression basis at the AMR
                            # commit boundary.  It was authored in an earlier accepted step, but
                            # native rehydration gives it a fresh attempt-local identity; budget
                            # it alongside the current RHS rather than treating it as a zero.
                            expression = environment.get(
                                ("history_flux", value.attrs["history"]), {}
                            )
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
                            expression = environment.get(value.inputs[alias_input[value.op]].id, {})
                        if value.op == "store_history":
                            environment[("history_flux", value.attrs["history"])] = expression
                    environment[value.id] = expression
                    next_paths.append(
                        (environment, basis_count, note(expression, maximum_terms), next_basis)
                    )
                active = next_paths
            return active

        paths = execute(program._values, paths)
        return (
            max(
                (
                    max(
                        basis_count,
                        max(
                            (len(expression) for expression in environment.values()
                             if isinstance(expression, dict)),
                            default=0,
                        ),
                    )
                    for environment, basis_count, _, _ in paths
                ),
                default=0,
            ),
            max((maximum_terms for _, _, maximum_terms, _ in paths), default=0),
        )

    budgets = [analyze(block) for block in ordered_blocks]

    # A retained flux history is materialized by native code in a later attempt, so it cannot be
    # counted as an authored RHS occurrence in the single-step walk above.  Distinct live sources
    # and retained lag samples coexist during rehydration, so both sides contribute to the finite
    # aggregate bound.
    def history_flux_blocks(
        values: Any,
        stores: dict[tuple[Any, str], Any],
        reads: set[tuple[Any, str]],
    ) -> None:
        for value in values:
            if value.op == "store_history" and value.inputs:
                stores[(value.block, value.attrs["history"])] = value.inputs[0]
            elif value.op == "history":
                reads.add((value.block, value.attrs["history"]))
            for key in (
                "cond_block", "body_block", "apply_block", "residual_block", "true_block", "false_block"
            ):
                nested = value.attrs.get(key)
                if isinstance(nested, (list, tuple)):
                    history_flux_blocks(nested, stores, reads)

    stored_histories: dict[tuple[Any, str], Any] = {}
    read_histories: set[tuple[Any, str]] = set()
    history_flux_blocks(program._values, stored_histories, read_histories)
    for index, block in enumerate(ordered_blocks):
        stored = {name: source for (candidate, name), source in stored_histories.items()
                  if candidate == block}
        read = {name for candidate, name in read_histories if candidate == block}
        if stored.keys() & read:
            bases, terms = budgets[index]
            # The current RHS and its retained lag are independently authenticated at the AMR
            # attempt boundary.  A source-only RHS still needs that pair: whether it contributes
            # spatial faces is a runtime sample fact, not permission to erase the exact temporal
            # basis bound from the frozen AB2 program.
            retained = 0
            live = 0
            for name, source in stored.items():
                if name not in read:
                    continue
                # The frozen persistence policy is the authoritative retained depth.  The
                # source may be an aliased/combined expression, so count every distinct RHS
                # ancestor conservatively before multiplying by the readable lag window.
                def source_bases(value: Any, seen: set[int]) -> int:
                    if value.id in seen:
                        return 0
                    seen.add(value.id)
                    if value.op == "rhs":
                        return 1
                    return sum(source_bases(input_value, seen) for input_value in value.inputs)

                source_count = max(1, source_bases(source, set()))
                live += source_count
                retained += source_count * int(program._histories[name])
            budgets[index] = (max(bases, live, 1) + retained, max(terms, 1))
    return tuple(budgets)


def _emit_flux_expression_budget(program: Any) -> str:
    budgets = _flux_expression_budgets(program)
    rhs_bounds = ", ".join("UINT64_C(%d)" % rhs for rhs, _ in budgets)
    coefficient_bounds = ", ".join("UINT64_C(%d)" % coefficient for _, coefficient in budgets)
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
        f"{str(has_flux).lower()}; }}\n"
        f'extern "C" int pops_program_flux_expression_budget_count() {{ return {count}; }}\n'
        + lookup("pops_program_flux_rhs_basis_bound", rhs_bounds)
        + lookup("pops_program_flux_coefficient_term_bound", coefficient_bounds)
        # Generated Program IR currently has no interface-coupling node.  The explicit zero is an
        # authenticated finite authority, not absence of metadata; hand-authored Program DSOs must
        # export their own exact non-zero bound when they call apply_coupling_operators().
        + 'extern "C" std::uint64_t '
        "pops_program_interface_coupling_application_bound() { return UINT64_C(0); }\n"
        + 'extern "C" std::uint64_t '
        "pops_program_interface_coupling_identity_character_bound() { return UINT64_C(0); }\n"
        + _emit_checkpoint_shape_metadata(program)
    )


def _emit_checkpoint_shape_metadata(program: Any) -> str:
    """Emit the frozen POPSAND4 shape before any native prelude allocation."""
    block_idx = program._block_indices()
    temporal = program.temporal_manifest()
    history_manifest = {row["name"]: row for row in temporal["histories"]}
    histories = []
    for name, lag in sorted(program._histories.items()):
        owner = getattr(program, "_history_blocks", {}).get(name)
        owner_index = block_idx.get(owner) if owner is not None else None
        if owner_index is None:
            raise ValueError("AMR history %r requires explicit block owner provenance" % name)
        state_ref = getattr(program, "_history_state_refs", {}).get(name)
        state_identity = (
            state_ref.qualified_id if state_ref is not None else "scalar-history:" + name
        )
        space = getattr(program, "_history_spaces", {}).get(name)
        space_identity = (
            json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
            if space is not None
            else "scalar-field"
        )
        row = history_manifest[name]
        interpolation = json.dumps(row["interpolation"], sort_keys=True, separators=(",", ":"))
        component = getattr(program, "_histories_ncomp", {}).get(name)
        histories.append(
            (
                str(name),
                int(owner_index),
                str(state_identity),
                str(space_identity),
                str(row["clock"]),
                interpolation,
                int(lag) + 1,
                -1 if component is None else int(component),
            )
        )

    def string_accessor(symbol: str, column: int, rows: Any) -> str:
        cases = "".join(
            "    case %d: return %s;\n" % (index, json.dumps(row[column]))
            for index, row in enumerate(rows)
        )
        return (
            f'extern "C" const char* {symbol}(int index) {{\n'
            "  switch (index) {\n" + cases + '    default: return "";\n'
            "  }\n"
            "}\n"
        )

    def integer_accessor(symbol: str, column: int) -> str:
        cases = "".join(
            "    case %d: return %d;\n" % (index, row[column])
            for index, row in enumerate(histories)
        )
        return (
            f'extern "C" int {symbol}(int index) {{\n'
            "  switch (index) {\n" + cases + "    default: return 0;\n"
            "  }\n"
            "}\n"
        )

    clocks = tuple((str(row["id"]),) for row in temporal["clocks"])
    return (
        "// Frozen Program checkpoint descriptors and logical-clock authority.\n"
        f'extern "C" int pops_program_checkpoint_history_count() {{ return {len(histories)}; }}\n'
        + string_accessor("pops_program_checkpoint_history_name", 0, histories)
        + integer_accessor("pops_program_checkpoint_history_owner", 1)
        + string_accessor("pops_program_checkpoint_history_state_identity", 2, histories)
        + string_accessor("pops_program_checkpoint_history_space_identity", 3, histories)
        + string_accessor("pops_program_checkpoint_history_clock_identity", 4, histories)
        + string_accessor("pops_program_checkpoint_history_interpolation_identity", 5, histories)
        + integer_accessor("pops_program_checkpoint_history_depth", 6)
        + integer_accessor("pops_program_checkpoint_history_components", 7)
        + f'extern "C" int pops_program_checkpoint_logical_clock_count() {{ return {len(clocks)}; }}\n'
        + string_accessor("pops_program_checkpoint_logical_clock_identity", 0, clocks)
        + 'extern "C" const char* pops_program_checkpoint_temporal_provider_identity() {\n'
        + (
            '  return "pops.amr.same-level-transport-euler-stage-flux@2";\n}\n'
            if program.cell_local_time_contract() is not None
            else '  return "pops.temporal-partition.global@1";\n}\n'
        )
        + 'extern "C" std::uint64_t pops_program_checkpoint_temporal_cell_capacity() {\n'
        + "  return UINT64_C(0);\n}\n"
        + 'extern "C" std::uint64_t '
        + "pops_program_checkpoint_temporal_cells_per_topology_cell() {\n"
        + "  return UINT64_C(%d);\n}\n"
        % (len(block_idx) if program.cell_local_time_contract() is not None else 0)
    )


def _require_bounded_cell_local_program(program: Any, target: Any, hierarchy_bodies: Any) -> Any:
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
        raise ValueError("Program.cell_local_time currently requires the default Program cadence")
    if hierarchy_bodies is not None:
        raise ValueError("Program.cell_local_time does not support hierarchy-scoped field solves")
    if getattr(program, "_dt_bound", None) is not None:
        raise ValueError("Program.cell_local_time does not support a Program dt-bound body")
    if getattr(program, "_histories", None):
        raise ValueError("Program.cell_local_time does not support history operators")

    values = tuple(program._values)
    states = tuple(value for value in values if value.op == "state")
    rhs_values = tuple(value for value in values if value.op == "rhs")
    results = tuple(value for value in values if value.op == "linear_combine")
    blocks = program._block_indices()
    if (
        not states
        or len(states) != len(blocks)
        or len(rhs_values) != len(states)
        or len(results) != len(states)
        or len(values) != 3 * len(states)
    ):
        raise ValueError(
            "Program.cell_local_time requires one transport-only ForwardEuler route per "
            "Program block"
        )
    commits = dict(program._commits)
    routes = []
    for state in states:
        matching_rhs = tuple(
            rhs
            for rhs in rhs_values
            if len(rhs.inputs) == 1 and rhs.inputs[0] is state
        )
        if len(matching_rhs) != 1:
            raise ValueError(
                "Program.cell_local_time requires one typed default-flux RHS per accepted state"
            )
        rhs = matching_rhs[0]
        if (
            rhs.block != state.block
            or rhs.attrs.get("flux") is not True
            or rhs.attrs.get("fluxes") is not None
            or tuple(rhs.attrs.get("sources", ())) != ()
        ):
            raise ValueError(
                "Program.cell_local_time requires default-flux RHS routes without sources or fields"
            )
        result = commits.get(state.state_ref)
        if result is None or not any(result is candidate for candidate in results):
            raise ValueError(
                "Program.cell_local_time ForwardEuler commit must consume its accepted state and RHS"
            )
        if (
            len(result.inputs) != 2
            or result.inputs[0] is not state
            or result.inputs[1] is not rhs
        ):
            raise ValueError(
                "Program.cell_local_time ForwardEuler commit must consume its accepted state and RHS"
            )
        coefficients = tuple(result.attrs.get("coeffs", ()))
        if (
            len(coefficients) != 2
            or dict(coefficients[0]) != {0: 1}
            or dict(coefficients[1]) != {1: 1}
        ):
            raise ValueError(
                "Program.cell_local_time requires the exact update U_next = U + dt * rhs(U)"
            )
        routes.append((int(blocks[state.block]), int(rhs.id)))
    if len(commits) != len(routes) or len({block for block, _ in routes}) != len(routes):
        raise ValueError(
            "Program.cell_local_time routes must commit every Program block exactly once"
        )
    routes.sort()
    return contract, tuple(routes)


def _emit_amr_install(
    program: Any,
    target: Any,
    prelude: Any,
    body: Any,
    hierarchy_bodies: Any = None,
    provider_plan_install: str = "",
) -> str:
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
    cell_local_time = _require_bounded_cell_local_program(program, target, hierarchy_bodies)
    if target != "amr_system":
        return ""
    flux_expression_budget = _emit_flux_expression_budget(program)
    if cell_local_time is not None:
        cell_local_contract, cell_local_routes = cell_local_time
        clock_identity = json.dumps(program.clock.qualified_id)
        route_rows = ", ".join(
            "{%d, -1, %d}" % (program_block, rhs_id) for program_block, rhs_id in cell_local_routes
        )
        route_count = len(cell_local_routes)
        return (
            flux_expression_budget + "\n#include <pops/runtime/program/amr_program_context.hpp>\n"
            'extern "C" void pops_install_program_amr('
            "pops::AmrSystem<pops::kNativeDimension>* sys) {\n"
            + provider_plan_install
            + ("\n" if provider_plan_install else "")
            + "  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);\n"
            "  auto& ctx = *ctx_owner;\n"
            f"  ctx.configure_primary_clock({clock_identity});\n"
            "  static constexpr std::array<"
            "pops::runtime::program::SameLevelCellTemporalForwardEulerRoute, "
            f"{route_count}> cell_temporal_routes{{{{{route_rows}}}}};\n"
            "  ctx.prepare_same_level_cell_temporal_execution("
            f"{clock_identity}, {cell_local_contract.tick_denominator}, "
            f"{cell_local_contract.rung}, cell_temporal_routes);\n"
            "  ctx.install([ctx_owner](double dt) {\n"
            "    ctx_owner->advance_same_level_cell_temporal(dt);\n"
            "  }, ctx_owner);\n"
            "}\n"
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
            "  auto _require_local_transform_level_contract = [ctx_owner]() {\n"
            "    auto& ctx = *ctx_owner;\n"
            "    if (ctx.program_resource_topology().levels > 1)\n"
            '      throw std::runtime_error("local_transform on multi-level AMR requires a typed '
            'post-synchronization Program phase; refusing pre-reflux execution");\n'
            "  };\n"
            "  _require_local_transform_level_contract();\n"
        )
        transform_refresh_guard = "    _require_local_transform_level_contract();\n"
    if hierarchy_bodies is None:
        phase_fields = "    std::function<void(double)> step;\n"
        phase_initializers = (
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + body + "\n"
            "      }\n"
        )
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
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + body + "\n"
            "      },\n"
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + gather + "\n"
            "      },\n"
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + solve + "\n"
            "      },\n"
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + publish + "\n"
            "      }\n"
        )
        installed_driver = (
            "    auto _advance_hierarchy = [&](double hierarchy_dt) {\n"
            "      _refresh_level_programs();\n"
            "      const int _nlev = ctx.program_resource_topology().levels;\n"
            "      if (ctx.uses_prepared_krylov_fallback()) {\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_resource_level(_k, [&]() {\n"
            "            _level_programs->at(static_cast<std::size_t>(_k)).step(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "      } else {\n"
            "        // Gather every level before the unique hierarchy-scoped solve.\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_resource_level(_k, [&]() {\n"
            "            _level_programs->at(static_cast<std::size_t>(_k)).gather(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "        ctx.with_program_resource_level(0, [&]() {\n"
            "          _level_programs->front().solve(hierarchy_dt);\n"
            "        });\n"
            "        // The composite solution is complete before any level reconstructs or commits.\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_resource_level(_k, [&]() {\n"
            "            _level_programs->at(static_cast<std::size_t>(_k)).publish(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "      }\n"
            "    };\n"
            "    ctx.advance_synchronized_hierarchy(dt, _advance_hierarchy);\n"
        )

    # Every generated prelude allocation is layout-bound. Materialize one complete closure bundle
    # per level before the first advance, and rebuild the set exactly once after a topology epoch or
    # process-local materialization-generation change. This is deliberately generic: scalar scratch,
    # condensed coefficients, matrix-free
    # apply captures, prepared problems and Krylov workspaces all follow the same lifetime protocol.
    level_resources = (
        "  struct _PopsAmrLevelProgram {\n" + phase_fields + "  };\n"
        "  auto _make_level_program = [ctx_owner]() {\n"
        "    auto& ctx = *ctx_owner;\n" + prelude + "\n"
        "    return _PopsAmrLevelProgram{\n" + phase_initializers + "    };\n"
        "  };\n"
        "  auto _level_programs = std::make_shared<std::vector<_PopsAmrLevelProgram>>();\n"
        "  auto _level_program_epoch = std::make_shared<std::uint64_t>(\n"
        "      std::numeric_limits<std::uint64_t>::max());\n"
        "  auto _level_program_generation = std::make_shared<std::uint64_t>(\n"
        "      std::numeric_limits<std::uint64_t>::max());\n"
        "  auto _refresh_level_programs = [=]() {\n"
        "    auto& ctx = *ctx_owner;\n"
        "    const auto topology = ctx.program_resource_topology();\n"
        "    const std::uint64_t epoch = topology.epoch;\n"
        "    const std::uint64_t generation = topology.generation;\n"
        "    const int levels = topology.levels;\n"
        + transform_refresh_guard
        + "    if (*_level_program_epoch == epoch &&\n"
        "        *_level_program_generation == generation &&\n"
        "        _level_programs->size() == static_cast<std::size_t>(levels))\n"
        "      return;\n"
        "    _level_programs->clear();\n"
        "    _level_programs->reserve(static_cast<std::size_t>(levels));\n"
        "    ctx.for_each_program_resource_level([&](int) {\n"
        "      _level_programs->emplace_back(_make_level_program());\n"
        "    });\n"
        "    *_level_program_epoch = epoch;\n"
        "    *_level_program_generation = generation;\n"
        "  };\n"
        "  _refresh_level_programs();\n"
    )
    return (
        flux_expression_budget
        + "\n#include <pops/runtime/program/amr_program_context.hpp>  // registers the AmrSystem provider\n"
        "// AMR install entry (epic ADC-511 / ADC-508, Spec 6): the target='amr_system' counterpart\n"
        "// of pops_install_program. AmrSystem::install_program resolves + calls it after binding the\n"
        "// blocks by name and seeding the runtime params. The shared factory selects the provider and\n"
        "// the wrapper installs only the parent/child clock driver: the SAME\n"
        "// lowered body is recursively subcycled, temporally interpolated and conservatively synced.\n"
        'extern "C" void pops_install_program_amr('
        "pops::AmrSystem<pops::kNativeDimension>* sys) {\n"
        + provider_plan_install
        + ("\n" if provider_plan_install else "")
        + "  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);\n"
        "  auto& ctx = *ctx_owner;\n"
        + transform_guard
        + level_resources
        + "\n  ctx.install([=](double dt) {\n"
        "    auto& ctx = *ctx_owner;\n"
        "    _refresh_level_programs();\n"
        + installed_driver
        + "  }, ctx_owner, _refresh_level_programs);\n"
        "}\n"
    )
