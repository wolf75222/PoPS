"""pops.codegen.program_emit_amr -- the AMR install-entry emitter (epic ADC-511 / ADC-508, Spec 6).

Split out of :mod:`pops.codegen.program_codegen` so that module stays under the Spec-4 500-line
budget. ``_emit_amr_install`` is the only public name; ``program_codegen`` re-imports it and calls
it from ``emit_cpp_program`` when ``target='amr_system'``.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any

from .amr_program_context_unit import amr_program_context_include


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
    """Choose per-block resource ceilings for exact FluxExpression bookkeeping.

    An explicit ceiling admits dynamic support, including repeated dense history
    remaps. Exceeding it must reject the native transaction before publication.
    Without one, estimate the current invocation and declared retained windows;
    that estimate is not a proof that arbitrary histories fit for unlimited steps.
    """
    blocks = program._block_indices()
    ordered_blocks = sorted(blocks, key=blocks.get)
    from pops.time._program.resource_budgets import validate_flux_expression_budget
    explicit = program.flux_expression_budget
    validate_flux_expression_budget(explicit)
    if explicit is not None:
        return tuple(explicit for _ in ordered_blocks)

    def flux_basis_count(value: Any) -> int:
        if value.op in {"rhs", "principal_rate", "stencil_flux_divergence"}:
            return 1 if value.attrs.get("flux", True) else 0
        if value.op == "diffusive_rhs":
            from pops.codegen.program_emit_diffusion import diffusive_flux_basis_count

            return diffusive_flux_basis_count(value)
        return 0

    def accepted_flux_polynomials(value: Any) -> list[dict[int, Fraction]]:
        # Newton residual probes only evaluate faces; the consumed solve publishes
        # each prepared diffusive provider once. Budget that exact publication,
        # including its authenticated temporal polynomial, independently of the
        # nonlinear iteration limit. Branch/range traversal below still counts
        # actual possible solve invocations on each execution path.
        if value.op == "solve_spatial_nonlinear":
            from pops.time._program.spatial_solve import spatial_rate_weight

            return [
                _flux_polynomial(spatial_rate_weight(value, rate))
                for rate in value.attrs["residual_block"]
                if rate.op == "diffusive_rhs"
                for _ in range(flux_basis_count(rate))
            ]
        return [{0: Fraction(1)} for _ in range(flux_basis_count(value))]

    def region_values(values: Any) -> Any:
        # Preserve lexical order and repeated region occurrences. A completed
        # region may be shared; only an active ancestor constitutes a cycle.
        pending = [("enter", values)]
        active_regions: set[int] = set()
        while pending:
            action, item = pending.pop()
            if action == "exit":
                active_regions.remove(item)
                continue
            if action == "enter":
                identity = id(item)
                if identity in active_regions:
                    raise ValueError("AMR flux budget contains a cyclic region")
                active_regions.add(identity)
                pending.append(("exit", identity))
                pending.extend(("value", value) for value in reversed(item))
                continue
            yield item
            for key in reversed((
                "cond_block", "body_block", "apply_block", "residual_block",
                "true_block", "false_block",
            )):
                nested = item.attrs.get(key)
                if isinstance(nested, (list, tuple)):
                    pending.append(("enter", nested))

    def contains_rhs(values: Any, block: Any, *, flux_only: bool = False) -> bool:
        return any(
            value.op in {"rhs", "diffusive_rhs", "principal_rate", "stencil_flux_divergence"}
            and value.block == block and (not flux_only or flux_basis_count(value) != 0)
            for value in region_values(values)
        )

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

        def leaf_expressions(expression: Any) -> Any:
            # Products carry independent field expressions. Traverse the abstract
            # carrier iteratively; its depth is not a resource ceiling.
            pending = [expression]
            while pending:
                current = pending.pop()
                if isinstance(current, tuple):
                    pending.extend(member for _, member in current)
                else:
                    yield current

        def note(expression: Any, maximum: int) -> int:
            for leaf in leaf_expressions(expression):
                maximum = max(maximum, *(len(polynomial) for polynomial in leaf.values()), 0)
            return maximum

        def execute_region(values: Any, active: Any) -> Any:
            for value in values:
                next_paths = []
                for environment, basis_count, maximum_terms, next_basis in active:
                    if value.op == "branch":
                        for arm_key, result_key in (
                            ("true_block", "true_result"),
                            ("false_block", "false_result"),
                        ):
                            fork = yield (
                                value.attrs[arm_key],
                                [(dict(environment), basis_count, maximum_terms, next_basis)],
                            )
                            for arm_env, arm_count, arm_maximum, arm_next in fork:
                                arm_result = arm_env.get(value.attrs[result_key].id, {})
                                arm_env[value.id] = arm_result
                                next_paths.append(
                                    (arm_env, arm_count, note(arm_result, arm_maximum), arm_next)
                                )
                        continue

                    if value.op in ("range", "subcycle", "structured_fold"):
                        loop_paths = [(dict(environment), basis_count, maximum_terms, next_basis)]
                        loop_input = value.inputs[0]
                        if value.op == "structured_fold":
                            bound = value.inputs[2]
                            if bound.op != "int64_constant":
                                if (contains_rhs(value.attrs["body_block"], block, flux_only=True)
                                        or any(leaf_expressions(environment.get(loop_input.id, {})))):
                                    raise ValueError("AMR dynamic fold flux storage requires an explicit flux_expression_budget")
                                iterations = 1
                            else:
                                iterations = int(bound.attrs["value"])
                            argument = value.attrs["body_argument"]
                        else:
                            iterations = int(value.attrs["count"])
                            argument = loop_input
                        for _ in range(iterations):
                            seeded = []
                            for loop_env, loop_count, loop_maximum, loop_next in loop_paths:
                                loop_env = dict(loop_env)
                                loop_env[argument.id] = loop_env.get(loop_input.id, {})
                                seeded.append((loop_env, loop_count, loop_maximum, loop_next))
                            loop_paths = yield value.attrs["body_block"], seeded
                            for loop_env, _, _, _ in loop_paths:
                                loop_env[loop_input.id] = loop_env.get(
                                    value.attrs["body"].id, {}
                                )
                        for loop_env, loop_count, loop_maximum, loop_next in loop_paths:
                            expression = loop_env.get(loop_input.id, {})
                            loop_env[value.id] = expression
                            next_paths.append(
                                (loop_env, loop_count, note(expression, loop_maximum), loop_next)
                            )
                        continue

                    if value.op == "while":
                        expression = environment.get(value.inputs[0].id, {})
                        if (
                            contains_rhs(value.attrs["cond_block"], block, flux_only=True)
                            or contains_rhs(value.attrs["body_block"], block, flux_only=True)
                            or any(leaf_expressions(expression))
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
                        probes = yield (
                            value.attrs["cond_block"],
                            [(dict(environment), basis_count, maximum_terms, next_basis)],
                        )
                        probes = yield value.attrs["body_block"], probes
                        for probe_env, probe_count, probe_maximum, probe_next in probes:
                            probe_env[value.id] = {}
                            next_paths.append((probe_env, probe_count, probe_maximum, probe_next))
                        continue

                    expression = {}
                    if value.op == "product":
                        expression = tuple(
                            (name, environment.get(source.id, {}))
                            for name, source in zip(value.attrs["fields"], value.inputs, strict=True)
                        )
                    elif value.op == "product_get":
                        source = environment.get(value.inputs[0].id, ())
                        expression = dict(source).get(value.attrs["field"], {})
                    elif value.op == "region_argument":
                        expression = environment.get(value.id, {})
                    elif value.block == block:
                        polynomials = accepted_flux_polynomials(value)
                        multiplicity = len(polynomials)
                        if multiplicity:
                            expression = {
                                (value.id, next_basis + offset): polynomial
                                for offset, polynomial in enumerate(polynomials)
                            }
                            next_basis += multiplicity
                            basis_count += multiplicity
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

        def execute(values: Any, active: Any) -> Any:
            # Trampoline the exact same path interpreter: no recursion limit,
            # branch merge, sampling, or loss of loop multiplicity.
            stack = [(id(values), execute_region(values, active))]
            active_regions = {id(values)}
            result = None
            while stack:
                try:
                    nested, nested_paths = stack[-1][1].send(result)
                except StopIteration as completed:
                    result = completed.value
                    active_regions.remove(stack.pop()[0])
                else:
                    identity = id(nested)
                    if identity in active_regions:
                        raise ValueError("AMR flux budget contains a cyclic region")
                    active_regions.add(identity)
                    stack.append((identity, execute_region(nested, nested_paths)))
                    result = None
            return result

        paths = execute(program._values, paths)
        return (
            max(
                (
                    max(
                        basis_count,
                        max(
                            (len(leaf) for expression in environment.values()
                             for leaf in leaf_expressions(expression)),
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
        for value in region_values(values):
            if value.op == "store_history" and value.inputs:
                stores[(value.block, value.attrs["history"])] = value.inputs[0]
            elif value.op == "history":
                reads.add((value.block, value.attrs["history"]))

    stored_histories: dict[tuple[Any, str], Any] = {}
    read_histories: set[tuple[Any, str]] = set()
    history_flux_blocks(program._values, stored_histories, read_histories)
    history_manifest = {
        row["name"]: row for row in program.temporal_manifest()["histories"]
    }
    for index, block in enumerate(ordered_blocks):
        stored = {name: source for (candidate, name), source in stored_histories.items()
                  if candidate == block}
        read = {name for candidate, name in read_histories if candidate == block}
        # Pure state histories have no flux ancestry. Inspect structured regions too: branch and
        # loop results can carry real RHS values through attrs rather than ordinary input edges.
        # Keep the existing conservative retained bound whenever any actual RHS exists, including
        # source-only RHS calls; only a proved RHS-free block can retain its zero analyzed budget.
        if stored.keys() & read and contains_rhs(program._values, block):
            bases, terms = budgets[index]
            # The current RHS and its retained lag are independently authenticated at the AMR
            # attempt boundary.  A source-only RHS still needs that pair: whether it contributes
            # spatial faces is a runtime sample fact, not permission to erase the exact temporal
            # basis bound from the frozen AB2 program.
            retained = 0
            live = 0
            temporal_slots = 1
            for name, source in stored.items():
                if name not in read:
                    continue
                # The frozen persistence policy is the authoritative retained depth.  The
                # source may be an aliased/combined expression, so count every distinct RHS
                # ancestor conservatively before multiplying by the readable lag window.
                def source_bases(value: Any, seen: set[int]) -> int:
                    pending = [value]
                    count = 0
                    while pending:
                        current = pending.pop()
                        if current.id in seen:
                            continue
                        seen.add(current.id)
                        multiplicity = flux_basis_count(current)
                        if multiplicity:
                            count += multiplicity
                        else:
                            pending.extend(reversed(current.inputs))
                    return count

                source_count = max(1, source_bases(source, set()))
                interpolation = history_manifest[name]["interpolation"]
                slots = (int(interpolation["slots"])
                         if interpolation.get("kind") == "history_transfer" else 1)
                # An affine macro publication binds the complete immutable entry
                # and candidate rings, plus explicitly authorized state endpoints.
                # Its rectangular row can select every source; scalar parameters
                # are runtime values, so no branch/weight is assumed to be zero.
                from pops.codegen.program_lowerability import all_ops
                for operation in all_ops(program):
                    if operation.op != "accept_increment":
                        continue
                    row = operation.attrs.get("histories", {}).get(name)
                    if row is not None:
                        from pops.time._history.transfer import transfer_dimensions
                        sources, _targets, _parameters = transfer_dimensions(row["transfer"])
                        slots = max(slots, sources)
                temporal_slots = max(temporal_slots, slots)
                live += source_count
                retained += source_count * int(program._histories[name]) * slots
            budgets[index] = (max(bases, live, 1) + retained, max(terms, 1) * temporal_slots)
    return tuple(budgets)


def _emit_flux_expression_budget(program: Any) -> str:
    budgets = _flux_expression_budgets(program)
    from pops.time._program.resource_budgets import validate_flux_expression_abi
    validate_flux_expression_abi(budgets)
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
    # One retained basis can carry at most one sample of each installed physical interface.
    # Summing endpoint bounds is conservative (a paired sample is counted twice), and includes
    # every declared readable history lag. The scheduler supplies the independent route ceiling.
    interface_samples = sum(rhs for rhs, _ in budgets)
    interface_identity_characters = (
        len(program._ir_hash()) + len("shared-rhs/") + len(str(2**31 - 1))
        + len("program-rhs-group") if interface_samples else 0
    )
    return (
        "// Frozen-IR FluxExpression budgets in exact pops_program_block_name order.\n"
        f'extern "C" bool pops_program_has_flux_expression() {{ return '
        f"{str(has_flux).lower()}; }}\n"
        f'extern "C" int pops_program_flux_expression_budget_count() {{ return {count}; }}\n'
        + lookup("pops_program_flux_rhs_basis_bound", rhs_bounds)
        + lookup("pops_program_flux_coefficient_term_bound", coefficient_bounds)
        + 'extern "C" std::uint64_t pops_program_flux_exact_coefficient_bytes() { return '
        f"UINT64_C({program.max_exact_coefficient_bytes}); }}\n"
        + 'extern "C" std::uint64_t '
        f"pops_program_interface_coupling_application_bound() {{ return UINT64_C({interface_samples}); }}\n"
        + 'extern "C" std::uint64_t '
        f"pops_program_interface_coupling_identity_character_bound() {{ return UINT64_C({interface_identity_characters}); }}\n"
        + _emit_checkpoint_shape_metadata(program)
    )


def _emit_checkpoint_shape_metadata(program: Any) -> str:
    """Emit the frozen checkpoint shape before any native prelude allocation."""
    from .program_history_metadata import emit_checkpoint_history_metadata

    block_idx = program._block_indices()
    temporal = program.temporal_manifest()
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

    clocks = tuple((str(row["id"]),) for row in temporal["clocks"])
    return (
        "// Frozen Program checkpoint descriptors and logical-clock authority.\n"
        + emit_checkpoint_history_metadata(program, require_owner=True)
        + f'extern "C" int pops_program_checkpoint_logical_clock_count() {{ return {len(clocks)}; }}\n'
        + string_accessor("pops_program_checkpoint_logical_clock_identity", 0, clocks)
        + 'extern "C" const char* pops_program_checkpoint_primary_clock_identity() {\n'
        + "  return %s;\n}\n" % json.dumps(temporal["primary_clock"])
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


def _emit_amr_program_provider_register(provider_plan_install: str) -> str:
    """Emit the Program consumer-plan registrar.

    ``AmrSystem::install_program`` calls this before ``ensure_engine()`` materializes the
    hierarchy. Native block install may already have sealed the auxiliary registry; consumer
    plans are appended to that sealed image and re-resolved in place.
    """
    return (
        'extern "C" void pops_register_program_provider_routes_amr('
        "pops::AmrSystem<pops::kNativeDimension>* sys) {\n"
        "  if (sys == nullptr)\n"
        '    throw std::invalid_argument("program provider-route registrar received null AmrSystem");\n'
        + (provider_plan_install + "\n" if provider_plan_install else "")
        + "}\n"
    )


def _emit_flux_temporal_family_install(program: Any) -> str:
    from pops.codegen.program_emit_diffusion import _diffusive_flux_families
    from pops.codegen.program_emit_kernels import _named_fluxes
    from pops.codegen.program_emit_ops import _rhs_flux_temporal_family
    from pops.codegen.program_emit_field_routes import _walk_program_nodes

    blocks = program._block_indices()
    rows = []
    for value in _walk_program_nodes(tuple(program._values)):
        if value.op == "rhs" and value.attrs.get("flux", True):
            named = _named_fluxes(value)
            requested = value.attrs.get("sources")
            provider = 3 if named is not None else (0 if requested is None or "default" in requested else 1)
            rows.append((blocks[value.block], value.id, provider,
                         _rhs_flux_temporal_family(value, named)))
        elif value.op == "principal_rate":
            rows.append((blocks[value.block], value.id, 1, _rhs_flux_temporal_family(value)))
        elif value.op == "stencil_flux_divergence":
            from pops.codegen.program_emit_stencil_flux import stencil_temporal_family
            rows.append((blocks[value.block], value.id, 5, stencil_temporal_family(value)))
        elif value.op == "diffusive_rhs":
            constitutive, transport = _diffusive_flux_families(value)
            rows.append((blocks[value.block], value.id, 4, constitutive))
            if transport is not None:
                rows.append((blocks[value.block], value.id, 1, transport))
    rows.sort(key=lambda row: row[:3])
    if len({row[:3] for row in rows}) != len(rows):
        raise ValueError("AMR flux temporal-family table aliases one resolved producer")
    encoded = ", ".join("{%d, %d, %d, %s}" % (block, rhs, provider, json.dumps(family))
                        for block, rhs, provider, family in rows)
    return "  ctx.install_flux_temporal_families({%s});\n" % encoded if rows else ""


def _emit_amr_install(
    program: Any,
    target: Any,
    prelude: Any,
    body: Any,
    hierarchy_bodies: Any = None,
    provider_plan_install: str = "",
    post_synchronization: Any = None,
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
    flux_temporal_families = _emit_flux_temporal_family_install(program)
    provider_register = _emit_amr_program_provider_register(provider_plan_install)
    if cell_local_time is not None:
        from .program_source_map import operation_group

        cell_local_contract, cell_local_routes = cell_local_time
        clock_identity = json.dumps(program.clock.qualified_id)
        route_rows = ", ".join(
            "{%d, -1, %d}" % (program_block, rhs_id) for program_block, rhs_id in cell_local_routes
        )
        route_count = len(cell_local_routes)
        # The bounded-shape check above proves that this native provider owns
        # every state read, default-flux RHS and U + dt*RHS update in the IR.
        # They execute together in this call, with no separate generated SSA
        # variables. Record the coalesced call, not the discarded global body.
        advance_lines = []
        with operation_group(program._values, advance_lines, environment={},
                             channel="cell_local_provider_advance"):
            advance_lines.append("    ctx_owner->advance_same_level_cell_temporal(dt);")
        return (
            flux_expression_budget + "\n" + amr_program_context_include()
            + provider_register
            + 'extern "C" void pops_install_program_amr('
            "pops::AmrSystem<pops::kNativeDimension>* sys) {\n"
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
            + "\n".join(advance_lines) + "\n"
            + "  }, ctx_owner);\n"
            "}\n"
        )

    from .program_amr_finite_scope import scope_body, uses_finite_values
    from .program_amr_continuation import configuration_install, checkpoint_exports
    from pops.time._program.continuation_forest import forest_node
    continuation_forest = forest_node(program)
    regrid_forest = bool(continuation_forest is not None and
                         continuation_forest.attrs["continuation_forest"].get("regrid"))
    finite_scopes = uses_finite_values(program)
    body = scope_body(body, finite_scopes)
    if hierarchy_bodies is not None:
        hierarchy_bodies = tuple(scope_body(part, finite_scopes) for part in hierarchy_bodies)
    if regrid_forest and post_synchronization:
        post_synchronization = ("if (ctx.continuation_regrid_phase()) ctx.continuation_regrid_bind_target_();\n"
                                + post_synchronization)
    post_sync_src = scope_body(post_synchronization, finite_scopes) if post_synchronization else ""
    from pops.codegen.program_lowerability import all_ops
    affine_finalization = any(value.op == "accept_increment" for value in all_ops(program))
    post_sync_field = (
        "    std::function<void(double)> post_synchronization;\n" if post_sync_src else ""
    )
    post_sync_initializer = (
        "      , [=](double dt) {\n"
        "        auto& ctx = *ctx_owner;\n"
        + ("        if (ctx.continuation_regrid_phase()) ctx.begin_regrid_invocation();\n"
           "        else { ctx.begin_step(dt); ctx.set_stage_time(1, 1); }\n"
           if regrid_forest else
           "        ctx.begin_step(dt);\n        ctx.set_stage_time(1, 1);\n")
        + post_sync_src
        + "\n"
        "      }\n"
        if post_sync_src
        else ""
    )
    post_sync_driver = (
        "    const int _nlev_post = ctx.program_resource_topology().levels;\n"
        "    for (int _k = 0; _k < _nlev_post; ++_k) {\n"
        "      ctx.with_program_resource_level(_k, [&]() {\n"
        "        _level_programs->at(static_cast<std::size_t>(_k))"
        ".post_synchronization(dt);\n"
        "      });\n"
        "    }\n"
        if post_sync_src
        else ""
    )
    if affine_finalization:
        post_sync_driver = (
            "    ctx.prepare_affine_finalization();\n"
            "    ctx.with_program_resource_level(0, [&]() {\n"
            "      _level_programs->front().post_synchronization(dt);\n"
            "    });\n"
            "    ctx.finish_affine_finalization();\n"
        )
    if continuation_forest is not None:
        if affine_finalization:
            raise ValueError("AMR forest must express its complete transformation before parent assessment")
        post_sync_driver += (
            "    ctx.begin_continuation_parent();\n"
            "    ctx.with_program_resource_level(0, [&]() {\n"
            "      _level_programs->front().post_synchronization(dt);\n"
            "    });\n"
        )

    transform_guard = ""
    transform_refresh_guard = ""
    # Installation and hierarchy refresh must use the same authenticated phase
    # classification as resolution, including the composite solve's internal Q maps.
    from pops.codegen._resolution import _uses_local_transform

    if _uses_local_transform(program):
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
    has_maps = any(value.op in ("layout_map_export", "layout_map_import")
                   for value in program._values)
    from pops.codegen.program_emit_hierarchy_regions import has_hierarchy_continuations
    has_hierarchy_regions = has_hierarchy_continuations(program)
    has_continuations = has_maps or has_hierarchy_regions
    if has_maps and hierarchy_bodies is not None:
        raise NotImplementedError("AMR mapping and field barriers require one combined region schedule")
    if hierarchy_bodies is None:
        phase_fields = "    std::function<void(double)> step;\n" + post_sync_field
        phase_initializers = (
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + body + "\n"
            "      }\n"
            + post_sync_initializer
        )
        if has_continuations:
            installed_driver = (
                "    ctx.advance_mapping_hierarchy(dt, [=](double level_dt) {\n"
                "      auto& ctx = *ctx_owner;\n"
                "      const auto topology = ctx.program_resource_topology();\n"
                "      if (*_level_program_epoch != topology.epoch ||\n"
                "          *_level_program_generation != topology.generation ||\n"
                "          _level_programs->size() != static_cast<std::size_t>(topology.levels))\n"
                '        throw std::logic_error("AMR continuation level resources lost their exact hierarchy generation");\n'
                "      _level_programs->at(static_cast<std::size_t>(ctx.level())).step(level_dt);\n"
                "    }, ctx_owner, [=]() {\n"
                "      auto& ctx = *ctx_owner;\n"
                + post_sync_driver +
                "    });\n"
            )
        else:
            installed_driver = (
                "    auto _advance_level = [&](double level_dt) {\n"
                "      _refresh_level_programs();\n"
                "      _level_programs->at(static_cast<std::size_t>(ctx.level())).step(level_dt);\n"
                "    };\n"
                "    ctx.advance_hierarchy(dt, _advance_level);\n"
                + post_sync_driver
            )
    else:
        if len(hierarchy_bodies) == 4:
            gather, solve, observe, publish = hierarchy_bodies
        else:
            gather, solve, publish = hierarchy_bodies
            observe = None
        spatial_solve = any(value.op in ("solve_spatial_nonlinear", "solve_spatial_field") for value in program._values)
        direct_field_solve = any("hierarchy_field_identity" in value.attrs for value in program._values)
        hierarchy_solve_driver = (
            # The spatial solve checks out each prepared level itself for predictor
            # reconciliation and every residual/JVP. A surrounding level-0 checkout
            # would remove that envelope from the registry before its first traversal.
            "        _level_programs->front().solve(hierarchy_dt);\n"
            if spatial_solve else
            "        ctx.with_program_attempt_level(0, [&]() {\n"
            "          _level_programs->front().solve(hierarchy_dt);\n"
            "        });\n"
        )
        phase_fields = (
            "    std::function<void(double)> step;\n"
            "    std::function<void(double)> gather;\n"
            "    std::function<void(double)> solve;\n"
            "    std::function<void(double)> publish;\n"
            + ("    std::function<void(double)> observe;\n" if observe is not None else "")
            + post_sync_field
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
            + (",\n      [=](double dt) {\n"
               "        auto& ctx = *ctx_owner;\n"
               "        (void)dt;\n" + observe + "\n      }\n" if observe is not None else "")
            + post_sync_initializer
        )
        installed_driver = (
            "    auto _advance_hierarchy = [&](double hierarchy_dt) {\n"
            "      _refresh_level_programs();\n"
            "      // The synchronized engine prepares every complete level envelope before this callback.\n"
            "      // Gather/solve/publish run once through the root callback.\n"
            "      if (ctx.level() != 0)\n"
            "        return;\n"
            "      const int _nlev = ctx.program_resource_topology().levels;\n" +
            ("      if (false) {\n" if spatial_solve or direct_field_solve
             else "      if (ctx.uses_prepared_krylov_fallback()) {\n") +
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_attempt_level(_k, [&]() {\n"
            "            _level_programs->at(static_cast<std::size_t>(_k)).step(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "      } else {\n"
            "        // Gather every level before the unique hierarchy-scoped solve.\n"
            "        ctx.with_synchronized_field_gather([&]() {\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_attempt_level(_k, [&]() {\n"
            "            _level_programs->at(static_cast<std::size_t>(_k)).gather(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "        });\n"
            + hierarchy_solve_driver
            + ("        ctx.begin_staged_field_publications();\n"
               "        for (int _k = 0; _k < _nlev; ++_k) {\n"
               "          ctx.with_program_attempt_level(_k, [&]() {\n"
               "            _level_programs->at(static_cast<std::size_t>(_k)).observe(hierarchy_dt);\n"
               "          });\n"
               "        }\n"
               "        ctx.publish_staged_field_components();\n" if observe is not None else "") +
            "        // The composite solution is complete before any level reconstructs or commits.\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_attempt_level(_k, [&]() {\n"
            "            _level_programs->at(static_cast<std::size_t>(_k)).publish(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "      }\n"
            "    };\n"
            "    ctx.advance_synchronized_hierarchy(dt, _advance_hierarchy, true);\n"
            + post_sync_driver
        )

    if affine_finalization:
        blocks = ", ".join(str(index) for index in program._block_indices().values())
        installed_driver = ("    ctx.begin_affine_finalization(dt, {%s});\n" % blocks) + installed_driver
    if continuation_forest is not None:
        blocks = ", ".join(str(index) for index in program._block_indices().values())
        installed_driver = ("    ctx.begin_continuation_forest(dt, {%s});\n" % blocks) + installed_driver

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
        "  auto _refresh_level_programs = [=](bool force = false) {\n"
        "    if (force) {\n"
        "      *_level_program_epoch = std::numeric_limits<std::uint64_t>::max();\n"
        "      *_level_program_generation = std::numeric_limits<std::uint64_t>::max();\n"
        "    }\n"
        "    auto& ctx = *ctx_owner;\n"
        "    const auto topology = ctx.program_resource_topology();\n"
        "    const std::uint64_t epoch = topology.epoch;\n"
        "    const std::uint64_t generation = topology.generation;\n"
        "    const int levels = topology.levels;\n"
        + transform_refresh_guard
        + "    const auto& lane = ctx.prepared_execution_lane();\n"
        "    const bool stale = force || *_level_program_epoch != epoch ||\n"
        "        *_level_program_generation != generation ||\n"
        "        _level_programs->size() != static_cast<std::size_t>(levels);\n"
        "    if (pops::all_reduce_max(stale ? 1L : 0L, lane) == 0)\n"
        "      return;\n"
        # Same-generation restore destroys scratch borrows too. Invalidate before any throwing
        # preparation so a failed restore cannot execute the previous, now dangling captures.
        "    *_level_program_epoch = std::numeric_limits<std::uint64_t>::max();\n"
        "    *_level_program_generation = std::numeric_limits<std::uint64_t>::max();\n"
        "    std::vector<_PopsAmrLevelProgram> next;\n"
        "    std::exception_ptr allocation_error;\n"
        "    try { next.reserve(static_cast<std::size_t>(levels)); }\n"
        "    catch (...) { allocation_error = std::current_exception(); }\n"
        "    pops::collectively_rethrow_exception(allocation_error, lane,\n"
        '        "AMR Program level resource allocation failed collectively");\n'
        "    ctx.for_each_program_resource_level([&](int) {\n"
        "      std::exception_ptr level_error;\n"
        "      try { next.emplace_back(_make_level_program()); }\n"
        "      catch (...) { level_error = std::current_exception(); }\n"
        "      pops::collectively_rethrow_exception(level_error, lane,\n"
        '          "AMR Program level resource capture failed collectively");\n'
        "    });\n"
        + configuration_install(program)
        + "    _level_programs->swap(next);\n"
        "    *_level_program_epoch = epoch;\n"
        "    *_level_program_generation = generation;\n"
        "  };\n"
        "  _refresh_level_programs();\n"
    )
    return (
        flux_expression_budget + checkpoint_exports(program)
        + "\n" + amr_program_context_include("  // registers the AmrSystem provider")
        + "// AMR install entry (epic ADC-511 / ADC-508, Spec 6): the target='amr_system' counterpart\n"
        "// of pops_install_program. AmrSystem::install_program resolves + calls it after binding the\n"
        "// blocks by name and seeding the runtime params. The shared factory selects the provider and\n"
        "// the wrapper installs only the parent/child clock driver: the SAME\n"
        "// lowered body is recursively subcycled, temporally interpolated and conservatively synced.\n"
        + provider_register
        + 'extern "C" void pops_install_program_amr('
        "pops::AmrSystem<pops::kNativeDimension>* sys) {\n"
        + "  auto ctx_owner = pops::runtime::program::make_program_execution_provider(sys);\n"
        "  auto& ctx = *ctx_owner;\n"
        + flux_temporal_families
        + transform_guard
        + level_resources
        + "\n  ctx.install([=](double dt) {\n"
        "    auto& ctx = *ctx_owner;\n"
        # Map continuations use the bundles materialized at install/regrid/restart.
        # Their collective level region checks the exact generation before use.
        + ("" if has_continuations else "    _refresh_level_programs();\n")
        + installed_driver
        + "  }, ctx_owner, [=]() { _refresh_level_programs(); },\n"
        "     [=]() { _refresh_level_programs(true); });\n"
        + ("  ctx.install_continuation_regrid_graph([=]() {\n"
           "    auto& ctx = *ctx_owner;\n"
           "    _refresh_level_programs(true);\n"
           "    const int _levels = ctx.program_resource_topology().levels;\n"
           "    for (int _k = 0; _k < _levels; ++_k) {\n"
           "      ctx.with_program_resource_level(_k, [&]() {\n"
           "        _level_programs->at(static_cast<std::size_t>(_k)).post_synchronization(0.);\n"
           "      });\n"
           "    }\n"
           "    ctx.begin_continuation_parent();\n"
           "    ctx.with_program_resource_level(0, [&]() {\n"
           "      _level_programs->front().post_synchronization(0.);\n"
           "    });\n"
           "  });\n" if regrid_forest else "")
        + "}\n"
    )
