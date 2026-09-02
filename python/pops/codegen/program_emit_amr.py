"""pops.codegen.program_emit_amr -- the AMR install-entry emitter (epic ADC-511 / ADC-508, Spec 6).

Split out of :mod:`pops.codegen.program_codegen` so that module stays under the Spec-4 500-line
budget. ``_emit_amr_install`` is the only public name; ``program_codegen`` re-imports it and calls
it from ``emit_cpp_program`` when ``target='amr_system'``.
"""

from __future__ import annotations

from fractions import Fraction
import json
from typing import Any


_AMR_EXACT_COEFFICIENT_TYPE = (
    "std::initializer_list<pops::runtime::program::ExactCoefficientTerm>"
)


def _rewrite_amr_public_exact_coefficient_calls(source: str) -> str:
    """Make AMR's exact-coefficient calls deducible through the public service template.

    The AMR adapter has the exact five-argument ``axpy``/``lincomb`` implementation, while the
    common ``ProgramExecutionServices`` facade forwards variadic calls through a public template.
    A bare braced initializer cannot be deduced by that template, so emit the same immutable
    metadata with its explicit public type.  This preserves the native flux ledger rather than
    silently dropping the exact coefficient authority to reach the three-argument fallback.
    """
    lines = []
    for line in source.splitlines(keepends=True):
        if "ctx.axpy(" in line or "ctx.lincomb(" in line:
            line = line.replace(
                ", dt, {{",
                ", dt, " + _AMR_EXACT_COEFFICIENT_TYPE + "{{",
            )
            # ``lincomb`` carries one exact metadata list per side; the first replacement above
            # types the left list and this one types the right list without touching unrelated
            # aggregate initializers in generated kernels.
            line = line.replace(
                "}}, {{",
                "}}, " + _AMR_EXACT_COEFFICIENT_TYPE + "{{",
            )
        lines.append(line)
    return "".join(lines)


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
    addition, and exact cancellation as ``ProgramExecutionServices``. An unbounded ``while`` may remain in
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


def _validate_hierarchy_scoped_solve_barrier(program: Any, target: Any) -> None:
    """Reject hierarchy solves whose region shape is not a top-level barrier.

    Resource-plan lowering needs exact component metadata. A hierarchy solve nested in a control
    region is not a legal cross-level lifetime, however, so allowing the resource planner to inspect
    that IR first can replace the authoritative barrier refusal with an incidental ``unknown
    component count`` error. Validate the structural barrier before any plan is built; the actual
    phase emitter repeats the same checks when it is called directly.
    """
    if target != "amr_system":
        return
    from pops.codegen.program_lowerability import all_ops

    solves = [value for value in all_ops(program) if value.op == "solve_linear"]
    scoped = [value for value in solves if value.attrs.get("scope") == "hierarchy"]
    if not scoped:
        return
    top_level_ids = {id(value) for value in program._values}
    nested_scoped = [value.name for value in scoped if id(value) not in top_level_ids]
    if nested_scoped:
        raise NotImplementedError(
            "a hierarchy-scoped solve must be a top-level barrier; nested solve_linear values %r "
            "cannot cross the gather/solve/publish boundary" % nested_scoped
        )
    if len(scoped) != 1 or len(solves) != 1:
        raise NotImplementedError(
            "AMR hierarchy-scoped lowering supports exactly one top-level solve_linear; multiple "
            "hierarchy barriers require an explicit region schedule"
        )
    nested_controls = [
        value.name for value in program._values if value.op in {"while", "range", "branch"}
    ]
    if nested_controls:
        raise NotImplementedError(
            "a hierarchy-scoped solve must be a top-level barrier; control-flow regions %r cannot "
            "cross the gather/solve/publish boundary" % nested_controls
        )


def _emit_amr_candidate_support(artifact_identity: str, route_manifest: str, program_name: str) -> str:
    """DSO-private v5 candidate callbacks shared by every AMR Program shape."""
    artifact = json.dumps(artifact_identity)
    route = json.dumps(route_manifest)
    name = json.dumps(program_name)
    return (
        "namespace {\n"
        "struct ProgramCandidateState final {\n"
        "  std::shared_ptr<pops::runtime::program::ProgramExecutionServices<\n"
        "      pops::kNativeDimension>> ctx_owner;\n"
        "  std::function<void(double)> step;\n"
        "  std::function<void()> hierarchy_resource_refresh;\n"
        "  std::function<std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot>()>\n"
        "      accepted_snapshot;\n"
        "};\n"
        "\n"
        "struct ProgramStepRejectSentinel final : pops::runtime::program::ProgramStepRejectSignal {\n"
        "  using ProgramStepRejectSignal::ProgramStepRejectSignal;\n"
        "};\n"
        "struct ProgramStepRejectPublishFailure final {};\n"
        "template <class Context>\n"
        "[[noreturn]] void program_reject_step(Context& ctx, pops::SolveStatus status,\n"
        "                                      pops::runtime::program::StepAttemptDisposition disposition,\n"
        "                                      std::uint32_t reason_code, std::string_view phase,\n"
        "                                      std::string_view detail = {}) {\n"
        "  pops::runtime::program::ProgramStepRejectRecord record{};\n"
        "  if (!ctx.publish_step_attempt_rejection(status, disposition, reason_code, phase, detail, record))\n"
        "    throw ProgramStepRejectPublishFailure{};\n"
        "  throw ProgramStepRejectSentinel{record};\n"
        "}\n"
        "\n"
        "void program_candidate_step(void* opaque, double dt) {\n"
        "  try {\n"
        "    static_cast<ProgramCandidateState*>(opaque)->step(dt);\n"
        "  } catch (const pops::runtime::program::ProgramStepRejectSignal& signal) {\n"
        "    auto* state = static_cast<ProgramCandidateState*>(opaque);\n"
        "    if (!state->ctx_owner->adopt_step_attempt_rejection(signal.record))\n"
        "      throw ProgramStepRejectPublishFailure{};\n"
        "  }\n"
        "}\n"
        "void program_candidate_hierarchy_refresh(void* opaque) {\n"
        "  auto& state = *static_cast<ProgramCandidateState*>(opaque);\n"
        "  state.ctx_owner->refresh_accepted_hierarchy(state.hierarchy_resource_refresh);\n"
        "}\n"
        "void program_candidate_history_remap(void* opaque, const void* descriptor) {\n"
        "  if (descriptor == nullptr)\n"
        "    throw std::invalid_argument(\"AMR Program candidate received a null history-remap descriptor\");\n"
        "  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->accept_history_remap(\n"
        "      *static_cast<const pops::runtime::program::AmrProgramHistoryRemapDescriptor*>(descriptor));\n"
        "}\n"
        "void program_candidate_restart_preflight(void* opaque) {\n"
        "  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->preflight_restart_regrid();\n"
        "}\n"
        "void program_candidate_restart_regrid(void* opaque) {\n"
        "  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->restart_regrid();\n"
        "}\n"
        "void program_candidate_restart_resync(void* opaque) {\n"
        "  static_cast<ProgramCandidateState*>(opaque)->ctx_owner->resync_after_restart();\n"
        "}\n"
        "pops::runtime::program::AcceptedProgramExecutionServicesSnapshot*\n"
        "program_candidate_accepted_snapshot(void* opaque) {\n"
        "  auto& state = *static_cast<ProgramCandidateState*>(opaque);\n"
        "  if (!state.accepted_snapshot)\n"
        "    throw std::logic_error(\"AMR Program candidate has no accepted snapshot factory\");\n"
        "  return state.accepted_snapshot().release();\n"
        "}\n"
        "void program_candidate_destroy(void* opaque) noexcept {\n"
        "  delete static_cast<ProgramCandidateState*>(opaque);\n"
        "}\n"
        "\n"
        "static constexpr char kProgramCandidateArtifactIdentity[] = " + artifact + ";\n"
        "static constexpr char kProgramCandidateName[] = " + name + ";\n"
        "static constexpr char kProgramCandidateAbiKey[] = POPS_ABI_KEY_LITERAL;\n"
        "static constexpr char kProgramCandidateRouteManifest[] = " + route + ";\n"
        "static constexpr char kProgramCandidateBoundaryManifest[] = \"pops.boundary.manifest.v1\";\n"
        "static constexpr char kProgramCandidatePersistentResourceManifest[] =\n"
        "    \"pops.persistent-resource.manifest.v1\";\n"
        "static constexpr char kProgramCandidateCheckpointIdentity[] = \"pops.checkpoint.identity.v1\";\n"
        "constexpr std::uint64_t kProgramCandidateCapabilities =\n"
        "    pops::runtime::program::kProgramCapabilityHierarchy |\n"
        "    pops::runtime::program::kProgramCapabilitySchedules |\n"
        "    pops::runtime::program::kProgramCapabilityCellTemporal |\n"
        "    pops::runtime::program::kProgramCapabilityPersistentValues |\n"
        "    pops::runtime::program::kProgramCapabilityTransactions;\n"
        "constexpr std::uint64_t kProgramCandidateRequiredServices =\n"
        "    pops::runtime::program::kProgramServiceState |\n"
        "    pops::runtime::program::kProgramServiceFields |\n"
        "    pops::runtime::program::kProgramServiceSpatial |\n"
        "    pops::runtime::program::kProgramServiceHierarchy |\n"
        "    pops::runtime::program::kProgramServiceHistory |\n"
        "    pops::runtime::program::kProgramServiceClock |\n"
        "    pops::runtime::program::kProgramServiceReduction |\n"
        "    pops::runtime::program::kProgramServiceTransaction |\n"
        "    pops::runtime::program::kProgramServicePersistentValues;\n"
        "}  // namespace\n"
    )


def _emit_amr_candidate_entry_prefix() -> str:
    return (
        "bool program_candidate_prepare(void* opaque,\n"
        "    const pops::runtime::program::ProgramHostDescriptor* host,\n"
        "    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {\n"
        "  using namespace pops::runtime::program;\n"
        "  if (opaque == nullptr || host == nullptr || diagnostic == nullptr) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,\n"
        "                                     \"AMR Program prepare requires state and host descriptors\");\n"
        "    return false;\n"
        "  }\n"
        "  if (!valid_program_host_descriptor(*host)) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,\n"
        "                                     \"AMR Program install received an invalid host descriptor\");\n"
        "    return false;\n"
        "  }\n"
        "  if (host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||\n"
        "      host->runtime_kind != ProgramRuntimeKind::amr ||\n"
        "      host->execution_lane != ProgramExecutionLane::host ||\n"
        "      host->services.state_store == nullptr) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::unsupported_runtime,\n"
        "                                     \"Program artifact requires the native AMR host runtime\");\n"
        "    return false;\n"
        "  }\n"
        "  auto* state = static_cast<ProgramCandidateState*>(opaque);\n"
        "  if (state->ctx_owner || state->step) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     \"AMR Program candidate was prepared twice\");\n"
        "    return false;\n"
        "  }\n"
        "  try {\n"
        "    state->ctx_owner = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(host->preparation);\n"
        "    auto ctx_owner = state->ctx_owner;\n"
        "    auto& ctx = *ctx_owner;\n"
    )


def _emit_amr_candidate_entry_suffix(maximum_bytes: int | None = None) -> str:
    # The candidate descriptor's maximum is a resource-plan authority, not the
    # size of its private callback state.  Runtime-sized plans carry the v5
    # unknown sentinel until the host seals the post-prepare layouts.
    candidate_maximum_bytes = (
        "pops::runtime::program::kProgramResourcePlanUnknownExtent"
        if maximum_bytes is None else "UINT64_C(%d)" % int(maximum_bytes)
    )
    return (
        "    return true;\n"
        "  } catch (const std::exception& exc) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     exc.what());\n"
        "    return false;\n"
        "  } catch (...) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     \"AMR Program candidate preparation failed\");\n"
        "    return false;\n"
        "  }\n"
        "}\n\n"
        'extern "C" bool pops_install_program(\n'
        "    const pops::runtime::program::ProgramHostDescriptor* host,\n"
        "    pops::runtime::program::ProgramCandidateDescriptor* candidate,\n"
        "    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {\n"
        "  using namespace pops::runtime::program;\n"
        "  if (host == nullptr || candidate == nullptr || diagnostic == nullptr ||\n"
        "      !valid_program_host_descriptor(*host) ||\n"
        "      host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||\n"
        "      host->runtime_kind != ProgramRuntimeKind::amr ||\n"
        "      host->execution_lane != ProgramExecutionLane::host) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,\n"
        "                                     \"AMR Program install received an invalid host descriptor\");\n"
        "    return false;\n"
        "  }\n"
        "  *candidate = {};\n"
        "  try {\n"
        "    auto state = std::make_unique<ProgramCandidateState>();\n"
        "    ProgramCandidateDescriptor descriptor{};\n"
        "    descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));\n"
        "    descriptor.abi_version = kProgramInstallAbiVersion;\n"
        "    descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);\n"
        "    descriptor.runtime_kind = ProgramRuntimeKind::amr;\n"
        "    descriptor.provided_capability_bits = kProgramCandidateCapabilities;\n"
        "    descriptor.required_capability_bits = kProgramCandidateCapabilities;\n"
        "    descriptor.required_service_bits = kProgramCandidateRequiredServices;\n"
        "    descriptor.program_name = {kProgramCandidateName,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidateName) - 1)};\n"
        "    descriptor.artifact_identity = {kProgramCandidateArtifactIdentity,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidateArtifactIdentity) - 1)};\n"
        "    descriptor.abi_key = {kProgramCandidateAbiKey,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidateAbiKey) - 1)};\n"
        "    descriptor.route_manifest = {kProgramCandidateRouteManifest,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidateRouteManifest) - 1)};\n"
        "    descriptor.boundary_manifest = {kProgramCandidateBoundaryManifest,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidateBoundaryManifest) - 1)};\n"
        "    descriptor.persistent_resource_manifest = {kProgramCandidatePersistentResourceManifest,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidatePersistentResourceManifest) - 1)};\n"
        "    descriptor.checkpoint_identity = {kProgramCandidateCheckpointIdentity,\n"
        "        static_cast<std::uint64_t>(sizeof(kProgramCandidateCheckpointIdentity) - 1)};\n"
        "    descriptor.blocks = kProgramCandidateBlocks;\n"
        "    descriptor.parameters = kProgramCandidateParameters;\n"
        "    descriptor.operator_authorities = kProgramCandidateOperatorAuthorities;\n"
        "    descriptor.history_authorities = kProgramCandidateHistoryAuthorities;\n"
        "    descriptor.checkpoint_shape = kProgramCandidateCheckpointShape;\n"
        "    descriptor.flux_budgets = kProgramCandidateFluxBudgets;\n"
        "    descriptor.flux_basis_occurrences = kProgramCandidateFluxBasisOccurrences;\n"
        "    descriptor.face_flux_stages = kProgramCandidateFaceFluxStages;\n"
        "    descriptor.resource_plan = kProgramCandidateResourcePlan;\n"
        "    descriptor.boundary_routes = kProgramCandidateBoundaryRoutes;\n"
        "    descriptor.provider_routes = kProgramCandidateProviderRoutes;\n"
        "    descriptor.module_operators = kProgramCandidateModuleOperators;\n"
        "    descriptor.module_state_spaces = kProgramCandidateModuleStateSpaces;\n"
        "    descriptor.module_field_spaces = kProgramCandidateModuleFieldSpaces;\n"
        "    descriptor.maximum_bytes = " + candidate_maximum_bytes + ";\n"
        "    descriptor.context = state.get();\n"
        "    descriptor.prepare = &program_candidate_prepare;\n"
        "    descriptor.step = &program_candidate_step;\n"
        "    descriptor.dt_bound = nullptr;\n"
        "    descriptor.hierarchy_refresh = &program_candidate_hierarchy_refresh;\n"
        "    descriptor.history_remap_accepted = &program_candidate_history_remap;\n"
        "    descriptor.restart_regrid_preflight = &program_candidate_restart_preflight;\n"
        "    descriptor.restart_regrid = &program_candidate_restart_regrid;\n"
        "    descriptor.restart_resync = &program_candidate_restart_resync;\n"
        "    descriptor.create_accepted_snapshot = &program_candidate_accepted_snapshot;\n"
        "    descriptor.destroy = &program_candidate_destroy;\n"
        "    if (!valid_program_candidate_descriptor(descriptor)) {\n"
        "      write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_candidate,\n"
        "                                       \"AMR Program artifact produced an invalid candidate descriptor\");\n"
        "      return false;\n"
        "    }\n"
        "    *candidate = descriptor;\n"
        "    (void)state.release();\n"
        "    return true;\n"
        "  } catch (...) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     \"AMR Program artifact installation failed\");\n"
        "    return false;\n"
        "  }\n"
        "}\n"
    )


def _emit_amr_install(
    program: Any,
    target: Any,
    prelude: Any,
    body: Any,
    hierarchy_bodies: Any = None,
    provider_plan_install: str = "",
    post_synchronization: Any = None,
    *,
    artifact_identity: str,
    route_manifest: str,
    program_name: str,
    maximum_bytes: int | None,
) -> str:
    """C++ source of the AMR install entry the .so exports (epic ADC-511 / ADC-508, Spec 6).

    ``target='system'`` emits NOTHING (a System-only .so carries only ``pops_install_program``).
    ``target='amr_system'`` emits ``pops_install_program``, the entry ``AmrSystem::install_program``
    resolves (it dlopens the .so, validates the ABI key + section-24 requirements, binds the blocks by
    name, seeds the runtime params, then calls this). Candidate preparation resolves the retained
    provider only from the host-owned preparation image; it never receives an ``AmrSystem`` facade.
    The retained provider then executes the recursively subcycled per-level macro-step driver: the
    IDENTICAL lowered ``{body}`` -- the common
    ``pops_install_program`` candidate lifecycle is used for both runtime kinds; the
    ordinary AMR shape is wrapped in an explicit level-clock scheduler.
    A bounded ``Program.cell_local_time`` shape keeps that same candidate lifecycle but assigns
    preparation and stepping to the provider's typed cell-temporal executor instead of nesting it
    under the ordinary global hierarchy scheduler.  The bounded shape is intentionally kept narrow
    so unsupported extensions are refused before materializing any source or artifact.
    Its install-time prelude is materialized once per native level, not once per hierarchy: each
    closure therefore owns fields/workspaces with the exact level layout. A topology-epoch or
    process-local materialization-generation change rematerializes the bundles before the next
    advance. The body references only the variable ``ctx``
    (never the provider type), so codegen has no concrete Uniform/AMR context dispatch.

    Shape: one macro-step recursively advances each child on its declared parent/child clock relation,
    with exact stage abscissae and mandatory temporal interpolation from parent old/new snapshots, then
    synchronizes finest-first by conservative reflux followed by average-down. Authored single-state
    field nodes use the exact point/provider-qualified solve at each active level. The services expose
    only an explicitly level-0-only default-field route for manual drivers, so coarse auxiliary
    injection can never masquerade as a requested fine-level solve. The C/F interface is now conservative to round-off: the
    per-level effective flux is captured through the Program's own linear combination and routed
    through the native ``route_reflux`` at level sync (ADC-639), so mass/momentum/energy are conserved
    across the interface on a genuinely multilevel run; a coarse-only / flat Program stays
    bit-identical."""
    cell_local_time = _require_bounded_cell_local_program(program, target, hierarchy_bodies)
    if target != "amr_system":
        return ""
    if not all(hasattr(prelude, name) for name in (
        "render_global_install", "render_level_install", "render_bindings"
    )):
        raise TypeError(
            "AMR installation requires explicit PreludeSections; a flat prelude would "
            "replay cold preparation through an accepted topology"
        )
    body = _rewrite_amr_public_exact_coefficient_calls(body)
    if hierarchy_bodies is not None:
        hierarchy_bodies = tuple(
            _rewrite_amr_public_exact_coefficient_calls(fragment)
            for fragment in hierarchy_bodies
        )
    candidate_support = _emit_amr_candidate_support(artifact_identity, route_manifest, program_name)
    candidate_entry_prefix = _emit_amr_candidate_entry_prefix()
    candidate_entry_suffix = _emit_amr_candidate_entry_suffix(maximum_bytes)
    global_install = prelude.render_global_install(4)
    level_install = prelude.render_level_install(6)
    bindings = prelude.render_bindings(4)
    # Cell-local time is a Program execution operation, not a second installation ABI.  Keep its
    # typed route preparation in the common candidate prepare callback and select the provider-owned
    # executor as the common candidate step.  In particular, do not return a separate source shape
    # here: every AMR Program, including the bounded Forward-Euler envelope, must pass through the
    # same v5 ``pops_install_program`` candidate lifecycle and diagnostics.
    cell_local_prepare = ""
    cell_local_step = ""
    cell_local_refresh = ""
    cell_local_accepted_snapshot = ""
    if cell_local_time is not None:
        cell_local_contract, cell_local_routes = cell_local_time
        clock_identity = json.dumps(program.clock.qualified_id)
        route_rows = ", ".join(
            "{%d, -1, %d}" % (program_block, rhs_id) for program_block, rhs_id in cell_local_routes
        )
        route_count = len(cell_local_routes)
        cell_local_prepare = (
            f"    ctx.configure_primary_clock({clock_identity});\n"
            "    static constexpr std::array<"
            "pops::runtime::program::SameLevelCellTemporalForwardEulerRoute, "
            f"{route_count}> cell_temporal_routes{{{{{route_rows}}}}};\n"
            "    ctx.prepare_same_level_cell_temporal_execution("
            f"{clock_identity}, {cell_local_contract.tick_denominator}, "
            f"{cell_local_contract.rung}, cell_temporal_routes);\n"
        )
        cell_local_step = (
            "    state->step = [ctx_owner = state->ctx_owner](double dt) {\n"
            "      ctx_owner->advance_same_level_cell_temporal(dt);\n"
            "    };\n"
        )
        # The executor owns topology requalification and accepted-state refresh.  The common AMR
        # candidate hook still needs a callable, but there is no per-level Program bundle to rebuild.
        cell_local_refresh = "    state->hierarchy_resource_refresh = []() {};\n"
        # Cell-local time still enters through the common v5 Program candidate.  It has no ordinary
        # level-resource bundle, but transaction/replacement/restart authority still requires the
        # accepted service image.  Capture the retained provider directly; do not route this through
        # a second installer or manufacture an empty level authority.
        cell_local_accepted_snapshot = (
            "    state->accepted_snapshot = [ctx_owner = state->ctx_owner]() {\n"
            "      return ctx_owner->create_accepted_context_snapshot();\n"
            "    };\n"
        )

    def walk(values: Any) -> Any:
        for value in values:
            yield value
            if value.op == "post_synchronization":
                continue
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

    post_sync_src = post_synchronization if post_synchronization else ""
    post_sync_field = (
        "    std::function<void(double)> post_synchronization;\n" if post_sync_src else ""
    )
    post_sync_initializer = (
        "      , [=](double dt) {\n"
        "        auto& ctx = *ctx_owner;\n"
        "        ctx.begin_step(dt);\n"
        "        ctx.set_stage_time(1, 1);\n"
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
        "        _active_level_authority(ctx)->programs.at(static_cast<std::size_t>(_k))"
        ".post_synchronization(dt);\n"
        "      });\n"
        "    }\n"
        if post_sync_src
        else ""
    )

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
    # Keep the ordinary phase variables defined even when the cell-local executor owns the whole
    # hierarchy traversal and therefore does not materialize a per-level Program bundle below.
    phase_fields = phase_initializers = installed_driver = ""
    if cell_local_time is not None:
        # ``advance_same_level_cell_temporal`` recursively traverses the prepared hierarchy itself.
        # It must not be nested under the ordinary per-level scheduler, whose global driver would
        # otherwise consume the cell-local accepted image with a global ``dt``.
        level_resources = ""
    elif hierarchy_bodies is None:
        phase_fields = "    std::function<void(double)> step;\n" + post_sync_field
        phase_initializers = (
            "      [=](double dt) {\n"
            "        auto& ctx = *ctx_owner;\n"
            "        (void)dt;\n" + body + "\n"
            "      }\n"
            + post_sync_initializer
        )
        installed_driver = (
            "    auto _advance_level = [&](double level_dt) {\n"
            "      _refresh_level_programs(false);\n"
            "      _active_level_authority(ctx)->programs.at(static_cast<std::size_t>(ctx.level())).step(level_dt);\n"
            "    };\n"
            "    ctx.advance_hierarchy(dt, _advance_level);\n"
            + post_sync_driver
        )
    else:
        gather, solve, publish = hierarchy_bodies
        phase_fields = (
            "    std::function<void(double)> step;\n"
            "    std::function<void(double)> gather;\n"
            "    std::function<void(double)> solve;\n"
            "    std::function<void(double)> publish;\n"
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
            + post_sync_initializer
        )
        installed_driver = (
            "    auto _advance_hierarchy = [&](double hierarchy_dt) {\n"
            "      _refresh_level_programs(false);\n"
            "      // The subcycling engine invokes this body once per level. The candidate tower\n"
            "      // is complete before the root callback, so gather/solve/publish run there once.\n"
            "      if (ctx.level() != 0)\n"
            "        return;\n"
            "      const int _nlev = ctx.program_resource_topology().levels;\n"
            "      if (ctx.uses_prepared_krylov_fallback()) {\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_resource_level(_k, [&]() {\n"
            "            _active_level_authority(ctx)->programs.at(static_cast<std::size_t>(_k)).step(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "      } else {\n"
            "        // Gather every level before the unique hierarchy-scoped solve.\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_resource_level(_k, [&]() {\n"
            "            _active_level_authority(ctx)->programs.at(static_cast<std::size_t>(_k)).gather(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "        ctx.with_program_resource_level(0, [&]() {\n"
            "          _active_level_authority(ctx)->programs.front().solve(hierarchy_dt);\n"
            "        });\n"
            "        // The composite solution is complete before any level reconstructs or commits.\n"
            "        for (int _k = 0; _k < _nlev; ++_k) {\n"
            "          ctx.with_program_resource_level(_k, [&]() {\n"
            "            _active_level_authority(ctx)->programs.at(static_cast<std::size_t>(_k)).publish(hierarchy_dt);\n"
            "          });\n"
            "        }\n"
            "      }\n"
            "    };\n"
            "    ctx.advance_synchronized_hierarchy(dt, _advance_hierarchy);\n"
            + post_sync_driver
        )

    # Every generated binding is layout-bound.  The install collector separates
    # candidate-only preparation from bindings, so requalification rebuilds a
    # lexically complete closure bundle without invoking any ``prepare_*`` API
    # through the accepted service image.
    # per level before the first advance, and rebuild the set exactly once after a topology epoch or
    # process-local materialization-generation change. This is deliberately generic: scalar scratch,
    # condensed coefficients, matrix-free
    # apply captures, prepared problems and Krylov workspaces all follow the same lifetime protocol.
    level_resources = (
        "  struct _PopsAmrLevelProgram {\n" + phase_fields + "  };\n"
        "  auto _make_level_program = [](auto ctx_owner, auto& ctx, bool prepare_resources) {\n"
        "    if (!ctx_owner)\n"
        "      throw std::logic_error(\"AMR Program level owner is empty\");\n"
        "    if (prepare_resources) {\n" + level_install + "\n"
        "    }\n"
        + bindings + "\n"
        "    return _PopsAmrLevelProgram{\n" + phase_initializers + "    };\n"
        "  };\n"
        "  // Keep the complete topology-bound level tower in one immutable authority.  Separating\n"
        "  // the vector from its epoch/generation made a failed forward rebuild observable as a\n"
        "  // mixed authority; Candidate now replaces this single image only after every level\n"
        "  // resource has been cold-prepared.\n"
        "  struct _PopsAmrLevelProgramAuthority final {\n"
        "    std::vector<_PopsAmrLevelProgram> programs;\n"
        "    std::uint64_t topology_epoch = std::numeric_limits<std::uint64_t>::max();\n"
        "    std::uint64_t materialization_generation = std::numeric_limits<std::uint64_t>::max();\n"
        "  };\n"
        "  struct _PopsAmrLevelProgramAuthoritySlot final {\n"
        "    std::shared_ptr<const _PopsAmrLevelProgramAuthority> active;\n"
        "  };\n"
        "  auto _level_program_authority_slot =\n"
        "      std::make_shared<_PopsAmrLevelProgramAuthoritySlot>();\n"
        "  auto _refresh_level_programs = [=](bool prepare_resources) {\n"
        "    auto& ctx = *ctx_owner;\n"
        "    const auto topology = ctx.program_resource_topology();\n"
        "    const std::uint64_t epoch = topology.epoch;\n"
        "    const std::uint64_t generation = topology.generation;\n"
        "    const int levels = topology.levels;\n"
        + transform_refresh_guard
        + "    const auto active_authority = _level_program_authority_slot->active;\n"
        "    if (active_authority && active_authority->topology_epoch == epoch &&\n"
        "        active_authority->materialization_generation == generation &&\n"
        "        active_authority->programs.size() == static_cast<std::size_t>(levels))\n"
        "      return;\n"
        "    auto next_level_authority = std::make_shared<_PopsAmrLevelProgramAuthority>();\n"
        "    next_level_authority->programs.reserve(static_cast<std::size_t>(levels));\n"
        "    ctx.for_each_program_resource_level([&](int) {\n"
        "      next_level_authority->programs.emplace_back(\n"
        "          _make_level_program(ctx_owner, ctx, prepare_resources));\n"
        "    });\n"
        "    next_level_authority->topology_epoch = epoch;\n"
        "    next_level_authority->materialization_generation = generation;\n"
        "    // Terminal no-throw publication: a failed candidate build leaves the entire\n"
        "    // accepted authority untouched; vector and generation can never diverge.\n"
        "    _level_program_authority_slot->active = std::move(next_level_authority);\n"
        "  };\n"
        "  _refresh_level_programs(true);\n"
        "  auto _active_level_authority = [=](const auto& active_ctx) {\n"
        "    const auto authority = _level_program_authority_slot->active;\n"
        "    const auto topology = active_ctx.program_resource_topology();\n"
        "    if (!authority || authority->topology_epoch != topology.epoch ||\n"
        "        authority->materialization_generation != topology.generation ||\n"
        "        authority->programs.size() != static_cast<std::size_t>(topology.levels))\n"
        "      throw std::logic_error(\"AMR Program level authority is absent or stale\");\n"
        "    return authority;\n"
        "  };\n"
        "  auto _build_forward_level_authority = [=](const auto& owner) {\n"
        "    auto next = std::make_shared<_PopsAmrLevelProgramAuthority>();\n"
        "    const auto topology = owner->program_resource_topology();\n"
        "    if (topology.levels <= 0)\n"
        "      throw std::logic_error(\"AMR Program forward authority has no levels\");\n"
        "    next->programs.reserve(static_cast<std::size_t>(topology.levels));\n"
        "    owner->for_each_program_resource_level([&](int) {\n"
        "      next->programs.emplace_back(_make_level_program(ctx_owner, *owner, true));\n"
        "    });\n"
        "    next->topology_epoch = topology.epoch;\n"
        "    next->materialization_generation = topology.generation;\n"
        "    return std::shared_ptr<const _PopsAmrLevelProgramAuthority>(std::move(next));\n"
        "  };\n"
        "  class _PopsAcceptedProgramExecutionServicesSnapshot final\n"
        "      : public pops::runtime::program::AcceptedProgramExecutionServicesSnapshot {\n"
        "   public:\n"
        "    using _Authority = _PopsAmrLevelProgramAuthority;\n"
        "    using _Slot = _PopsAmrLevelProgramAuthoritySlot;\n"
        "    using _Services = pops::runtime::program::ProgramExecutionServices<pops::kNativeDimension>;\n"
        "    using _Builder = std::function<std::shared_ptr<const _Authority>(std::shared_ptr<_Services>)>;\n"
        "    _PopsAcceptedProgramExecutionServicesSnapshot(\n"
        "        std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot> inner,\n"
        "        std::shared_ptr<_Services> owner, std::shared_ptr<_Slot> slot,\n"
        "        std::shared_ptr<const _Authority> accepted, _Builder builder)\n"
        "        : inner_(std::move(inner)), owner_(std::move(owner)), slot_(std::move(slot)),\n"
        "          accepted_(std::move(accepted)), builder_(std::move(builder)) {\n"
        "      if (!inner_ || !owner_ || !slot_ || !accepted_ || !builder_)\n"
        "        throw std::logic_error(\"AMR Program accepted bundle snapshot is incomplete\");\n"
        "    }\n"
        "    std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot>\n"
        "    prepare_restore() const override {\n"
        "      auto restored = inner_->prepare_restore();\n"
        "      if (!restored) throw std::logic_error(\"AMR Program accepted service restore is empty\");\n"
        "      return std::make_unique<_PopsAcceptedProgramExecutionServicesSnapshot>(\n"
        "          std::move(restored), owner_, slot_, staged_ ? staged_ : accepted_, builder_);\n"
        "    }\n"
        "    void refresh_from_owner_preallocated() override { inner_->refresh_from_owner_preallocated(); }\n"
        "    std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot>\n"
        "    detach_for_forward(std::uint64_t epoch, std::uint64_t generation, void*& token) const override {\n"
        "      auto detached = inner_->detach_for_forward(epoch, generation, token);\n"
        "      if (!detached) throw std::logic_error(\"AMR Program detached service image is empty\");\n"
        "      return std::make_unique<_PopsAcceptedProgramExecutionServicesSnapshot>(\n"
        "          std::move(detached), owner_, slot_, accepted_, builder_);\n"
        "    }\n"
        "    void rebind_after_forward_publish(void* token) noexcept override { inner_->rebind_after_forward_publish(token); }\n"
        "    void prepare_forward_hierarchy_refresh(std::uint64_t e, std::uint64_t g) override { inner_->prepare_forward_hierarchy_refresh(e, g); }\n"
        "    void prepare_forward_history_remap(const pops::runtime::program::AmrProgramHistoryRemapDescriptor& d) override { inner_->prepare_forward_history_remap(d); }\n"
        "    void prepare_forward_full_history_reseed(const pops::runtime::program::AmrProgramFullHistoryReseedDescriptor& d) override { inner_->prepare_forward_full_history_reseed(d); }\n"
        "    void prepare_forward_temporal_partition(const pops::runtime::program::PreparedForwardAmrTemporalAuthority& a) override { inner_->prepare_forward_temporal_partition(a); }\n"
        "    void prepare_forward_scratch_rematerialization(const pops::runtime::program::PreparedForwardAmrScratchTopology& t) override { inner_->prepare_forward_scratch_rematerialization(t); }\n"
        "    void prepare_forward_execution_bundle(const pops::runtime::program::PreparedForwardAmrExecutionAuthority& erased) override {\n"
        "      const auto* typed = dynamic_cast<const pops::runtime::program::PreparedForwardAmrExecutionAuthorityView<pops::kNativeDimension>*>(&erased);\n"
        "      if (typed == nullptr) throw std::logic_error(\"AMR Program forward bundle has the wrong dimension\");\n"
        "      inner_->prepare_forward_execution_bundle(erased);\n"
        "      staged_ = owner_->with_forward_execution_overlay(*typed, *inner_, [&](auto overlay) {\n"
        "        return builder_(std::move(overlay));\n"
        "      });\n"
        "    }\n"
        "    pops::runtime::program::PreparedForwardAmrAcceptedContext\n"
        "    prepare_forward_accepted_context(std::int64_t step) const override { return inner_->prepare_forward_accepted_context(step); }\n"
        "    void prime_at_bind() override { inner_->prime_at_bind(); }\n"
        "    void prime_copied_image_at_bind() override { inner_->prime_copied_image_at_bind(); }\n"
        "    void snapshot_transaction_diagnostics_noexcept() noexcept override { inner_->snapshot_transaction_diagnostics_noexcept(); }\n"
        "    void publish_transaction_diagnostics_noexcept() noexcept override { inner_->publish_transaction_diagnostics_noexcept(); }\n"
        "    void restore_transaction_diagnostics_noexcept() noexcept override { inner_->restore_transaction_diagnostics_noexcept(); }\n"
        "    void publish_restore() noexcept override {\n"
        "      inner_->publish_restore();\n"
        "      if (staged_) { using std::swap; swap(slot_->active, staged_); accepted_ = slot_->active; }\n"
        "    }\n"
        "   private:\n"
        "    std::unique_ptr<pops::runtime::program::AcceptedProgramExecutionServicesSnapshot> inner_;\n"
        "    std::shared_ptr<_Services> owner_;\n"
        "    std::shared_ptr<_Slot> slot_;\n"
        "    std::shared_ptr<const _Authority> accepted_;\n"
        "    std::shared_ptr<const _Authority> staged_;\n"
        "    _Builder builder_;\n"
        "  };\n"
        "  state->accepted_snapshot = [=]() {\n"
        "    const auto accepted = _level_program_authority_slot->active;\n"
        "    if (!accepted) throw std::logic_error(\"AMR Program has no initial level authority\");\n"
        "    return std::make_unique<_PopsAcceptedProgramExecutionServicesSnapshot>(\n"
        "        ctx_owner->create_accepted_context_snapshot(), ctx_owner,\n"
        "        _level_program_authority_slot, accepted, _build_forward_level_authority);\n"
        "  };\n"
    )
    if cell_local_time is not None:
        # The temporary ordinary bundle above is intentionally not emitted for this operation.  The
        # provider-owned CellTemporalExecutor allocates and refreshes its own topology-bound state.
        level_resources = ""
    return (
        "#include <pops/runtime/program/program_execution_services.hpp>  // common AMR execution service\n"
        "// AMR candidate entry (epic ADC-511 / ADC-508, Spec 6): the target='amr_system'\n"
        "// implementation of the common pops_install_program ABI. AmrSystem::install_program\n"
        "// resolves and calls it after binding blocks by name and seeding runtime parameters. The\n"
        "// shared factory selects the provider. Ordinary candidates install the parent/child clock\n"
        "// driver; a bounded cell-local candidate binds its typed executor step directly. The SAME\n"
        "// lowered body is recursively subcycled, temporally interpolated and conservatively synced.\n"
        + candidate_support
        + candidate_entry_prefix
        + (provider_plan_install + "\n" if provider_plan_install else "")
        + (global_install + "\n" if global_install else "")
        + cell_local_prepare
        + transform_guard
        + level_resources
        + cell_local_accepted_snapshot
        + (cell_local_refresh or
           "\n    state->hierarchy_resource_refresh = [=]() { _refresh_level_programs(false); };\n")
        + (cell_local_step or (
            "    state->step = [=](double dt) {\n"
            "    auto& ctx = *ctx_owner;\n"
            "    _refresh_level_programs(false);\n"
            + installed_driver
            + "    };\n"
        ))
        + candidate_entry_suffix
    )
