"""pops.codegen.program_codegen -- C++ emission for a pops.time.Program.

FREE FUNCTIONS taking the ``program`` (and an optional physical ``model``), mirroring
``pops.codegen.module_codegen`` for models. ``emit_cpp_program(program, model=None)`` lowers
the Program SSA IR to the C++ source of a ``problem.so`` (the stable .so ABI installed by
``System::install_program``); the ``_emit_*`` / ``_check_*`` helpers and the per-cell kernel
emitters are the lowering machinery. This module exclusively owns compiler materialization;
``pops.time.Program`` remains an inert authoring and IR object.

This is the THIN public module of the program emitter.  The lowering machinery is split
across sibling modules so each file fits the Spec-4 size budget, and every name re-imported
below so the public surface of ``pops.codegen.program_codegen`` is unchanged:

  - ``program_emit_kernels``       -- op tables, text helpers, model-free per-cell kernels,
                                      the ``_PROGRAM_CPP_TEMPLATE``;
  - ``program_emit_model_kernels`` -- the model-coefficient per-cell kernels;
  - ``program_emit_solve``         -- the matrix-free Krylov + condensed-Schur emitters;
  - ``program_emit_schedule``      -- the unified scheduler wrap (ADC-458);
  - ``program_emit_control``       -- the body walk + control-flow (while/range/if) emitters;
  - ``program_emit_ops``           -- the per-op ``_emit_op`` dispatcher;
  - ``program_emit_amr``           -- the AMR install-entry emitter (target='amr_system', ADC-508).
  - ``program_metadata``           -- candidate-table module records.

The orchestration (``emit_cpp_program`` + candidate-table lowering checks) stays here.

The emission-only tables (_MODEL_OPS / _ALLOWED_OPS / _PROFILE_SKIP_OPS / _AUX_OUTPUT_OPS)
are module-level constants in ``program_emit_kernels``, re-exported here.
"""

from __future__ import annotations

from typing import Any
from pops.time.references import block_name

import json  # noqa: F401  (kept for any external reference to program_codegen.json)

# Re-export every moved name so the public surface of this module is unchanged.
from pops.codegen.program_emit_kernels import (  # noqa: F401
    _ALLOWED_OPS,
    _AUX_OUTPUT_OPS,
    _MODEL_OPS,
    _PROFILE_SKIP_OPS,
    _PROGRAM_CPP_TEMPLATE,
    _block_inverse_include,
    _prepared_native_component_includes,
    ProgramValue,
    _apply_in_arg,
    _cell_locals,
    _coeff_cpp,
    _has_runtime_param,
    _deref,
    _emit_cell_compare_kernel,
    _emit_field_combine,
    _emit_where_kernel,
    _kernel_close,
    _kernel_open,
    _model_impl,
    _named_fluxes,
    _to_affine,
)
from pops.codegen.program_emit_model_kernels import (  # noqa: F401
    _emit_apply_kernel,
    _emit_coupled_rate_kernel,
    _emit_flux_kernel,
    _emit_local_transform_kernel,
    _emit_residual_eval,
    _emit_solve_local_linear_kernel,
    _emit_solve_local_nonlinear_kernel,
    _emit_solve_coupled_implicit_kernel,
    _emit_source_kernel,
    _linear_source_rows,
    _residual_term_exprs,
)
from pops.codegen.program_emit_solve import (  # noqa: F401
    _emit_matrix_free_operator,
    _emit_solve_linear,
    _validate_matrix_free_contract,
)
from pops.codegen.program_emit_schedule import (  # noqa: F401
    _emit_schedule_wrap,
    _schedule_due_test,
    _split_output_decl,
)
from pops.codegen.program_emit_control import (  # noqa: F401
    _coupled_rate_components,
    _emit_amr_hierarchy_bodies,
    _emit_body,
    _emit_branch,
    _emit_range,
    _emit_while,
    _walk_expr,
)
from pops.codegen.program_emit_ops import _emit_op  # noqa: F401
from pops.codegen.program_emit_params import program_param_entries as _program_param_entries
from pops.codegen.program_emit_amr import (  # noqa: F401
    _emit_amr_install,
    _validate_hierarchy_scoped_solve_barrier,
)
from pops.codegen.program_metadata import program_module_records as _program_module_records
from pops.codegen.program_persistent_plan import (
    ProgramResourcePlan,
    get_program_resource_plan,
)
from pops.codegen.temporal_manifest import render_temporal_manifest
from pops.codegen._compile_emit import _emit_route_manifest  # noqa: F401 (ADC-599 embedded manifest)
from pops.codegen.program_lowerability import (
    all_ops as _all_ops,
    check_model_owner_dispatch as _check_model_owner_dispatch,
    check_schedules_lowerable as _check_schedules_lowerable,
)
from pops.codegen.program_models import ProgramModelGraph, model_for_node


# --- Program -> C++ lowering (free functions taking `program`) ------------------------------
# --- C++ codegen (Phase 2c-ii / Phase 4b): lower the IR to a problem.so source ---
def emit_cpp_program(
    program: Any,
    model: Any = None,
    target: str = "system",
    *,
    model_graph: Any = None,
    field_plans: Any = None,
    balance_due_contract: Any = None,
) -> str:
    """Lower the public low-level Program route without privileged resolve evidence."""
    return _emit_cpp_program_impl(
        program,
        model=model,
        target=target,
        model_graph=model_graph,
        field_plans=field_plans,
        balance_due_contract=balance_due_contract,
        has_shared_interface_implicit_jacvec=False,
    )


def _emit_resolved_cpp_program(
    program: Any,
    model: Any = None,
    target: str = "system",
    *,
    model_graph: Any = None,
    field_plans: Any = None,
    balance_due_contract: Any = None,
    shared_interface_codegen_evidence: Any,
) -> str:
    """Lower the private resolve-authenticated shared-interface route."""
    from pops.codegen._shared_interface_evidence import (
        _ResolvedSharedInterfaceCodegenEvidence,
    )

    if type(shared_interface_codegen_evidence) is not _ResolvedSharedInterfaceCodegenEvidence:
        raise TypeError(
            "resolved shared-interface lowering requires exact nominal codegen evidence"
        )
    shared_interface_codegen_evidence.require(program, target=target)
    return _emit_cpp_program_impl(
        program,
        model=model,
        target=target,
        model_graph=model_graph,
        field_plans=field_plans,
        balance_due_contract=balance_due_contract,
        has_shared_interface_implicit_jacvec=True,
    )


def _validate_resource_slot_calls(source: str, plan: ProgramResourcePlan) -> None:
    """Validate that every generated resource call already contains a dense slot.

    Resource slots are resolved by each emitter through ``persistent_slot_token``.  Keeping a
    source-level guard is useful for shared/control emitters: an accidentally retained placeholder,
    a value-id spelling, or an arbitrary expression must fail before an artifact is returned rather
    than being repaired by a lossy post-lowering map.
    """

    if not source:
        return
    import re

    api = (
        "prepare_rhs_scratch|prepare_state_scratch|prepare_scalar_scratch|prepare_cache_slot|"
        "rhs_scratch|scratch_state|scalar_scratch|"
        "cache_effective_dt|cache_store_scratch|cache_accumulate_dt|"
        "cache_restore_scratch|schedule_is_due|schedule_decision"
    )
    allowed_slots = {row.slot for row in plan.entries}
    def validate_token(token: str, *, operation: str) -> None:
        if "__POPS_PERSISTENT_SLOT_" in token:
            raise ValueError(
                "unresolved Program resource slot placeholder remains in generated source"
            )
        if not token.isdigit():
            raise ValueError(
                "Program resource %s must contain a compile-time dense slot, got %r"
                % (operation, token)
            )
        slot = int(token)
        if slot not in allowed_slots:
            raise ValueError(
                "Program resource %s uses an absent/legacy resource value id %d"
                % (operation, slot)
            )

    pattern = re.compile(r"ctx\.(?:%s)\(\s*([^,\s\)]+)" % api)
    for match in pattern.finditer(source):
        validate_token(match.group(1), operation="call")

    # The field-route solve keeps its point as the first argument.  Its second
    # argument is the same dense resource slot validated above; inspect it
    # separately so an SSA/value-id spelling cannot survive in this hot path.
    field_route_pattern = re.compile(
        r"ctx\.solve_fields_from_blocks_at\(\s*[^,\s\)]+\s*,\s*([^,\s\)]+)"
    )
    for match in field_route_pattern.finditer(source):
        validate_token(match.group(1), operation="field-route solve")


def _emit_cpp_program_impl(
    program: Any,
    model: Any = None,
    target: str = "system",
    *,
    model_graph: Any = None,
    field_plans: Any = None,
    balance_due_contract: Any = None,
    has_shared_interface_implicit_jacvec: bool,
) -> str:
    """Generate the C++ source of a problem.so implementing this Program (codegen).

    Exports the stable v5 install entry, its candidate descriptor and immutable tables, then prepares
    the macro step as a closure over the provider retained by the host-owned preparation image (no
    facade pointer and no MultiFab / flux / solver reimplementation).

    @p target selects the install entry the .so exports. ``"system"`` (default) emits only
    ``pops_install_program`` (the single-level ``System`` macro-step closure). ``"amr_system"``
    (epic ADC-511 / ADC-508, Spec 6) emits only ``pops_install_program``, the entry
    ``AmrSystem::install_program`` resolves: the shared preparation-image factory selects the AMR
    storage/topology adapter and the candidate owns the real per-level synchronized macro-step. A
    mismatched runtime is refused by the descriptor before preparation; no dead closure is compiled
    against a facade.

    Lowers the Program by a topological walk of the SSA IR: each block's current state is its base
    (``ctx.state(idx)``); each field node runs its exact point/provider-qualified solve; each RHS
    becomes a scratch + ``rhs_into``; each intermediate ``linear_combine`` becomes a zero scratch
    accumulated with ``axpy``; the committed combine writes the block state via ``lincomb``.
    Forward Euler, SSPRK2/SSPRK3 and RK4 all lower this way -- no per-scheme class.

    Multi-block (ADC-426): N typed ``T.state(block[U])`` declarations + N ``T.commit``
    are lowered -- each op routes to its own block's runtime index (``_block_indices``, in the order
    the blocks are first declared via ``T.state``). The candidate descriptor carries block NAMES in
    that order; ``System::install_program`` binds them to the instantiated System blocks BY NAME
    (Spec 3 criterion 23, ADC-457), so the
    System blocks (through the private ``sim.add_equation`` install seam) may be added in ANY
    order -- a Program
    block whose name has no instantiated System block fails loud (``Program requires block instance
    '<name>', but simulation did not instantiate it``). A block declared but never committed is a
    READ-ONLY block (allowed; e.g. a passive field whose charge couples the others through the shared
    Poisson). A commit of a block no ``T.state`` declares is rejected. A single-block Program lowers
    byte-identically (its one block is index 0; an order-matching multi-block Program too -- the
    name map is the identity).

    Phase-4b also lowers the SPLIT-SOURCE / LOCAL-LINEAR ops -- ``source`` (a named ``m.source_term``
    evaluated per cell), ``apply`` (LU for a named ``m.linear_source``) and ``solve_local_linear``
    ((I -/+ a*L) U = rhs solved cell by cell via a dense per-cell inverse) -- but ONLY when the
    physical ``model`` (the ``pops.dsl`` model whose ``source_term`` / ``linear_source`` they name)
    is provided: the codegen reads the model's symbolic coefficients to emit the per-cell kernels.
    Without ``model`` those ops raise NotImplementedError (the Program cannot be lowered in
    isolation); ``model=None`` still lowers FE / SSPRK / RK4 (no model needed). A ``rhs`` routes its
    base on its ``flux`` flag and whether ``"default"`` is among the requested ``sources`` (ADC-425 /
    ADC-430, spec criterion 17 -- flux and sources are explicit, never summed implicitly). With
    ``flux=True``: ``"default"`` present -> ``ctx.rhs_into`` (= ``-div F`` + the model's
    default/composite source, the historical path); ``"default"`` absent (incl. the empty list
    ``[]``) -> ``ctx.neg_div_flux_default_into`` (= ``-div F`` only, NO default source). With
    ``flux=False`` (SOURCE-ONLY, ADC-430): NO ``-div F`` base -- ``"default"`` present (or ``None``)
    -> ``ctx.source_default_into`` (= S only, the exact mirror); ``"default"`` absent -> the zeroed
    scratch (the named sources, if any, are the whole RHS). Each NAMED source (``sources=[...]``
    beyond ``"default"``) then lowers with a model: the same per-cell ``m.source_term`` kernel as the
    standalone ``source`` op, accumulated onto ``R`` via ``axpy``. So ``flux=True,sources=[]`` is flux
    only, ``flux=True,sources=["default"]`` is flux + default source (unchanged),
    ``flux=False,sources=["default"]`` is the default source only, ``flux=False,sources=["s"]`` is
    just ``s`` -- the named ones never double-count the default (it is folded in iff "default" was
    listed). More than one block now lowers (ADC-426): each op routes to its block's runtime index
    (``_block_indices``, in T.state declaration order) and control flow (while/range/if) inside a
    block lowers per block; a SIMULTANEOUS multi-target coupled field solve
    (``solve_fields_from_blocks([Ua, Ub])``) lowers to
    ``ctx.solve_fields_from_blocks_at(point, resource_slot, <pack>)`` (see below), after an
    install-time ``ctx.prepare_generated_field_route(resource_slot, field, {program_blocks})``.

    Each ``solve_fields(state=...)`` op lowers to the owner-qualified
    ``ctx.solve_fields_from_state_at(point, field, idx, <stage state>)`` route (ADC-409/ADC-759):
    the exact provider is re-solved at the active hierarchy level and logical stage time from THAT
    stage's state, not the block's current state. So a field-coupled multi-stage scheme (Poisson
    feedback into the flux) is exact: stage k's RHS reads phi solved from stage k's own state. For
    the first stage the stage state is U^n, so this is identical to the historical
    ``solve_fields()``; for an uncoupled model the field solve is inert either way. This is already a
    COUPLED multi-block solve:
    the system Poisson RHS is ``Sum_s elliptic_rhs_s(U_s)`` (``assemble_poisson_rhs``), so block
    ``idx`` reads its stage state while every OTHER block contributes its LIVE state into the one
    shared phi/aux. A per-block callable field operator therefore sees all blocks' charge. A
    SIMULTANEOUS multi-target override (several blocks at their stage states in ONE solve) lowers to
    ``ctx.solve_fields_from_blocks_at(point, resource_slot, <pack>)`` (Spec 3 criterion 24,
    ADC-457/ADC-759): the RHS is
    ``Sum_s elliptic_rhs_s(U_s)`` reading EVERY listed block's stage state at once
    (``assemble_poisson_rhs_from_blocks``), each slotted at its block index (nullptr = the block's
    live state) -- the coupled multi-species field solve."""
    if model is not None and model_graph is not None:
        raise TypeError("emit_cpp_program received competing model and model_graph authorities")
    if model_graph is not None and type(model_graph) is not ProgramModelGraph:
        raise TypeError("model_graph must be an exact ProgramModelGraph")
    authority = model_graph if model_graph is not None else model
    if target not in ("system", "amr_system"):
        raise ValueError("emit_cpp_program: target 'system' | 'amr_system' (got %r)" % (target,))
    if type(has_shared_interface_implicit_jacvec) is not bool:
        raise TypeError(
            "emit_cpp_program shared-interface implicit-JVP evidence must be an exact bool"
        )
    from pops._balance_due_contract import BalanceDueContract
    if balance_due_contract is None:
        balance_due_contract = BalanceDueContract.from_consumer_graph(None)
    if type(balance_due_contract) is not BalanceDueContract:
        raise TypeError(
            "emit_cpp_program balance_due_contract must be an exact BalanceDueContract"
        )
    program.validate()
    # The region boundary is a structural lowering contract, not a resource-shape diagnostic. A
    # nested hierarchy solve must be refused before persistent-plan lowering can inspect its opaque
    # state and report a secondary component-count error.
    _validate_hierarchy_scoped_solve_barrier(program, target)
    # Seal the complete persistent-resource table before any lowering emits
    # source. This is also where schedule policy, transfer-provider, exact byte
    # and memory-bound refusals happen, so invalid plans cannot leave an artifact.
    resource_plan = get_program_resource_plan(program, target=target)
    # The dedicated AMR install emitter validates this contract too, but it
    # must be rejected for a uniform target before ordinary body materialization.
    from pops.codegen.program_emit_amr import _require_bounded_cell_local_program

    _require_bounded_cell_local_program(program, target, None)
    _check_lowerable(program, authority, field_plans or {}, target=target)
    from pops.codegen.program_emit_kernels import ProgramProviderPlans

    provider_plans = ProgramProviderPlans()
    prelude_sections, body, post_synchronization, operator_authorities = _emit_body(
        program,
        authority,
        target=target,
        field_plans=field_plans or {},
        balance_due_contract=balance_due_contract,
        has_shared_interface_implicit_jacvec=has_shared_interface_implicit_jacvec,
        provider_plans=provider_plans,
    )
    _validate_resource_slot_calls(prelude_sections.render_uniform(), resource_plan)
    _validate_resource_slot_calls(body, resource_plan)
    persistent_manifest = render_temporal_manifest(
        program, target=target, plan=resource_plan)
    from pops.codegen.program_emit_field_boundaries import emit_field_boundaries

    field_boundary_helpers = emit_field_boundaries(
        program, authority, field_plans or {}, target)
    if field_boundary_helpers:
        prelude_sections.prepend_global_install(
            "program_candidate_prepare_field_boundaries(ctx);"
        )
    prelude = prelude_sections.render_uniform()
    from pops.runtime.routes import route_registry_signature

    amr_body = (
        _emit_amr_hierarchy_bodies(
            program,
            authority,
            field_plans or {},
            has_shared_interface_implicit_jacvec=has_shared_interface_implicit_jacvec,
            provider_plans=provider_plans,
        )
        if target == "amr_system"
        else None
    )
    if amr_body:
        # Hierarchy lowering returns one source fragment per gather/solve/publish
        # phase. Validate every phase before the AMR install template combines
        # them; treating the tuple as one string would defer the refusal to
        # template formatting and leave a legacy value id in one phase.
        for fragment in amr_body:
            _validate_resource_slot_calls(fragment, resource_plan)
    amr_install = _emit_amr_install(
        program,
        target,
        prelude_sections,
        body,
        amr_body,
        provider_plans.cpp_install(target),
        post_synchronization,
        artifact_identity=program._ir_hash(),
        route_manifest=route_registry_signature(),
        program_name=program.name,
        # Host-only prepared carriers are runtime-shaped even when generated value rows are
        # exact, so their aggregate cannot be a DSO descriptor ceiling.
        maximum_bytes=None,
    )
    # Keep AMR's dedicated install emitter as the source of its lifecycle hooks,
    # while replacing its fixed marker with the authenticated manifest payload.
    if amr_install:
        amr_install = amr_install.replace(
            '"pops.persistent-resource.manifest.v1"',
            json.dumps(persistent_manifest),
        )
    return _PROGRAM_CPP_TEMPLATE.format(
        name=json.dumps(program.name),
        hash=program._ir_hash(),
        candidate_tables=_emit_candidate_tables(
            program,
            authority,
            operator_authorities,
            route_registry_signature(),
            target,
            plan=resource_plan,
        ),
        prelude=prelude,
        body=body,
        model_helpers=_emit_program_model_helpers(program, authority) + field_boundary_helpers,
        system_install=_emit_system_install(
            target,
            prelude,
            body,
            provider_plans.cpp_install(target),
            artifact_identity=program._ir_hash(),
            route_manifest=route_registry_signature(),
            program_name=program.name,
            persistent_manifest=persistent_manifest,
            # Generated rows retain their exact manifest; the descriptor remains symbolic until
            # host preparation materializes every detached carrier family.
            maximum_bytes=None,
        ),
        prepared_native_component_includes=_prepared_native_component_includes(program),
        block_inverse_include=_block_inverse_include(program),
        amr_install=amr_install,
    )


def _program_checkpoint_rows(program: Any, block_indices: dict[Any, int]) -> list[tuple[Any, ...]]:
    """Return the exact checkpoint rows consumed by the detached native prelude.

    The POD checkpoint table and ``ctx.register_history`` describe one authority.  In particular,
    the table must not replace the authored clock, state or interpolation identities with historical
    capacity-only placeholders: the host compares these rows with the detached accepted image before
    publishing the Program.
    """
    history_manifest = {
        row["name"]: row for row in program.temporal_manifest()["histories"]
    }
    histories_ncomp = getattr(program, "_histories_ncomp", {})
    rows = []
    for name, lag in sorted(getattr(program, "_histories", {}).items()):
        owner = getattr(program, "_history_blocks", {}).get(name)
        owner_index = block_indices.get(owner, -1)
        state_ref = getattr(program, "_history_state_refs", {}).get(name)
        state_identity = (
            state_ref.qualified_id
            if state_ref is not None
            else "scalar-history:" + str(name)
        )
        space = getattr(program, "_history_spaces", {}).get(name)
        space_identity = (
            json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
            if space is not None
            else "scalar-field"
        )
        manifest = history_manifest[str(name)]
        interpolation = json.dumps(
            manifest["interpolation"], sort_keys=True, separators=(",", ":")
        )
        ncomp = histories_ncomp.get(name)
        rows.append(
            (
                str(name),
                str(state_identity),
                str(space_identity),
                str(manifest["clock"]),
                interpolation,
                int(owner_index),
                -1 if ncomp is None else int(ncomp),
                int(lag) + 1,
            )
        )
    return rows


def _emit_candidate_tables(
    program: Any,
    authority: Any,
    operator_authorities: tuple[tuple[int, ...], ...],
    route_manifest: str,
    target: str,
    *,
    plan: ProgramResourcePlan | None = None,
) -> str:
    """Emit the complete v5 POD metadata carried by ``ProgramCandidateDescriptor``.

    This replaces the fan-out of metadata ``dlsym`` calls.  Tables are intentionally immutable
    arrays of ABI PODs: the host copies them while the DSO remains resident, validates every
    string and only then invokes the candidate prepare callback.
    """
    if plan is None:
        plan = get_program_resource_plan(program, target=target)
    if type(plan) is not ProgramResourcePlan:
        raise TypeError("candidate tables require an exact ProgramResourcePlan")
    block_indices = program._block_indices()
    blocks = [block_name(block) for block in sorted(block_indices, key=block_indices.get)]
    params = _program_param_entries(program, authority)
    histories = [
        (str(name), int(depth))
        for name, (depth, policy) in sorted((getattr(program, "_history_persistence", None) or {}).items())
        if not policy.degenerate_to_dense(depth)
    ]
    checkpoints = _program_checkpoint_rows(program, block_indices)
    # A fixed row per block makes zero an explicit bound.  AMR carries the exact frozen-IR basis
    # and coefficient limits in the candidate, rather than exposing one accessor per bound.
    if target == "amr_system":
        from pops.codegen.program_flux_lowering import flux_table_budgets, lower_amr_flux_tables
        if program.cell_local_time_contract() is not None:
            # ExactFace is consumed by the cell-temporal executor, not the ordinary two-table
            # materializer.  Do not declare a non-zero host budget without a final consumer.
            flux = [(0, 0, 0, 0) for _ in blocks]
            flux_basis_occurrences, face_flux_stages = (), ()
        else:
            flux_basis_occurrences, face_flux_stages = lower_amr_flux_tables(program, plan)
            flux = [(*bounds, 0, 0) for bounds in flux_table_budgets(
                len(blocks), flux_basis_occurrences, face_flux_stages
            )]
    else:
        flux = [(0, 0, 0, 0) for _ in blocks]
        flux_basis_occurrences = ()
        face_flux_stages = ()
    resource = plan.abi_rows()
    boundary_routes = [(route_manifest, "boundary-manifest", 0)]
    provider_routes = [(route_manifest, "provider-manifest", 0)]
    module_operators, module_states, module_fields = _program_module_records(program, authority)

    lines = ["namespace {", "// v5 Program candidate POD metadata; no auxiliary Program export."]
    lines.append(
        "static constexpr char kProgramCandidateResourcePlanSchema[] = %s;"
        % json.dumps(plan.schema)
    )
    lines.append(
        "static constexpr char kProgramCandidateResourcePlanDigest[] = %s;"
        % json.dumps(plan.digest)
    )
    # Keep a lossless, deterministic copy beside the native POD table.  The
    # native record is intentionally compact, while this authenticated payload
    # preserves the complete key/path and typed policy contract for hosts that
    # materialize the richer ProgramResourcePlanRecord version.
    lines.append(
        "static constexpr char kProgramCandidateResourcePlanJson[] = %s;"
        % json.dumps(json.dumps(plan.abi_data(), sort_keys=True, separators=(",", ":")))
    )
    if resource:
        lines.append(
            "static constexpr std::uint32_t kProgramCandidateResourcePlanSlots[] = {%s};"
            % ", ".join(str(row.slot) for row in resource)
        )
    else:
        lines.append(
            "static constexpr std::uint32_t kProgramCandidateResourcePlanSlots[] = {0};"
        )
    lines.append(
        "static constexpr std::uint64_t kProgramCandidateResourcePlanCount = %d;"
        % len(resource)
    )
    def u64(value: int | None) -> str:
        # ``None`` is a declaration-time unknown, not a byte count.  Keep it
        # explicit in the v5 POD image so the host can seal the exact ceiling
        # only after prepare_* has exposed all layouts.
        if value is None:
            return "pops::runtime::program::kProgramResourcePlanUnknownExtent"
        return "UINT64_C(%d)" % int(value)

    lines.append(
        "static constexpr std::uint64_t kProgramCandidateResourcePlanMaximumBytes = %s;"
        % u64(plan.maximum_bytes)
    )
    serial = 0

    def literal(value: str) -> str:
        nonlocal serial
        name = f"kProgramCandidateString{serial}"
        serial += 1
        lines.append(f"static constexpr char {name}[] = {json.dumps(value)};")
        return "{" + name + ", static_cast<std::uint64_t>(sizeof(" + name + ") - 1)}"

    def table(name: str, record_type: str, rows: list[str]) -> None:
        if not rows:
            lines.append(f"static constexpr pops::runtime::program::ProgramAbiTable {name}{{}};")
            return
        values = ",\n    ".join(rows)
        array = name + "Rows"
        lines.append(f"static constexpr pops::runtime::program::{record_type} {array}[] = {{\n    {values}\n}};")
        lines.append(
            f"static constexpr pops::runtime::program::ProgramAbiTable {name}{{{array}, "
            f"static_cast<std::uint64_t>(sizeof({array}) / sizeof({array}[0])), "
            f"static_cast<std::uint64_t>(sizeof({array}[0]))}};"
        )

    table("kProgramCandidateBlocks", "ProgramBlockRecord", ["{" + literal(name) + "}" for name in blocks])
    table(
        "kProgramCandidateParameters", "ProgramParameterRecord",
        ["{%d, %d, %.17g, %s}" % (block, index, default, literal(str(name)))
         for block, name, index, default in params],
    )
    table(
        "kProgramCandidateOperatorAuthorities", "ProgramAuthorityRecord",
        ["{{" + ", ".join("UINT64_C(%d)" % int(word) for word in words) + "}}"
         for words in operator_authorities],
    )
    table(
        "kProgramCandidateHistoryAuthorities", "ProgramHistoryAuthorityRecord",
        ["{%s, %d, 0}" % (literal(identity), depth) for identity, depth in histories],
    )
    table(
        "kProgramCandidateCheckpointShape", "ProgramCheckpointRecord",
        ["{%s, %s, %s, %s, %s, %d, %d, UINT64_C(%d)}" % (
            literal(identity), literal(owner), literal(space), literal(clock), literal(transfer),
            block, components, retained,
        ) for identity, owner, space, clock, transfer, block, components, retained in checkpoints],
    )
    table(
        "kProgramCandidateFluxBudgets", "ProgramFluxBudgetRecord",
        ["{UINT64_C(%d), UINT64_C(%d), UINT64_C(%d), UINT64_C(%d)}" % row for row in flux],
    )
    table(
        "kProgramCandidateFluxBasisOccurrences", "ProgramFluxBasisOccurrenceRecord",
        ["{sizeof(pops::runtime::program::ProgramFluxBasisOccurrenceRecord), "
         "pops::runtime::program::kProgramFluxBasisOccurrenceSchemaVersion, "
         "%d, %d, %d, %d, %d, %d, INT64_C(%d), INT64_C(%d), %s, %s, %s, %s}" % (
             slot, expression_slot, block, level, rhs_identity, provider, stage_numerator,
             stage_denominator, literal(identity), literal(occurrence_path), literal(owner), literal(clock),
         ) for slot, expression_slot, block, level, rhs_identity, provider, stage_numerator,
         stage_denominator, identity, occurrence_path, owner, clock in flux_basis_occurrences],
    )
    table(
        "kProgramCandidateFaceFluxStages", "ProgramFaceFluxStageRecord",
        ["{sizeof(pops::runtime::program::ProgramFaceFluxStageRecord), "
         "pops::runtime::program::kProgramFaceFluxStageSchemaVersion, %d, %d, %d, "
         "%d, INT64_C(%d), INT64_C(%d), %s, %s, %s, %s}" % (
             slot, basis_slot, expression_slot, dt_power, coefficient_numerator,
             coefficient_denominator,
             literal(identity), literal(occurrence_path), literal(owner), literal(clock),
         ) for slot, basis_slot, expression_slot, dt_power, coefficient_numerator,
         coefficient_denominator, identity,
         occurrence_path, owner, clock in face_flux_stages],
    )
    table(
        "kProgramCandidateResourcePlan", "ProgramResourcePlanRecord",
        ["{static_cast<std::uint32_t>(sizeof(pops::runtime::program::ProgramResourcePlanRecord)), "
         "pops::runtime::program::kProgramResourcePlanSchemaVersion, %d, %d, UINT64_C(%d), UINT64_C(%d), %d, "
         "%d, %d, 0, %s, %s, %s, %s, "
         "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s}" % (
            row.slot,
            (1 if row.lifetime in {"persistent", "persistent_schedule"} else 0)
            | (2 if row.communicates or row.communication != "none" else 0)
            | (4 if row.restart_required else 0)
            | (8 if row.cells is not None else 0)
            | (16 if row.itemsize is not None else 0)
            | (32 if row.runtime_sized else 0),
            row.key.value_id, row.key.occurrence_path_id,
            -1 if row.key.level is None else row.key.level,
            row.components, row.ghosts,
            u64(row.bytes), u64(row.maximum_bytes),
            u64(row.cells), u64(row.itemsize),
            literal(row.schema), literal(row.plan_digest), literal(row.identity),
            literal(row.key.occurrence_path), literal(row.key.owner), literal(row.key.space),
            literal(row.key.clock), literal(row.lifetime), literal(row.centering),
            literal(row.off_policy), literal(row.communication), literal(row.transfer_provider),
            literal(row.restart_provider),
            literal(json.dumps(list(row.component_names), separators=(",", ":"))),
            literal(json.dumps(list(row.shape), separators=(",", ":"))),
            "pops::runtime::program::ProgramResourcePlanType::runtime_sized"
            if row.runtime_sized
            else "pops::runtime::program::ProgramResourcePlanType::exact",
        ) for row in resource],
    )
    for name, rows in (("kProgramCandidateBoundaryRoutes", boundary_routes),
                       ("kProgramCandidateProviderRoutes", provider_routes)):
        table(name, "ProgramRouteRecord", ["{%s, %s, UINT64_C(%d)}" % (
            literal(identity), literal(kind), capabilities) for identity, kind, capabilities in rows])
    for name, rows in (("kProgramCandidateModuleOperators", module_operators),
                       ("kProgramCandidateModuleStateSpaces", module_states),
                       ("kProgramCandidateModuleFieldSpaces", module_fields)):
        table(name, "ProgramModuleRecord", ["{%s, %s, %s, %s, %s}" % (
            literal(identity), literal(kind), literal(signature), literal(requirements), literal(owner))
            for identity, kind, signature, requirements, owner in rows])
    lines.append("}  // namespace")
    return "\n".join(lines) + "\n"


def _emit_program_model_helpers(program: Any, authority: Any) -> str:
    """Emit device helpers required by model expressions in Program-inline kernels."""

    from pops.codegen.cpp_writer import _collect_eig_witnesses, _eig_witness_helpers

    expressions = []
    seen = set()
    for value in _all_ops(program):
        if value.op != "local_transform":
            continue
        node_model = model_for_node(authority, value)
        impl = _model_impl(node_model)
        name = value.attrs["transform"]
        identity = (id(impl), name)
        if identity in seen:
            continue
        seen.add(identity)
        declaration = impl._local_transforms[name]
        expressions.extend(declaration["expressions"])
        expressions.append(declaration["valid_if"])
    lines = _eig_witness_helpers(_collect_eig_witnesses(expressions), indent="")
    return ("\n".join(lines) + "\n") if lines else ""


def _emit_system_install(
    target: str,
    prelude: str,
    body: str,
    provider_plan_install: str,
    *,
    artifact_identity: str,
    route_manifest: str,
    program_name: str,
    persistent_manifest: str | None = None,
    maximum_bytes: int | None = None,
) -> str:
    """Emit only the install entry matching the artifact's declared runtime target.

    An AMR artifact may contain hierarchy-only providers. Emitting the uniform entry as well would
    compile the AMR prelude against the wrong topology provider and, more
    importantly, advertise a System route the artifact cannot execute.  The native loaders already
    resolve distinct mandatory symbols, so a mismatched target now fails by a missing entry instead
    of carrying a dead or partially compilable implementation.
    """
    if target != "system":
        return ""
    artifact = json.dumps(artifact_identity)
    name = json.dumps(program_name)
    route = json.dumps(route_manifest)
    if persistent_manifest is None:
        persistent_manifest = "pops.persistent-resource.manifest.v1"
    persistent = json.dumps(persistent_manifest)
    max_bytes = (
        "pops::runtime::program::kProgramResourcePlanUnknownExtent"
        if maximum_bytes is None else "UINT64_C(%d)" % int(maximum_bytes)
    )
    return (
        "namespace {\n"
        "\n"
        "struct ProgramCandidateState final {\n"
        "  std::shared_ptr<pops::runtime::program::ProgramExecutionServices<"
        "pops::kNativeDimension>> ctx_owner;\n"
        "  std::function<void(double)> step;\n"
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
        "  auto* state = static_cast<ProgramCandidateState*>(opaque);\n"
        "  try {\n"
        "    state->step(dt);\n"
        "  } catch (const pops::runtime::program::ProgramStepRejectSignal& signal) {\n"
        "    if (!state->ctx_owner->adopt_step_attempt_rejection(signal.record))\n"
        "      throw ProgramStepRejectPublishFailure{};\n"
        "  }\n"
        "}\n"
        "\n"
        "void program_candidate_destroy(void* opaque) noexcept {\n"
        "  delete static_cast<ProgramCandidateState*>(opaque);\n"
        "}\n"
        "\n"
        "static constexpr char kProgramCandidateArtifactIdentity[] = " + artifact + ";\n"
        "static constexpr char kProgramCandidateName[] = " + name + ";\n"
        "static constexpr char kProgramCandidateAbiKey[] = POPS_ABI_KEY_LITERAL;\n"
        "static constexpr char kProgramCandidateRouteManifest[] = " + route + ";\n"
        "static constexpr char kProgramCandidateBoundaryManifest[] = \"pops.boundary.manifest.v1\";\n"
        "static constexpr char kProgramCandidatePersistentResourceManifest[] = " + persistent + ";\n"
        "static constexpr char kProgramCandidateCheckpointIdentity[] = \"pops.checkpoint.identity.v1\";\n"
        "\n"
        "constexpr std::uint64_t kProgramCandidateCapabilities =\n"
        "    pops::runtime::program::kProgramCapabilitySchedules |\n"
        "    pops::runtime::program::kProgramCapabilityPersistentValues |\n"
        "    pops::runtime::program::kProgramCapabilityTransactions;\n"
        "constexpr std::uint64_t kProgramCandidateRequiredServices =\n"
        "    pops::runtime::program::kProgramServiceState |\n"
        "    pops::runtime::program::kProgramServiceFields |\n"
        "    pops::runtime::program::kProgramServiceSpatial |\n"
        "    pops::runtime::program::kProgramServiceHistory |\n"
        "    pops::runtime::program::kProgramServiceClock |\n"
        "    pops::runtime::program::kProgramServiceReduction |\n"
        "    pops::runtime::program::kProgramServiceTransaction |\n"
        "    pops::runtime::program::kProgramServicePersistentValues;\n"
        "\n"
        "}  // namespace\n"
        "\n"
        "bool program_candidate_prepare(void* opaque,\n"
        "    const pops::runtime::program::ProgramHostDescriptor* host,\n"
        "    pops::runtime::program::ProgramInstallDiagnostic* diagnostic) noexcept {\n"
        "  using namespace pops::runtime::program;\n"
        "  if (opaque == nullptr || host == nullptr || diagnostic == nullptr) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,\n"
        "                                     \"Program prepare requires state and host descriptors\");\n"
        "    return false;\n"
        "  }\n"
        "  if (!valid_program_host_descriptor(*host)) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,\n"
        "                                     \"Program install received an invalid host descriptor\");\n"
        "    return false;\n"
        "  }\n"
        "  if (host->native_dimension != static_cast<std::uint32_t>(pops::kNativeDimension) ||\n"
        "      host->runtime_kind != ProgramRuntimeKind::uniform ||\n"
        "      host->execution_lane != ProgramExecutionLane::host) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::unsupported_runtime,\n"
        "                                     \"Program artifact requires the native Uniform host runtime\");\n"
        "    return false;\n"
        "  }\n"
        "  auto* state = static_cast<ProgramCandidateState*>(opaque);\n"
        "  if (state->ctx_owner || state->step) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     \"Program candidate was prepared twice\");\n"
        "    return false;\n"
        "  }\n"
        "  try {\n"
        "    state->ctx_owner = pops::runtime::program::make_program_execution_provider<pops::kNativeDimension>(host->preparation);\n"
        "    auto& ctx = *state->ctx_owner;\n"
        + provider_plan_install + ("\n" if provider_plan_install else "")
        + prelude + "\n"
        "    state->step = [ctx_owner = state->ctx_owner](double dt) {\n"
        "      auto& ctx = *ctx_owner;\n"
        "      (void)dt;\n"
        "      ctx.begin_step(dt);\n" + body + "\n"
        "    };\n"
        "    return true;\n"
        "  } catch (const std::exception& exc) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     exc.what());\n"
        "    return false;\n"
        "  } catch (...) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     \"Program candidate preparation failed\");\n"
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
        "      host->runtime_kind != ProgramRuntimeKind::uniform ||\n"
        "      host->execution_lane != ProgramExecutionLane::host) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_host_descriptor,\n"
        "                                     \"Program install received an invalid host descriptor\");\n"
        "    return false;\n"
        "  }\n"
        "  *candidate = {};\n"
        "  try {\n"
        "    auto state = std::make_unique<ProgramCandidateState>();\n"
        "    ProgramCandidateDescriptor descriptor{};\n"
        "    descriptor.struct_size = static_cast<std::uint32_t>(sizeof(ProgramCandidateDescriptor));\n"
        "    descriptor.abi_version = kProgramInstallAbiVersion;\n"
        "    descriptor.native_dimension = static_cast<std::uint32_t>(pops::kNativeDimension);\n"
        "    descriptor.runtime_kind = ProgramRuntimeKind::uniform;\n"
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
        "    descriptor.maximum_bytes = " + max_bytes + ";\n"
        "    descriptor.context = state.get();\n"
        "    descriptor.prepare = &program_candidate_prepare;\n"
        "    descriptor.step = &program_candidate_step;\n"
        "    descriptor.dt_bound = nullptr;\n"
        "    descriptor.destroy = &program_candidate_destroy;\n"
        "    if (!valid_program_candidate_descriptor(descriptor)) {\n"
        "      write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::invalid_candidate,\n"
        "                                       \"Program artifact produced an invalid candidate descriptor\");\n"
        "      return false;\n"
        "    }\n"
        "    *candidate = descriptor;\n"
        "    (void)state.release();\n"
        "    return true;\n"
        "  } catch (...) {\n"
        "    write_program_install_diagnostic(diagnostic, ProgramInstallErrorCode::artifact_rejected,\n"
        "                                     \"Program artifact installation failed\");\n"
        "    return false;\n"
        "  }\n"
        "}\n"
    )


def _check_lowerable(
    program: Any,
    model: Any = None,
    field_plans: Any = None,
    *,
    target: str | None = None,
) -> None:
    """Raise NotImplementedError if the IR uses a construct the current codegen cannot lower yet,
    naming the offending construct (never a silent mis-lowering). @p model: the physical model that
    declares the named sources / linear sources; required for the Phase-4b ops.

    Multi-block (ADC-426): N ``T.state`` blocks + N ``T.commit`` are supported -- each op routes to
    its block's index (``_block_indices``). Validation: a block is committed AT MOST once (enforced
    at ``commit`` time); a read-only block (declared via ``T.state`` but never committed) is allowed
    (e.g. a passive field whose charge couples the others); a commit of a block that was never
    declared by ``T.state`` is rejected (an unknown-block commit cannot route to an index)."""
    _check_model_owner_dispatch(program, model)
    blocks = program._block_indices()
    for state_ref in list(program._commits) + list(getattr(program, "_post_sync_commits", {})):
        block = state_ref.block_ref
        if block not in blocks:
            raise ValueError(
                "commit of unknown block %r: no T.state(block[U]) declares it "
                "(declared blocks: %s)"
                % (block_name(block), sorted(block_name(item) for item in blocks))
            )
    _check_schedules_lowerable(program, target=target)
    for v in program._values:
        _check_op_lowerable(program, v, model, field_plans or {})
    # Dense local linear solves and prepared local nonlinear solves are specialized to the complete
    # manifest-sized system. Their stack storage and loops use exactly N; no central component-count
    # allowlist, truncating buffer, or scientific-model branch selects N.


# 'linear_source' is a pure NAME-reference SSA node (vtype 'operator'): it carries no runtime work
# (consumed by apply / solve_local_linear, which read the model coefficients), so it lowers to
# nothing -- always allowed, model or not. 'reduce' / 'compare' / 'while' are the ADC-404a control
# flow / reduction ops (lowered inline via pops::dot; no model needed). 'matrix_free_operator' /
# 'scalar_field' / 'vector_field' / 'laplacian' / 'gradient' / 'divergence' / 'solve_linear' are the ADC-405 / ADC-412
# matrix-free Krylov ops (the operator declaration carries an apply sub-block; solve_linear lowers to
# pops::*_solve; divergence is the centered FV divergence of a gradient field).

# Ops NOT wrapped in a per-node profile scope (ADC-459): they bind a reference or read a cached
# scalar and do no per-step numerical work, so timing them only adds always-zero noise to
# sim.profile_report(). Every other op that emits a statement is wrapped (rhs / solve_fields /
# linear_combine / source / apply / reductions / loops / Schur kernels / ...).


def _check_op_lowerable(program: Any, v: Any, model: Any, field_plans: Any) -> None:
    """Lowerability check for a single op (used for both the top-level walk and a while sub-block).
    Raises NotImplementedError / ValueError naming the offending construct (never a mis-lowering)."""
    node_model = (
        model_for_node(model, v)
        if model is not None and (v.block is not None or v.attrs.get("operator_handle") is not None)
        else model
    )
    _validate_matrix_free_contract(v, node_model)
    if v.op in _MODEL_OPS:
        if model is None:
            raise NotImplementedError(
                "emit_cpp_program cannot lower op '%s' (value '%s') without the physical model "
                "that declares its named source / linear source; pass model= "
                "(compile_problem threads it through)" % (v.op, v.name)
            )
        if v.op == "solve_local_nonlinear":  # recurse: the residual sub-block ops must lower too
            for w in v.attrs["residual_block"]:
                _check_op_lowerable(program, w, model, field_plans)
        return  # _emit_op lowers it from the model's symbolic coefficients
    if v.op not in _ALLOWED_OPS:
        raise NotImplementedError(
            "emit_cpp_program cannot lower op '%s' (value '%s') yet; supported ops are %s "
            "(+ %s with a model; nested control flow / Krylov are later phases)"
            % (v.op, v.name, sorted(_ALLOWED_OPS), sorted(_MODEL_OPS))
        )
    if v.op in ("coupled_rate", "solve_coupled_implicit"):
        # A coupled_rate (collisions / ionization, Spec 3 criterion 27) lowers to ONE multi-state
        # for_each_cell kernel (see _emit_coupled_rate_kernel). The lowering reaches the operator
        # body (its per-block component formulas) through the BOUND registry, and binds each input
        # state's cons names from that input's StateSpace -- so the operator must be bound and the
        # formulas must be cons-only (the MVP). Validate both here so a non-lowerable coupled_rate
        # fails loud naming ADC-457, never emits an undefined reference.
        _coupled_rate_components(program, v, model)
        return
    if v.op == "coupled_rate_out":
        # A pure projection of one block out of the coupled bundle: it emits nothing (its var
        # aliases that block's rate scratch). Lowerable iff its producing coupled_rate is (checked
        # when that node is walked); nothing to validate here.
        return
    if v.op in ("while", "range", "branch", "post_synchronization"):
        keys = ("true_block", "false_block") if v.op == "branch" else ("cond_block", "body_block")
        for key in keys:
            for w in v.attrs.get(key, []):
                _check_op_lowerable(program, w, model, field_plans)
        return
    if v.op == "matrix_free_operator":  # recurse into the apply sub-block (set by set_apply)
        if v.attrs.get("apply_block") is None:
            raise ValueError(
                "matrix_free_operator '%s' has no apply; call P.set_apply before lowering" % v.name
            )
        for w in v.attrs["apply_block"]:
            _check_op_lowerable(program, w, model, field_plans)
        return
    if v.op in ("solve_fields", "solve_fields_from_blocks"):
        # Every field solve is routed by the exact authenticated install plan. An OperatorHandle
        # may denote the model's default provider or any named provider; neither status is inferred
        # from a spelling or from ``field is None``. The resolved Case field owns discretization,
        # solver, BC, hierarchy and the native slot, so Program-only emission must refuse when that
        # authority is absent instead of silently selecting the historical default solver.
        field_ref = v.attrs.get("field")
        if field_ref is None:
            raise ValueError("field solve %r has no exact field identity" % v.name)
        from pops.codegen.program_emit_field_routes import resolved_field_route

        resolved_field_route(field_ref, field_plans)
        return
    if v.op == "rhs":
        named_fluxes = _named_fluxes(v)
        # ADC-430: flux=False is SOURCE-ONLY -- no -div F base. Named fluxes (a -div of selected
        # flux_terms) contradict "no flux": reject the combination loud rather than silently picking
        # one (request flux=True for named fluxes, or flux=False for a source-only stage).
        if not v.attrs.get("flux", True) and named_fluxes is not None:
            raise ValueError(
                "rhs '%s' sets flux=False (source-only) but also requests named fluxes %r; a "
                "source-only stage has no flux divergence -- drop fluxes= or set flux=True"
                % (v.name, named_fluxes)
            )
        if named_fluxes is not None:  # NAMED fluxes (ADC-419): need the model's flux_term coeffs
            if model is None:
                raise NotImplementedError(
                    "emit_cpp_program cannot lower rhs '%s' with named fluxes %r without the "
                    "physical model that declares them (m.flux_term); pass model= "
                    "(compile_problem threads it through)" % (v.name, named_fluxes)
                )
            impl_f = _model_impl(node_model)
            ft = impl_f._flux_terms
            for f in named_fluxes:
                if f not in ft:
                    raise ValueError(
                        "unknown flux_term '%s' in rhs '%s'; declared flux_terms: %s"
                        % (f, v.name, sorted(ft))
                    )
            # The named-flux path emits -div(selected fluxes) only (no ctx.rhs_into), so the model's
            # DEFAULT source would be silently dropped -- reject it (it must be requested as a named
            # source_term instead). The named sources below are still axpy'd on top.
            if getattr(impl_f, "_source", None):
                raise NotImplementedError(
                    "rhs with named fluxes %r needs a model whose default source is empty (no "
                    "m.source); rhs '%s' has a non-empty default source that the named-flux path "
                    "would drop (declare it as a source_term instead)" % (named_fluxes, v.name)
                )
        extra = [s for s in (v.attrs.get("sources") or []) if s != "default"]
        if not extra:
            return
        # A named source in an rhs reads the model's symbolic source_term coefficients (same as the
        # standalone 'source' op): lowering needs the model.
        if model is None:
            raise NotImplementedError(
                "emit_cpp_program cannot lower rhs '%s' with named sources %r without the "
                "physical model that declares them (m.source_term); pass model= "
                "(compile_problem threads it through)" % (v.name, extra)
            )
        impl = _model_impl(node_model)
        # ADC-425: the named sources are axpy'd on top of an EXPLICIT base. With "default" requested
        # the base is ctx.rhs_into (flux + the model's default/composite source); without it the base
        # is ctx.neg_div_flux_default_into (flux only). Either way the default source is folded in iff
        # the caller listed "default", so adding distinct named source_terms cannot double-count it --
        # the old "model default source must be empty" rejection is gone (the routing is now exact).
        for s in extra:
            if s not in impl._source_terms:
                raise ValueError(
                    "unknown source_term '%s' in rhs '%s'; declared source_terms: %s"
                    % (s, v.name, sorted(impl._source_terms))
                )
