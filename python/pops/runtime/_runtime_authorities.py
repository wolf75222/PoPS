"""Install resolved runtime authorities before native block construction.

This seam is intentionally protocol-driven: layout selection stays in ``_runtime_executor`` while
authorities describe the data the chosen engine must install.  A provider that cannot execute an
authority rejects it here, before native blocks freeze their configuration.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, cast


def _install_boundary_authorities(engine: Any, install_plan: Any) -> None:
    compiled_by_name = {row.name: row for row in install_plan.artifact.blocks}
    reports = {}
    native = getattr(engine, "_s", None)
    install = getattr(native, "_install_boundary_plan", None)
    install_state_route = getattr(native, "_install_block_state_route", None)
    from pops.runtime._component_execution_context import component_execution_data

    execution_data = component_execution_data(install_plan.execution_context)
    component_installers = {
        "apply_region_batch": getattr(native, "_install_ghost_boundary_component", None),
        "residual": getattr(native, "_install_field_boundary_residual_component", None),
        "jvp": getattr(native, "_install_field_boundary_jvp_component", None),
    }
    prepared = []
    state_routes: dict[str, str] = {}
    required_states: set[str] = set()
    required_fields: set[str] = set()
    for block in install_plan.artifact.plan.blocks:
        state_identities = tuple(getattr(block, "state_identities", ()))
        if len(state_identities) != 1:
            raise TypeError(
                "each installed native block requires one exact qualified state identity")
        state_identity = state_identities[0]
        if not isinstance(state_identity, str) or not state_identity:
            raise TypeError("native block state identity must be a non-empty qualified id")
        previous_state = state_routes.setdefault(state_identity, block.name)
        if previous_state != block.name:
            raise ValueError("one qualified state identity is routed to multiple native blocks")
        if not block.boundaries:
            continue
        if len(block.boundaries) != 1:
            raise ValueError(
                "one block must resolve to exactly one composed GhostProducerPlan; got %d "
                "boundary authorities for %r" % (len(block.boundaries), block.name)
            )
        authority = block.boundaries[0]
        protocol = getattr(authority, "runtime_boundary_data", None)
        if not callable(protocol):
            raise TypeError("resolved boundary authority lacks runtime_boundary_data(params)")
        first, second = protocol(install_plan.params), protocol(install_plan.params)
        if type(first) is not dict or first != second \
                or first.get("schema_version") != 1 \
                or first.get("authority_type") != "prepared_boundary_plan":
            raise TypeError(
                "runtime_boundary_data(params) must return one deterministic prepared v1 plan"
            )
        if not callable(install):
            raise NotImplementedError(
                "the selected native provider cannot install resolved ghost-production plans"
            )
        component = compiled_by_name[block.name].model
        ncomp = getattr(component, "n_vars", None)
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
            raise TypeError("compiled block lacks an authenticated positive n_vars")
        faces = first.get("faces")
        if not isinstance(faces, list) or len(faces) != 4 \
                or [row.get("ordinal") for row in faces] != [0, 1, 2, 3]:
            raise ValueError("prepared boundary plan must contain canonical xlo/xhi/ylo/yhi rows")
        types = [row.get("type") for row in faces]
        if any(value not in {"periodic", "foextrap", "dirichlet", "external"}
               for value in types):
            raise NotImplementedError("prepared boundary plan selected an unavailable face producer")
        values = []
        for comp in range(ncomp):
            for row in faces:
                row_values = row.get("values")
                if not isinstance(row_values, list) or len(row_values) != ncomp:
                    raise ValueError("prepared boundary face values must exactly cover every component")
                values.append(float(row_values[comp]))
        boundary_state_identity = _canonical_qualified_id(
            first.get("state"), where="prepared boundary state")
        if boundary_state_identity != state_identity:
            raise ValueError("prepared boundary state differs from its owning block route")
        required_depth = first.get("required_depth")
        if isinstance(required_depth, bool) or not isinstance(required_depth, int):
            raise TypeError("prepared boundary required_depth must be an exact integer")
        base_arguments = (
            block.name,
            str(first.get("identity")),
            required_depth,
            types,
            values,
            ncomp,
            list(first.get("omitted_interface_faces", [])),
            state_identity,
        )
        component_rows = first.get("component_regions", [])
        if not isinstance(component_rows, list):
            raise TypeError("prepared boundary component_regions must be a list")
        component_jobs = []
        for row in component_rows:
            if not isinstance(row, dict):
                raise TypeError("prepared boundary component region must be a dict")
            component_id = row.get("component_id")
            installed = install_plan.components.get(component_id)
            if installed is None:
                raise ValueError(
                    "boundary Handle %s requires exact component %r; it is not installed"
                    % (row.get("target", {}).get("qualified_id"), component_id)
                )
            if installed.component_manifest.token != row.get(
                    "component_manifest_identity"):
                raise ValueError(
                    "boundary Handle %s changed installed component manifest identity"
                    % row.get("target", {}).get("qualified_id")
                )
            interface = row.get("native_interface")
            if not isinstance(interface, dict) or interface != installed.interface.to_data() \
                    or row.get("interface_version") != installed.interface.version:
                raise ValueError(
                    "boundary Handle %s changed installed interface identity/version"
                    % row.get("target", {}).get("qualified_id")
                )
            if installed.native_handle is None:
                raise ValueError("boundary components must be loaded before native installation")
            region = row.get("region")
            if not isinstance(region, dict):
                raise TypeError("boundary component region descriptor must be a dict")
            parameters = row.get("parameters")
            if not isinstance(parameters, list) or any(
                    not isinstance(value, dict)
                    or set(value) != {"qualified_id", "value"}
                    for value in parameters):
                raise TypeError("boundary component parameter table is not canonical")
            operation = row.get("operation")
            if not isinstance(operation, str):
                raise TypeError("prepared boundary component operation must be text")
            install_component = component_installers.get(operation)
            if not callable(install_component):
                raise NotImplementedError(
                    "the selected native provider cannot install typed boundary operation %r"
                    % operation
                )
            for table_name in ("states", "directions", "fields", "outputs"):
                table = row.get(table_name)
                if not isinstance(table, list) or any(
                        not isinstance(identity, str) or not identity for identity in table):
                    raise TypeError(
                        "boundary component %s table must contain qualified identities"
                        % table_name)
            if row.get("state_identity") != state_identity:
                raise ValueError(
                    "boundary component primary state differs from its owning block route")
            required_states.update(row["states"])
            required_fields.update(row["fields"])
            if any(identity != state_identity for identity in row["directions"]):
                raise NotImplementedError(
                    "native boundary JVP directions must use the owning block state storage")
            if operation in {"residual", "jvp"} and len(row["outputs"]) != 1:
                raise NotImplementedError(
                    "native boundary residual/JVP currently requires one exact mutable output")
            component_jobs.append((
                install_component,
                block.name,
                installed.native_handle,
                row,
                "",
                "",
                execution_data,
            ))
        reports[block.name] = MappingProxyType(dict(first))
        prepared.append((base_arguments, tuple(component_jobs)))

    missing_states = required_states - set(state_routes)
    if missing_states:
        raise ValueError(
            "boundary component state dependencies lack exact native block routes: %s"
            % sorted(missing_states))
    available_fields = {}
    for field_plan in install_plan.artifact.plan.field_plans.values():
        unknown = getattr(getattr(field_plan, "operator", None), "unknown", None)
        identity = getattr(unknown, "qualified_id", None)
        options = getattr(field_plan, "native_options", None)
        slot = options.get("provider_slot") if isinstance(options, Mapping) else None
        if not isinstance(identity, str) or not identity or not isinstance(slot, str) or not slot:
            raise TypeError("resolved field plan lacks exact output identity/provider storage route")
        previous = available_fields.setdefault(identity, slot)
        if previous != slot:
            raise ValueError("one solved field identity has competing native provider routes")
    missing_fields = required_fields - set(available_fields)
    if missing_fields:
        raise ValueError(
            "boundary component field dependencies lack exact solved-field routes: %s"
            % sorted(missing_fields))
    field_routes = tuple(
        (identity, available_fields[identity]) for identity in sorted(required_fields))

    # Installation is an all-authorities transaction from Python's point of
    # view.  Validate and authenticate every block/component row before the
    # first native mutation, then roll the pre-build plan registry back if a
    # native constructor/prepare rejects any item.  Retrying therefore never
    # encounters a half-installed duplicate plan.
    discard = getattr(native, "_discard_boundary_plans", None)
    if state_routes and not callable(install_state_route):
        raise NotImplementedError(
            "the selected native provider cannot bind qualified block state storage")
    if state_routes and not callable(discard):
        raise NotImplementedError(
            "the selected native provider cannot roll back boundary authority installation")
    try:
        for state_identity, block_name in sorted(state_routes.items()):
            cast(Callable[..., Any], install_state_route)(block_name, state_identity)
        install_field_route = getattr(native, "_install_boundary_field_route", None)
        if field_routes and not callable(install_field_route):
            raise NotImplementedError(
                "the selected native provider cannot bind qualified boundary field storage")
        for field_identity, provider_slot in field_routes:
            cast(Callable[..., Any], install_field_route)(field_identity, provider_slot)
        for base_arguments, component_jobs in prepared:
            cast(Callable[..., Any], install)(*base_arguments)
            for job in component_jobs:
                installer, *arguments = job
                cast(Callable[..., Any], installer)(*arguments)
    except BaseException:
        cast(Callable[..., Any], discard)()
        raise
    engine._boundary_authorities = MappingProxyType(reports)


def _canonical_qualified_id(value: Any, *, where: str) -> str:
    if not isinstance(value, dict):
        raise TypeError("%s must be one canonical Handle identity" % where)
    identity = value.get("qualified_id")
    if not isinstance(identity, str) or not identity:
        raise TypeError("%s has no owner-qualified identity" % where)
    return identity


def _require_interface_component(install_plan: Any, binding: dict[str, Any]) -> Any:
    if not isinstance(binding, dict) or binding.get("operation") != "evaluate_faces":
        raise TypeError(
            "shared conservative flux requires one typed evaluate_faces component binding")
    component_id = binding.get("component_id")
    installed = install_plan.components.get(component_id)
    if installed is None:
        raise ValueError(
            "shared interface requires exact component %r; it is not installed" % component_id)
    if installed.component_manifest.token != binding.get("component_manifest_identity"):
        raise ValueError("shared interface changed installed component manifest identity")
    interface = binding.get("native_interface")
    # Detaching a compiled plan converts tuple carriers to their JSON-equivalent lists.  The
    # canonical encoder intentionally gives both the same ordered-array identity, so compare that
    # authenticated structure instead of Python container implementation details.
    from pops.identity import canonical_bytes

    if not isinstance(interface, dict) or canonical_bytes(interface) != canonical_bytes(
            installed.interface.to_data()) \
            or binding.get("interface_version") != installed.interface.version:
        raise ValueError("shared interface changed native interface identity/version")
    if installed.native_handle is None:
        raise ValueError("shared NumericalFlux component must be loaded before native installation")
    return installed


def finalize_runtime_authorities(engine: Any, install_plan: Any) -> None:
    """Install authorities that require materialized native block storage.

    Physical ghost plans are installed before block construction so generated closures capture them.
    A shared NumericalFlux is different: both exact endpoint MultiFabs must exist before the scheduler
    can prove their BoxArray, DistributionMapping and face geometry.  This finalizer is therefore called
    by the unified install seam after blocks/Program materialization and before the bind freeze.
    """
    from pops.runtime._component_execution_context import component_execution_data

    reports = getattr(engine, "_boundary_authorities", None)
    if reports is None:
        raise RuntimeError("post-block authority finalization lost pre-build boundary reports")
    native = getattr(engine, "_s", None)
    install = getattr(native, "_install_interface_flux_component", None)
    rows: dict[str, dict[str, Any]] = {}
    owners: dict[str, set[str]] = {}
    endpoint_owners: dict[str, dict[str, set[str]]] = {}
    for block_name, report in reports.items():
        bindings = report.get("interface_component_bindings", [])
        if not isinstance(bindings, list):
            raise TypeError("prepared interface_component_bindings must be a list")
        for row in bindings:
            if not isinstance(row, dict) or set(row) != {"interface", "component"}:
                raise TypeError("prepared interface component binding is not canonical")
            interface = row["interface"]
            identity = _canonical_qualified_id(
                interface.get("handle") if isinstance(interface, dict) else None,
                where="shared interface")
            previous = rows.setdefault(identity, row)
            if previous != row:
                raise ValueError(
                    "shared interface %s has competing runtime declarations" % identity)
            owners.setdefault(identity, set()).add(block_name)
        endpoints = report.get("interface_endpoints", [])
        if not isinstance(endpoints, list):
            raise TypeError("prepared interface_endpoints must be a list")
        for endpoint in endpoints:
            if not isinstance(endpoint, dict) or set(endpoint) != {"interface", "owned_sides"}:
                raise TypeError("prepared shared-interface endpoint row is not canonical")
            identity = endpoint["interface"]
            if not isinstance(identity, str) or not identity:
                raise TypeError("prepared shared-interface endpoint identity is invalid")
            sides = endpoint["owned_sides"]
            if not isinstance(sides, list) or any(side not in {"left", "right"} for side in sides):
                raise TypeError("prepared shared-interface endpoint sides are invalid")
            table = endpoint_owners.setdefault(
                identity, {"left": set(), "right": set()})
            for side in sides:
                table[side].add(block_name)
    if not rows:
        engine._interface_authorities = MappingProxyType({})
        return
    if not callable(install):
        raise NotImplementedError(
            "the selected native provider cannot install shared NumericalFlux components")

    block_layouts: dict[str, str] = {}
    for assignment in install_plan.artifact.layout_plan.assignments:
        if assignment.subject_kind != "block":
            continue
        name = assignment.subject.local_id
        if name in block_layouts:
            raise ValueError("native block has multiple LayoutPlan assignments")
        block_layouts[name] = assignment.layout.qualified_id
    block_names_provider = getattr(native, "block_names", None)
    if not callable(block_names_provider):
        raise TypeError("native shared-interface provider must expose block_names()")
    block_names = tuple(cast(Any, block_names_provider()))
    if len(block_names) != len(set(block_names)):
        raise ValueError("native block registry contains duplicate names")
    block_indices = {name: index for index, name in enumerate(block_names)}
    execution_data = component_execution_data(install_plan.execution_context)
    adaptive = {row.adaptive for row in install_plan.artifact.layout_plan.layouts}
    levels = (0,)
    if adaptive == {True}:
        hierarchy = install_plan.resolved_hierarchy.plan
        if hierarchy.level_count != 1:
            raise NotImplementedError(
                "shared interface runtime finalization requires one frozen AMR level")
    elif adaptive != {False}:
        raise ValueError("shared interface finalization requires one coherent layout capability")

    installed_reports = {}
    jobs = []
    for identity, row in sorted(rows.items()):
        interface = row["interface"]
        endpoints = []
        for side_name in ("left", "right"):
            side = interface.get(side_name)
            if not isinstance(side, dict):
                raise TypeError("shared interface %s endpoint is not canonical" % side_name)
            layout_id = _canonical_qualified_id(
                side.get("layout"), where="shared interface %s layout" % side_name)
            matches = endpoint_owners.get(
                identity, {"left": set(), "right": set()})[side_name]
            if len(matches) != 1:
                raise ValueError(
                    "shared interface %s BoundaryHandle must identify exactly one native block"
                    % side_name)
            endpoint = next(iter(matches))
            if block_layouts.get(endpoint) != layout_id:
                raise ValueError(
                    "shared interface %s endpoint layout differs from native block assignment"
                    % side_name)
            endpoints.append(endpoint)
        left, right = endpoints
        if owners[identity] != {left, right}:
            raise ValueError(
                "shared interface %s runtime ownership differs from its endpoint plans" % identity)
        if block_layouts[left] != block_layouts[right]:
            raise NotImplementedError(
                "native shared NumericalFlux requires co-located endpoint blocks in one layout")
        try:
            left_index, right_index = block_indices[left], block_indices[right]
        except KeyError as error:
            raise ValueError(
                "shared interface endpoint block %r was not materialized" % error.args[0]) from None
        installed = _require_interface_component(install_plan, row["component"])
        # Empty overrides are deliberate: LoadedComponent owns the authenticated
        # parameters/target JSON captured from the installed component manifest.
        # Boundary binding scalars travel independently in the typed invocation
        # request and must never replace that component preparation contract.
        parameters_json = ""
        target_json = ""
        for level in levels:
            jobs.append((
                left_index, right_index, level, installed.native_handle,
                interface, row["component"], parameters_json, target_json,
                execution_data,
            ))
        installed_reports[identity] = MappingProxyType({
            "left_block": left,
            "right_block": right,
            "levels": levels,
            "component_id": row["component"]["component_id"],
        })
    discard = getattr(native, "_discard_interface_flux_components", None)
    if jobs and not callable(discard):
        raise NotImplementedError(
            "the selected native provider cannot roll back shared interface installation")
    try:
        for job in jobs:
            cast(Callable[..., Any], install)(*job)
    except BaseException:
        cast(Callable[..., Any], discard)()
        raise
    engine._interface_authorities = MappingProxyType(installed_reports)


def _install_amr_provider_authorities(engine: Any, install_plan: Any) -> None:
    """Authenticate and install external AMR providers as one pre-build transaction."""

    providers = install_plan.amr_providers
    if not isinstance(providers, Mapping) or tuple(providers) != ("clustering", "tagger"):
        raise ValueError("adaptive runtime requires exact clustering and tagger providers")
    native = getattr(engine, "_s", None)
    from pops import interfaces
    from pops.runtime._component_execution_context import component_execution_data

    expected_interfaces = {
        "clustering": interfaces.Clustering.to_data(),
        "tagger": interfaces.Tagger.to_data(),
    }
    expected_builtin_ids = {
        "clustering": "pops.lib.amr::berger_rigoutsos",
        "tagger": "pops.lib.amr::symbolic_tagger",
    }
    layout_identity = install_plan.artifact.layout_plan.qualified_id
    execution_data = component_execution_data(install_plan.execution_context)
    jobs = []
    reports = {}
    resolved_tagging = getattr(
        getattr(install_plan, "bootstrap_plan", None), "tagging", None)
    resolved_tagging_identity = getattr(resolved_tagging, "qualified_id", None)
    for slot, frozen in providers.items():
        if not isinstance(frozen, Mapping):
            raise TypeError("AMR %s provider binding must be an immutable mapping" % slot)
        binding = dict(frozen)
        if binding.get("schema_version") != 1 \
                or not isinstance(binding.get("provider_identity"), str) \
                or not binding["provider_identity"] \
                or binding.get("layout_identity") != layout_identity \
                or binding.get("native_interface") != expected_interfaces[slot]:
            raise ValueError("AMR %s provider binding is incomplete or unauthenticated" % slot)
        if slot == "tagger":
            from pops._generated_component_interfaces import NATIVE_TAGGING_PROGRAM_ABI

            capability = binding.get("tagging_capability")
            maximum_stencil_terms = (
                capability.get("maximum_stencil_terms")
                if isinstance(capability, Mapping)
                else None
            )
            if not isinstance(capability, Mapping) \
                    or tuple(capability.get("candidate_outputs", ())) != tuple(
                        NATIVE_TAGGING_PROGRAM_ABI["candidate_outputs"]) \
                    or not set(capability.get("indicator_stencil_routes", ())) <= set(
                        NATIVE_TAGGING_PROGRAM_ABI["indicator_stencil_routes"]) \
                    or not capability.get("indicator_stencil_routes") \
                    or isinstance(maximum_stencil_terms, bool) \
                    or not isinstance(maximum_stencil_terms, int) \
                    or maximum_stencil_terms < 1 \
                    or maximum_stencil_terms \
                    > NATIVE_TAGGING_PROGRAM_ABI["maximum_stencil_terms"] \
                    or capability.get("non_finite_policy") \
                    != NATIVE_TAGGING_PROGRAM_ABI["non_finite_policy"] \
                    or capability.get("persistent_hysteresis") is not \
                    NATIVE_TAGGING_PROGRAM_ABI["persistent_hysteresis"] \
                    or not isinstance(binding.get("tagging_graph_identity"), str) \
                    or (resolved_tagging_identity is not None
                        and binding.get("tagging_graph_identity")
                        != resolved_tagging_identity):
                raise ValueError(
                    "AMR Tagger lacks the exact resolved candidate-program authority")
        provider_type = binding.get("provider_type")
        if provider_type == "builtin_amr_%s" % slot:
            if binding.get("provider_id") != expected_builtin_ids[slot] \
                    or any(name in binding for name in (
                        "component_id", "component_manifest_identity", "component")):
                raise ValueError("builtin AMR %s provider is not canonical" % slot)
        elif provider_type == "external_amr_%s" % slot:
            component_id = binding.get("component_id")
            installed = install_plan.components.get(component_id)
            if installed is None:
                raise ValueError(
                    "AMR %s provider requires exact component %r; it is not installed"
                    % (slot, component_id))
            if installed.component_manifest.token != binding.get(
                    "component_manifest_identity"):
                raise ValueError("AMR %s provider changed component manifest identity" % slot)
            if installed.interface.to_data() != binding.get("native_interface") \
                    or installed.interface.version != binding.get("interface_version"):
                raise ValueError("AMR %s provider changed native interface/version" % slot)
            if installed.native_handle is None:
                raise ValueError("AMR %s component must be loaded before installation" % slot)
            component = binding.get("component")
            if not isinstance(component, Mapping) \
                    or component.get("component_id") != component_id \
                    or component.get("component_manifest") != installed.component_manifest.token \
                    or component.get("interface") != installed.interface.to_data():
                raise ValueError("AMR %s provider lost its exact component declaration" % slot)
            if slot == "tagger":
                from pops.amr.providers import _normalize_tagger_capability
                from pops.identity.semantic import semantic_value

                if semantic_value(
                        binding.get("tagging_capability"),
                        where="installed AMR Tagger capability") != semantic_value(
                            _normalize_tagger_capability(
                                installed.runtime_contract.capabilities),
                            where="manifest AMR Tagger capability") \
                        or (resolved_tagging_identity is not None
                            and binding.get("tagging_graph_identity")
                            != resolved_tagging_identity) \
                        or not isinstance(binding.get("clock_identity"), str) \
                        or not binding["clock_identity"]:
                    raise ValueError(
                        "external AMR Tagger lacks its exact graph/capability/clock contract")
            installer = getattr(native, "_install_amr_%s_component" % slot, None)
            if not callable(installer):
                raise NotImplementedError(
                    "the selected native provider cannot install external AMR %s" % slot)
            jobs.append((installer, installed.native_handle, binding, execution_data))
        else:
            raise ValueError("AMR %s provider kind is not supported" % slot)
        reports[slot] = MappingProxyType(binding)

    discard = getattr(native, "_discard_amr_provider_components", None)
    if jobs and not callable(discard):
        raise NotImplementedError(
            "the selected native provider cannot roll back AMR provider installation")
    try:
        for installer, handle, binding, execution in jobs:
            cast(Callable[..., Any], installer)(handle, binding, execution)
    except BaseException:
        cast(Callable[..., Any], discard)()
        raise
    engine._amr_provider_authorities = MappingProxyType(reports)


def install_runtime_authorities(engine: Any, install_plan: Any) -> None:
    """Install every pre-build authority carried by one normalized install plan."""
    _install_boundary_authorities(engine, install_plan)
    adaptive = {row.adaptive for row in install_plan.artifact.layout_plan.layouts}
    if adaptive == {False}:
        return
    if adaptive != {True}:
        raise ValueError("runtime authorities require one coherent layout capability")

    _install_amr_provider_authorities(engine, install_plan)

    execution = install_plan.amr_execution
    protocol = getattr(execution, "runtime_execution_data", None)
    if not callable(protocol):
        raise TypeError("adaptive execution authority must implement runtime_execution_data()")
    first, second = protocol(), protocol()
    if type(first) is not dict or first != second \
            or set(first) != {"schema_version", "authority_type", "mode", "relations"} \
            or first.get("schema_version") != 2 \
            or first.get("authority_type") != "amr_execution":
        raise TypeError("AMR runtime_execution_data() must return one deterministic v2 dict")
    relations = first["relations"]
    if not isinstance(relations, list):
        raise TypeError("AMR execution relations must be a list")
    if first.get("mode") == "synchronous":
        nlevels = len(install_plan.resolved_hierarchy.plan.transitions) + 1
        relations = [
            {
                "parent_level": parent, "child_level": parent + 1,
                "temporal_ratio": {"numerator": 1, "denominator": 1},
                "remainder_policy": "integral_only",
            }
            for parent in range(nlevels - 1)
        ]
    elif first.get("mode") != "subcycled":
        raise ValueError("AMR execution mode must be subcycled or synchronous")
    expected = len(install_plan.resolved_hierarchy.plan.transitions)
    if len(relations) != expected:
        raise ValueError("AMR execution requires one temporal relation per hierarchy transition")
    for index, row in enumerate(relations):
        if (not isinstance(row, dict) or set(row) != {
                "parent_level", "child_level", "temporal_ratio", "remainder_policy"}):
            raise ValueError("AMR execution temporal relation has incomplete keys")
        ratio = row["temporal_ratio"]
        if (row["parent_level"] != index or row["child_level"] != index + 1
                or not isinstance(ratio, dict)
                or set(ratio) != {"numerator", "denominator"}
                or isinstance(ratio["numerator"], bool)
                or not isinstance(ratio["numerator"], int)
                or isinstance(ratio["denominator"], bool)
                or not isinstance(ratio["denominator"], int)
                or ratio["denominator"] <= 0 or ratio["numerator"] < ratio["denominator"]
                or row["remainder_policy"] not in {
                    "integral_only", "explicit_final_substep"}):
            raise ValueError("AMR execution temporal relation is not canonical")
        if (ratio["numerator"] % ratio["denominator"] != 0
                and row["remainder_policy"] == "integral_only"):
            raise ValueError("non-integral AMR temporal relation requires an explicit remainder")
    engine.set_temporal_relations(
        [int(row["temporal_ratio"]["numerator"]) for row in relations],
        [int(row["temporal_ratio"]["denominator"]) for row in relations],
        [str(row["remainder_policy"]) for row in relations],
    )
    installed_execution = dict(first)
    installed_execution["relations"] = [
        {
            **row,
            "temporal_ratio": dict(row["temporal_ratio"]),
        }
        for row in relations
    ]
    engine._amr_execution_authority = MappingProxyType(installed_execution)

    if install_plan.bootstrap_plan is not None:
        from pops.runtime._runtime_mesh_lowering import flow_bootstrap_tagging

        flow_bootstrap_tagging(
            engine, install_plan.bootstrap_plan, install_plan.params,
            clock_identity=install_plan.amr_providers["tagger"]["clock_identity"])


__all__ = ["finalize_runtime_authorities", "install_runtime_authorities"]
